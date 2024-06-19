import numpy as np
import joblib
import threading
import time

import pylsl
import tomli
from dareplane_utils.general.time import sleep_s
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from scipy.signal import decimate

from dareplane_utils.signal_processing.filtering import FilterBank
from mb_regression.utils.logging import logger


def init_lsl_outlet(cfg: dict, fb: FilterBank) -> pylsl.StreamOutlet:
    """Initialize the LSL outlet"""

    cfg_out = cfg["lsl_outlet"]

    channels = [f"{cn}_{band}" for cn in fb.ch_names for band in fb.zis.keys()]
    info = pylsl.StreamInfo(
        cfg_out["name"],
        cfg_out["type"],
        len(channels) + 1,   # 
        cfg_out["nominal_freq_hz"],
        cfg_out["format"],
    )

    # enrich a channel name
    chns = info.desc().append_child("channels")
    for chn in channels + ['model_output']:
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"{chn}")
        ch.append_child_value("unit", "AU")
        ch.append_child_value("type", "filter_bank")
        ch.append_child_value("scaling_factor", "1")

    outlet = pylsl.StreamOutlet(info)
    return outlet


def process_loop(
    stop_event: threading.Event = threading.Event(),
):
    """Process the given pipeline in a loop with a given freq"""

    cfg = tomli.load(open("./configs/config.toml", "rb"))
    sw = StreamWatcher(
        name=cfg["stream_to_query"]["stream"],
        buffer_size_s=cfg["stream_to_query"]["buffer_size_s"],

    )
    sw.chunk_buffer_size = 1024 * 512    # Increase for pull from LSL
    sw.connect_to_stream()

    # in_sfreq = sw.inlet.info().nominal_srate()
    #
    # Hard coded to 22000 kHz for now using the older dp AO module
    in_sfreq = 22000 

    fb = FilterBank(
        bands=cfg["frequency_bands"],
        sfreq=in_sfreq,
        n_in_channels=len(cfg["stream_to_query"]["channels"]),
        # output="abs_ma",
        output="signal",
        filter_buffer_s=2,
        n_lookback=int(in_sfreq // 10),   # lookback for moving average 10th of a second like this
    )
    # model = joblib.load("./configs/model.joblib")
    model = joblib.load("./configs/model_day3.joblib")

    outlet = init_lsl_outlet(cfg, fb)
    ch_to_watch = cfg["stream_to_query"]["channels"]

    out_sfreq = cfg["lsl_outlet"]["nominal_freq_hz"]
    qfactor = max(int(in_sfreq // out_sfreq), 1)

    tlast = time.perf_counter_ns()

    # Warmup
    while (time.perf_counter_ns() - tlast) * 1e-9 < cfg["others"][
        "warm_up_time_s"
    ]:
        sleep_s(0.1)
        sw.update()
        new_data = sw.unfold_buffer()[-sw.n_new :, ch_to_watch]
        new_times = sw.unfold_buffer_t()[-sw.n_new :]
        fb.filter(new_data, new_times)

    logger.debug("Filter Warmup finished")

    sent_samples = 0
    start_time = pylsl.local_clock()

    while not stop_event.is_set():
        cycle_start = pylsl.local_clock()

        dt_s = cycle_start - start_time

        req_samples = int(dt_s * out_sfreq) - sent_samples

        if req_samples > 1:
            sw.update()
            new_data = sw.unfold_buffer()[-sw.n_new :, ch_to_watch]
            new_times = sw.unfold_buffer_t()[-sw.n_new :]

            fb.filter(new_data, new_times)

            # # we might have additional samples after decimating since we floor
            # the frequency fb.sfreq/freq_hz
            # if qfactor > 1:
            #     try:
            #         # Decimate becomes very slow once a lot of data accumulates
            #         xf = decimate(fb.get_data(), qfactor, axis=0, ftype="fir")[
            #             -req_samples:
            #         ]

            #         logger.debug(f"Decimation done")
            #     except Exception as e:
            #         # With iir filter design for decimate this broke often
            #         logger.error(f"Error in decimation: {e}")
            #         breakpoint()
            # else:
            #    xf = fb.get_data()[-req_samples:]
            
            # How about pushing the mean for n samples? rectified signal?
            #
            xf = np.abs(fb.get_data()[-req_samples:]).mean(axis=0).reshape(1, -1)

            xfl = np.nan_to_num(np.log10(xf)).reshape(xf.shape[0], -1)

            preds = model.predict(xfl)

            data = np.hstack([xfl, preds.reshape(-1, 1)])

            # outlet.push_chunk(data)
            for _ in range(req_samples):
                # Will stack: ch1_band1, ch1_band2, ch1_band3, ..., chn_bandm, prediction
                # Rectified in samples for now, later maybe chunk and mean per chunk?
                outlet.push_sample(data[0])

            sent_samples += req_samples

            sw.n_new = 0
            fb.n_new = 0


        else:
            tsleep = (1 / out_sfreq)
            # keeping the clock more simple for better resource usage
            time.sleep(tsleep)
            # sleep_s(tsleep)


def run_multiband_regression() -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(
        target=process_loop,
        kwargs={"stop_event": stop_event},
    )

    logger.debug(f"Created {thread=}")
    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    logger.setLevel(10)
    Fire(run_multiband_regression)
