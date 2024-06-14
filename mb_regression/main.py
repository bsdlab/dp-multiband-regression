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

    info = pylsl.StreamInfo(
        cfg_out["name"],
        cfg_out["type"],
        len(fb.ch_names) + 1,   # 
        cfg_out["nominal_freq_hz"],
        cfg_out["format"],
    )

    # enrich a channel name
    chns = info.desc().append_child("channels")
    for chn in fb.ch_names + ['model_output']:
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
    sw.connect_to_stream()
    # in_sfreq = sw.inlet.info().nominal_srate()
    #
    # Hard coded to 22000 kHz for now using the older dp AO module
    in_sfreq = 22_000 

    fb = FilterBank(
        bands=cfg["frequency_bands"],
        sfreq=in_sfreq,
        n_in_channels=len(cfg["stream_to_query"]["channels"]),
        # output="abs_ma",
        output="signal",
        filter_buffer_s=2,
        n_lookback=int(in_sfreq // 10),   # lookback for moving average 10th of a second like this
    )

    model = joblib.load("./configs/model.joblib")

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

    tlast = time.perf_counter_ns()
    while not stop_event.is_set():
        dt_s = (time.perf_counter_ns() - tlast) * 1e-9
        req_samples = int(dt_s * out_sfreq)

        if req_samples > 1:
            sw.update()
            tlast = time.perf_counter_ns()

            new_data = sw.unfold_buffer()[-sw.n_new :, ch_to_watch]
            new_times = sw.unfold_buffer_t()[-sw.n_new :]
            fb.filter(new_data, new_times)

            # we might have additional samples after decimating since we floor
            # the frequency fb.sfreq/freq_hz
            if qfactor > 1:
                try:
                    xf = decimate(fb.get_data(), qfactor, axis=0, ftype="fir")[
                        -req_samples:
                    ]
                except Exception as e:
                    # With iir filter design for decimate this broke often
                    logger.error(f"Error in decimation: {e}")
                    breakpoint()
            else:
                xf = fb.get_data()[-req_samples:]

            for x in xf:
                # transform the x (features) to the model
                xl = np.log10(x)
                y = model.predict(xl)

                # push features and label for reconstruction
                outlet.push_sample(np.hstack([xl , [y]]))

            sw.n_new = 0
            fb.n_new = 0

        else:
            sleep_s(0.8 * (1 / out_sfreq - dt_s))


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
    Fire(run_bandpass_filter)
