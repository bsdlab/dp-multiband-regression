from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from mb_regression.main import run_multiband_regression
from mb_regression.utils.logging import logger


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)
    pcommand_map = {
        "STARTMULTIBAND": run_multiband_regression,
    }

    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="multiband_regression_module"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)
