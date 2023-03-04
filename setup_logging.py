import logging
from pathlib import Path
from typing import Union

import daiquiri


def setup_logging(level: str, log_dirpath: Union[Path, str]):
    logging.getLogger("dcm_prep").setLevel(level)
    logging.getLogger("brain_mask").setLevel(level)
    logging_formatter = daiquiri.formatter.ColorFormatter(
        fmt=(
            "%(color)s"
            "%(asctime)s "
            "[PID %(process)d] "
            "[%(levelname)s] "
            "%(name)s.%(funcName)s.%(lineno)d -> %(message)s"
            "%(color_stop)s"
        )
    )
    daiquiri.setup(
        level=level,
        outputs=(
            daiquiri.output.Stream(formatter=logging_formatter),
            daiquiri.output.RotatingFile(
                filename="bratsaas.log",
                directory=log_dirpath,
                max_size_bytes=5 * 1024 * 1024,
                backup_count=2,
                formatter=logging_formatter,
            ),
        ),
    )
