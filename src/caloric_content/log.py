import logging
import sys
from datetime import datetime
from pathlib import Path

import transformers

logger = logging.getLogger(__name__)


def setup_logging(config, logger_name="main"):
    logs_dir = Path(config.logs_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

    logs_file_path = logs_dir.joinpath(
        datetime.now().strftime(f"%Y_%m_%dT%H_%M_%S-{logger_name}.log")
    )

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    fh = logging.handlers.RotatingFileHandler(
        logs_file_path,
        maxBytes=10 * 2**20,
        backupCount=5,
        mode="w",
        encoding="utf-8",
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[ch, fh],
    )
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Configured logging to {logs_file_path}.")

    return None
