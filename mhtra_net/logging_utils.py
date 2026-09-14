"""
Central logging setup for MHTRA-Net.

Scripts (resample / train / evaluate) call `setup_logging()` once at startup;
every module gets its own logger via `get_logger(__name__)`.

Console output goes to **stdout** on purpose: tqdm writes its progress bars to
stderr, so the two never overwrite each other. An optional file handler mirrors
the same records to a log file.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level="INFO", log_file=None):
    """
    Configure the root logger. Safe to call more than once (handlers are reset).

    level    : "DEBUG" / "INFO" / "WARNING" / ... or a logging.* constant. It is
               set on the root logger (not just the handlers) so that
               `log.isEnabledFor(DEBUG)` guards really do skip the expensive
               debug formatting in hot loops.
    log_file : optional path; the directory is created if needed. Receives the
               same records as the console.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
    root.addHandler(console)

    if log_file:
        parent = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, FILE_DATEFMT))
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    logger = get_logger(__name__)
    logger.debug("logging configured: console=%s file=%s", logging.getLevelName(level), log_file or "-")
    return root


def get_logger(name):
    """Module-level logger. `mhtra_net.dataset` -> shown as `mhtra_net.dataset`."""
    return logging.getLogger(name)


def add_logging_args(parser, default_log_file=None):
    """Adds --log_level / --log_file to an argparse parser, so every script agrees."""
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="verbosity; DEBUG adds per-batch/per-file detail")
    parser.add_argument("--log_file", default=default_log_file,
                        help="mirror all logs to this file")
    return parser


def format_duration(seconds):
    """3725.4 -> '1h 02m 05s'; 65.2 -> '1m 05s'; 4.13 -> '4.1s'"""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_count(n):
    """1234567 -> '1,234,567'"""
    return f"{n:,}"


@contextmanager
def log_stage(logger, what, level=logging.INFO):
    """
    Brackets a step with start/finish lines and a duration, and logs a traceback
    if it raises, so a crash always says which stage died.

        with log_stage(log, "loading datasets"):
            ...
    """
    logger.log(level, "%s ...", what)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException:
        logger.exception("%s FAILED after %s", what, format_duration(time.perf_counter() - t0))
        raise
    logger.log(level, "%s done in %s", what, format_duration(time.perf_counter() - t0))


def log_environment(logger, device=None):
    """One-time dump of interpreter / library / hardware versions."""
    logger.info("python      %s", sys.version.split()[0])
    logger.info("executable  %s", sys.executable)
    logger.info("cwd         %s", os.getcwd())
    try:
        import numpy
        logger.info("numpy       %s", numpy.__version__)
    except ImportError:                                    # pragma: no cover
        logger.warning("numpy not importable")
    try:
        import torch
    except ImportError:                                    # pragma: no cover
        logger.warning("torch not importable")
        return
    logger.info("torch       %s (cuda build: %s)", torch.__version__, torch.version.cuda or "cpu-only")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("gpu         %s | %.1f GB | %d SMs",
                    props.name, props.total_memory / 1e9, props.multi_processor_count)
    else:
        logger.info("gpu         none available - running on CPU (training will be slow)")
    if device is not None:
        logger.info("device      %s", device)


def log_args(logger, args):
    """Logs every parsed argparse value, one per line, alphabetically."""
    logger.info("arguments:")
    for key in sorted(vars(args)):
        logger.info("    %-16s = %s", key, getattr(args, key))


def log_cuda_memory(logger, prefix=""):
    """Current / peak GPU allocation, if CUDA is in use. No-op on CPU."""
    try:
        import torch
    except ImportError:                                    # pragma: no cover
        return
    if not torch.cuda.is_available():
        return
    logger.info("%sgpu memory: %.2f GB allocated, %.2f GB peak reserved",
                prefix, torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9)
