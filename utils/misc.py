import logging
import sys


def normalize_param_name(name: str) -> str:
    """Strip torch.compile / DDP wrapper prefixes from a parameter name."""
    prefixes = ("_orig_mod.", "module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def setup_logging(log_file=None):
    logger = logging.getLogger("nv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger
