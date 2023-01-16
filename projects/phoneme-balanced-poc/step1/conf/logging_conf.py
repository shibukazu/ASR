import logging
import os
import sys
from pathlib import Path


def config_generator(dir_name: str):
    """
    logging config generator
    """
    os.makedirs(dir_name, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "minimal": {"format": "%(message)s"},
            "detailed": {
                "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "minimal",
                "level": logging.DEBUG,
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(dir_name, "info.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.INFO,
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": Path(dir_name, "error.log"),
                "maxBytes": 10485760,  # 1 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.ERROR,
            },
        },
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    }
