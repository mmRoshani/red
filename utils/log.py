import logging
import logging.config
import os
import os.path
import sys

from constants.framework import LOG_PATH


class Log:
    def __init__(self, log_file: str, model_type: str, distance_metric: str):

        os.makedirs(f"{LOG_PATH}/{model_type}/{distance_metric}", exist_ok=True)

        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                print(f"Old log file '{log_file}' deleted.")
            except PermissionError as _:
                print("Log file deletion can cause data lost, if you are sure please restart you session")

        self.log_instance = logging.getLogger("SAFE_PFL_LOGGER")
        self.log_instance.setLevel(logging.DEBUG)
        self.log_instance.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s, %(levelname)8s | %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        self.log_instance.addHandler(file_handler)

        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        self.log_instance.addHandler(screen_handler)

        self.log_instance.info("Logger object created successfully...")
        self.log_instance.warning(f"The {log_file} will be truncated at each run")

    def info(self, info: str):
        self.log_instance.info(info)
        self.flush()

    def warn(self, warn: str):
        self.log_instance.warning(warn)
        self.flush()

    def debug(self, debug: str):
        self.log_instance.debug(debug)
        self.flush()

    def critical(self, critical: str):
        self.log_instance.critical(critical)
        self.flush()

    def error(self, error: str):
        self.log_instance.error(error)
        self.flush()

    def flush(self):
        for handler in self.log_instance.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

    def close(self):
        self.log_instance.handlers.close()
