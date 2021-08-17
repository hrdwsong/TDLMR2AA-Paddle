"""
    logger.py saves a global logger instance that should be initialized at the code start point. Later then it can be
    used in any file that import logger. Write msgs to both stdout and file + can disable stdout.
"""

import time
from datetime import datetime

logger_instance = None


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, "w")
        self.stdout_print = True
        self.init_time = time.time()
        self.print_format = "Time passed [minutes]: {:.2f}.     {}"
        self.log_print("execution date (d/m/y): {}".format(datetime.now().strftime("%d/%m/%Y, %H:%M:%S")))

    def enable_stdout_prints(self):
        self.stdout_print = True

    def disable_stdout_prints(self):
        self.stdout_print = False

    def log_print(self, msg):
        minutes_passed = (time.time() - self.init_time) / 60
        formatted_msg = self.print_format.format(minutes_passed, msg)
        if self.stdout_print:
            print(formatted_msg)
        self.log_file.write(formatted_msg + "\n")

    def flush(self):
        self.log_file.flush()

    def new_section(self):
        self.log_file.write("\n" * 3)

    # def close(self):
    #     self.log_print("Logger Terminated")
    #     self.log_file.close()


def init_log(log_path):
    global logger_instance
    logger_instance = Logger(log_path)


def log_print(msg):
    assert logger_instance is not None, "should initialize logger before using it. use init_log"
    logger_instance.log_print(msg)

def new_section():
    logger_instance.new_section()


def enable_stdout_prints():
    logger_instance.enable_stdout_prints()


def disable_stdout_prints():
    logger_instance.disable_stdout_prints()


def flush():
    logger_instance.flush()

#
# def close():
#     if logger_instance is not None:
#         logger_instance.close()
