import os
import csv
from datetime import datetime


class Logger:
    def __init__(self, filepath, filename, field_names):
        self.filepath = filepath
        self.filename = filename
        self.field_names = field_names

        self.logfile, self.logwriter = csv_log(
            file_name=os.path.join(filepath, filename + ".csv"), field_names=field_names
        )
        self.logwriter.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.field_names)):
            dict[self.field_names[i]] = args[i]
        self.logwriter.writerow(dict)

    def close(self):
        self.logfile.close()


def csv_log(file_name, field_names):
    assert file_name is not None
    assert field_names is not None
    logfile = open(file_name, "w")
    logwriter = csv.DictWriter(logfile, fieldnames=field_names)
    return logfile, logwriter


def get_timestamp():
    return datetime.now().strftime("%y-%m-%d.%H-%M-%S")
