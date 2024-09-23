"""
Xplore DS :: File Tools Package
"""

import os


def create_folder(folder_path: str) -> bool:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True


def get_name_and_extension_from_file(filename: str) -> list:
    return os.path.splitext(filename)


def get_filename_from_path(path: str) -> str:
    from pathlib import Path

    return Path(path).name


def load_parameters_from_file(path_file: str) -> dict:

    try:
        with open(path_file) as json_file:
            data = json.load(json_file)
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise

    return data


def get_nrows_from_file(filepath) -> int:
    f = open(filepath)
    try:
        lines = 1
        buf_size = 1024 * 1024
        read_f = f.read  # loop optimization
        buf = read_f(buf_size)

        # Empty file
        if not buf:
            return 0

        while buf:
            lines += buf.count("\n")
            buf = read_f(buf_size)
    finally:
        f.close()
    return lines
