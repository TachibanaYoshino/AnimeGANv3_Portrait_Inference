import os


def check_folder(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)