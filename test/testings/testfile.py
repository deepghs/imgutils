import os.path

TESTFILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'testfile'))


def get_testfile(filename, *segs):
    return os.path.normpath(os.path.join(TESTFILE_DIR, filename, *segs))
