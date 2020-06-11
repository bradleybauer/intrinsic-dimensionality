import sys


def rprint(s):
    if type(s) != str:
        s = str(s)
    sys.stdout.write("\033[2K\033[1G\r")
    sys.stdout.flush()
    sys.stdout.write(s)
    sys.stdout.flush()
