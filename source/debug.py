# I took this from http://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
# and: http://stackoverflow.com/questions/460586/simulating-a-local-static-variable-in-python
# All credit belongs there.
import sys
import traceback
import pdb


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        # we are NOT in interactive mode, print the exception
        traceback.print_exception(type, value, tb)
        print
        # then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more modern


class Debugger(object):
    def __init__(self):
        self.times_invoked = 0

    def __call__(self, times):
        if times <= self.times_invoked:
            return
        self.times_invoked += 1
        pdb.set_trace()


sys.excepthook = info
_DEBUG = Debugger()
