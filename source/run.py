"""
This script receives a python parameters file as an input argument
and then loads the files and runs the experiments described in the
file.

The parameters file should be in the form "folder/file.py"
"""
from core.debug import setup_debug
import sys
from optparse import OptionParser


def main(package_name, load_file):
    setup_debug()
    _temp = __import__(package_name, globals(), locals(), fromlist=['experiment, params'])
    experiment = _temp.experiment(_temp.params)
    # Figure out how to reload data and view it...
    # if load_file is not None:
    #     experiment.load(load_file)
    #     experiment.visualize()
    #     return
    experiment.compute()
    experiment.visualize()
    experiment.save()
    return experiment

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-l", "--load", action="store", type="string", dest="load_file",
                      help='Load already computed data from a file and do not compute'
                           ' new results')
    options, args = parser.parse_args(sys.argv)
    # print options, args
    # print args[1][-3:]
    if len(args) < 2 or args[1][-3:] != '.py':
        print "Please provide a python file for the parameters."
        print "For example: 'python run.py blahblah_params.py'"
        sys.exit()
    package_name = args[1].replace('/', '.')[:-3]
    main(package_name, options.load_file)
