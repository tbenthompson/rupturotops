import os
import os.path
import cPickle
# To be designed as follows:
#   functions receive an object with the relevant parameters
#   parameter categories are defined in other python files
#   the output values are saved in a specific location
#   the run is given a specific name
#   then, the data is saved in the relevant folder for that name
#   allow options for a name prefix and then the lowest number not taken


class DataController(dict):
    def __init__(self, items=[]):
        for item in items:
            self[item[0]] = item[1]

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, val):
        self[attr] = val

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)

    def __reduce__(self):
        #makes the object pickleable
        items = [[k, self[k]] for k in self]
        inst_dict = vars(self).copy()
        for k in vars(DataController()):
            inst_dict.pop(k, None)
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return cPickle.load(f)


test_file = '../data/test/data'
def test_data_controller_access():
    d = DataController()
    d['abc'] = 1
    assert d.abc == 1

    d.ghi = 2
    assert d['ghi'] == 2


def test_data_save():
    if os.path.exists(test_file):
        os.remove(test_file)
    d = DataController()
    d.abc = 1
    d.defghi = DataController()
    d.defghi.abc = 2
    d.save(test_file)
    assert(os.path.exists(test_file))


def test_data_load():
    d = DataController()
    d.abc = 1
    d.save(test_file)
    e = DataController.load(test_file)
    assert(e.abc == 1)

    d.abc = 2
    d.defghi = DataController()
    d.defghi.abc = 3
    d.save(test_file)
    e = DataController.load(test_file)
    assert(e.abc == 2)
    assert(e.defghi.abc == 3)


def test_real_parameters():
    from material import wetdiabase
    wetdiabase.save(test_file)
    new = DataController.load(test_file)
    assert(new.activationenergy == wetdiabase.activationenergy)
