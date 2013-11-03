import os
import os.path
import cPickle

data_root = '/home/tbent/projects/viscosaur/data'


class DataController(dict):
    """
    The class extends the python dictionary to support using friendlier syntax:
        data.key = value
    instead of
        data['key'] = value

    The class also provides pickling capacities to store data and load data.
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return super(DataController,
                     self).__getitem__(key)

    def __setitem__(self, key, value):
        return super(DataController,
                     self).__setitem__(key, value)

    def __delitem__(self, key):
        return super(DataController,
                     self).__delitem__(key)

    def __contains__(self, key):
        return super(DataController,
                     self).__contains__(key)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, val):
        self[attr] = val

    def __reduce__(self):
        # This function ensures that pickling is successful.
        items = [[k, self[k]] for k in self]
        inst_dict = vars(self).copy()
        # Throw away all the variables that exist for the bare class
        # where no parameters have been set. We just want to save parameters.
        for k in vars(DataController()):
            inst_dict.pop(k, None)
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def save(self, filename):
        #binary mode is important for pickling
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)

    @staticmethod
    def load(filename):
        #binary mode is important for pickling
        with open(filename, 'rb') as f:
            return cPickle.load(f)


##############################################################################
# TESTS BELOW HERE
##############################################################################

test_file = data_root + '/test/data'


def test_data_controller_in():
    a = DataController(bcd=1)
    exists = 'bcd' in a
    assert exists is True
    exists = 'ghi' in a
    assert exists is False


def test_data_controller_init():
    b = DataController([('a', 1)])
    assert b.a == 1
    c = DataController(foo='bar')
    assert c.foo == 'bar'


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
    from parameters.material import wetdiabase
    wetdiabase.save(test_file)
    new = DataController.load(test_file)
    assert(new.activation_energy == wetdiabase.activation_energy)
