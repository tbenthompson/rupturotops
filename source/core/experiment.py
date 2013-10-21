import datetime
import shutil
import os
import os.path
from data_controller import DataController, data_root


class Experiment(object):
    """
    The experiment class provides a general framework for defining a numerical
    experiment and running it.

    Any experiment should be subclassed from this "Experiment" cass. The functions
    to implement are "_initialize", "_compute" and "_visualize".

    The class provides methods to save data automatically to a central location.
    """
    def __init__(self, params):
        self.params = params
        self.data = DataController()
        self.data_loc = None
        self._assign_data_location()

        self._initialize()

    def _initialize(self):
        raise Exception('_initialize is still a stub method')

    def compute(self):
        """
        Abstract base for the computation of an experiment
        """
        return self._compute()

    def _compute(self):
        raise Exception('_compute is still a stub method')

    def visualize(self):
        """
        Abstract base for the visualization of an experiment
        """
        return self._visualize()

    def _visualize(self):
        raise Exception('_visualize is still a stub method')

    def save(self):
        """
        Save the data and the parameters
        """
        self.params.save(self.data_loc + '/params.pkl')
        self.data.save(self.data_loc + '/data.pkl')

    def _assign_data_location(self):
        """
        This assigns a data folder to the experiment. The folder is chosen
        as the data_root/proj_name/run_name# where # is the lowest number
        that hasn't already been chosen.
        """
        if self.data_loc is not None:
            return
        folder_name = data_root + '/' + self.params.proj_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_run_folder_name = folder_name + '/' + self.params.run_name + '_' + timestamp
        os.mkdir(new_run_folder_name)
        self.data_loc = new_run_folder_name

##############################################################################
# TESTS BELOW HERE
##############################################################################

test_params = DataController(bar=1, proj_name='test', run_name='test')


class ExperimentTester(Experiment):
    def _initialize(self):
        pass

    def _compute(self):
        self.data.abc = 1

    def _visualize(self):
        self.data.fgi = 2


def test_experiment():
    foo = ExperimentTester(test_params)
    assert foo.params.bar == 1
    foo.compute()
    assert foo.data.abc == 1
    foo.visualize()
    assert foo.data.fgi == 2
    shutil.rmtree(foo.data_loc)


def test_experiment_save():
    foo = ExperimentTester(test_params)
    foo.compute()
    foo.visualize()
    foo.save()
    root_loc = foo.data_loc
    params_saved = os.path.exists(root_loc + '/params.pkl')
    data_saved = os.path.exists(root_loc + '/data.pkl')
    assert params_saved is True
    assert data_saved is True
    shutil.rmtree(root_loc)


def test_assign_data_location():
    e = ExperimentTester(test_params)
    e._assign_data_location()
    root_loc = e.data_loc
    dir_exists = os.path.exists(root_loc)
    assert dir_exists
    correct_path = (data_root + '/test/test_') in root_loc
    assert correct_path is True
    shutil.rmtree(root_loc)
