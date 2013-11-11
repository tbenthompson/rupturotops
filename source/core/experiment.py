import datetime
import shutil
import os
import os.path
from mpi4py import MPI
from core.data_controller import DataController, data_root
from core.debug import _DEBUG
assert(_DEBUG)


class Experiment(object):

    """
    The experiment class provides a general framework for defining a numerical
    experiment and running it.

    Any experiment should be subclassed from this "Experiment" cass. The functions
    to implement are "_initialize", "_compute" and "_visualize".

    The class provides methods to save data automatically to a central location.

    By convention a subclass should never modify the parameters passed to it. Use
    self.data.var_name or self.var_name depending on whether the data should be saved
    or not
    """

    def __init__(self, params=DataController()):
        self.proj_name = 'test'
        self.run_name = 'test'
        if 'proj_name' in params:
            self.proj_name = params.proj_name
        if 'run_name' in params:
            self.run_name = params.run_name
        if 'material' in params:
            self.material = params.material

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

    def load(self, filename):
        self.data = DataController.load(filename)

    def _assign_data_location(self):
        """
        This assigns a data folder to the experiment. The folder is chosen
        as the data_root/proj_name/run_name# where # is the lowest number
        that hasn't already been chosen.
        """
        if self.data_loc is not None:
            return
        if MPI.COMM_WORLD.Get_rank() is not 0:
            return
        folder_name = data_root + '/' + self.proj_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        new_run_folder_name = folder_name + '/' + \
            self.run_name + \
            '_' + timestamp
        # We intentionally ignore the case where the folder already exists.
        # This means data may be overwritten, but it is the user's resp
        # to make sure this doesn't happen.
        if not os.path.exists(new_run_folder_name):
            os.mkdir(new_run_folder_name)
        self.data_loc = new_run_folder_name

# ----------------------------------------------------------------------------
# TESTS BELOW HERE
# ----------------------------------------------------------------------------
class ExperimentTester(Experiment):

    def _initialize(self):
        pass

    def _compute(self):
        self.data.abc = 1

    def _visualize(self):
        self.data.fgi = 2


def test_experiment():
    foo = ExperimentTester(DataController(bar = 1))
    assert foo.params.bar == 1
    foo.compute()
    assert foo.data.abc == 1
    foo.visualize()
    assert foo.data.fgi == 2
    shutil.rmtree(foo.data_loc)


def test_experiment_save():
    foo = ExperimentTester()
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
    e = ExperimentTester()
    e._assign_data_location()
    root_loc = e.data_loc
    dir_exists = os.path.exists(root_loc)
    assert dir_exists
    correct_path = (data_root + '/test/test_') in root_loc
    assert correct_path is True
    shutil.rmtree(root_loc)
