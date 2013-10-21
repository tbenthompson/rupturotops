import shutil
import os
import os.path
from data_controller import DataController

data_root = '/home/tbent/projects/viscosaur/data/'


class Experiment(object):
    def __init__(self, params):
        self.params = params
        self.data = DataController()

    def compute(self):
        self._compute()

    def _compute(self):
        raise Exception('_compute is still a stub method')

    def visualize(self):
        self._visualize()

    def _visualize(self):
        raise Exception('_visualize is still a stub method')

    def save(self):
        self.data.save()

    def assign_data_location(self):
        folder_name = data_root + self.params.proj_name
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        existing_runs = os.listdir(folder_name)
        new_run_folder_name = folder_name + '/' + \
            self.lowest_open_space(self.params.run_name, existing_runs)
        os.mkdir(new_run_folder_name)
        self.params.data_loc = new_run_folder_name

    @staticmethod
    def lowest_open_space(prefix, runs_list):
        similar_runs = [run.replace(prefix, '') for run in
                        runs_list if run.startswith(prefix)]
        #chopped runs should only be numbers now, throw out the others
        final_runs = [int(run) for run in similar_runs if run.isdigit()]
        if len(final_runs) == 0:
            return prefix + '0'
        max_run = max(final_runs)
        return prefix + str(max_run + 1)


def test_lowest_open_space():
    result = Experiment.lowest_open_space('abc', ['abc1', 'abc2', 'abc3'])
    assert result == 'abc4'

    result = Experiment.lowest_open_space('abc', [])
    assert result == 'abc0'

    result = Experiment.lowest_open_space('abc', ['def'])
    assert result == 'abc0'


def test_assign_data_location():
    params = DataController()
    params['proj_name'] = 'test'
    params['run_name'] = 'test'
    e = Experiment(params)
    e.assign_data_location()
    dir_exists = os.path.exists(data_root + '/test/test0')
    assert dir_exists
    shutil.rmtree(data_root + '/test/test0')
