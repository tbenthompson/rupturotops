"""
This file contains general tests that didn't fit well in some other file
"""

from core.experiment import Experiment
from core.data_controller import DataController

class TestExperiment(Experiment):
    def _initialize(self):
        if self.params.bar == 1:
            self.initialized = True

    def _compute(self):
        self.computed = True

    def _visualize(self):
        self.visualized = True

params = DataController(bar=1)
experiment = TestExperiment

def test_run():
    from run import main
    exp = main('testing', '')
    assert(exp.initialized == True)
    assert(exp.computed == True)
    assert(exp.visualized == True)

