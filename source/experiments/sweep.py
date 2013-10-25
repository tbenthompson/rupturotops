from core.experiment import Experiment


class Sweep(Experiment):
    """
    Sweep runs another experiment multiple time with different parameter values.

    By setting parallelize = True, Sweep will parallelize the
    loop that test all parameter values. Ensure that the experiment being tested
    does not parallelize already. You would get suboptimal behavior.

    Currently not yet implemented.
    """

    def __init__(self, sub_experiment, param_name, values):
        self.sub_experiment = sub_experiment
        self.param_name = param_name
        self.values = values

    def abc():
        pass
