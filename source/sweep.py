from core.experiment import Experiment

class Sweep(Experiment):
    """
    Sweep runs another experiment multiple time with different parameter values.

    By setting parallelize = True, Sweep will parallelize the
    loop that test all parameter values. Ensure that the experiment being tested
    does not parallelize already. You would get suboptimal behavior.
    """
    def __init__(self, sub_exp, param_name, values):
        """
        sub_exp = the subordinate experiment to be tested
        param_name = the name of the parameter to be varied
        values = the values to assign to the parameter
        """
        self.sub_exp = sub_exp
        self.param_name = param_name
        self.values = values

    def abc():
        pass
