# To be designed as follows:
#   functions receive an object with the relevant parameters
#   parameter categories are defined in other python files
#   the output values are saved in a specific location
#   the run is given a specific name
#   then, the data is saved in the relevant folder for that name
#   allow options for a name prefix and then the lowest number not taken


class DataController(dict):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, val):
        self[attr] = val


def test_data_controller():
    d = DataController()
    d['abc'] = 1
    assert d.abc == 1

    d.ghi = 2
    assert d['ghi'] == 2
