from core.data_controller import DataController
from experiments.fenics import ShearHeatingFenics

params = DataController()

experiment = ShearHeatingFenics(params)
