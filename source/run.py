# from core.debug import _DEBUG
from shear_heating.shear_heating import ShearHeating
from shear_heating.shear_heating_params import params


def main():
    experiments = [ShearHeating(params)]
    for e in experiments:
        e.compute()
        e.visualize()
        e.save()

if __name__ == '__main__':
    main()
