from core.debug import setup_debug
from shear_heating.shear_heating import ShearHeating
from shear_heating.shear_heating_params import params


def main():
    setup_debug()
    experiments = [ShearHeating(params)]
    for e in experiments:
        e.compute()
        e.visualize()
        e.save()

if __name__ == '__main__':
    main()
