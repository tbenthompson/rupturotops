import numpy as np

class Derivator(object):
    def __init__(self, bc, reconstructor, riemann):
        pass

    # @autojit()
    def compute(self, t, now):
        now = self.bc.compute(t, now)
        recon_left = self.reconstructor.compute(now, 'left')
        recon_right = self.reconstructor.compute(now, 'right')

        rightwards_flux = self.riemann.compute(recon_left,
                                               recon_right,
                                               self.v)
        # The total flux should be the flux coming in from the right
        # minus the flux going out the left.
        leftwards_flux = -np.roll(rightwards_flux, -1)
        total_flux = rightwards_flux + leftwards_flux
        return total_flux[2:-2] / self.delta_x
