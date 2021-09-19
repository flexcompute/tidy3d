import numpy as np

class Mode(object):
    """
    Eigenmode of a 2D cross-section in a 3D simulation, assuming translational 
    symmetry in the third dimesnion.
    """
    def __init__(self, E, H, neff, keff):
        """Construct.
        
        Parameters
        ----------
        E : array_like
            Electric field of the mode.
        H : array_like
            Magnetic field of the mode, must be same shape as ``E```.
        neff : float
            Real part of the effective index.
        keff : float
            Imaginary part of the effective index.
        """
        self.E = E
        self.H = H
        self.neff = neff
        self.keff = keff

        # Effective k-vector
        # TODO: this is taken to be purely real later on and scaled to 
        # self.neff, but generally, it can be complex.
        self.kvector = None

    def fields(self):
        return (self.E, self.H)

    def fields_to_center(self):
        """ Return the E and H fields interpolated at the center of the Yee 
        cells in the 2D cross-section. """

        Ex_roll = np.roll(self.E[0, :, :], -1, 1)
        Ex_roll[:, -1] = 0
        Ey_roll = np.roll(self.E[1, :, :], -1, 0)
        Ey_roll[-1, :] = 0
        Hx_roll = np.roll(self.H[0, :, :], -1, 0)
        Hx_roll[-1, :] = 0
        Hy_roll = np.roll(self.H[1, :, :], -1, 1)
        Hy_roll[:, -1] = 0

        E = np.zeros_like(self.E)
        H = np.zeros_like(self.H)
        E[0, :, :] = (self.E[0, :, :] + Ex_roll)/2
        E[1, :, :] = (self.E[1, :, :] + Ey_roll)/2
        H[0, :, :] = (self.H[0, :, :] + Hx_roll)/2
        H[1, :, :] = (self.H[1, :, :] + Hy_roll)/2

        return (E, H)

    def to_dict(self):
        """ Returns a dictionary with the Mode properties. """
        return {'E': self.E, 
                'H': self.H,
                'neff': self.neff,
                'keff': self.keff
                }

    def fix_efield_phase(self):
        """ Apply a global phase based on the point with the largest E-field 
        value. This value is fixed to be real and positive. For lossless, 
        localized modes, this global phase should then make all of E and H real.
        """

        ind_max = np.argmax(np.abs(self.E))
        Emax = self.E.ravel()[ind_max]
        phi = np.angle(Emax)
        self.E *= np.exp(-1j*phi)
        self.H *= np.exp(-1j*phi)