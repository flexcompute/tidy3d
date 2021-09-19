import numpy as np
import h5py

from ..constants import float_, pec_val
from ..utils import inside_box_coords, listify, log_and_raise
from ..utils.log import Tidy3DError
from ..grid import SubGrid
from .solver import compute_modes as get_modes
from .dot_product import dot_product

# from ...run.coefficients import get_mat_params

class ModePlane(object):
    """2D plane for computation of modes used in ModeSource and ModeMonitor.
    The coordinate system of the ModePlane.grid is rotated such that the 
    z-axis is normal to the plane.
    """
    def __init__(self, span, norm_ind):
        """Construct.
        
        Parameters
        ----------
        span : np.ndarray of shape (3, 2)
            (micron) Defines (xmin, xmax), (ymin, ymax), (zmin, zmax) of the 
            mode plane.
        norm_ind : int
            Specifies the normal direction. We must then also have 
            ``span[mode_ind, 0] = span[mode_ind, 1]``.
        """
        self.span = span
        self.norm_ind = norm_ind
        
        """ Everything is stored in axes oriented as 
        (in_plane1, in_plane2, normal). Array self.new_ax defines how to do the
        switching between simulation axes and ModePlane axes:
            sim_axes[self.new_ax] -> ModePlane axes
            mpl_axes[self.old_ax] -> Simulation axes
        """
        self.new_ax = [0, 1, 2]
        self.new_ax.pop(self.norm_ind)
        self.new_ax.append(self.norm_ind)
        self.old_ax = np.argsort(self.new_ax).tolist()

        # Grid is to be set later based on a Simulation.
        self.grid = None

        # Permittivity at the Yee grid locations
        self.eps_ex = None
        self.eps_ey = None
        self.eps_ez = None

        """List of modes, set by a call to ``compute_modes()``. The first list 
        dimension is equal to the number of sampling frequencies, while the 
        second dimension is the number of computed modes. Each mode is given by 
        a dictionary with the fields and propagation constants."""
        self.modes = [[]]
        self.freqs = []

    def _set_sim(self, sim, freqs):
        """ Set the grid of the ModePlane based on a global simulation grid.
        The ModePlane grid is rotated such that ``z`` is the normal direction.
        Also set the ModePlane frequencies and the ``modes`` attribute as a
        list of Nfreqs empty lists.
        """
        self.freqs = listify(freqs)
        self.modes = [[] for i in range(len(self.freqs))]
        indsx, indsy, indsz = inside_box_coords(self.span, sim.grid.coords)
        if np.any([inds[0]==inds[1] for inds in (indsx, indsy, indsz)]):
            raise Tidy3DError("Mode plane position is outside simulation domain.")

        """Array of shape (3, 2) of int defining the starting and stopping 
        index in the global simulation grid of the ModePlane span."""
        self.span_inds = np.array([[inds[0], inds[1]] for inds in (indsx, indsy, indsz)])

        # Cut the mode plane span if symmetries are applied
        self.symmetries = [0, 0]
        for i, d in enumerate(self.new_ax[:2]):
            self.symmetries[i] = sim.symmetries[d]
            if sim.symmetries[d] != 0:
                Nd = sim.grid.Nxyz[d]
                self.span_inds[d, 0] = max(Nd//2, self.span_inds[d, 0])

        # Space and time resolution from global grid.
        self.time_step = sim.dt

        self.grid = SubGrid(sim.grid, span_inds=self.span_inds)
        self.grid.moveaxis(self.new_ax, (0, 1, 2))


    def _get_eps_cent(self, sim, freq):
        """Get the (non-averaged) permittivity at the center of the Yee cells,
        in ModePlane axes, at a given frequency. Used for plotting.
        """

        sim_mesh = [self.grid.mesh[a] for a in self.old_ax]
        eps = sim._get_eps(sim_mesh, edges='in', freq=freq)
        eps = np.squeeze(eps, axis=self.norm_ind)

        # Return as shape (N_cross_ind1, N_cross_ind2)
        return eps


    def _set_yee_sim(self, sim):
        """Set the permittivity at the Yee grid positions by passing the 
        simulation in which the mode plane is embedded.
        """

        epses = []
        meshes = [self.grid.mesh_ex, self.grid.mesh_ey, self.grid.mesh_ez]
        comps = ['xx', 'yy', 'zz']

        for im, mesh in enumerate(meshes):
            eps_freqs = []
            # Mesh rotated back in simulation axis
            sim_mesh = [mesh[a] for a in self.old_ax]

            for freq in self.freqs:
                eps = sim._get_eps(sim_mesh, edges='average', freq=freq, syms=False,
                    pec_val=pec_val, component=comps[self.new_ax[im]])
                eps = np.squeeze(eps, axis=self.norm_ind)
                eps_freqs.append(eps)

            epses.append(np.stack(eps_freqs, axis=0))

        [self.eps_ex, self.eps_ey, self.eps_ez] = epses


    def _set_yee_sim1(self, sim):
        """Set the permittivity at the Yee grid positions by passing the 
        simulation in which the mode plane is embedded.
        """

        epses = []
        meshes = [self.grid.mesh_ex, self.grid.mesh_ey, self.grid.mesh_ez]

        for im, mesh in enumerate(meshes):
            eps_freqs = []
            # Mesh rotated back in simulation axis
            sim_mesh = [mesh[a] for a in self.old_ax]

            for freq in self.freqs:
                # eps = sim._get_eps(sim_mesh, edges='average', freq=freq, syms=False)
                eps, _, _ = get_mat_params(sim.structures, sim._mat_inds, sim_mesh,
                    component=self.new_ax[im])
                eps = np.squeeze(eps, axis=self.norm_ind)
                eps_freqs.append(eps)

            epses.append(np.stack(eps_freqs, axis=0))

        [self.eps_ex, self.eps_ey, self.eps_ez] = epses


    def _set_yee_arr(self, eps_yee):
        """Set the permittivity at the Yee grid positions by passing an 
        array of shape (Nfreqs, Nx, Ny, Nz, 3) in Simulation axes.
        """

        eps_new = np.moveaxis(eps_yee, 1 + np.array(self.new_ax), [1, 2, 3])
        self.eps_ex = eps_new[:, :, :, 0, self.new_ax[0]]
        self.eps_ey = eps_new[:, :, :, 0, self.new_ax[1]]
        self.eps_ez = eps_new[:, :, :, 0, self.new_ax[2]]


    def compute_modes(self, Nmodes, target_neff=None, pml_layers=(0, 0)):
        """ Compute the ``Nmodes`` eigenmodes in decreasing order of 
        propagation constant at every frequency in the list ``freqs``.
        """

        for (ifreq, freq) in enumerate(self.freqs):
            modes = self._compute_modes_ifreq(
                ifreq, Nmodes, target_neff, pml_layers)
            self.modes[ifreq] = modes 

    def _compute_modes_ifreq(self, ifreq, Nmodes, target_neff=None, pml_layers=[0, 0]):
        """ Compute the ``Nmodes`` eigenmodes in decreasing order of 
        propagation constant for frequency index ``ifreq``.
        """

        if self.grid is None:
            raise Tidy3DError("Mode plane has not been added to a simulation yet.")

        freq = self.freqs[ifreq]
        # Get permittivity. Slightly break the c1-c2 symmetry to avoid 
        # complex-valued degenerate modes.
        epses = [self.eps_ex[ifreq],
                 self.eps_ey[ifreq] + 1e-6,
                 self.eps_ez[ifreq]]

        # Get modes
        modes = get_modes(
            epses,
            freq,
            mesh_step=self.grid.mesh_step,
            pml_layers=pml_layers,
            num_modes=Nmodes,
            target_neff=target_neff,
            symmetries=self.symmetries,
            coords=self.grid.coords[:2])
        
        for mode in modes:
            # Normalize to unit power flux
            fields_cent = mode.fields_to_center()
            flux = dot_product(fields_cent, fields_cent, self.grid.coords)
            flux *= 2**np.sum([sym != 0 for sym in self.symmetries])
            mode.E /= np.sqrt(flux)
            mode.H /= np.sqrt(flux)
            # Make largest E-component real
            mode.fix_efield_phase()

        return modes