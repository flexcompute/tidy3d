import queue

import numpy as np

import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
from multiprocessing import Pool, cpu_count
import time

ITERATIONS = 5

# !!! td.CustomMedium can't be extended with additional fields
class CellMedium(td.CustomMedium):
    cell_permittivity: np.ndarray

class CellBox(td.Box):
    def split(self, nx: int, ny: int, nz: int):
        sx, sy, sz = self.size # type: ignore
        dx = sx / nx
        dy = sy / ny
        dz = sz / nz
        boxes = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = self.center[0] - sx / 2 + (i + 0.5) * dx
                    y = self.center[1] - sy / 2 + (j + 0.5) * dy
                    z = self.center[2] - sz / 2 + (k + 0.5) * dz
                    boxes.append(CellBox(center=(x, y, z), size=(dx, dy, dz)))
        return boxes

    def split_evenly(self):
        return self.split(2, 2, 2)

    def volume(self):
        return super().volume()

    def vertices(self)-> np.ndarray:
        (x0, y0, z0), (x1, y1, z1) = self.bounds
        return np.array([
            (x0, y0, z0),
            (x0, y0, z1),
            (x0, y1, z0),
            (x0, y1, z1),
            (x1, y0, z0),
            (x1, y0, z1),
            (x1, y1, z0),
            (x1, y1, z1),
        ]) * (1.0 - 1e-5)

class CellGrid(td.Grid):
    """Defines a grid for the simulation domain."""

    def cell_at(self, x: int, y: int, z: int) -> CellBox:
        """Returns the cell at the given coordinates."""
        x = int(x)
        y = int(y)
        z = int(z)
        if x < 0 or x > self.num_cells[0]:
            raise ValueError(f"x={x} is out of bounds")
        if y < 0 or y > self.num_cells[1]:
            raise ValueError(f"y={y} is out of bounds")
        if z < 0 or z > self.num_cells[2]:
            raise ValueError(f"z={z} is out of bounds")
        lx = self.boundaries.x[x + 1] - self.boundaries.x[x]
        ly = self.boundaries.y[y + 1] - self.boundaries.y[y]
        lz = self.boundaries.z[z + 1] - self.boundaries.z[z]
        cx = self.boundaries.x[x] + lx / 2
        cy = self.boundaries.y[y] + ly / 2
        cz = self.boundaries.z[z] + lz / 2
        return CellBox(center=(cx, cy, cz), size=(lx, ly, lz))

    def n_cells(self):
        return np.prod(self.num_cells)

    # column order
    def decode_cell(self, cell: int):
        if cell < 0 or cell >= self.n_cells():
            raise ValueError(f"cell={cell} is out of bounds")

        nx, ny, _ = self.num_cells
        x = cell % nx
        y = (cell // nx) % ny
        z = cell // (nx * ny)
        return x, y, z

    def encode_cell(self, x: int, y: int, z: int):
        nx, ny, nz = self.num_cells
        assert 0 <= x < nx
        assert 0 <= y < ny
        assert 0 <= z < nz
        return x + nx * (y + ny * z)

    @staticmethod
    def reshape(linear: np.ndarray, shape: tuple[int, int, int])->np.ndarray:
        return linear.reshape(shape, order='F')

    def adjacent_cell_inds(self, x: int, y: int, z: int)->list[int]:
        """Returns the adjacent cells to the given grid point."""
        cells = []
        for dx in [-1, 0]:
            for dy in [-1, 0]:
                for dz in [-1, 0]:
                    if (
                        0 <= x + dx < self.num_cells[0]
                        and 0 <= y + dy < self.num_cells[1]
                        and 0 <= z + dz < self.num_cells[2]
                    ):
                        cells.append(self.encode_cell(x + dx, y + dy, z + dz))
        return cells



class SpatialMediumCreator:
    def __init__(self, box: td.Box, dl: float, wave_guide: td.Structure):
        self.box = box
        self.dl = dl
        self.wave_guide = wave_guide

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        threshold =  cell.size[0] / 2 ** ITERATIONS
        while not q.empty():
            tmp_cell = q.get()
            if not tmp_cell.intersects(geometry, strict_inequality=[True, True, True]):
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            if all(geometry.inside(**vertices)):
                filled_cells.append(tmp_cell)
            else:
                # split the tmp_cell if necessary
                if tmp_cell.size[0] > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v

    def create(self) -> td.CustomMedium:
        def linespace(min_val, max_val, dl):
            if min_val > max_val:
                raise ValueError("min_val must be less than or equal to max_val")
            num_points = int((max_val - min_val) / dl) + 1
            return np.linspace(min_val, max_val, num_points)

        def make_grid(lx, ly, lz, dl) -> CellGrid:
            x = linespace(-lx / 2, lx / 2, dl)
            y = linespace(-ly / 2, ly / 2, dl)
            z = linespace(-lz / 2, lz / 2, dl)
            boundaries = td.Coords(x=x, y=y, z=z)
            return CellGrid(boundaries=boundaries)

        def calc_cell_permittivity(cell: CellBox, wave_guide: td.Structure):
            geometry = wave_guide.geometry
            medium = wave_guide.medium
            eps = medium.eps_model(None)
            p = SpatialMediumCreator.calc_cell_factor(cell, geometry)
            return (1-p) * 1.0 + p * eps.real


        box = self.box
        wave_guide = self.wave_guide
        dl = self.dl

        bg_permittivity = wave_guide.background_permittivity or 1.
        cell_grid = make_grid(box.size[0], box.size[1], box.size[2], dl)
        cell_permittivity = np.full(cell_grid.n_cells(), bg_permittivity)
        total_cells = cell_grid.n_cells()
        for idx in range(total_cells):
            x, y, z = cell_grid.decode_cell(idx)
            cell = cell_grid.cell_at(x, y, z)
            cell_permittivity[idx] = calc_cell_permittivity(
                cell, wave_guide) or bg_permittivity

        # initial grid points with cell permittivity
        boundaries = cell_grid.boundaries
        data = np.zeros((boundaries.x.size, boundaries.y.size, boundaries.z.size))
        for x in range(boundaries.x.size):
            for y in range(boundaries.y.size):
                for z in range(boundaries.z.size):
                    adj_cells = cell_grid.adjacent_cell_inds(x, y, z)
                    vals = [cell_permittivity[c] for c in adj_cells]
                    data[x, y, z] = np.average(vals)

        permittivity = SpatialDataArray(data, coords=boundaries.to_dict)
        return td.CustomMedium(permittivity=permittivity, conductivity=None)

class SpatialMediumCreator2:
    def __init__(self, sim: td.Simulation):
        self._sim = sim

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        x, y, z = cell.size[0], cell.size[1], cell.size[2]
        min_size = min(x, y, z)
        threshold =  min_size / 2 ** ITERATIONS
        while not q.empty():
            tmp_cell = q.get()
            b1 = tmp_cell.bounds
            b2 = geometry.bounds
            b3 = td.Box.bounds_intersection(b1, b2)
            size = tuple((pt_max - pt_min) for pt_min, pt_max in zip(b3[0], b3[1]))
            center = b3[0] + np.array(size) / 2
            if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            in_ids = geometry.inside(**vertices)
            if in_ids.all():
                filled_cells.append(tmp_cell)
            else:
                min_size = min(tmp_cell.size[0], tmp_cell.size[1], tmp_cell.size[2])
                # BUG:
                if (in_ids.any() or (np.array(b2) == np.array(b3)).all() or tmp_cell.inside([center[0]], [center[1]], [center[2]])) and min_size > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)
                # if min_size > threshold:
                #     for c in tmp_cell.split_evenly():
                #         q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v


    def create(self) -> tuple[td.CustomMedium, np.ndarray]:
        def make_grid() -> CellGrid:
            return CellGrid(boundaries=self._sim.grid.boundaries)

        def calc_cell_permittivity(cell: CellBox, wave_guide: td.Structure):
            geometry = wave_guide.geometry
            medium = wave_guide.medium
            eps = medium.eps_model(None)
            p = SpatialMediumCreator2.calc_cell_factor(cell, geometry)
            return (1-p) * 1.0 + p * eps.real

        cell_grid = make_grid()
        cell_permittivity = np.full(cell_grid.n_cells(), 0.0)
        total_cells = cell_grid.n_cells()
        print(f"--cell shape: {cell_grid.num_cells}={cell_grid.n_cells()}")
        wave_guide = self._sim.structures[0] # HACK: assuming only one structure
        start = time.perf_counter()
        for idx in range(total_cells):
            x, y, z = cell_grid.decode_cell(idx)
            cell = cell_grid.cell_at(x, y, z)
            cell_permittivity[idx] = calc_cell_permittivity(
                cell, wave_guide) or 1.0
        print(f"Calc Time elapsed: {time.perf_counter() - start}")

        start = time.perf_counter()
        boundaries = cell_grid.boundaries
        data = np.zeros((boundaries.x.size, boundaries.y.size, boundaries.z.size))
        for x in range(boundaries.x.size):
            for y in range(boundaries.y.size):
                for z in range(boundaries.z.size):
                    adj_cells = cell_grid.adjacent_cell_inds(x, y, z)
                    vals = [cell_permittivity[c] for c in adj_cells]
                    data[x, y, z] = np.average(vals)
        print(f"Average Time elapsed: {time.perf_counter() - start}")

        permittivity = SpatialDataArray(data, coords=boundaries.to_dict)
        return td.CustomMedium(permittivity=permittivity), cell_grid.reshape(cell_permittivity, cell_grid.num_cells)

from typing import Tuple, Callable, TypeAlias
# weight and medium
CellProp: TypeAlias = Tuple[float, td.Medium]
# for every structure, the cell properties
CellInfo: TypeAlias = list[CellProp]

ValueType = float

# function to calculate the effective permittivity, return the effective permittivity
CellFunc: TypeAlias = Callable[[CellInfo, ValueType], ValueType]
# effective permittivity for cell index
CellEff: TypeAlias = Tuple[int, ValueType]

class SpatialMediumCreator3:
    """
    使用多进程并行
    """
    MAX_ITERATIONS = None

    def __init__(self, sim: td.Simulation):
        self._sim = sim

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        x, y, z = cell.size[0], cell.size[1], cell.size[2]
        min_size = min(x, y, z)
        threshold = min_size / 2**(cls.MAX_ITERATIONS or ITERATIONS)
        while not q.empty():
            tmp_cell = q.get()
            b1 = tmp_cell.bounds
            b2 = geometry.bounds
            b3 = td.Box.bounds_intersection(b1, b2)
            size = tuple(
                (pt_max - pt_min) for pt_min, pt_max in zip(b3[0], b3[1]))
            center = b3[0] + np.array(size) / 2
            if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            in_ids = geometry.inside(**vertices)
            if in_ids.all():
                filled_cells.append(tmp_cell)
            else:
                min_size = min(tmp_cell.size[0], tmp_cell.size[1],
                               tmp_cell.size[2])
                # BUG:
                if (in_ids.any() or (np.array(b2) == np.array(b3)).all() or
                        tmp_cell.inside([center[0]], [center[1]],
                                        [center[2]])) and min_size > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)
                # if min_size > threshold:
                #     for c in tmp_cell.split_evenly():
                #         q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v

    @staticmethod
    def calc_cell_permittivity(args):
        idx, structs, cell_grid, bg_permittivity = args
        x, y, z = cell_grid.decode_cell(idx)
        cell = cell_grid.cell_at(x, y, z)
        cell_info: CellInfo = []
        for struct in structs:
            geometry = struct.geometry
            medium = struct.medium
            p = SpatialMediumCreator3.calc_cell_factor(cell, geometry)
            if p == 0:
                continue
            cell_info.append((p, medium))

        def calc_eff(cell_info: CellInfo, bg_permittivity: ValueType):
            if not cell_info:
                return bg_permittivity
            ci = cell_info[-1]
            p, medium = ci
            eps = medium.eps_model(None)
            return (1 - p) * bg_permittivity + p * eps.real

        return idx, calc_eff(cell_info, bg_permittivity)

    @staticmethod
    def average_cell_permittivity(args):
        point, cell_grid, cell_permittivity = args
        x, y, z = point
        adj_cells = cell_grid.adjacent_cell_inds(x, y, z)
        vals = [cell_permittivity[c] for c in adj_cells]
        return x,y,z,np.average(vals)

    def create(self) -> tuple[td.CustomMedium, np.ndarray]:

        def make_grid() -> CellGrid:
            return CellGrid(boundaries=self._sim.grid.boundaries)

        def try_process_cell_point(idx, cell_grid, cell_permittivity, data: np.ndarray):
            x, y, z = cell_grid.decode_cell(idx)
            X, Y, Z = np.meshgrid((0,1), (0,1), (0,1))
            for dx, dy, dz in zip(X.ravel(), Y.ravel(), Z.ravel()):
                px, py, pz = x + dx, y + dy, z + dz
                adj_cells = cell_grid.adjacent_cell_inds(px, py, pz)
                vals = []
                for c in adj_cells:
                    p = cell_permittivity[c]
                    if p == 0:
                        continue
                    vals.append(p)
                if not vals:
                    continue
                data[px, py, pz] = np.average(vals)

        cell_grid = make_grid()
        structures = self._sim.structures
        bg_permittivity = 1.0
        cell_permittivity = np.full(cell_grid.n_cells(), 0.0)
        boundaries = cell_grid.boundaries
        data = np.zeros((boundaries.x.size, boundaries.y.size, boundaries.z.size))

        total_cells = cell_grid.n_cells()
        print(f"process total cells: {total_cells} on {cpu_count()} cores")
        p = Pool(cpu_count())
        start = time.perf_counter()
        for idx, perm in p.imap_unordered(self.calc_cell_permittivity,
                                            ((idx, structures, cell_grid, bg_permittivity) for idx in range(total_cells))):
            cell_permittivity[idx] = perm
            # t = time.time()
            # try_process_cell_point(idx, cell_grid, cell_permittivity, data)
            # print(f"process {idx} time: {time.time() - t}")
        print(f"Calc Time elapsed: {time.perf_counter() - start}")

        start = time.perf_counter()
        for x,y,z,v in p.imap_unordered(self.average_cell_permittivity,
                                        ((point, cell_grid, cell_permittivity) for point in np.ndindex(data.shape))):
            data[x, y, z] = v
        print(f"Average Time elapsed: {time.perf_counter() - start}")
        p.close()

        permittivity = SpatialDataArray(data, coords=boundaries.to_dict)
        return td.CustomMedium(permittivity=permittivity), cell_grid.reshape(cell_permittivity, cell_grid.num_cells)



class SpatialMediumCreator4:
    """ Creates a spatially varying medium based on the structures in the simulation.
        这个版本不对周围8个cell进行平均
    """
    MAX_ITERATIONS = None
    def __init__(self, sim: td.Simulation):
        self._sim = sim

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        x, y, z = cell.size[0], cell.size[1], cell.size[2]
        min_size = min(x, y, z)
        threshold = min_size / 2**(cls.MAX_ITERATIONS or ITERATIONS)
        while not q.empty():
            tmp_cell = q.get()
            b1 = tmp_cell.bounds
            b2 = geometry.bounds
            b3 = td.Box.bounds_intersection(b1, b2)
            size = tuple(
                (pt_max - pt_min) for pt_min, pt_max in zip(b3[0], b3[1]))
            center = b3[0] + np.array(size) / 2
            if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            in_ids = geometry.inside(**vertices)
            if in_ids.all():
                filled_cells.append(tmp_cell)
            else:
                min_size = min(tmp_cell.size[0], tmp_cell.size[1],
                               tmp_cell.size[2])
                # BUG:
                if (in_ids.any() or (np.array(b2) == np.array(b3)).all() or
                        tmp_cell.inside([center[0]], [center[1]],
                                        [center[2]])) and min_size > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)
                # if min_size > threshold:
                #     for c in tmp_cell.split_evenly():
                #         q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v

    @staticmethod
    def calc_cell_permittivity(args):
        idx, structs, cell_grid, bg_permittivity = args
        x, y, z = cell_grid.decode_cell(idx)
        cell = cell_grid.cell_at(x, y, z)
        cell_info: CellInfo = []
        for struct in structs:
            geometry = struct.geometry
            medium = struct.medium
            p = SpatialMediumCreator4.calc_cell_factor(cell, geometry)
            print('idx: ', idx, 'p: ', p, 'geometry:', geometry)
            if p == 0:
                continue
            cell_info.append((p, medium))

        def calc_eff(cell_info: CellInfo, bg_permittivity: ValueType):
            if not cell_info:
                return bg_permittivity
            ci = cell_info[-1]
            p, medium = ci
            eps = medium.eps_model(None)
            return max((1 - p) * bg_permittivity + p * eps.real, 1.0)

        return idx, calc_eff(cell_info, bg_permittivity)

    def create(self) -> tuple[td.CustomMedium, np.ndarray]:

        def make_grid() -> CellGrid:
            return CellGrid(boundaries=self._sim.grid.boundaries)

        def try_process_cell_point(idx, cell_grid, cell_permittivity,
                                   data: np.ndarray):
            x, y, z = cell_grid.decode_cell(idx)
            X, Y, Z = np.meshgrid((0, 1), (0, 1), (0, 1))
            for dx, dy, dz in zip(X.ravel(), Y.ravel(), Z.ravel()):
                px, py, pz = x + dx, y + dy, z + dz
                adj_cells = cell_grid.adjacent_cell_inds(px, py, pz)
                vals = []
                for c in adj_cells:
                    p = cell_permittivity[c]
                    if p == 0:
                        continue
                    vals.append(p)
                if not vals:
                    continue
                data[px, py, pz] = np.average(vals)

        cell_grid = make_grid()
        structures: list[td.Structure] = []
        for s in self._sim.structures:
            if isinstance(s.geometry,td.Box):
                x = 0 in s.geometry.zero_dims and td.inf or s.geometry.size[0]
                y = 1 in s.geometry.zero_dims and td.inf or s.geometry.size[1]
                z = 2 in s.geometry.zero_dims and td.inf or s.geometry.size[2]
                s = s.updated_copy(geometry=td.Box(center=s.geometry.center, size=(x, y, z)))
            structures.append(s)

        bg_permittivity = 1.0
        cell_permittivity = np.full(cell_grid.n_cells(),
                                    self._sim.medium.eps_model(None).real or bg_permittivity)
        boundaries = cell_grid.boundaries

        total_cells = cell_grid.n_cells()
        print(f"process total cells: {total_cells} on {cpu_count()} cores, iterations: {self.MAX_ITERATIONS}")
        print('grid shape: ', cell_grid.num_cells)
        p = Pool(cpu_count() - 1)
        start = time.perf_counter()
        # for idx in range(total_cells):
        #     cell_permittivity[idx] = self.calc_cell_permittivity(
        #         (idx, structures, cell_grid, bg_permittivity))[1]
        for idx, perm in p.imap_unordered(
                self.calc_cell_permittivity,
            ((idx, structures, cell_grid, bg_permittivity)
             for idx in range(total_cells))):
            cell_permittivity[idx] = perm
        print(f"Calc Time elapsed: {time.perf_counter() - start}")

        # data = np.ones(
        #     (boundaries.x.size, boundaries.y.size, boundaries.z.size))
        # start = time.perf_counter()
        # for x,y,z,v in p.imap_unordered(self.average_cell_permittivity,
        #                                 ((point, cell_grid, cell_permittivity) for point in np.ndindex(data.shape))):
        #     data[x, y, z] = v
        # print(f"Average Time elapsed: {time.perf_counter() - start}")
        p.close()
        permittivity_3d = cell_grid.reshape(cell_permittivity,
                                            cell_grid.num_cells)
        # 不用boundaries, 有个问题，如果是非均匀网格，下面这个方法会有问题

        if False:
            x = np.linspace(boundaries.x[0], boundaries.x[-1], permittivity_3d.shape[0])
            y = np.linspace(boundaries.y[0], boundaries.y[-1], permittivity_3d.shape[1])
            z = np.linspace(boundaries.z[0], boundaries.z[-1], permittivity_3d.shape[2])
            new_coords = td.Coords(x=x, y=y, z=z).to_dict
        else:
            def interpolate_coords(coords: td.Coords, n):
                def _slice_coords(c):
                    nn = len(c) - 1
                    vv = np.zeros(nn * n)
                    for i in range(nn):
                        v = np.linspace(c[i], c[i + 1], n+2)
                        vv[i * n:(i + 1) * n] = v[1:-1]
                    return vv
                return {k: _slice_coords(v) for k, v in coords.to_dict.items()}
            # new_coords = cell_grid.centers.to_dict
            new_coords = interpolate_coords(cell_grid.boundaries, 1)

            def interpolate_data(data: np.ndarray, n):
                sp = data.shape
                new_data = np.zeros((sp[0] * n, sp[1] * n, sp[2] * n))
                for i in range(sp[0]):
                    for j in range(sp[1]):
                        for k in range(sp[2]):
                            new_data[i * n:(i + 1) * n, j * n:(j + 1) * n, k * n:(k + 1) * n] = data[i, j, k]
                return new_data

        # permittivity = SpatialDataArray(interpolate_data(permittivity_3d, 3), coords=new_coords)
        permittivity = SpatialDataArray(interpolate_data(permittivity_3d, 1), coords=new_coords)
        return td.CustomMedium(permittivity=permittivity, interp_method='nearest'), permittivity_3d

class SpatialMediumCreator5:
    MAX_ITERATIONS = None
    def __init__(self, sim: td.Simulation):
        self._sim = sim

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        x, y, z = cell.size[0], cell.size[1], cell.size[2]
        min_size = min(x, y, z)
        threshold = min_size / 2**(cls.MAX_ITERATIONS or ITERATIONS)
        while not q.empty():
            tmp_cell = q.get()
            b1 = tmp_cell.bounds
            b2 = geometry.bounds
            b3 = td.Box.bounds_intersection(b1, b2)
            size = tuple(
                (pt_max - pt_min) for pt_min, pt_max in zip(b3[0], b3[1]))
            center = b3[0] + np.array(size) / 2
            if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            in_ids = geometry.inside(**vertices)
            if in_ids.all():
                filled_cells.append(tmp_cell)
            else:
                min_size = min(tmp_cell.size[0], tmp_cell.size[1],
                               tmp_cell.size[2])
                # BUG:
                if (in_ids.any() or (np.array(b2) == np.array(b3)).all() or
                        tmp_cell.inside([center[0]], [center[1]],
                                        [center[2]])) and min_size > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)
                # if min_size > threshold:
                #     for c in tmp_cell.split_evenly():
                #         q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v

    @staticmethod
    def calc_cell_permittivity(args):
        idx, structs, cell_grid, bg_permittivity = args
        x, y, z = cell_grid.decode_cell(idx)
        cell = cell_grid.cell_at(x, y, z)
        cell_info: CellInfo = []
        for struct in structs:
            geometry = struct.geometry
            medium = struct.medium
            p = SpatialMediumCreator5.calc_cell_factor(cell, geometry)
            if p == 0:
                continue
            cell_info.append((p, medium))

        def calc_eff(cell_info: CellInfo, bg_permittivity: ValueType):
            if not cell_info:
                return bg_permittivity
            ci = cell_info[-1]
            p, medium = ci
            eps = medium.eps_model(None)
            return max((1 - p) * bg_permittivity + p * eps.real, 1.0)

        return idx, calc_eff(cell_info, bg_permittivity)

    @staticmethod
    def average_cell_permittivity(args):
        point, cell_grid, cell_permittivity = args
        x, y, z = point
        adj_cells = cell_grid.adjacent_cell_inds(x, y, z)
        vals = [cell_permittivity[c] for c in adj_cells]
        return x,y,z,np.average(vals)

    def create(self, sub_divide = 1) -> tuple[td.CustomMedium, np.ndarray]:

        def make_grid() -> CellGrid:
            def interpolate_coords(coords: td.Coords, n)->td.Coords:
                def _slice_coords(c):
                    nn = len(c) * (n+1) - n
                    vv = np.zeros(nn)
                    k = n+1
                    for i in range(len(c)-1):
                        v = np.linspace(c[i], c[i + 1], k+1)
                        vv[i * k:(i + 1) * k] = v[0:-1]
                    vv[-1] = c[-1]
                    return vv
                return td.Coords(x=_slice_coords(coords.x), y=_slice_coords(coords.y), z=_slice_coords(coords.z))
            grid:td.Grid = self._sim.grid
            return CellGrid(boundaries=interpolate_coords(grid.boundaries, 2**sub_divide-1))

        def try_process_cell_point(idx, cell_grid, cell_permittivity,
                                   data: np.ndarray):
            x, y, z = cell_grid.decode_cell(idx)
            X, Y, Z = np.meshgrid((0, 1), (0, 1), (0, 1))
            for dx, dy, dz in zip(X.ravel(), Y.ravel(), Z.ravel()):
                px, py, pz = x + dx, y + dy, z + dz
                adj_cells = cell_grid.adjacent_cell_inds(px, py, pz)
                vals = []
                for c in adj_cells:
                    p = cell_permittivity[c]
                    if p == 0:
                        continue
                    vals.append(p)
                if not vals:
                    continue
                data[px, py, pz] = np.average(vals)

        cell_grid = make_grid()
        structures = self._sim.structures
        bg_permittivity = 1.0
        cell_permittivity = np.full(cell_grid.n_cells(),
                                    self._sim.medium.eps_model(None).real or bg_permittivity)
        boundaries = cell_grid.boundaries

        total_cells = cell_grid.n_cells()
        print(f"process total cells: {total_cells} on {cpu_count()} cores, iterations: {self.MAX_ITERATIONS}")
        p = Pool(cpu_count() - 1)
        start = time.perf_counter()
        for idx, perm in p.imap_unordered(
                self.calc_cell_permittivity,
            ((idx, structures, cell_grid, bg_permittivity)
             for idx in range(total_cells))):
            cell_permittivity[idx] = perm
        print(f"Calc Time elapsed: {time.perf_counter() - start}")

        data = np.ones(
            (boundaries.x.size, boundaries.y.size, boundaries.z.size))
        start = time.perf_counter()
        for x,y,z,v in p.imap_unordered(self.average_cell_permittivity,
                                        ((point, cell_grid, cell_permittivity) for point in np.ndindex(data.shape))):
            data[x, y, z] = v
        print(f"Average Time elapsed: {time.perf_counter() - start}")
        p.close()
        permittivity_3d = cell_grid.reshape(cell_permittivity,
                                            cell_grid.num_cells)

        permittivity = SpatialDataArray(data, coords=cell_grid.boundaries.to_dict)
        return td.CustomMedium(permittivity=permittivity, interp_method='linear'), permittivity_3d

class SpatialMediumCreator6(SpatialMediumCreator5):
    """ Creates a spatially varying medium based on the structures in the simulation.
    """
    def create(self, sub_divide=2):
        return super().create(sub_divide)

