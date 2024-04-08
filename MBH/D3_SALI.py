import numpy as np

from C_plotting import *
from D2_escape_time import *
from numba import prange
from time import perf_counter


def normalize(array):
    return array/np.linalg.norm(array)


@cuda.jit
def cuda_SALI(B, L, E, T, qp, v1, v2, dqp):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < qp.shape[0])
    j_in_range = (0 <= j) and (j < qp.shape[1])

    if i_in_range and j_in_range:
        # print(i, j)
        if qp[i, j, 1, 1] == qp[i, j, 1, 1]:
            calculate_SALI(
                B, L, E, T,
                qp[i, j],
                v1[i, j],
                v2[i, j],
                dqp[i, j]
            )


@cuda.jit(device=True)
def calculate_SALI(
    B: float, L: float, E: float, T_max: float,
    qp: np.ndarray, v1: ndarray, v2: ndarray, dqp: np.ndarray
):
    dt = 0.001
    T = 0.
    while T <= T_max:
        symplectic_cuda(qp, dqp, B, L, E, dt)
        symplectic_var_cuda(v1, dqp, qp, B, L, E, dt)
        symplectic_var_cuda(v2, dqp, qp, B, L, E, dt)

        v1_norm = (v1[0, 0]**2 + v1[0, 1]**2 + v1[1, 0]**2 + v1[1, 1]**2)**0.5
        v2_norm = (v2[0, 0]**2 + v2[0, 1]**2 + v2[1, 0]**2 + v2[1, 1]**2)**0.5

        v1[0, 0] = v1[0, 0]/v1_norm
        v1[0, 1] = v1[0, 1]/v1_norm
        v1[1, 0] = v1[1, 0]/v1_norm
        v1[1, 1] = v1[1, 1]/v1_norm

        v2[0, 0] = v2[0, 0]/v2_norm
        v2[0, 1] = v2[0, 1]/v2_norm
        v2[1, 0] = v2[1, 0]/v2_norm
        v2[1, 1] = v2[1, 1]/v2_norm

        T += dt


class PlotSALI(PlotValues):

    def __init__(
            self,
            r_esc: float, p: float, E: float,
            T: float = 500., resolution: int = 100,
            r_bounds=None, pr_bounds=None,
            path_name: str = 'SALI/'
    ):
        self.input_E = E
        self.input_resc = r_esc
        self.input_p = p
        super().__init__(r_esc, p, E, path_name=path_name)
        if not r_bounds:
            r_bounds = self.get_r_bounds()
            if len(r_bounds) == 1:
                self.r_bounds = np.array([1, r_bounds[0]])
            else:
                self.r_bounds = r_bounds[-2:]
        else:
            self.r_bounds = r_bounds
        if not pr_bounds:
            U_val = self.get_U_eq()
            if U_val.shape[0] == 2 and U_val[0] > self.E:
                self.pr_bounds = self.get_pr_bounds()
            else:
                self.pr_bounds = PLUS_MINUS*np.sqrt(E)
        else:
            self.pr_bounds = pr_bounds
        self.resolution = resolution
        self.T = T

        self.r_mesh, self.pr_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.v1_mesh, self.v2_mesh, self.SALI_mesh = self.create_var_meshes()
        self.name = f'SALI_resc={r_esc}_p={p}_E={self.E}_res={resolution}'

    def create_io_meshes(self):
        r_mesh, pr_mesh = np.meshgrid(
           np.linspace(*self.r_bounds, self.resolution+2)[1:-1],
           np.linspace(*self.pr_bounds, self.resolution+2)[1:-1]
        )
        return r_mesh, pr_mesh

    def create_qp_mesh(self):
        pth_mesh = np.full_like(self.r_mesh, np.nan)
        for i, row in enumerate(self.r_mesh):
            for j, r in enumerate(row):
                pth_sq = r ** 2 * (self.E - self.get_potential_energy(r, PI / 2) - (self.pr_mesh[i, j]) ** 2) / (
                            1 - 1 / r)
                if pth_sq >= 0:
                    pth_mesh[i, j] = -np.sqrt(pth_sq)
        qp_mesh = np.empty((*np.shape(self.r_mesh), 2, 2))
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pth_mesh
        )

    def create_var_meshes(self):
        v1_mesh = np.full((*np.shape(self.r_mesh), 2, 2), 0.)
        v2_mesh = np.full((*np.shape(self.r_mesh), 2, 2), 0.)
        v1_mesh[:, :, 0, 0].fill(1.)
        v2_mesh[:, :, 0, 1].fill(1.)
        SALI_mesh = np.full_like(self.r_mesh, np.nan)
        return v1_mesh, v2_mesh, SALI_mesh

    def generate_SALI(self, block_size: int = 8):
        grid_size = self.resolution//block_size + 1
        cuda_qp = cuda.to_device(self.qp_mesh)
        cuda_v1 = cuda.to_device(self.v1_mesh)
        cuda_v2 = cuda.to_device(self.v2_mesh)
        cuda_dqp_mesh = cuda.device_array_like(np.empty_like(self.qp_mesh))
        time = perf_counter()
        cuda_SALI[
            (grid_size, grid_size), (block_size, block_size)
        ](
            self.B, self.L, self.E, self.T,
            cuda_qp, cuda_v1, cuda_v2, cuda_dqp_mesh
        )
        self.v1_mesh = cuda_v1.copy_to_host().reshape(*cuda_v1.copy_to_host().shape[:-2], -1)
        self.v2_mesh = cuda_v2.copy_to_host().reshape(*cuda_v2.copy_to_host().shape[:-2], -1)
        print(f'time taken = {perf_counter()-time}')
        print("Calculating SALI")
        for i, row in enumerate(self.SALI_mesh):
            for j, val in enumerate(row):
                if self.qp_mesh[i, j, 1, 1] == self.qp_mesh[i, j, 1, 1]:
                    sum = np.linalg.norm(normalize(self.v1_mesh[i, j])+normalize(self.v2_mesh[i, j]))
                    diff = np.linalg.norm(normalize(self.v1_mesh[i, j])-normalize(self.v2_mesh[i, j]))
                    SALI_value = np.nanmin([sum, diff])
                    if SALI_value > 0:
                        self.SALI_mesh[i, j] = np.log10(SALI_value)
                    else:
                        self.SALI_mesh[i, j] = -17.

    def plot_SALI(
            self, save_plot=True, save_array=True, load_array=False, block_size: int = 8,
            name: str = ''
    ):
        if name:
            self.name = name
        if load_array:
            try:
                self.SALI_mesh = self.load(self.name)
            except FileNotFoundError:
                print('file not found, generating SALI')
                self.generate_SALI(block_size)
        else:
            print('generating SALI')
            self.generate_SALI(block_size)

        if save_array:
            self.save(self.name, self.SALI_mesh)

        plot = EscapePlot(self.input_resc, self.input_p, self.input_E, resolution=self.resolution)
        name = plot.name
        try:
            pass_mesh = plot.load(f'escape_pass_{name}')
            basin_mesh = plot.load(f'escape_basin_{name}')
            self.SALI_mesh[basin_mesh == basin_mesh] = np.nan
            self.SALI_mesh[pass_mesh == 0] = np.nan
        except FileNotFoundError:
            print("file not found")

        plt.figure(figsize=figsize7)
        self.plot_zzc()
        # cmap = matplotlib.cm.get_cmap("plasma", 3)
        plt.pcolormesh(self.r_mesh, self.pr_mesh, self.SALI_mesh, cmap="plasma", vmin=-12, vmax=0)
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        plt.colorbar()
        if save_plot:
            self.savefig(self.name)
        plt.close("all")


def generate_SALI_data():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    r_esc_values.reverse()
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]
    for E in E_values:
        for p in p_values:
            for r_esc in r_esc_values:
                sali = PlotSALI(r_esc, p, E)
                try:
                    sali.load(sali.name)
                except FileNotFoundError:
                    print(E, p, r_esc)
                    sali.plot_SALI()


def SALI_value_plot():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]

    r_mesh, E_mesh = np.meshgrid(r_esc_values, E_values)
    for _, p in enumerate(p_values):
        num = np.empty_like(r_mesh)
        for i, row in enumerate(r_mesh):
            for j, r_esc in enumerate(row):
                E = E_mesh[i, j]
                escape_plot = EscapePlot(r_esc, p, E)
                name = escape_plot.name
                pass_mesh = escape_plot.load(f'escape_pass_{name}')
                basin_mesh = escape_plot.load(f'escape_basin_{name}')
                SALI_plot = PlotSALI(r_esc, p, E)
                sali = SALI_plot.load(SALI_plot.name)
                total = np.count_nonzero(pass_mesh == pass_mesh)
                sali[basin_mesh == basin_mesh] = np.nan
                sali[pass_mesh == 0] = np.nan
                ordered = np.count_nonzero(sali >= -1.)
                num[i, j] = 100*ordered / total
        plt.figure(figsize=figsize3)
        plt.pcolormesh(r_mesh, E_mesh, num, vmin=0., vmax=20, rasterized=True)
        plt.xlabel("$r_{esc}$")
        plt.ylabel("$E$")
        plt.colorbar()
        isco = p * ((3 * p - 1) / (3 - p)) ** (1 / 4)
        if np.min(r_esc_values) < isco < np.max(r_esc_values):
            unstable_values = np.linspace(isco, np.max(r_esc_values))[1:]
            E_unstable = np.empty_like(unstable_values)
            for i, r_esc in enumerate(unstable_values):
                L = r_esc * ((3 - p) * (3 * p - 1)) ** (1 / 4) / np.sqrt(
                    2 * (4 * p ** 2 - 9 * p + 3 + np.sqrt((3 * p - 1) * (3 - p))))
                B = L / r_esc ** 2
                r_eq = np.roots([2*B**2, -B**2, 0, 1-2*B*L, -2*L**2, 3*L**2])
                r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))[0]
                E_unstable[i] = potential_energy(B, L, r_eq, PI/2)
            plt.plot(unstable_values, E_unstable, c='w')
        plt.ylim(bottom=np.min(E_values), top=np.max(E_values))
        savefig(f"order_plot_p={p}")
        plt.close("all")
        print(f"{p} done")