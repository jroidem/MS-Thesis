import numpy as np

from C_plotting import *
from numba import prange
from time import perf_counter


@njit(parallel=True)
def create_mesh(qp_mesh, r_mesh, pr_mesh, pz_mesh):
    n = qp_mesh.shape
    for i in prange(n[0]):
        for j in prange(n[1]):
            if pz_mesh[i, j] == pz_mesh[i, j]:
                qp_mesh[i, j] = np.array((
                    [r_mesh[i, j], 0.],
                    [pr_mesh[i, j], pz_mesh[i, j]]
                ))
    return qp_mesh


@cuda.jit
def cuda_SALI(L, T, qp, v1, v2, dqp):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < qp.shape[0])
    j_in_range = (0 <= j) and (j < qp.shape[1])

    if i_in_range and j_in_range:
        # print(i, j)
        if qp[i, j, 1, 1] == qp[i, j, 1, 1]:
            calculate_SALI(
                L, T,
                qp[i, j],
                v1[i, j],
                v2[i, j],
                dqp[i, j]
            )


@cuda.jit(device=True)
def calculate_SALI(
    L: float, T_max: float,
    qp: np.ndarray, v1: ndarray, v2: ndarray, dqp: np.ndarray
):
    dt = 0.001
    T = 0.
    while T <= T_max:
        symplectic_cuda(qp, dqp, L, diff_eqn_cuda, dt)
        symplectic_var_cuda(v1, dqp, qp, L, var_eqn_cuda, dt)
        symplectic_var_cuda(v2, dqp, qp, L, var_eqn_cuda, dt)
        T += dt


@njit
def calculate_SALI_trajectory(
    qp: np.ndarray, T: float, dt: float = 0.001
):
    tpoints = np.arange(0, T, dt)

    q_points = np.empty((qp.shape[1], tpoints.shape[0]))
    v1_mesh = np.zeros_like(qp)
    v1_mesh[0, 0] = 1.
    v2_mesh = np.zeros_like(v1_mesh)
    v2_mesh[0, 2] = 1.
    p_points = np.empty_like(q_points)
    var_sum = np.empty_like(tpoints)
    var_diff = np.empty_like(tpoints)
    LCN = np.zeros_like(tpoints)
    SALI = np.empty_like(tpoints)

    for i, t in enumerate(tpoints):
        qp = symplectic(qp, diff_eqn, dt)
        v1_mesh = symplectic_var(v1_mesh, qp, dt)
        v2_mesh = symplectic_var(v2_mesh, qp, dt)
        q_points[:, i], p_points[:, i] = qp[:, :]
        if i != 0:
            LCN[i] = np.log(np.linalg.norm(v1_mesh))/t
        var_sum[i] = np.linalg.norm(normalize(v1_mesh)+normalize(v2_mesh))
        var_diff[i] = np.linalg.norm(normalize(v1_mesh)-normalize(v2_mesh))

        min = np.min(np.array([var_sum[i], var_diff[i]]))
        if min > 0.:
            SALI[i] = np.log10(min)
        else:
            SALI[i] = -17.
    return q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN


class PlotSALI(PlotValues):

    def __init__(
            self,
            L: float, P: float,
            T: float = 1000., resolution: int = 100,
            r_bounds=None, pr_bounds=None,
            path_name: str = 'SALI/'
    ):
        """
        :param L: Canonical angular momentum
        :param P: Rescaled energy
        :param resolution: Resolution of the arrays
        """

        self.P = P
        """
        if not (0 <= P < 1):
            raise Warning('P is not in the bounded range')
        """

        super().__init__(L, P, is_E=True, path_name=path_name)

        if not r_bounds:
            self.r_bounds = self.get_r_bounds()
        else:
            self.r_bounds = r_bounds
        if not pr_bounds:
            self.pr_bounds = self.get_pr_bounds()
        else:
            self.pr_bounds = pr_bounds
        self.resolution = resolution
        self.T = T

        self.r_mesh, self.pr_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.v1_mesh, self.v2_mesh, self.SALI_mesh = self.create_var_meshes()
        self.name = f'L={self.L}_P={self.P}_T={T}_res={resolution}'

    def create_io_meshes(self):
        r_mesh, pr_mesh = np.meshgrid(
           np.linspace(*self.r_bounds, self.resolution+2)[1:-1],
           np.linspace(*self.pr_bounds, self.resolution+2)[1:-1]
        )
        return r_mesh, pr_mesh

    def create_qp_mesh(self):
        pz_mesh = np.full_like(self.r_mesh, np.nan)
        for i, row in enumerate(self.r_mesh):
            for j, r_value in enumerate(row):
                pz_sq = -2 * kinetic_energy(
                    self.L,
                    r_value,
                    0.,
                    self.h
                ) - self.pr_mesh[i, j] ** 2
                if pz_sq >= 0:
                    pz_mesh[i, j] = np.sqrt(pz_sq)
        qp_mesh = np.full((*np.shape(self.r_mesh), 2, 2), np.nan)
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pz_mesh
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
        print(np.shape(self.v1_mesh))
        time = perf_counter()
        cuda_SALI[
            (grid_size, grid_size), (block_size, block_size)
        ](
            self.L, self.T,
            cuda_qp, cuda_v1, cuda_v2, cuda_dqp_mesh
        )
        self.v1_mesh = cuda_v1.copy_to_host().reshape(*cuda_v1.copy_to_host().shape[:-2], -1)
        self.v2_mesh = cuda_v2.copy_to_host().reshape(*cuda_v2.copy_to_host().shape[:-2], -1)
        print(f'time taken = {perf_counter()-time}')
        print("Calculating SALI")
        for i, row in enumerate(self.SALI_mesh):
            for j, val in enumerate(row):
                if self.qp_mesh[i, j, 1, 1] == self.qp_mesh[i, j, 1, 1]:
                    self.SALI_mesh[i, j] = np.log10(np.min([
                        np.linalg.norm(normalize(self.v1_mesh[i, j])+normalize(self.v2_mesh[i, j])),
                        np.linalg.norm(normalize(self.v1_mesh[i, j])-normalize(self.v2_mesh[i, j]))
                    ]))

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
                print('file not found, generating generating SALI')
                self.generate_SALI(block_size)
        else:
            print('generating SALI')
            self.generate_SALI(block_size)

        if save_array:
            self.save(self.name, self.SALI_mesh)
        plt.figure(figsize=figsize7)
        self.plot_zzc(r_buff=0, pr_buff=0)
        cmap = matplotlib.cm.get_cmap("plasma", 3)
        plt.title(f"SALI (L={self.L}, P={self.P})")
        plt.pcolormesh(self.r_mesh, self.pr_mesh, self.SALI_mesh, cmap="plasma", vmin=-12, vmax=0)
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        plt.colorbar()
        if save_plot:
            self.savefig(self.name)
        plt.clf()


def SALI_value_plot():
    P_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    L_values = [
        0.2, 0.4, 0.6, 0.8,
        1.0, 1.2, 1.4, 1.6, 1.8,
        2.0, 2.2, 2.4, 2.6, 2.8,
        3.0, 3.2, 3.4, 3.6, 3.8,
        4.0, 4.2, 4.4, 4.6, 4.8, 5.0
    ]
    L_mesh, P_mesh = np.meshgrid(L_values, P_values)
    percent_mesh = np.empty_like(P_mesh)
    for i, P in enumerate(P_values):
        for j, L in enumerate(L_values):
            sali = PlotSALI(L, P, resolution=100)
            if P > 1.0:
                sali.name += "_trap"
            qp_mesh = sali.qp_mesh
            SALI_MESH = sali.load(sali.name)
            total = np.count_nonzero(qp_mesh[:, :, 1, 1] == qp_mesh[:, :, 1, 1])
            ordered = np.count_nonzero(SALI_MESH >= -0.9)
            percent_mesh[i, j] = 100*ordered / total
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(L_mesh, P_mesh, percent_mesh)
    plt.colorbar()
    plt.xlabel("L")
    plt.ylabel("P")
    savefig(fr"order_increase", r"\analysis\figures\\")


def generate_SALI_plots():
    L_values = [
        1.0, 1.2, 1.4, 1.6, 1.8,
        2.0, 2.2, 2.4, 2.6, 2.8,
        3.0, 3.2, 3.4, 3.6, 3.8,
        4.0, 4.2, 4.4, 4.6, 4.8, 5.0
    ]
    for P in [0.7]:
        for L in L_values:
            #escape = EscapePlot(L, P, resolution=100)
            #name = f"escape_time/L={L}_P={P}"
            #escape_mesh = escape.load(name)
            sali = PlotSALI(L, P, resolution=100)
            #sali.qp_mesh[:, :, 1, 1][escape_mesh == escape_mesh] = np.nan
            #sali.plot_SALI(name=sali.name+"_trap")
            sali.plot_SALI()


def generate_SALI_data():
    P_values = [0.1, 0.2, 0.3, 0.4]
    L_values = [
        0.4, 0.6, 0.8,
        1.0, 1.2, 1.4, 1.6, 1.8,
        2.0, 2.2, 2.4, 2.6, 2.8,
        3.0, 3.2, 3.4, 3.6, 3.8,
        4.0, 4.2, 4.4, 4.6, 4.8, 5.0
    ]
    for P in P_values:
        for L in L_values:
            name = f"escape_time/L={L}_P={P}"
            sali = PlotSALI(L, P, resolution=100)
            sali.plot_SALI(name=sali.name)


def plot_SALI_slice():
    P = 0.5
    percent_sali = []
    L = 3.6
    sali = PlotSALI(L, P, resolution=100)
    SALI_MESH = sali.load(sali.name)
    size = np.shape(SALI_MESH)[0]
    plt.plot(sali.r_mesh[size//2, :], SALI_MESH[size//2, :])
    plt.plot(sali.r_mesh[size//2, :], np.full_like(sali.r_mesh[size//2, :], -0.9), linestyle="--")
    plt.show()


def sample_SALI_trajectory():
    plt.ylim(bottom=-17.0)
    plt.xlabel("t")
    plt.ylabel("$\\log_{10}$(SALI)")

    L = 1
    E = 0.5
    r = 1.7844903
    pr = 0
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, SALI, label="periodic", alpha=0.8)

    r = 1.8
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, SALI, label="quasi periodic", alpha=0.8)

    r = 1.85
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, SALI, label="chaotic", alpha=0.8)

    plt.legend()
    plot.savefig("SALI_trajectory")
    plt.close("all")


def sample_LCN_trajectory():
    plt.xlabel("t")
    plt.ylabel("LCN")

    L = 1
    E = 0.5
    r = 1.7844903
    pr = 0
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, LCN, label="periodic")

    r = 1.8
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, LCN, label="quasi periodic")

    r = 1.85
    plot = PlotTrajectory(L, E, 1000, r, pr)
    q_points, p_points, var_sum, var_diff, SALI, tpoints, LCN = calculate_SALI_trajectory(qp=plot.qp, T=plot.T)
    plt.plot(tpoints, LCN, label="chaotic")

    plt.legend()
    plot.savefig("LCN_trajectory")
    plt.close("all")
