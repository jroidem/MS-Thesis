from C_plotting import *
from numba import prange
from time import perf_counter


@njit(parallel=True)
def create_mesh(qp_mesh, r_mesh, pr_mesh, pz_mesh):
    n = qp_mesh[0].shape
    for i in prange(n[0]):
        for j in prange(n[1]):
            if pz_mesh[i, j] == pz_mesh[i, j]:
                qp_mesh[0, i, j] = np.array((
                    [r_mesh[i, j], 0.],
                    [pr_mesh[i, j], pz_mesh[i, j]]
                ))
    return qp_mesh


@cuda.jit
def poincare(qp_mesh, L, qp, dqp):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < qp_mesh.shape[1])
    j_in_range = (0 <= j) and (j < qp_mesh.shape[2])

    if i_in_range and j_in_range:
        # print(i, j)
        if qp[i, j, 1, 1] == qp[i, j, 1, 1]:
            calculate_poincare(
                i, j,
                qp_mesh,
                L,
                qp[i, j],
                dqp[i, j]
            )


@cuda.jit(device=True)
def calculate_poincare(
    i: int, j: int, qp_array: np.ndarray, L: float,
    qp: np.ndarray, dqp: np.ndarray
):
    dt = 0.001
    T_max = 5_000.
    max_time = False
    T = 0.
    iterations = qp_array.shape[0]
    for m in range(1, iterations):
        symplectic_cuda(qp, dqp, L, diff_eqn_cuda, dt)
        z0 = qp[0, 1]
        while qp[0, 1]*z0 > 0:
            z0 = qp[0, 1]
            symplectic_cuda(qp, dqp, L, diff_eqn_cuda, dt)
            T += dt
            if T > T_max:
                max_time = True
                break
        if max_time:
            break
        qp_array[m, i, j, 0, 0] = qp[0, 0]
        qp_array[m, i, j, 0, 1] = qp[0, 1]
        qp_array[m, i, j, 1, 0] = qp[1, 0]
        qp_array[m, i, j, 1, 1] = qp[1, 1]
        T += dt
        #qp[0, 1] = -qp[0, 1]
        #qp[1, 1] = -qp[1, 1]
    if qp_array[1, i, j, 0, 0] != qp_array[1, i, j, 0, 0]:
        qp_array[0, i, j, 0, 0] = np.nan
        qp_array[0, i, j, 0, 1] = np.nan
        qp_array[0, i, j, 1, 0] = np.nan
        qp_array[0, i, j, 1, 1] = np.nan


class PlotPoincare(PlotValues):

    def __init__(
            self,
            L: float, E: float,
            iterations: int = 200, resolution: int = 20,
            r_bounds=None, pr_bounds=None,
            path_name: str = 'poincare/'
    ):
        """
        :param L: Canonical angular momentum
        :param E: Rescaled energy
        :param iterations: Number of Poincare points per initial conditions
        :param resolution: Resolution of the arrays
        """

        self.E = E
        """
        if not (0 <= E < 1):
            raise Warning('P is not in the bounded range')
        """

        super().__init__(L, E, is_E=True, path_name=path_name)

        if not r_bounds:
            self.r_bounds = self.get_r_bounds()
        else:
            self.r_bounds = r_bounds
        if not pr_bounds:
            self.pr_bounds = self.get_pr_bounds()
        else:
            self.pr_bounds = pr_bounds
        self.resolution = resolution
        self.iterations = iterations

        self.r_mesh, self.pr_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.name = f'L={self.L}_E={self.E}_ite={iterations}_res={resolution}'

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
                    0,
                    self.h
                ) - self.pr_mesh[i, j] ** 2
                if pz_sq >= 0:
                    pz_mesh[i, j] = np.sqrt(pz_sq)
        qp_mesh = np.full((self.iterations, *np.shape(self.r_mesh), 2, 2), np.nan)
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pz_mesh
        )

    def generate_poincare(self, block_size: int = 2):
        grid_size = self.resolution//block_size + 1

        cuda_qp_mesh = cuda.to_device(self.qp_mesh)
        cuda_qp = cuda.to_device(self.qp_mesh[0])
        cuda_dqp_mesh = cuda.device_array_like(np.empty_like(self.qp_mesh[0]))
        time = perf_counter()
        poincare[
            (grid_size, grid_size), (block_size, block_size)
        ](
            cuda_qp_mesh,
            self.L,
            cuda_qp, cuda_dqp_mesh
        )
        self.qp_mesh = cuda_qp_mesh.copy_to_host()
        print(f'time taken = {perf_counter()-time}')

    def plot_poincare(
            self, pz_slice=False, show_plot=False, save_plot=True, save_array=True, load_array=False, pdf=True,
            name: str = ''
    ):
        if name:
            self.name = name
        if pz_slice:
            self.name += "_pz_slice"
        if load_array:
            try:
                poincare_points = self.load(self.name)
            except FileNotFoundError:
                print('file not found, generating poincare points')
                self.generate_poincare()
                poincare_points = self.qp_mesh
        else:
            print('generating poincare points')
            self.generate_poincare()
            poincare_points = self.qp_mesh

        if save_array:
            self.save(self.name, poincare_points)

        plt.figure(figsize=figsize4)
        if pz_slice:
            plt.xlabel('$p_z$')
        else:
            plt.xlabel('$r$')
            self.plot_zzc()
        plt.ylabel('$p_r$')
        # plt.title(f"Poincare (L={self.L}, E={self.E})")
        for i, _ in enumerate(poincare_points):
            if pz_slice:
                plt.plot(
                    poincare_points[i, :, :, 1, 1], poincare_points[i, :, :, 1, 0],
                    '.', c='k', markersize=0.5, mew=0.5, linewidth=0., rasterized=True
                )
            else:
                plt.plot(
                    poincare_points[i, :, :, 0, 0], poincare_points[i, :, :, 1, 0],
                    '.', c='k', markersize=0.5, mew=0.5, linewidth=0., rasterized=True
                )

        if save_plot:
            self.savefig(self.name, pdf=pdf)

        if show_plot:
            plt.show()
        plt.clf()
        return poincare_points

    def custom_trajectory(self, r0: float, pr0: float, T:float, name:str, load_array=True):
        trajectory = PlotTrajectory(self.L, self.E, T, r0, pr=pr0)

        plt.figure(figsize=figsize4)
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        if load_array:
            self.plot_zzc(color='gray', alpha=0.5)
            poincare_points = self.load(self.name)
            for i, _ in enumerate(poincare_points):
                plt.plot(
                    poincare_points[i, :, :, 0, 0], poincare_points[i, :, :, 1, 0],
                    '.', c='gray', alpha=0.25, markersize=0.5, mew=0.5, linewidth=0., rasterized=True
                )
        q_points, p_points, t_points = trajectory.get_trajectory()
        r_points, _, z_points = q_points
        pr_points, _, _ = p_points
        index_array = np.where(z_points*np.roll(z_points, 1) <= 0)
        plt.plot(r_points[index_array], pr_points[index_array], '.', c='C0', markersize=1., mew=1., rasterized=True)
        self.savefig(name)


def sample_ordered():
    L = 1
    E = 0.5
    r = 1.7844903
    pr = 0
    poinc = PlotPoincare(L, E)
    # poinc.plot_poincare()
    poinc.custom_trajectory(r, pr, 100, "periodic_poincare")
    plt.clf()
    plt.figure(figsize=figsize4)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plot = PlotTrajectory(L, E, 100, r)
    plot.plot_trajectory()
    plot.savefig("periodic_orbit")
    plt.clf()


def sample_quasi():
    L = 1
    E = 0.5
    r = 1.8
    pr = 0
    poinc = PlotPoincare(L, E)
    # poinc.plot_poincare()
    poinc.custom_trajectory(r, pr, 5000, "quasi_poincare")
    plt.clf()
    plt.figure(figsize=figsize4)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plot = PlotTrajectory(L, E, 5000, r)
    plot.plot_trajectory()
    plot.savefig("quasi_orbit")
    plt.clf()


def sample_chaotic():
    L = 1
    E = 0.5
    r = 1.85
    pr = 0
    poinc = PlotPoincare(L, E)
    # poinc.plot_poincare()
    poinc.custom_trajectory(r, pr, 5000, "chaotic_poincare")
    plt.clf()
    plt.figure(figsize=figsize4)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plot = PlotTrajectory(L, E, 1000, r)
    plot.plot_trajectory()
    plot.savefig("chaotic_orbit")
    plt.clf()
