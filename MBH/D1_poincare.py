from C_plotting import *
from numba import prange
from time import perf_counter


@njit(parallel=True)
def create_mesh(qp_mesh, r_mesh, pr_mesh, pth_mesh):
    n = qp_mesh[0].shape
    for i in prange(n[0]):
        for j in prange(n[1]):
            if pth_mesh[i, j] == pth_mesh[i, j]:
                qp_mesh[0, i, j] = np.array((
                    [r_mesh[i, j], PI/2],
                    [pr_mesh[i, j]/(1-1/r_mesh[i, j]), pth_mesh[i, j]]
                ))
    return qp_mesh


@cuda.jit
def poincare(qp_mesh, B, L, E, qp, dqp):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < qp_mesh.shape[1])
    j_in_range = (0 <= j) and (j < qp_mesh.shape[2])

    if i_in_range and j_in_range:
        # print(i, j)
        if qp[i, j, 1, 1] == qp[i, j, 1, 1]:
            calculate_poincare(
                i, j,
                qp_mesh,
                B, L, E,
                qp[i, j],
                dqp[i, j]
            )


@cuda.jit(device=True)
def calculate_poincare(
    i: int, j: int, qp_array: np.ndarray, b: float, L: float, E: float,
    qp: np.ndarray, dqp: np.ndarray
):
    dt = 0.001
    T_max = 5_000.
    max_time = False
    T = 0.
    iterations = qp_array.shape[0]
    for m in range(1, iterations):
        symplectic_cuda(qp, dqp, b, L, E, dt)
        th0 = qp[0, 1]
        while (qp[0, 1]-PI/2)*(th0-PI/2) > 0:
            th0 = qp[0, 1]
            symplectic_cuda(qp, dqp, b, L, dt)
            T += dt
            if T > T_max:
                max_time = True
                break
        if max_time or qp[0, 0] <= 1:
            break

        qp_array[m, i, j, 0, 0] = qp[0, 0]
        qp_array[m, i, j, 0, 1] = qp[0, 1]
        qp_array[m, i, j, 1, 0] = qp[1, 0]
        qp_array[m, i, j, 1, 1] = qp[1, 1]
        T += dt
    if qp_array[1, i, j, 0, 0] != qp_array[1, i, j, 0, 0]:
        qp_array[0, i, j, 0, 0] = np.nan
        qp_array[0, i, j, 0, 1] = np.nan
        qp_array[0, i, j, 1, 0] = np.nan
        qp_array[0, i, j, 1, 1] = np.nan


class PlotPoincare(PlotValues):

    def __init__(
            self,
            r_esc: float, p: float, E: float,
            iterations: int = 200, resolution: int = 20,
            r_bounds=None, pr_bounds=np.array([-1.0, 1.0]),
            path_name: str = 'poincare/'
    ):
        super().__init__(r_esc, p, E, path_name=path_name)

        if not r_bounds:
            r_bounds = self.get_r_bounds()
            if len(r_bounds) == 1:
                self.r_bounds = np.array([1, r_bounds[0]])
            else:
                self.r_bounds = r_bounds[-2:]
        else:
            self.r_bounds = r_bounds
        self.pr_bounds = pr_bounds
        self.resolution = resolution
        self.iterations = iterations

        self.r_mesh, self.pr_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.name = f'b={self.B}_L={self.L}_E={self.E}_ite={iterations}_res={resolution}'

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
                pth_sq = r**2*(self.E-self.get_potential_energy(r, PI/2)-(self.pr_mesh[i, j])**2)/(1-1/r)
                if pth_sq >= 0:
                    pth_mesh[i, j] = np.sqrt(pth_sq)
        qp_mesh = np.full((self.iterations, *np.shape(self.r_mesh), 2, 2), np.nan)
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pth_mesh
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
            self.B,
            self.L,
            self.E,
            cuda_qp, cuda_dqp_mesh
        )
        self.qp_mesh = cuda_qp_mesh.copy_to_host()
        print(f'time taken = {perf_counter()-time}')

    def plot_poincare(
            self, show_plot=False, save_plot=True, save_array=True, load_array=False, pdf=True,
            name: str = ''
    ):
        if name:
            self.name = name
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
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        for i, _ in enumerate(poincare_points):
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
