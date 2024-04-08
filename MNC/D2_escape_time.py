import matplotlib.pyplot as plt

from C_plotting import *
from numba import prange, cuda
from time import perf_counter


@njit(parallel=True)
def create_mesh(qp_mesh, r_mesh, pr_mesh, pz_mesh):
    n = qp_mesh.shape
    for i in prange(n[0]):
        for j in prange(n[1]):
            qp_mesh[i, j] = np.array((
                [r_mesh[i, j], 0.],
                [pr_mesh[i, j], pz_mesh[i, j]]
            ))
    return qp_mesh


@cuda.jit(device=True)
def calculate_escape_time(qp, L, asymptotes, dqp, t_points, min_r, max_r):
    min_r_index = 0
    max_r_index = 0
    pass_index = -1
    for i, j in enumerate(t_points):
        pr_i = qp[1, 0]
        z_i = qp[0, 1]
        symplectic_cuda(qp, dqp, L, diff_eqn_cuda, 0.001)
        pr_f = qp[1, 0]
        z_f = qp[0, 1]

        if z_i*z_f <= 0:
            pass_index += 1

        if pr_i*pr_f < 0 < z_f*qp[1, 1]:
            if 0 < pr_i:
                max_r[max_r_index] = qp[0, 0]
                if max_r_index >= max_r.shape[0]-1:
                    max_r_index = 0
                else:
                    max_r_index += 1
            else:
                min_r[min_r_index] = qp[0, 0]
                if min_r_index >= max_r.shape[0] - 1:
                    min_r_index = 0
                else:
                    min_r_index += 1

        inner_in_range = \
            asymptotes[0] < min_r[0] and \
            asymptotes[0] < min_r[1] and \
            asymptotes[0] < min_r[2] and \
            asymptotes[0] < min_r[3]

        outer_in_range = \
            asymptotes[1] > max_r[0] and \
            asymptotes[1] > max_r[1] and \
            asymptotes[1] > max_r[2] and \
            asymptotes[1] > max_r[3]

        if inner_in_range and outer_in_range:
            return j, pass_index
    return np.nan, pass_index


@cuda.jit(device=True)
def calculate_escape_quantities(qp, L, asymptotes, dqp, t_points, min_r, max_r):
    min_r_index = 0
    max_r_index = 0
    check_max_z = True
    max_z = 0.
    return_time = 0
    dt = t_points[1]-t_points[0]
    for t in t_points:
        pr_i = qp[1, 0]
        z_i = qp[0, 1]
        symplectic_cuda(qp, dqp, L, diff_eqn_cuda, dt)
        pr_f = qp[1, 0]
        z_f = qp[0, 1]

        if z_i*z_f < 0:
            return_time = t
            break

        if check_max_z:
            max_z = z_f
            if qp[1, 1] < 0:
                check_max_z = False

        if pr_i*pr_f < 0:
            if 0 < pr_i:
                max_r[max_r_index] = qp[0, 0]
                if max_r_index >= max_r.shape[0]-1:
                    max_r_index = 0
                else:
                    max_r_index += 1
            else:
                min_r[min_r_index] = qp[0, 0]
                if min_r_index >= max_r.shape[0] - 1:
                    min_r_index = 0
                else:
                    min_r_index += 1

        inner_in_range = asymptotes[0] < min_r[0] and \
            asymptotes[0] < min_r[1] and \
            asymptotes[0] < min_r[2] and \
            asymptotes[0] < min_r[3]

        outer_in_range = asymptotes[1] > max_r[0] and \
            asymptotes[1] > max_r[1] and \
            asymptotes[1] > max_r[2] and \
            asymptotes[1] > max_r[3]

        if inner_in_range and outer_in_range and z_f*qp[1, 1] > 0:
            return np.nan, np.nan
    return max_z, return_time


@cuda.jit
def cuda_escape_quantities(
        max_z_mesh, return_time_mesh, qp_mesh, L, asymptote, dqp, t_points,
        max_r, min_r
        ):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < max_z_mesh.shape[0])
    j_in_range = (0 <= j) and (j < max_z_mesh.shape[1])

    if i_in_range and j_in_range:
        if qp_mesh[i, j, 1, 1] == qp_mesh[i, j, 1, 1]:
            max_z_mesh[i, j], return_time_mesh[i, j] = \
                calculate_escape_quantities(
                qp_mesh[i, j],
                L,
                asymptote[i, j],
                dqp[i, j],
                t_points,
                min_r[i, j],
                max_r[i, j]
            )


@cuda.jit
def cuda_escape_time(
        esc_time_mesh, pass_mesh, qp_mesh, L, asymptote, dqp, t_points, max_r, min_r
        ):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < esc_time_mesh.shape[0])
    j_in_range = (0 <= j) and (j < esc_time_mesh.shape[1])

    if i_in_range and j_in_range:
        if qp_mesh[i, j, 1, 1] == qp_mesh[i, j, 1, 1]:
            esc_time_mesh[i, j], pass_mesh[i, j] = calculate_escape_time(
                qp_mesh[i, j],
                L,
                asymptote[i, j],
                dqp[i, j],
                t_points,
                min_r[i, j],
                max_r[i, j]
            )


class EscapePlot(PlotValues):
    def __init__(
            self,
            L: float,
            E: float,
            resolution: int = 300,
            T: float = 1000,
            dt: float = 0.001,
            r_bounds=None,
            pr_bounds=None,
            path_name: str = 'escapes/'
    ):
        super().__init__(L, E, path_name=path_name)
        self.resolution = resolution
        if not r_bounds:
            self.r_bounds = self.get_r_bounds()
        else:
            self.r_bounds = r_bounds
        if not pr_bounds:
            self.pr_bounds = self.get_pr_bounds()
        else:
            self.pr_bounds = pr_bounds
        self.r_mesh, self.pr_mesh, self.time_mesh, self.pass_mesh, \
            self.return_mesh, self.max_z_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.asymptote_mesh = self.create_asymptote()
        self.T = T
        self.t_points = np.arange(0, T, dt)
        self.name = f"L={self.L}_E={self.E}"

    def create_io_meshes(self):
        r_mesh, pr_mesh = np.meshgrid(
            np.linspace(*self.r_bounds, self.resolution+2)[1:-1],
            np.linspace(*self.pr_bounds, self.resolution + 2)[1:-1]
        )

        return r_mesh, pr_mesh, *[np.full_like(r_mesh, np.nan) for _ in range(4)]

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
        qp_mesh = np.empty((*self.time_mesh.shape, 2, 2))
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pz_mesh
        )

    def create_asymptote(self):
        asymptote_mesh = np.empty((*self.time_mesh.shape, 2))
        for i, row in enumerate(self.r_mesh):
            for j, _ in enumerate(row):
                asymptote_mesh[i, j] = np.sqrt(
                    2 * (2 * self.h + self.L) - PLUS_MINUS * 4 * np.sqrt(
                        self.h ** 2 + self.L * self.h)
                )
        return asymptote_mesh

    def generate_meshes(self):
        block_size = 8
        grid_size = self.resolution//block_size + 1

        cuda_time_mesh = cuda.to_device(self.time_mesh)
        cuda_pass_mesh = cuda.to_device(self.pass_mesh)
        cuda_qp = cuda.to_device(self.qp_mesh)
        cuda_dqp = cuda.device_array_like(np.empty_like(self.qp_mesh))
        cuda_asym = cuda.to_device(self.asymptote_mesh)
        cuda_time = cuda.to_device(self.t_points)
        max_r = np.empty((*self.time_mesh.shape, 4))
        min_r = np.empty((*self.time_mesh.shape, 4))
        for i in range(4):
            min_r[:, :, i] = self.asymptote_mesh[:, :, 0]-1
            max_r[:, :, i] = self.asymptote_mesh[:, :, 1]+1
        cuda_max = cuda.to_device(max_r)
        cuda_min = cuda.to_device(min_r)
        time = perf_counter()
        cuda_escape_time[
            (grid_size, grid_size), (block_size, block_size)
        ](
            cuda_time_mesh,
            cuda_pass_mesh,
            cuda_qp,
            self.L,
            cuda_asym,
            cuda_dqp,
            cuda_time,
            cuda_max,
            cuda_min
        )
        self.time_mesh = cuda_time_mesh.copy_to_host()
        self.pass_mesh = cuda_pass_mesh.copy_to_host()

        print(f'time taken = {perf_counter()-time}')

    def generate_quantity_meshes(self):
        block_size = 8
        grid_size = self.resolution//block_size + 1

        cuda_max_z_mesh = cuda.to_device(self.max_z_mesh)
        cuda_return_mesh = cuda.to_device(self.return_mesh)
        cuda_qp = cuda.to_device(self.qp_mesh)
        cuda_dqp = cuda.device_array_like(np.empty_like(self.qp_mesh))
        cuda_asym = cuda.to_device(self.asymptote_mesh)
        cuda_time = cuda.to_device(self.t_points)
        max_r = np.empty((*self.time_mesh.shape, 4))
        min_r = np.empty((*self.time_mesh.shape, 4))
        for i in range(4):
            min_r[:, :, i] = self.asymptote_mesh[:, :, 0]-1
            max_r[:, :, i] = self.asymptote_mesh[:, :, 1]+1
        cuda_max = cuda.to_device(max_r)
        cuda_min = cuda.to_device(min_r)
        time = perf_counter()
        cuda_escape_quantities[
            (grid_size, grid_size), (block_size, block_size)
        ](
            cuda_max_z_mesh,
            cuda_return_mesh,
            cuda_qp,
            self.L,
            cuda_asym,
            cuda_dqp,
            cuda_time,
            cuda_max,
            cuda_min
        )

        self.max_z_mesh = cuda_max_z_mesh.copy_to_host()
        self.return_mesh = cuda_return_mesh.copy_to_host()
        for i, row in enumerate(self.r_mesh):
            for j, _ in enumerate(row):
                self.asymptote_mesh[i, j] = np.sqrt(
                    2 * (2 * self.h + self.L) - PLUS_MINUS * 4 * np.sqrt(
                        self.h ** 2 + self.L * self.h)
                )

        print(f'time taken = {perf_counter()-time}')


def escape_plots(L=1., h=1., resolution=300, T=300):
    P = energy_h_to_E(L, h)[0]
    plot = EscapePlot(
        L,
        P,
        resolution,
        T
    )
    plot.generate_meshes()

    name = plot.name
    plot.save(f'escape_time_{name}', plot.time_mesh)
    plot.save(f'escape_pass_{name}', plot.pass_mesh)

    plot.plot_zzc(r_buff=0, pr_buff=0)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plt.figure(figsize=figsize3)
    plot.savefig(f'escape_time_{name}')
    plt.clf()

    plot.plot_zzc(r_buff=0, pr_buff=0)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.pass_mesh)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig(f'escape_pass_{name}')
    plt.clf()


def generate_escape_plots_template(L, E, resolution=300, T=1_000, discrete=True, name=""):
    plot = EscapePlot(
        L,
        E,
        resolution,
        T
    )
    plot.generate_meshes()
    if not name:
        name = plot.name
    plot.save('escape_time/'+name, plot.time_mesh)
    plot.save('escape_pass/'+name, plot.pass_mesh)

    plt.figure(figsize=figsize7)
    plot.plot_zzc(r_buff=0, pr_buff=0)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig('escape_time/escape_time_'+name)
    plt.clf()

    plt.figure(figsize=figsize7)
    if discrete:
        N = 5
        plot.pass_mesh[plot.pass_mesh >= N] = np.nan
        plot.pass_mesh[plot.time_mesh != plot.time_mesh] = np.nan
        cmap = plt.cm.get_cmap("turbo", N)
        mesh_values = np.unique(plot.pass_mesh[~np.isnan(plot.pass_mesh)])
        plt.clf()
        mesh_plot = plt.pcolormesh(
            plot.r_mesh, plot.pr_mesh, plot.pass_mesh,
            cmap=cmap, vmin=np.min(mesh_values) - 0.5, vmax=np.max(mesh_values) + 0.5, rasterized=True
        )
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        plot.plot_zzc(r_buff=0, pr_buff=0)
        plt.colorbar(mesh_plot, ticks=mesh_values)

        plot.savefig('escape_pass/escape_pass_discrete_' + name)
        plt.clf()
    else:
        plot.plot_zzc(r_buff=0, pr_buff=0)
        plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.pass_mesh, rasterized=True)
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        plt.colorbar()
        plot.savefig('escape_pass/escape_pass_'+name)
        plt.clf()


def generate_quantity_plots_template(L, E, resolution=100, T=1000, name="", zoomed=False):
    plot = EscapePlot(
        L,
        E,
        resolution,
        T,
        r_bounds = [1.4, 1.6],
        pr_bounds = [0.2, 0.4]
    )
    plot.generate_quantity_meshes()
    if not name:
        name = f"L={L}_P={E}"
    plot.save(f'max_z/{name}', plot.max_z_mesh)
    plot.save(f'return_time/{name}', plot.return_mesh)

    plt.figure(figsize=figsize7)
    plot.plot_zzc(r_buff=0, pr_buff=0)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.max_z_mesh, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig(f'max_z/{name}')
    plt.clf()

    if zoomed:
        plot = EscapePlot(
            L,
            E,
            resolution,
            T,
            r_bounds=[1.4, 1.6],
            pr_bounds=[0.2, 0.4]
        )
        plot.generate_quantity_meshes()
        plt.figure(figsize=figsize7)
        plt.ylim([0.2, 0.4])
        plt.xlim([1.4, 1.6])
        plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.max_z_mesh, rasterized=True)
        plt.xlabel('$r$')
        plt.ylabel('$p_r$')
        plt.colorbar()
        plot.savefig(f'max_z/{name}_zoomed')
        plt.clf()

    """
    plt.figure(figsize=figsize7)
    plot.plot_zzc(r_buff=0, pr_buff=0)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.return_mesh, vmin=0, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig(f'return_time/{name}')
    plt.clf()
    """


def sample_escape_plot():
    L = 1.0
    E = 1.05
    generate_escape_plots_template(L, E, resolution=300, name="_sample")


def sample_quantity_plot():
    L = 1
    P = 1.05
    generate_quantity_plots_template(L, P, resolution=300, name="max_z_sample", zoomed=True)


def generate_escape_plots_zoomed_1():
    L = 1.
    E = 1.05
    resolution = 100
    T = 10_000
    plot = EscapePlot(
        L,
        E,
        resolution,
        T,
        r_bounds=[1.4, 1.6],
        pr_bounds=[0.2, 0.4]
    )
    plot.generate_meshes()
    plot.save('escape_time_zoomed', plot.time_mesh)
    plot.save('escape_pass_zoomed', plot.pass_mesh)

    plt.figure(figsize=figsize7)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig('escape_time_zoomed')
    plt.clf()

    plt.figure(figsize=figsize7)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.pass_mesh, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig('escape_pass_zoomed')
    plt.clf()


def generate_escape_plots_zoomed_2():
    L = 1.
    E = 1.05
    resolution = 100
    T = 10_000
    plot = EscapePlot(
        L,
        E,
        resolution,
        T,
        r_bounds= [1.05, 1.1],
        pr_bounds= [-0.625, -0.62]
    )
    plot.generate_meshes()
    plot.save('escape_time_zoomed_2', plot.time_mesh)
    plot.save('escape_pass_zoomed_2', plot.pass_mesh)

    plt.figure(figsize=figsize7)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig('escape_time_zoomed_2')
    plt.clf()

    plt.figure(figsize=figsize7)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.pass_mesh, rasterized=True)
    plt.xlabel('$r$')
    plt.ylabel('$p_r$')
    plt.colorbar()
    plot.savefig('escape_pass_zoomed_2')
    plt.clf()
