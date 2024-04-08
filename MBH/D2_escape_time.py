import matplotlib.pyplot as plt

from C_plotting import *
from numba import prange, cuda
from time import perf_counter


@njit(parallel=True)
def create_mesh(qp_mesh, r_mesh, pr_mesh, pth_mesh):
    n = qp_mesh.shape
    for i in prange(n[0]):
        for j in prange(n[1]):
            qp_mesh[i, j] = np.array((
                [r_mesh[i, j], PI/2],
                [pr_mesh[i, j]/(1-1/r_mesh[i, j]), pth_mesh[i, j]]
            ))
    return qp_mesh


@cuda.jit(device=True)
def calculate_escape_time(qp, b, L, E, asymptotes, dqp, t_points, min_r, max_r):
    min_r_index = 0
    max_r_index = 0
    pass_index = 0
    for i, j in enumerate(t_points):
        r_i = qp[0, 0]
        th_i = qp[0, 1]
        pr_i = qp[1, 0]
        pth_i = qp[1, 1]
        symplectic_cuda(qp, dqp, b, L, E, 0.001)
        r_f = qp[0, 0]
        pr_f = qp[1, 0]
        pth_f = qp[1, 1]
        th_f = qp[0, 1]

        if cos(th_i)*cos(th_f) <= 0:
            pass_index += 1

        if (pr_i*sin(th_i)+pth_i*r_i*cos(th_i))*(pr_f*sin(th_f)+pth_f*r_f*cos(th_f)) < 0 \
                < cos(th_f)*(pr_f*cos(th_f)-pth_f*r_f*sin(th_f)):
            if 0 < (pr_i*sin(th_i)+pth_i*r_i*cos(th_i)):
                max_r[max_r_index] = r_f*sin(th_f)
                if max_r_index >= max_r.shape[0]-1:
                    max_r_index = 0
                else:
                    max_r_index += 1
            else:
                min_r[min_r_index] = r_f*sin(th_f)
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
            if cos(th_f) > 0:
                return j, pass_index, 1
            elif cos(th_f) < 0:
                return j, pass_index, -1
        if r_f <= 1:
            return j, pass_index, 0
    return np.nan, pass_index, np.nan


@cuda.jit
def cuda_escape_time(
        esc_time_mesh, pass_mesh, basin_mesh, qp_mesh, b, L, E, asymptote, dqp, t_points, max_r, min_r
        ):
    i, j = cuda.grid(2)

    i_in_range = (0 <= i) and (i < esc_time_mesh.shape[0])
    j_in_range = (0 <= j) and (j < esc_time_mesh.shape[1])

    if i_in_range and j_in_range:
        if qp_mesh[i, j, 1, 1] == qp_mesh[i, j, 1, 1]:
            esc_time_mesh[i, j], pass_mesh[i, j], basin_mesh[i, j] = calculate_escape_time(
                qp_mesh[i, j],
                b,
                L,
                E,
                asymptote[i, j],
                dqp[i, j],
                t_points,
                min_r[i, j],
                max_r[i, j]
            )


class EscapePlot(PlotValues):
    def __init__(
            self,
            r_esc: float, p: float, E: float,
            resolution: int = 100,
            T: float = 1000,
            dt: float = 0.001,
            r_bounds=None,
            pr_bounds=None,
            path_name: str = 'escapes/'
    ):
        super().__init__(r_esc, p, E, path_name=path_name)
        self.resolution = resolution
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
        self.r_mesh, self.pr_mesh, self.time_mesh, self.pass_mesh, self.basin_mesh = self.create_io_meshes()
        self.qp_mesh = self.create_qp_mesh()
        self.asymptote_mesh = self.create_asymptote()
        self.T = T
        self.t_points = np.arange(0, T, dt)
        self.name = f"resc={r_esc}_p={p}_E={self.E}"

    def create_io_meshes(self):
        r_mesh, pr_mesh = np.meshgrid(
            np.linspace(*self.r_bounds, self.resolution+2)[1:-1],
            np.linspace(*self.pr_bounds, self.resolution + 2)[1:-1]
        )

        return r_mesh, pr_mesh, *[np.full_like(r_mesh, np.nan) for _ in range(3)]

    def create_qp_mesh(self):
        pth_mesh = np.full_like(self.r_mesh, np.nan)
        for i, row in enumerate(self.r_mesh):
            for j, r in enumerate(row):
                pth_sq = r**2*(self.E-self.get_potential_energy(r, PI/2)-(self.pr_mesh[i, j])**2)/(1-1/r)
                if pth_sq >= 0:
                    pth_mesh[i, j] = -np.sqrt(pth_sq)
        qp_mesh = np.empty((*np.shape(self.time_mesh), 2, 2))
        return create_mesh(
            qp_mesh,
            self.r_mesh,
            self.pr_mesh,
            pth_mesh
        )

    def create_asymptote(self):
        asymptote_mesh = np.empty((*self.time_mesh.shape, 2))
        for i, row in enumerate(self.r_mesh):
            for j, _ in enumerate(row):
                asymptote_mesh[i, j] = self.get_asymptotes()
        return asymptote_mesh

    def generate_meshes(self):
        block_size = 8
        grid_size = self.resolution//block_size + 1

        cuda_time_mesh = cuda.to_device(self.time_mesh)
        cuda_pass_mesh = cuda.to_device(self.pass_mesh)
        cuda_basin_mesh = cuda.to_device(self.basin_mesh)
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
            cuda_basin_mesh,
            cuda_qp,
            self.B,
            self.L,
            self.E,
            cuda_asym,
            cuda_dqp,
            cuda_time,
            cuda_max,
            cuda_min
        )
        self.time_mesh = cuda_time_mesh.copy_to_host()
        self.pass_mesh = cuda_pass_mesh.copy_to_host()
        self.basin_mesh = cuda_basin_mesh.copy_to_host()

        print(f'time taken = {perf_counter()-time}')


def EscapePlotTemplate(r_esc, p, E, resolution: int = 100, discrete=True):
    plot = EscapePlot(r_esc, p, E, resolution=resolution)
    name = "__hires_"+plot.name
    T = plot.T
    try:
        plot.time_mesh = plot.load(f'escape_time_{name}')
        plot.pass_mesh = plot.load(f'escape_pass_{name}')
        plot.basin_mesh = plot.load(f'escape_basin_{name}')
    except FileNotFoundError:
        plot.generate_meshes()
        plot.save(f'escape_time_{name}', plot.time_mesh)
        plot.save(f'escape_pass_{name}', plot.pass_mesh)
        plot.save(f'escape_basin_{name}', plot.basin_mesh)

    plt.figure(figsize=figsize3)
    plot.plot_zzc()
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    plt.colorbar()
    plot.savefig(f'escape_time/escape_time_{name}')
    plt.clf()
    if discrete:
        N = 5
        plot.pass_mesh[plot.pass_mesh >= N] = np.nan
        plot.pass_mesh[plot.time_mesh != plot.time_mesh] = np.nan
        cmap = plt.cm.get_cmap("turbo", N)
        mesh_values = np.unique(plot.pass_mesh[~np.isnan(plot.pass_mesh)])
        mesh_plot = plt.pcolormesh(
            plot.r_mesh, plot.pr_mesh, plot.pass_mesh,
            cmap=cmap, vmin=np.min(mesh_values) - 0.5, vmax=np.max(mesh_values) + 0.5, rasterized=True
        )
        plt.xlabel('$R$')
        plt.ylabel('$p^R$')
        plt.colorbar(mesh_plot, ticks=mesh_values)

        plot.savefig('escape_pass/escape_pass_discrete_' + name)
        plt.clf()

    plt.figure(figsize=figsize3)
    plot.plot_zzc()
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.pass_mesh, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    plt.colorbar()
    plot.savefig(f'escape_pass/escape_pass_{name}')
    plt.clf()

    plt.figure(figsize=figsize3)
    plot.plot_zzc()
    cmap = matplotlib.colors.ListedColormap(["blue", "black", "red"])
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.basin_mesh, cmap=cmap, vmin=-1, vmax=1, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    #plt.colorbar(basin_plot, ticks=[-1, 0, 1])
    plot.savefig(f'escape_basin/escape_basin_{name}')
    plt.close("all")


def generate_escape():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]
    for E in E_values:
        for p in p_values:
            for r_esc in r_esc_values:
                plot = EscapePlot(r_esc, p, E)
                try:
                    plot.load(f'escape_time_{plot.name}')
                except FileNotFoundError:
                    print(E, p, r_esc)
                    EscapePlotTemplate(r_esc, p, E)


def generate_hires_escape():
    E = 1.2
    r_esc = 1.8
    p = 1.2
    resolution = 300
    EscapePlotTemplate(r_esc, p, E, resolution)

    r_esc = 4.0
    p = 1.8
    EscapePlotTemplate(r_esc, p, E, resolution)

    r_esc = 3.0
    p = 2.8
    EscapePlotTemplate(r_esc, p, E, resolution)


def percent_plot():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]

    num = np.empty((np.shape(p_values)[0], np.shape(r_esc_values)[0], np.shape(E_values)[0]))
    width = np.empty_like(num)
    plt.figure(figsize=figsize3)
    for i, p in enumerate(p_values):
        for j, r_esc in enumerate(r_esc_values):
            for k, E in enumerate(E_values):
                plot = EscapePlot(r_esc, p, E)
                name = plot.name
                plot.pass_mesh = plot.load(f'escape_pass_{name}')
                plot.basin_mesh = plot.load(f'escape_basin_{name}')
                total = np.count_nonzero(plot.pass_mesh == plot.pass_mesh)
                plot.pass_mesh[plot.basin_mesh != 1] = np.nan
                one = np.count_nonzero(plot.pass_mesh == 0)
                num[i, j, k] = one / total
                val = plot.get_asymptotes()
                width[i, j, k] = np.abs(np.diff(val))
        plt.scatter(width[i], num[i], label=f"$\\rho$={p}", marker=".", alpha=0.8, rasterized=True)
    plt.legend(loc=(1, 0))
    plt.xlabel("channel width ($r_{esc,+}-r_{esc,-}$)")
    plt.ylabel("d$N_0/N_0$")
    savefig("percent_plot")


def sample_capture_trajectory():
    r_esc = 1.8
    p = 1.2
    E = 1.2
    T = 4
    PlotTrajectory(r_esc, p, E, T, r=1.25, pr=-0.1).plot_trajectory(label="$p^R=-0.1$")
    PlotTrajectory(r_esc, p, E, T, r=1.25, pr=-0.11).plot_trajectory(plotZVC=False, label="$p^R=-0.11$")
    plt.xlim(left=1.0, right=1.5)
    plt.legend()
    savefig("capture_scatter")


def capture_percent_plot():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]

    num = np.empty((np.shape(E_values)[0], np.shape(p_values)[0], np.shape(r_esc_values)[0]))
    resc = np.empty_like(num)
    plt.figure(figsize=figsize3)
    for i, E in enumerate(E_values):
        print(E)
        for j, p in enumerate(p_values):
            for k, r_esc in enumerate(r_esc_values):
                plot = EscapePlot(r_esc, p, E)
                name = plot.name
                plot.pass_mesh = plot.load(f'escape_pass_{name}')
                plot.basin_mesh = plot.load(f'escape_basin_{name}')
                total = np.count_nonzero(plot.pass_mesh == plot.pass_mesh)
                #plot.pass_mesh[plot.basin_mesh != 1] = np.nan
                capture = np.count_nonzero(plot.basin_mesh == 0)
                num[i, j, k] = 100*capture / total
                resc[i, j, k] = r_esc
        plt.scatter(resc[i], num[i], label=f"$E$={E}", marker=".", alpha=0.8, rasterized=True)
    plt.legend(loc=(1, 0))
    plt.xlabel("$r_{esc}$")
    plt.ylabel("% of captured trajectories")
    savefig("capture percent_plot")
