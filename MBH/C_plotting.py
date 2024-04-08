import matplotlib.pyplot as plt
import matplotlib
import os

from B_numerics import *

matplotlib.rcParams.update(
    {'font.size': 12,
     'lines.linewidth': 1,
     'figure.dpi': 300,
     'markers.fillstyle': 'full',
     'image.cmap': 'turbo',
     "figure.figsize": (4.5, 3)}
)
turbo_cmap = matplotlib.cm.get_cmap('turbo')


figsize1 = (4.0, 5.6)
figsize2 = (7, 4)
figsize3 = (4.5, 3)
figsize4 = (4.5, 4.5)
figsize5 = (8, 8)
figsize6 = (20, 20)
figsize7 = (5.5, 4.5)


folder = os.path.expanduser("~/Desktop/MS Thesis/BH") + '/Figures/'


def savefig(name: str, path_name: str = '', pdf=True):
    os.makedirs(os.path.dirname(f'{folder}{path_name}{name}.pdf'), exist_ok=True)
    if pdf:
        plt.savefig(f'{folder}{path_name}{name}.pdf', bbox_inches="tight")
    else:
        plt.savefig(f'{folder}{path_name}{name}.png', bbox_inches="tight")


def save(name: str, array: ndarray, path_name: str = ''):
    os.makedirs(os.path.dirname(f'{folder}{path_name}{name}.pdf'), exist_ok=True)
    np.save(f'{folder}{path_name}{name}', array)


def load(name: str, path_name: str = ''):
    return np.load(f'{folder}{path_name}{name}.npy')


class Plot:
    """
    Generic class for plots
    """

    def __init__(self, path_name=''):
        self.path_name = path_name

    def save(self, name: str, array: ndarray):
        save(name, array, self.path_name+'arrays/')

    def savefig(self, name: str, pdf=True):
        savefig(name, self.path_name+'figures/', pdf=pdf)

    def load(self, name: str):
        return load(name, self.path_name+'arrays/')


class PlotValues(Plot, Values):
    """
    Plots special curves for a given set of constants of motions.
    """

    def __init__(self, r_esc: float, p: float, E: float, path_name: str = ''):
        Plot.__init__(self, path_name)
        Values.__init__(self, r_esc, p, E)

    def plot_zvc(self, z_bounds: ndarray, r_buff: float = 0.1, resolution: int = 100, r_bounds=None, color='k'):
        """
        Plots the (r,z) curve with zero velocity for a given energy. Known as the zero velocity curve or zvc.
        :param z_bounds: any array whose minimum and maximum values gives up to where the curve is plotted along z.
        :param r_buff: amount of extra space along r outside the curve.
        :param resolution: resolution
        :return: all the points of the zvc; it is also plotted.
        """
        if not r_bounds:
            r_bounds = self.get_r_bounds()
        if len(r_bounds) == 1:
            bounds = np.array([0, r_bounds[0]])
        else:
            bounds = r_bounds[-2:]

        r_mesh, z_mesh = np.meshgrid(
            np.linspace(
                bounds[0] - r_buff * np.diff(bounds),
                bounds[1] + r_buff * np.diff(bounds),
                resolution
            ),
            np.linspace(np.min(z_bounds), np.max(z_bounds), resolution)
        )
        E = self.get_potential_energy(*cyl_to_sphere(r_mesh, z_mesh))
        cs = plt.contour(r_mesh, z_mesh, E, np.array([self.E]), colors=color, linestyles='--')
        if len(r_bounds) == 1:
            plt.contour(r_mesh, z_mesh, r_mesh**2+z_mesh**2, np.array([1]), colors="k")
        return cs.allsegs[0]

    def plot_zzc(self, r_buff: float = 0, pr_buff: float = 0, resolution: int = 100):
        """
        Plots the phase motion along the position and momentum of r of the purely equatorial motion.
        """
        r_bounds = self.get_r_bounds()
        if len(r_bounds) == 1:
            r_bounds = np.array([1, r_bounds[0]])
        else:
            r_bounds = r_bounds[-2:]
        pr_bounds = (1 + pr_buff) * self.get_pr_bounds()
        r_mesh, pr_mesh = np.meshgrid(
            np.linspace(r_bounds[0] - r_buff * np.diff(r_bounds),
                        r_bounds[1] + r_buff * np.diff(r_bounds),
                        resolution),
            np.linspace(*pr_bounds, resolution)
        )
        E = pr_mesh ** 2 + self.get_potential_energy(r_mesh, PI/2)
        cs = plt.contour(r_mesh, pr_mesh, E, np.array([self.E]), colors='k', linestyles='-', linewidths=1)
        return cs.allsegs[0][0]

    def plot_multi_zvc(self, z_bounds: ndarray, h_values: ndarray, r_bounds=None, r_buff: float = 0.1, resolution: int=100):
        """
        Plots multiple zero velocity curves of decreasing energy.
        """
        if not r_bounds:
            r_bounds = self.get_r_bounds()
        if len(r_bounds) == 1:
            r_bounds = np.array([0, r_bounds[0]])
        print(r_bounds)
        r_mesh, z_mesh = np.meshgrid(
            np.linspace(
                r_bounds[0] - r_buff * np.diff(r_bounds),
                r_bounds[1] + r_buff * np.diff(r_bounds),
                resolution
            ),
            np.linspace(np.min(z_bounds), np.max(z_bounds), resolution)
        )
        H = self.get_potential_energy_cyl(r_mesh, z_mesh)
        cs = plt.contour(r_mesh, z_mesh, H, h_values, colors='k', linestyles='--')
        return cs.allsegs[0][0], r_mesh, z_mesh, H

    def plot_asym(self):
        asym = self.get_asymptotes()
        for i in asym:
            plt.axvline(i, linestyle='--', c='r', zorder=2)

    def plot_equator(self):
        r_bounds = self.get_r_bounds()
        equator = np.zeros(2)
        if len(r_bounds) == 1:
            bounds = np.array([1, r_bounds[0]])
        else:
            bounds = r_bounds[-2:]

        plt.plot(bounds, equator, linestyle='--', c='orange')


class PlotTrajectory(PlotValues):
    """
    Generates the trajectory for a set of initial conditions.
    """
    def __init__(
            self,
            r_esc: float, p: float, E: float, T: float,
            r: float, th: float = PI/2, phi: float = 0, pr: float = 0, pth: float = 0,
            dt: float = 0.001,
            path_name: str = 'trajectory/'
    ):
        super().__init__(r_esc, p, E, path_name)
        r_bounds = self.get_r_bounds()
        if len(r_bounds) == 1:
            if not (1 < r <= r_bounds[0]):
                print(r_bounds)
                raise ValueError(f"r is invalid")
        else:
            if not (r_bounds[-2] < r <= r_bounds[-1]):
                print(r_bounds)
                raise ValueError(f"r is invalid")

        pr = np.sqrt(escape_energy(self.B, self.L))*pr/(1-1/r)
        pth = -r*np.sqrt((self.E-self.get_potential_energy(r, th)-((1-1/r)*pr)**2)/(1-1/r))
        self.qp = np.array((
            (r, th),
            (pr, pth)
        ), float)
        self.T = T/np.sqrt(escape_energy(self.B, self.L))
        self.dt = dt
        self.q_points = np.empty(2)
        self.p_points = np.empty(2)
        self.t_points = np.empty(1)

    def get_trajectory(self):
        self.q_points, self.p_points, self.t_points = integrate(self.qp, self.T, self.B, self.L, self.E, self.dt)
        return self.q_points, self.p_points, self.t_points

    def plot_trajectory(self, plotZVC=True, label=None):
        self.get_trajectory()
        if plotZVC:
            self.plot_zvc(sphere_to_cyl(*self.q_points)[1], resolution=1000)
        if self.E > escape_energy(self.B, self.L):
            self.plot_asym()

        plt.plot(*sphere_to_cyl(self.q_points[0], self.q_points[1]), rasterized=True, label=label)
        self.plot_equator()
        if not label:
            plt.legend()
        plt.xlabel('$r$')
        plt.ylabel('$z$')


def L_eq_plus_plot():
    resolution = 200
    p_values = [1.1, 1.5, 1.9, 2.3, 2.7]
    p_values.reverse()
    print(p_values)
    color = ["r", "orange", "y", "g", "b", "darkviolet"]
    r = np.linspace(1.0, 3.0, resolution + 1)[1:]
    L_plot = r * np.sqrt((3 * r - 1) / (2 * (4 * r ** 2 - 9 * r + 3 + np.sqrt((3 * r - 1) * (3 - r)))))
    y_min = L_plot[-1] * 0.5
    print(L_plot[-1])
    y_max = 20
    plt.figure(figsize=figsize3)
    plt.plot(r, L_plot, c="k", linestyle="dashed")
    L_values = [15, 10, 5, 3, 2, 1.733]
    for i, L in enumerate(L_values):
        p = np.round(r[np.argmin(np.abs(L_plot - L))], 2)
        B = np.sqrt((3 - p) / (2 * (4 * p ** 2 - 9 * p + 3 + np.sqrt((3 * p - 1) * (3 - p))))) / p
        L_min = p * np.sqrt((3 * p - 1) / (2 * (4 * p ** 2 - 9 * p + 3 + np.sqrt((3 * p - 1) * (3 - p)))))
        L_max = y_max
        L_points = np.linspace(L_min, L_max, resolution + 1)
        r_plus = np.empty_like(L_points)
        r_plus[0] = p
        r_minus = np.empty_like(L_points)
        r_minus[0] = p
        for j, L in enumerate(L_points[1:]):
            r_eq = np.roots([2 * B ** 2, -B ** 2, 0, 1 - 2 * B * L, -2 * L ** 2, 3 * L ** 2])
            r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))
            r_plus[j + 1] = r_eq[1]
            r_minus[j + 1] = r_eq[0]
        plt.plot(p, L_min, '.', c=color[i], markersize=6)
        plt.plot(r_plus, L_points, c=color[i])
        plt.plot(r_minus, L_points, c=color[i], label=f"$\\rho$={p}")
    plt.xlabel("$r_{eq}$")
    plt.ylabel("L")
    plt.ylim(bottom=y_min, top=y_max)
    plt.xlim(left=1, right=4)
    plt.legend(loc="upper right")
    savefig("L_eq_plus")


def L_eq_minus_plot():
    resolution = 1000
    p_values = [1.1, 1.5, 1.9, 2.3, 2.7]
    p_values.reverse()
    print(p_values)
    color = ["r", "orange", "y", "g", "b", "darkviolet"]
    r = np.linspace((5 + np.sqrt(13)) / 4, 3.0, resolution + 1)[1:]
    L_plot = -r * np.sqrt((3 * r - 1) / (2 * (4 * r ** 2 - 9 * r + 3 - np.sqrt((3 * r - 1) * (3 - r)))))
    y_max = np.min(L_plot)
    y_min = np.max(L_plot)
    print(L_plot[-1])
    y_max = -20
    plt.figure(figsize=figsize3)
    plt.plot(r, L_plot, c="k", linestyle="dashed")
    p_values = [2.2, 2.4, 2.6, 2.8]
    L_values = [-15, -10, -5, -3, -2]
    for i, L in enumerate(L_values):
        p = np.round(r[np.argmin(np.abs(L_plot - L))], 2)
        B = np.sqrt((3 - p) / (2 * (4 * p ** 2 - 9 * p + 3 - np.sqrt((3 * p - 1) * (3 - p))))) / p
        L_min = -p * np.sqrt((3 * p - 1) / (2 * (4 * p ** 2 - 9 * p + 3 - np.sqrt((3 * p - 1) * (3 - p)))))
        L_max = y_max
        L_points = np.linspace(L_min, L_max, resolution + 1)
        r_plus = np.empty_like(L_points)
        r_plus[0] = p
        r_minus = np.empty_like(L_points)
        r_minus[0] = p
        for j, L in enumerate(L_points[1:]):
            r_eq = np.roots([2 * B ** 2, -B ** 2, 0, 1 - 2 * B * L, -2 * L ** 2, 3 * L ** 2])
            r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))
            r_plus[j + 1] = r_eq[1]
            r_minus[j + 1] = r_eq[0]
        plt.plot(p, L_min, '.', c=color[i], markersize=6)
        plt.plot(r_plus, L_points, c=color[i])
        plt.plot(r_minus, L_points, c=color[i], label=f"$\\rho$={p}")
    plt.xlabel("$r_{eq}$")
    plt.ylabel("L")
    plt.ylim(bottom=y_max, top=y_min * 0.5)
    plt.xlim(left=1.3, right=10)
    plt.legend(loc="lower right")
    savefig("L_eq_minus")


def potential_plot():
    def U_points(L, B, r):
        return (1-1/r)*(1+(L/r-B*r)**2)

    r_points = np.linspace(1.0, 3.0, 100)
    p = 1.5
    B_isco = np.sqrt((3-p)/(2*(4*p**2-9*p+3+np.sqrt((3*p-1)*(3-p)))))/p
    L_isco = p*np.sqrt((3*p-1)/(2*(4*p**2-9*p+3+np.sqrt((3*p-1)*(3-p)))))
    plt.figure(figsize=figsize3)

    plt.plot(r_points, U_points(1.5, B_isco, r_points), c="C0", label="$|L|<|L_{isco}|$")

    plt.plot(r_points, U_points(L_isco, B_isco, r_points), c="C1", label="$|L|=|L_{isco}|$")
    plt.plot(p, U_points(L_isco, B_isco, p), '.', c="C1")

    B = B_isco
    L = 4
    r_eq = np.roots([2 * B ** 2, -B ** 2, 0, 1 - 2 * B * L, -2 * L ** 2, 3 * L ** 2])
    r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))
    for r in r_eq:
        plt.plot(r, U_points(4, B_isco, r), '.', c="C2")

    plt.plot(r_points, U_points(4, B_isco, r_points), c="C2", label="$|L|>|L_{isco}|$")
    plt.legend()
    xticks = [1, 2, 3]
    plt.xticks(np.append(xticks, 1.5), np.append(xticks, "$\\rho$"))
    plt.xlim(left=xticks[0], right=xticks[-1])
    plt.ylim(bottom=0)
    plt.xlabel("R")
    plt.ylabel("U")
    savefig("U_eqs")


def Contour():
    r_esc = 2.5
    p = 1.4
    E = 1.0
    values = PlotValues(r_esc, p, E)

    z_max = 4.
    z_min = -z_max
    z_bounds = np.array((z_min, z_max), float)
    print(values.get_U_eq())
    energies = np.sort(
        np.arange(values.get_U_eq()[1], values.get_U_eq()[0], -np.diff(values.get_U_eq()) / 5
        )
    )

    plt.figure(figsize=figsize1)
    _, r, z, H = values.plot_multi_zvc(z_bounds, energies, r_bounds=[0., 4.0], resolution=500)
    PlotValues(r_esc, p, values.get_U_eq()[0]).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500, color="red")
    PlotValues(r_esc, p, 1.0).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500, color="blue")
    PlotValues(r_esc, p, 0.).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500)
    PlotValues(r_esc, p, values.get_potential_energy_cyl(3.5, 0)).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500)
    ctr_f = plt.pcolormesh(r, z, H, vmin=0, vmax=1.6, cmap='bone', alpha=0.65, rasterized=True)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=ctr_f.norm, cmap=ctr_f.cmap), ticks=(0, 1, *values.get_U_eq()))
    cbar.ax.set_yticklabels(('$E=0$', '$E=U_{esc}$', '$E=U_{un}$', '$E=U_{st}$'))
    plt.xlabel('$r$')
    plt.ylabel('$z$', rotation='horizontal')
    plt.xlim([0., 4.0])
    for r in values.get_r_eq():
        plt.plot(r, 0, '.', c='k', rasterized=True)
    values.savefig('contour')


def Contour2():
    r_esc = 2.1
    p = 1.4
    E = 1.0
    values = PlotValues(r_esc, p, E)

    z_max = 4.
    z_min = -z_max
    z_bounds = np.array((z_min, z_max), float)
    print(values.get_U_eq())
    energies = np.sort(
        np.arange(values.get_U_eq()[1], values.get_U_eq()[0], -np.diff(values.get_U_eq()) / 5
        )
    )

    plt.figure(figsize=figsize1)
    _, r, z, H = values.plot_multi_zvc(z_bounds, energies, r_bounds=[0., 4.0], resolution=500)
    PlotValues(r_esc, p, values.get_U_eq()[0]).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500, color="red")
    PlotValues(r_esc, p, 1.0).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500, color="blue")
    PlotValues(r_esc, p, 0.).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500)
    PlotValues(r_esc, p, values.get_potential_energy_cyl(3, 0)).plot_zvc(z_bounds, r_bounds=[0, 4.0], resolution=500)
    ctr_f = plt.pcolormesh(r, z, H, vmin=0, vmax=1.6, cmap='bone', alpha=0.65, rasterized=True)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=ctr_f.norm, cmap=ctr_f.cmap), ticks=(0, 1, *values.get_U_eq()))
    cbar.ax.set_yticklabels(('$E=0$', '$E=U_{esc}$', '$E=U_{un}$', '$E=U_{st}$'))
    plt.xlabel('$r$')
    plt.ylabel('$z$', rotation='horizontal')
    plt.xlim([0., 4.0])
    for r in values.get_r_eq():
        plt.plot(r, 0, '.', c='k', rasterized=True)
    values.savefig('contour2')


def min_asym():
    r_esc = 2.1
    p = 1.4
    E = 1.0
    values = PlotValues(r_esc, p, E)
    z_max = 3
    z_bounds = np.array((-z_max, z_max))
    print(values.E)
    plt.figure(figsize=figsize1)
    values.plot_zvc(z_bounds)
    plt.axis('tight')
    plt.xlabel('$r$')
    plt.ylabel('$z$', rotation='horizontal')
    plt.axvline(values.get_asymptotes()[0], linestyle='--', c='r', rasterized=True)
    plt.xlim([0, 3])
    plt.xticks([values.get_asymptotes()[0]], ["$r_{esc}$"])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    values.savefig(f'{min_asym.__name__}')
    plt.clf()


def asym():
    r_esc = 2.1
    p = 1.4
    E = 1.3
    values = PlotValues(r_esc, p, E)
    z_max = 3
    z_bounds = np.array((-z_max, z_max))

    plt.figure(figsize=figsize1)
    values.plot_zvc(z_bounds)
    plt.axis('tight')
    plt.xlabel('$r$')
    plt.ylabel('$z$', rotation='horizontal')
    asymptotes = values.get_asymptotes()
    plt.axvline(asymptotes[0], linestyle='--', c='r', rasterized=True)
    plt.axvline(asymptotes[1], linestyle='--', c='r', rasterized=True)
    plt.fill_between(asymptotes, *z_bounds, color='r', alpha=0.2, rasterized=True)
    plt.xticks(asymptotes, ["$r_{esc,-}$", "$r_{esc,+}$"])
    plt.xlim([0, 3])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    values.savefig(f'{asym.__name__}')
    plt.clf()


def resc_plot():
    resolution = 500
    p_values = [1.1, 1.5, 1.9, 2.3, 2.7]
    p_values.reverse()
    print(p_values)
    color = ["r", "orange", "y", "g", "b", "darkviolet"]
    r = np.linspace(1.0, 3.0, resolution + 2)[1:-1]
    resc_plot_points = r*((3*r-1)/(3-r))**(1/4)
    print(resc_plot_points)
    y_min = 0
    y_max = 40
    plt.figure(figsize=figsize3)
    plt.plot(r, resc_plot_points, c="k", linestyle="dashed")
    r_esc_values = [1.1, 3, 5, 10, 15]
    for i, r_esc in enumerate(r_esc_values):
        p = np.round(r[np.argmin(np.abs(resc_plot_points - r_esc))], 2)
        resc_max = y_max
        resc_min = p*((3*p-1)/(3-p))**(1/4)
        resc_points = np.linspace(resc_min, resc_max, resolution + 1)
        r_plus = np.empty_like(resc_points)
        r_plus[0] = p
        r_minus = np.empty_like(resc_points)
        r_minus[0] = p
        for j, r_esc in enumerate(resc_points[1:]):
            L = r_esc * ((3-p)*(3*p-1))**(1/4)/np.sqrt(2*(4*p**2-9*p+3+np.sqrt((3*p-1)*(3-p))))
            B = L / r_esc ** 2
            r_eq = np.roots([2 * B ** 2, -B ** 2, 0, 1 - 2 * B * L, -2 * L ** 2, 3 * L ** 2])
            r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))
            r_plus[j + 1] = r_eq[1]
            r_minus[j + 1] = r_eq[0]
        plt.plot(p, resc_min, '.', c=color[i], markersize=6)
        plt.plot(r_plus, resc_points, c=color[i])
        plt.plot(r_minus, resc_points, c=color[i], label=f"$\\rho$={p}")
    plt.xlabel("$r_{eq}$")
    plt.ylabel("$r_{esc}$")
    plt.ylim(bottom=y_min, top=y_max)
    plt.xlim(left=1, right=5)
    plt.legend(loc="upper right")
    savefig("resc")
