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


figsize1 = (3.5, 5.6)
figsize2 = (7, 4)
figsize3 = (4.5, 3)
figsize4 = (4.5, 4.5)
figsize5 = (8, 8)
figsize6 = (20, 20)
figsize7 = (5.5, 4.5)


folder = os.path.expanduser("~/Desktop/MS Thesis") + '/Figures/'


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

    def __init__(self, L: float, energy: float, is_E: bool = True, path_name: str = ''):
        Plot.__init__(self, path_name)
        Values.__init__(self, L, energy, is_E)

    def plot_zvc(self, z_bounds: ndarray, r_buff: float = 0.1, resolution: int = 100):
        """
        Plots the (r,z) curve with zero velocity for a given energy. Known as the zero velocity curve or zvc.
        :param z_bounds: any array whose minimum and maximum values gives up to where the curve is plotted along z.
        :param r_buff: amount of extra space along r outside the curve.
        :param resolution: resolution
        :return: all the points of the zvc; it is also plotted.
        """
        r_bounds = self.get_r_bounds()
        r_mesh, z_mesh = np.meshgrid(
            np.linspace(
                r_bounds[0] - r_buff * np.diff(r_bounds),
                r_bounds[1] + r_buff * np.diff(r_bounds),
                resolution
            ),
            np.linspace(np.min(z_bounds), np.max(z_bounds), resolution)
        )
        H = kinetic_energy(self.L, r_mesh, z_mesh, 0)
        cs = plt.contour(r_mesh, z_mesh, H, np.array([self.h]), colors='k', linestyles='--')
        return cs.allsegs[0]

    def plot_multi_zvc(self, z_bounds: ndarray, h_values: ndarray, r_buff: float = 0.1, resolution: int = 100):
        """
        Plots multiple zero velocity curves of decreasing energy.
        """
        r_bounds = self.get_r_bounds()
        r_mesh, z_mesh = np.meshgrid(
            np.linspace(
                r_bounds[0] - r_buff * np.diff(r_bounds),
                r_bounds[1] + r_buff * np.diff(r_bounds),
                resolution
            ),
            np.linspace(np.min(z_bounds), np.max(z_bounds), resolution)
        )
        H = kinetic_energy(self.L, r_mesh, z_mesh, 0)
        cs = plt.contour(r_mesh, z_mesh, H, h_values, colors='k', linestyles='--')
        return cs.allsegs[0][0], r_mesh, z_mesh, H

    def plot_zzc(self, r_buff: float = 0.02, pr_buff: float = 0.04, resolution: int = 100, color='k', alpha=1):
        """
        Plots the (r,pr) curve along the equatorial plane. Used for the boundary of the Poincare plots.
        """
        r_bounds = self.get_r_bounds()
        pr_bounds = (1+pr_buff)*self.get_pr_bounds()
        r_mesh, pr_mesh = np.meshgrid(
            np.linspace(r_bounds[0] - r_buff * np.diff(r_bounds),
                        r_bounds[1] + r_buff * np.diff(r_bounds),
                        resolution),
            np.linspace(*pr_bounds, resolution)
        )
        H = kinetic_energy(self.L, r_mesh, 0, -pr_mesh ** 2 / 2)
        cs = plt.contour(r_mesh, pr_mesh, H, np.array([self.h]), colors=color, alpha=alpha, linestyles='-', linewidths=1)
        return cs.allsegs[0][0]

    def plot_r_bounds(self, resolution: int = 100):
        """
        Plots the toroidal bounds in the equatorial plane in Cartesian (x,y).
        """
        r_bounds = self.get_r_bounds()
        theta = np.linspace(0, 2*np.pi, resolution)
        r_min = np.full_like(theta, r_bounds[0])
        r_max = np.full_like(theta, r_bounds[1])
        x_min, y_min = p2c(r_min, theta)
        x_max, y_max = p2c(r_max, theta)
        plt.plot(x_min, y_min, '--', c='k')
        plt.plot(x_max, y_max, '--', c='k')
        return x_min, y_min, x_max, y_max

    def plot_asym(self):
        asym = self.get_asymptotes()
        for i in asym:
            plt.axvline(i, linestyle='--', c='r', zorder=2)

    def plot_equator(self):
        bounds = self.get_r_bounds()
        equator = np.zeros_like(bounds)
        plt.plot(bounds, equator, linestyle='--', c='orange')


class PlotTrajectory(PlotValues):
    """
    Generates the trajectory for a set of initial conditions.
    """
    def __init__(
            self,
            L: float, energy: float, T: float,
            r: float, theta: float = 0, z: float = 0, pr: float = 0, pz: float = 0,
            dt: float = 0.001, unknown: str = 'pz',
            is_E: bool = True, path_name: str = 'trajectory/', integrator=symplectic
    ):
        super().__init__(L, energy, is_E, path_name)
        r_bounds = self.get_r_bounds()
        if self.E == 0:
            if r != r_bounds:
                print('r is a singular value, and is set as such')
                r = self.get_r_eq()
        else:
            if not r_bounds[0] < r < r_bounds[1]:
                raise ValueError(f'r should be inbetween {r_bounds}')

        if unknown == 'pz':
            pz = np.sqrt(-2*self.get_kinetic_energy(r, z)-pr**2)
        elif unknown == 'pr':
            pr = np.sqrt(-2*self.get_kinetic_energy(r, z)-pz**2)
        elif unknown == '':
            self.h = total_energy(
                np.array((
                    (r, theta, z),
                    (pr, L, pz)
                ), float)
            )
            self.E = energy_h_to_E(L, self.h)
        else:
            raise ValueError('invalid unknown coordinate')
        self.qp = np.array((
            (r, theta, z),
            (pr, self.L, pz)
        ), float)
        self.T = T
        self.dt = dt
        self.q_points = np.empty(3)
        self.p_points = np.empty(3)
        self.t_points = np.empty(3)
        self.integrator = integrator

    def get_trajectory(self):
        self.q_points, self.p_points, self.t_points = integrate(self.qp, self.T, self.dt, self.integrator)
        return self.q_points, self.p_points, self.t_points

    def plot_trajectory(self):
        self.get_trajectory()
        self.plot_zvc(self.q_points[2])
        if self.E > 1:
            self.plot_asym()
        plt.plot(self.q_points[0], self.q_points[2], rasterized=True)
        self.plot_equator()
        plt.xlabel('$r$')
        plt.ylabel('$z$', rotation='horizontal')

    def plot_angular_motion(self):
        self.get_trajectory()
        r, theta, z = self.q_points
        x, y = p2c(r, theta)
        plt.plot(x, y)
        plt.axis('equal')
        self.plot_r_bounds()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def plot_all_figures():
    def L1():
        E = 1.
        L = 1.
        values = PlotValues(L, E, is_E=True)

        z_max = 5
        z_min = -z_max
        z_bounds = np.array((z_min, z_max), float)

        energies = np.linspace(values.h_eq, values.h_esc, 10)[1:]

        plt.figure(figsize=figsize1)
        _, r, z, H = values.plot_multi_zvc(z_bounds, energies)

        ctr_f = plt.pcolormesh(r, z, H, vmin=values.h_eq, vmax=values.h_esc, cmap='bone', alpha=0.65, rasterized=True)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=ctr_f.norm, cmap=ctr_f.cmap), ticks=(values.h_eq, values.h_esc))
        cbar.ax.set_yticklabels(('$U_{eq}$', '$U_{esc}$'))
        plt.xlabel('$r$')
        plt.ylabel('$z$', rotation='horizontal')
        plt.xlim([values.get_r_bounds()[0], values.get_r_bounds()[1]])
        plt.plot(values.get_r_eq(), 0, '.', c='k', rasterized=True)
        values.savefig(f'contourL={L}')

    def L10():
        E = 1
        L = 10.
        values = PlotValues(L, E, is_E=True)
        z_max = 5
        z_min = -z_max
        z_bounds = np.array((z_min, z_max))
        energies = np.linspace(values.h_eq, values.h_esc, 10)[1:]
        plt.figure(figsize=figsize1)
        _, r, z, H = values.plot_multi_zvc(z_bounds, energies)

        ctr_f = plt.pcolormesh(r, z, H, vmin=values.h_eq, vmax=values.h_esc, cmap='bone', alpha=0.65, rasterized=True)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=ctr_f.norm, cmap=ctr_f.cmap), ticks=(values.h_eq, values.h_esc))
        cbar.ax.set_yticklabels(('$U_{eq}$', '$U_{esc}$'))
        plt.xlabel('$r$')
        plt.ylabel('$z$', rotation='horizontal')
        plt.xlim(values.get_r_bounds())
        plt.plot(values.get_r_eq(), 0, '.', c='k', rasterized=True)
        values.savefig(f'contourL={L}')

    def min_asym():
        E = 1
        L = 1
        values = PlotValues(L, E)
        z_max = 5
        z_bounds = np.array((-z_max, z_max))
        print(values.h)
        plt.figure(figsize=figsize1)
        values.plot_zvc(z_bounds)
        plt.axis('tight')
        plt.xlabel('$r$')
        plt.ylabel('$z$', rotation='horizontal')
        plt.axvline(values.get_asymptotes()[0], linestyle='--', c='r', rasterized=True)
        plt.xticks([values.get_asymptotes()[0]], ["$r_{esc}$"])
        values.savefig(f'{min_asym.__name__}')
        plt.clf()

    def asym():
        E = 1.1
        L = 1.
        values = PlotValues(L, E, is_E=True)
        print(values.h)
        z_max = 5
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
        plt.xticks(asymptotes, ["$r_{esc,+}$", "$r_{esc,-}$"])
        values.savefig(f'{asym.__name__}')
        plt.clf()

    def escaping():
        L = 1.
        P = 1.1
        r = np.sqrt(2 * np.abs(L))
        print(r)
        T = 40
        plt.figure(figsize=figsize3)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_asym()
        plot.plot_trajectory()
        r = plot.q_points[0]
        r = r[len(r)//2:]
        plt.axvline(np.max(r), linestyle='--', c='g', zorder=2)
        plt.axvline(np.min(r), linestyle='--', c='g', zorder=2)
        plt.ylim(bottom=0.)
        plot.savefig(f'{escaping.__name__}')

    def inescaping():
        L = 1.
        P = 1.1
        r = np.sqrt(2 * np.abs(L)) + 0.2
        T = 40
        plt.figure(figsize=figsize3)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_asym()
        plot.plot_trajectory()
        plt.ylim(bottom=0.)
        r = plot.q_points[0]
        r = r[len(r)//2:]
        plt.axvline(np.max(r), linestyle='--', c='g', zorder=2)
        plt.axvline(np.min(r), linestyle='--', c='g', zorder=2)
        plot.savefig(f'{inescaping.__name__}')

    def im_esc():
        L = 1.
        P = 1.1
        r = np.sqrt(2*np.abs(L))
        T = 40
        plt.figure(figsize=figsize3)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_asym()
        plot.plot_trajectory()
        plt.ylim(bottom=0.)
        plot.savefig('im_esc')

    def alm_esc():
        L = 1.
        P = 1.08
        r = np.sqrt(2*np.abs(L))
        T = 40
        plt.figure(figsize=figsize3)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_asym()
        plot.plot_trajectory()
        plt.ylim(bottom=0.)
        plot.savefig('alm_esc')

    def im_esc_end():
        L = 1.
        P = 1.1
        r = np.sqrt(2*np.abs(L))
        T = 1650
        plt.figure(figsize=figsize3)
        plt.xlim(1, 2)
        plt.ylim(122.4, 122.6)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_asym()
        plot.plot_trajectory()
        plot.savefig('im_esc_end')

    def alm_esc_end():
        L = 1.
        P = 1.08
        r = np.sqrt(2*np.abs(L))
        T = 1550
        plt.figure(figsize=figsize3)
        plt.xlim(1, 2)
        plt.ylim(122.4, 122.6)
        plot = PlotTrajectory(L, P, T, r)
        plot.plot_zvc(np.array([122.4, 122.6]))
        plot.plot_asym()
        plot.plot_trajectory()
        plot.savefig('alm_esc_end')
    L1()
    L10()
    min_asym()
    asym()
    escaping()
    inescaping()
    im_esc()
    alm_esc()
    im_esc_end()
    alm_esc_end()
    return 0
