import matplotlib.colors
import numpy as np

from A_integrator import *
from B_numerics import *
from C_plotting import *
from D1_poincare import *
from D2_escape_time import *
from D3_SALI import *
import matplotlib.pyplot as plt


def PoincareTest():
    b = 1
    L = 3.5
    E = 0.46
    r = 1.8
    traj = PlotTrajectory(b, L, E, 1000, r)
    traj.plot_trajectory()
    plt.show()
    """
    poinc = PlotPoincare(b, L, E, pr_bounds=[-0.06, 0.06])
    poinc.plot_poincare()
    plt.clf()
    """


def plot_capture():
    r_esc_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0
    ]
    p_values = [
        1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
    ]
    E_values = [
        1.2, 1.4, 1.6, 1.8, 2.0
    ]
    r_esc_mesh, E_mesh = np.meshgrid(
        r_esc_values,
        E_values
    )
    for p in p_values:
        plot_mesh = np.empty_like(r_esc_mesh)
        for i, E in enumerate(E_values):
            for j, r_esc in enumerate(r_esc_values):
                plot = EscapePlot(r_esc, p, E)
                name = plot.name
                time_mesh = plot.load(f'escape_time_{name}')
                pass_mesh = plot.load(f'escape_pass_{name}')
                basin_mesh = plot.load(f'escape_basin_{name}')
                total = np.count_nonzero(pass_mesh == pass_mesh)
                capture = np.count_nonzero(basin_mesh == 0.)
                plot_mesh[i, j] = capture / total
        plt.figure(figsize=figsize3)
        plt.pcolormesh(r_esc_mesh, E_mesh, plot_mesh, rasterized=True)
        isco = p*((3*p-1)/(3-p))**(1/4)
        if 1 < isco < 4:
            plt.axvline(isco, linestyle='--', c='w')
        plt.ylabel('$E$')
        plt.xlabel('$r_{esc}$')
        plt.colorbar()
        savefig(name=f"percent_capture_p={p}.pdf", path_name=f"escapes/figures/analysis/")


def test(B, L, E):
    plot = EscapePlot(B, L, E)
    T = plot.T
    name = plot.name
    r_mesh = plot.r_mesh
    pr_mesh = plot.pr_mesh
    time_mesh = plot.load(f'escape_time_{name}')
    pass_mesh = plot.load(f'escape_pass_{name}')
    basin_mesh = plot.load(f'escape_basin_{name}')

    for i, row in enumerate(basin_mesh):
        for j, r in enumerate(row):
            if (basin_mesh[i, j] != basin_mesh[i, j]) and (pass_mesh[i, j] == pass_mesh[i, j]) and (1.1 < r_mesh[i, j] <1.25):
                PlotTrajectory(B, L, E, T=0.502, r=r_mesh[i, j], pr=pr_mesh[i, j]).plot_trajectory()
                plt.show()
                plt.close()


def trajtest():
    r_esc = 4.0

    T = 100.
    p_values = [2.2, 2.4, 2.8]
    pr_values = [-0.2, -0.1, 0., 0.1, 0.2]
    for p in p_values:
        for pr in pr_values:
            B_plus, L_plus, E_plus, B_minus, L_minus, E_minus = reparametrization(5, p, 1.0)
            plot = PlotTrajectory(B_minus, L_minus, E_minus, T, r_esc, pr=pr)
            plot.plot_trajectory()
            plot = PlotTrajectory(B_plus, L_plus, E_plus, T, r_esc, pr=pr)
            plot.plot_trajectory()
            plot.savefig(f"{p}, {pr}", pdf=False)
            plt.clf()


def generate_zoom_escape1():
    r_esc = 1.8
    p = 1.2
    E = 1.2
    plot = EscapePlot(
        r_esc, p, E,
        r_bounds=[1.65, 1.69],
        pr_bounds=[0.46, 0.5]
    )
    plot.generate_meshes()
    name = "_zoomed_" + plot.name
    T = plot.T
    plt.figure(figsize=figsize3)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    plt.colorbar()
    plot.savefig(f'escape_time/escape_time_{name}')
    plt.clf()


def generate_zoom_escape2():
    r_esc = 1.8
    p = 1.2
    E = 1.2
    plot = EscapePlot(
        r_esc, p, E,
        r_bounds=[2.035, 2.04],
        pr_bounds=[0.015, 0.02]
    )
    plot.generate_meshes()
    name = "_zoomed2_" + plot.name
    T = plot.T
    plt.figure(figsize=figsize3)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    plt.colorbar()
    plot.savefig(f'escape_time/escape_time_{name}')
    plt.clf()


def generate_zoom_escape3():
    r_esc = 1.8
    p = 1.2
    E = 1.2
    plot = EscapePlot(
        r_esc, p, E,
        r_bounds=[1.265, 1.27],
        pr_bounds=[0.055, 0.06]
    )
    plot.generate_meshes()
    name = "_zoomed3_" + plot.name
    T = plot.T
    plt.figure(figsize=figsize3)
    plt.pcolormesh(plot.r_mesh, plot.pr_mesh, plot.time_mesh, vmin=0, vmax=T, rasterized=True)
    plt.xlabel('$R$')
    plt.ylabel('$p^R$')
    plt.colorbar()
    plot.savefig(f'escape_time/escape_time_{name}')
    plt.clf()


def main():
    r_esc_values = [
        2.4, 2.8, 3.2
    ]
    p_values = [
        1.2
    ]
    E_values = [
        1.2
    ]
    for E in E_values:
        for p in p_values:
            for r_esc in r_esc_values:
                sali = PlotSALI(r_esc, p, E)
                print(E, p, r_esc)
                sali.plot_SALI()


if __name__ == '__main__':
    main()
