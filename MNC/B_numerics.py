from A_integrator import *

PLUS_MINUS = np.array([1, -1])
PI = np.pi


@njit
def potential_energy(L, r, z):
    """
    Potential energy of the system
    """
    R = np.sqrt(r * r + z * z)
    return (L / r - r / 2) ** 2 / 2 - 1 / R


@njit
def kinetic_energy(L: float, r: float, z: float, h: float):
    """
    Kinetic energy of the system
    """
    R = np.sqrt(r * r + z * z)
    return (L / r - r / 2) ** 2 / 2 - 1 / R - h


@njit
def total_energy(qp):
    q, p = qp
    r, _, z = q
    pr, L, pz = p
    return (pr ** 2 + pz ** 2) / 2 + potential_energy(L, r, z)


def get_h_eq(L: float):
    """
    Minimum Hamiltonian energy for a given L.
    """
    r_eq = np.roots([1, 0, 0, 4, -4 * L ** 2])
    r_eq = np.abs(
        r_eq[
            np.logical_and(r_eq >= 0, np.isreal(r_eq))
        ]
    )[0]
    return potential_energy(L, r_eq, 0)


def energy_h_to_E(L: float, h: float):
    """
    Rescales the Hamiltonian energy h into E. Where E=0 is the minimum energy
    and E=1 is the minimum energy needed for escape.

    :param L: Canonical angular momentum
    :param h: Hamiltonian energy
    :return: The rescaled energy, the minimum energy, and the minimum energy of escape
    """
    # computes the singular equilibrium point of the system
    r_eq = np.roots([1, 0, 0, 4, -4 * L ** 2])
    r_eq = np.abs(r_eq[np.logical_and(r_eq >= 0, np.isreal(r_eq))])[0]
    # computes the minimum energy
    h_eq = potential_energy(L, r_eq, 0)
    # computes the minimum energy of escape
    h_esc = (np.abs(L) - L) / 2
    # converts h to P
    E = (h - h_eq) / (h_esc - h_eq)
    return E, h_eq, h_esc


def energy_E_to_h(L: float, E: float):
    """
    Converts the energy rescale P back to the Hamiltonian energy h.

    :param L: Canonical angular momentum
    :param E: Rescaled energy
    :return: The Hamiltonian energy h, the minimum energy, and the minimum energy of escape
    """
    # computes the singular equilibrium point of the system
    r_eq = np.roots([1, 0, 0, 4, -4 * L ** 2])
    r_eq = np.abs(r_eq[np.logical_and(r_eq >= 0, np.isreal(r_eq))])[0]
    # computes the minimum energy
    h_eq = potential_energy(L, r_eq, 0)
    # computes the minimum energy of escape
    h_esc = (np.abs(L) - L) / 2
    # converts P to h
    h = h_eq + E * (h_esc - h_eq)
    return h, h_eq, h_esc


@njit
def p2c(r, theta):
    """
    Converts polar coordinates to cartesian.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


class Values:
    """
    Given the constants of motion, this class gives the special values like minima, bounds, etc.
    """
    def __init__(self, L: float, energy: float, is_E: bool = True):
        """
        :param L: Canonical angular momentum
        :param energy: Energy, can be either the Hamiltonian energy, or the rescaled energy
        :param is_E: Determines if the energy is the rescaled one
        """
        # Assigns the two energy scales
        if is_E:
            self.E = energy
            self.h, self.h_eq, self.h_esc = energy_E_to_h(L, energy)
        else:
            self.h = energy
            self.E, self.h_eq, self.h_esc = energy_h_to_E(L, energy)
        if self.E < 0:
            raise ValueError('energy given is below the minimum')
        self.L = L

    def get_potential_energy(self, r: float, z: float):
        return potential_energy(self.L, r, z)

    def get_kinetic_energy(self, r: float, z: float):
        return kinetic_energy(self.L, r, z, self.h)

    def get_r_eq(self):
        """
        Gets the minima of the potential energy along r, since the minima along z is 0.
        """
        r_eq = np.roots([1, 0, 0, 4, -4 * self.L ** 2])
        r_eq = np.abs(r_eq[np.logical_and(r_eq >= 0, np.isreal(r_eq))])[0]
        return r_eq

    def get_asymptotes(self):
        """
        If the energy is above h_esc, asymptotes of the ZVC along r will appear and is returned by this function.
        In other words the trajectories are unbounded along z in this r region.
        """
        # there are two asymptotes, the lesser one is first in the output array
        if self.E > 1:
            return np.sqrt(2 * (2 * self.h + self.L) + PLUS_MINUS * 4 * np.sqrt(self.h ** 2 + self.L * self.h))
        # there is only a single asymptote if the energy is h_esc itself
        elif self.E == 1.:
            return np.array([np.sqrt(2 * (2 * self.h + self.L))])
        elif self.E < 1:
            raise ValueError('Energy should be above the minimum energy of escape.')

    def get_r_bounds(self):
        """
        Gives the absolute bounds of the position along r.
        """
        if self.E == 0:
            r_bounds = self.get_r_eq()
        else:
            r_bounds = np.roots([1, 0, -4 * (self.L + 2 * self.h), -8, 4 * self.L ** 2])
            r_bounds = np.sort(np.abs(r_bounds[np.logical_and(r_bounds > 0, np.isreal(r_bounds))]))
        return r_bounds

    def get_pr_bounds(self):
        """
        Gives the absolute bounds of the momentum along r. The expression used was analytically derived
        """
        pr_bound = np.sqrt(-2 * self.get_kinetic_energy(self.get_r_eq(), 0))
        if self.E == 0:
            return pr_bound
        else:
            return PLUS_MINUS * pr_bound
