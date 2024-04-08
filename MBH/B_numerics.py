import matplotlib.pyplot as plt
import numpy as np

from A_integrator import *
PLUS_MINUS = np.array([1, -1])
PI = np.pi


def sphere_to_cyl(R, th):
    return R*np.sin(th), R*np.cos(th)


def cyl_to_sphere(r, z):
    return np.sqrt(r**2+z**2), np.arctan(r/z)


def potential_energy(B, L, r, th):
    return (1-1/r)*(1+(L/(r*np.sin(th))-B*(r*np.sin(th)))**2)


def potential_energy_cyl(B, L, r, z):
    return (1-1/np.sqrt(r**2+z**2))*(1+(L/r-B*r)**2)


def escape_energy(B, L):
    return 1+2*B*(abs(L)-L)


def reparametrization(r_esc:float, p:float, E:float):
    L_plus = r_esc*((3-p)*(3*p-1))**(1/4)/(np.sqrt(2*(4*p**2-9*p+3+np.sqrt((3*p-1)*(3-p)))))
    B_plus = L_plus/r_esc**2
    E_plus = E
    if p > (5+np.sqrt(13))/4:
        L_minus = -r_esc*((3-p)*(3*p-1))**(1/4)/(np.sqrt(2*(4*p**2-9*p+3-np.sqrt((3*p-1)*(3-p)))))
        B_minus = -L_minus/r_esc**2
        E_minus = E*(1-4*B_minus*L_minus)
        return B_plus, L_plus, E_plus, B_minus, L_minus, E_minus
    else:
        return B_plus, L_plus, E_plus


class Values:
    """
    Given the constants of motion, this class gives the special values like minima, bounds, etc.
    """
    def __init__(self, r_esc: float, p: float, E: float):
        self.L = r_esc*((3-p)*(3*p-1))**(1/4)/np.sqrt(2*(4*p**2-9*p+3+np.sqrt((3*p-1)*(3-p))))
        self.B = self.L / r_esc ** 2
        self.r_esc = r_esc
        if E < 0:
            raise ValueError('energy given is below the minimum')
        else:
            self.E = E*escape_energy(self.B, self.L)

    def get_potential_energy(self, r: float, th: float):
        return potential_energy(self.B, self.L, r, th)

    def get_potential_energy_cyl(self, r: float, z: float):
        return potential_energy_cyl(self.B, self.L, r, z)

    def get_kinetic_energy(self, r: float, th: float):
        return potential_energy(self.B, self.L, r, th)-self.E

    def get_asymptotes(self):
        if self.E < escape_energy(self.B, self.L):
            raise ValueError('Energy should be above the minimum energy of escape.')
        else:
            asym = np.roots(
                [self.B**2, 0, 1-self.E-2*self.L*self.B, 0, self.L**2]
            )
            return np.sort(np.abs(asym[np.logical_and(asym > 0, np.isreal(asym))]))

    def get_r_bounds(self):
        r_bounds = np.roots(
            [self.B ** 2, -self.B ** 2, 1 - self.E - 2 * self.B * self.L, 2 * self.B * self.L - 1, self.L ** 2,
             -self.L ** 2]
        )
        r_bounds = np.sort(np.abs(r_bounds[np.logical_and(r_bounds > 0, np.isreal(r_bounds))]))
        return r_bounds

    def get_r_eq(self):
        r_eq = np.roots([2*self.B**2, -self.B**2, 0, 1-2*self.B*self.L, -2*self.L**2, 3*self.L**2])
        r_eq = np.sort(np.abs(r_eq[np.logical_and(r_eq > 0, np.isreal(r_eq))]))
        return r_eq

    def get_U_eq(self):
        r_eq = self.get_r_eq()
        if r_eq.shape[0] == 1:
            return self.get_potential_energy(r_eq[0], PI/2)
        elif r_eq.shape[0] == 2:
            return np.array([self.get_potential_energy(r_eq[0], PI/2), self.get_potential_energy(r_eq[1], PI/2)])
        else:
            #print("No equilibrium points found")
            return np.array([])

    def get_pr_bounds(self):
        r_eq = self.get_r_eq()
        U_eq = self.get_U_eq()
        if U_eq.shape[0] == 2 and U_eq[0] > self.E:
            return PLUS_MINUS*np.sqrt(self.E-self.get_potential_energy(r_eq[1], PI/2))
        else:
            #print("No equilibrium points found")
            return PLUS_MINUS*np.sqrt(self.E)

    def get_capture_width(self):
        L = self.L
        U = self.E
        B = self.B
        width = np.roots([B**6, 0, -3*B**4*(2*B*L+U-1), 0, B**2*(12*B*L*(U-1)+3*(U-1)**2+B**2*(15*L**2+2*U)),
                          0, -6*B*L*(U-1)**2-(U-1)**3-4*B**3*L*(5*L**2+U)+2*B**2*(-9*L**2*(U-1)+U),
                          0, 3*L**2*(5*B**2*L**2+4*B*L*(U-1)+(U-1)**2),
                          0, 2*L**2*(-1+2*B*L)*U-3*L**4*(-1+2*B*L+U), 0, L**6-2*L**4*U])
        root = np.abs(width[np.logical_and(width > 0, np.isreal(width))])
        return np.sort(root[root < self.r_esc])
