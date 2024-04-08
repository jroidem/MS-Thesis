import numpy as np
from numba import njit, cuda

ndarray = np.ndarray


@njit
def diff_eqn(qp: ndarray):
    """
    Hamilton's equations of the uniformly magnetized Newtonian center, which will be the default system

    :param qp: 2x3 ndarray of the coordinates to be numerically integrated. First three are the position values,
    and the last three are the momentum values
    :return: Time derivative of each coordinate
    """
    # unpacking the array
    q, p = qp
    # further unpacking
    r, t, z = q
    pr, l, pz = p

    return np.array((
        (pr,
         l / r ** 2 - 1 / 2,
         pz),
        (l ** 2 / r ** 3 - r / 4 - r / (r ** 2 + z ** 2) ** 1.5,
         0.,
         -z / (r ** 2 + z ** 2) ** 1.5)
    ))


@njit
def var_eqn(v: ndarray, qp: ndarray):
    q, p = qp

    r, t, z = q
    pr, l, pz = p

    vq, vp = v

    vr, _, vz = vq
    vpr, _, vpz = vp

    return np.array((
        (vpr,
         0.,
         vpz),
        (-(1/4 + 3*l**2/r**4 + 1/(r ** 2 + z ** 2)**1.5 - 3*r**2/(r ** 2 + z ** 2)**2.5)*vr
         + (3*r*z/(r ** 2 + z ** 2)**2.5)*vz,
         0.,
         (3*r*z/(r ** 2 + z ** 2)**2.5)*vr - (1/(r ** 2 + z ** 2)**1.5 - 3*z**2/(r ** 2 + z ** 2)**2.5)*vz
    )))#/np.sqrt(vr**2+vz**2+vpr**2+vpz**2)


@cuda.jit(device=True)
def diff_eqn_cuda(qp: ndarray, dqp: ndarray, L):
    dqp[0][0] = qp[1, 0]
    dqp[0][1] = qp[1, 1]
    dqp[1][0] = L ** 2 / qp[0, 0] ** 3 - qp[0, 0] / 4 - qp[0, 0] / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 1.5
    dqp[1][1] = -qp[0, 1] / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 1.5


@cuda.jit(device=True)
def var_eqn_cuda(v: ndarray, dv: ndarray, qp: ndarray, L):
    dv[0][0] = v[1, 0]
    dv[0][1] = v[1, 1]
    dv[1][0] = -(1/4 + 3*L**2/qp[0, 0]**4 - 3*qp[0, 0]**2 / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 2.5
                + 1/(qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 1.5)*v[0, 0] \
                + (3*qp[0, 0]*qp[0, 1] / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 2.5)*v[0, 1]
    dv[1][1] = (3*qp[0, 0]*qp[0, 1] / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 2.5)*v[0, 0]\
                + ((2*qp[0, 1]**2-qp[0, 0]**2) / (qp[0, 0] ** 2 + qp[0, 1] ** 2) ** 2.5)*v[0, 1]


optimal_a = np.array([
    0.5153528374311229364,
    -0.085782019412973646,
    0.4415830236164665242,
    0.1288461583653841854
])
optimal_b = np.array([
    0.1344961992774310892,
    -0.2248198030794208058,
    0.7563200005156682911,
    0.3340036032863214255
])


@njit
def symplectic(qp: ndarray, df=diff_eqn, dt: float = 0.001):
    """
    4th order numerical integrator step that optimally conserves energy for the 4th order.

    :param qp: Coordinates
    :param df: A function that returns the time derivative of the coordinates
    :param dt: Time step
    :return: Coordinates after integrating one time step
    """
    output = np.copy(qp)
    for i, _ in enumerate(optimal_a):
        output[1] += dt * optimal_b[i] * df(output)[1]
        output[0] += dt * optimal_a[i] * df(output)[0]
    return output


@njit
def symplectic_var(v: ndarray, qp: ndarray, dt: float = 0.001):
    output = np.copy(v)
    for i, _ in enumerate(optimal_a):
        output[1] += dt * optimal_b[i] * var_eqn(output, qp)[1]
        output[0] += dt * optimal_a[i] * var_eqn(output, qp)[0]
    return output


@cuda.jit(device=True)
def symplectic_cuda(qp: ndarray, dqp: ndarray, L, df, dt: float):
    """
    CUDA implementation of symplectic()

    :param qp: Coordinates
    :param dqp: Time derivative output array
    :param L: Constant of motion canonical angular momentum
    :param df: Time derivative function
    :param dt: Time step
    """
    for i, _ in enumerate(optimal_a):
        df(qp, dqp, L)
        for j, k in enumerate(dqp[1]):
            qp[1, j] += dt * optimal_b[i] * k
        df(qp, dqp, L)
        for j, k in enumerate(dqp[0]):
            qp[0, j] += dt * optimal_a[i] * k


@cuda.jit(device=True)
def symplectic_var_cuda(v: ndarray, dv: ndarray, qp: ndarray, L, df, dt: float):
    """
    CUDA implementation of symplectic()

    :param v: Variation vector
    :param dv: Time derivative output array
    :param qp: Coordinates
    :param L: Constant of motion canonical angular momentum
    :param df: Variation function
    :param dt: Time step
    """
    for i, _ in enumerate(optimal_a):
        df(v, dv, qp, L)
        for j, k in enumerate(dv[1]):
            v[1, j] += dt * optimal_b[i] * k
        df(v, dv, qp, L)
        for j, k in enumerate(dv[0]):
            v[0, j] += dt * optimal_a[i] * k


@njit
def integrate(qp: ndarray, T: float, dt: float = 0.001, integrator=symplectic):
    """
    Numerically integrates a set of initial conditions for the given total time.

    :param qp: Coordinates to be numerically integrated
    :param T: Duration of numerical integration
    :param dt: Time step
    :param integrator: Numerical integrator used
    :returns: Position, momentum, and time values of the whole trajectory
    """
    # generate time points
    tpoints = np.arange(0, T, dt)
    # creates empty arrays for the position and momentum values
    q_points = np.empty((qp.shape[1], tpoints.shape[0]))
    p_points = np.copy(q_points)
    # getting the value of each integration step
    for t, _ in enumerate(tpoints):
        qp = integrator(qp, diff_eqn, dt)
        q_points[:, t], p_points[:, t] = qp[:, :]
    return q_points, p_points, tpoints

@njit
def normalize(array):
    return array/np.linalg.norm(array)
