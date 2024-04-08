import numpy as np
from numba import njit, cuda, prange
from math import sin, cos, sqrt

ndarray = np.ndarray
pi = np.pi


@njit
def diff_eqn(qp: ndarray, B, L, E):
    q, p = qp

    r, th = q
    pr, pth = p
    if r <= 1:
        return np.array([[0., 0.], [0., 0.]])
    else:
        return np.array((
            (
                pr*(1-1/r),
                pth/r**2
            ),
            (
                -(E/(1-1/r)**2+pr**2)/(2*r**2)+pth**2/r**3+L**2/(r**3*sin(th)**2)-B**2*r*sin(th)**2,
                L**2*cos(th)/(r**2*sin(th)**3)-B**2*r**2*sin(th)*cos(th)
            )
        ))


@cuda.jit(device=True)
def diff_eqn_cuda(qp: ndarray, dqp, B, L, E):
    q, p = qp
    r, th = q
    pr, pth = p
    if r <= 1:
        dqp[0, 0] = 0.
        dqp[0, 1] = 0.
        dqp[1, 0] = 0.
        dqp[1, 1] = 0.
    else:
        dqp[0, 0] = pr*(1-1/r)
        dqp[0, 1] = pth/r**2
        dqp[1, 0] = -(E/(1-1/r)**2+pr**2)/(2*r**2)+pth**2/r**3+L**2/(r**3*sin(th)**2)-B**2*r*sin(th)**2
        dqp[1, 1] = L**2*cos(th)/(r**2*sin(th)**3)-B**2*r**2*sin(th)*cos(th)


@cuda.jit(device=True)
def var_eqn_cuda(v: ndarray, dv: ndarray, qp: ndarray, B, L, E):
    q, p = qp
    r, th = q
    pr, pth = p
    dv[0][0] = v[1, 0]*(1-1/r)+v[0, 0]*pr/r**2
    dv[0][1] = v[1, 1]/r**2+v[0, 0]*(-2*pth)/r**3
    dv[1][0] = v[0, 0]*(E/(r-1)**3+pr**2/r**3-3*pth**2/r**4-3*L**2/(r**4*sin(th)**2)-(B*sin(th))**2) \
                + v[0, 1]*(-2*L**2*cos(th)/(r*sin(th))**3-B**2*r*sin(2*th)) + v[1, 0]*(-pr/r**2) + v[1, 1]*(2*pr/r**3)
    dv[1][1] = v[0, 0]*(-2*L**2*cos(th)/(r*sin(th))**3-B**2*r*sin(2*th)) \
                +v[0, 1]*(-L**2*(2*cos(th)**2+1)/(r**2*sin(th)**4)-B**2*r**2*cos(2*th))


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
def symplectic(qp: ndarray, b, L, E, dt: float = 0.001):
    output = np.copy(qp)
    for i, _ in enumerate(optimal_a):
        output[1] += dt * optimal_b[i] * diff_eqn(output, b, L, E)[1]
        output[0] += dt * optimal_a[i] * diff_eqn(output, b, L, E)[0]
    return output


@cuda.jit(device=True)
def symplectic_cuda(qp, dqp, B, L, E, dt):
    for i, _ in enumerate(optimal_a):
        diff_eqn_cuda(qp, dqp, B, L, E)
        for j, k in enumerate(dqp[1]):
            qp[1, j] += dt * optimal_b[i] * k
        diff_eqn_cuda(qp, dqp, B, L, E)
        for j, k in enumerate(dqp[0]):
            qp[0, j] += dt * optimal_a[i] * k


@cuda.jit(device=True)
def symplectic_var_cuda(v: ndarray, dv: ndarray, qp: ndarray, B, L, E, dt: float):

    for i, _ in enumerate(optimal_a):
        var_eqn_cuda(v, dv, qp, B, L, E)
        for j, k in enumerate(dv[1]):
            v[1, j] += dt * optimal_b[i] * k
        var_eqn_cuda(v, dv, qp, B, L, E)
        for j, k in enumerate(dv[0]):
            v[0, j] += dt * optimal_a[i] * k


@njit
def integrate(qp: ndarray, T: float, B: float, L: float, E: float, dt: float = 0.001):
    tpoints = np.arange(0, T, dt)
    q_points = np.zeros((qp.shape[1], tpoints.shape[0]))
    p_points = np.copy(q_points)
    for t, _ in enumerate(tpoints):
        qp = symplectic(qp, B, L, E, dt)
        if qp[0, 0] <= 1:
            break
        q_points[:, t], p_points[:, t] = qp[:, :]
    return q_points, p_points, tpoints
