import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

from jax.lib import xla_bridge

from .plot import visualize_data

#print(xla_bridge.get_backend().platform)


def lagrangian(q, q_dot, m1, m2, l1, l2, g):
    """
    Lagrangian of the double pendulum
    """
    t1, t2 = q     # theta 1 and theta 2
    w1, w2 = q_dot # omega 1 and omega 2

    # kinetic energy (T)
    T1 = 0.5 * m1 * (l1 * w1)**2
    T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 +
                    2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    T = T1 + T2

    # potential energy (V)
    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2

    return T - V


def f_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    """
    The analytical dynamics of the system in matrix form
    d/dt(theta1, theta2, omega1, omega2)
    """
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    a2 = (l1 / l2) * jnp.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - \
        (g / l1) * jnp.sin(t1)
    f2 = (l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return jnp.stack([w1, w2, g1, g2])


def equation_of_motion(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
                - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])

def solve_lagrangian(lagrangian, initial_state, **kwargs):
    """
    Obtain dynamics from the Lagrangian
    """
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(equation_of_motion, lagrangian),
                        initial_state, **kwargs)
    return f(initial_state)


@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=1, m2=1, l1=1, l2=1, g=9.8):
    """
    Double pendulum dynamics via the rewritten Euler-Lagrange
    """
    L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)

@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times):
    """
    Double pendulum dynamics via analytical forces taken from Diego's blog
    """
    return odeint(f_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

def rk4_step(f, x, t, h):
    """
    One step of runge-kutta integration
    """
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def main():
    time_step = 0.01
    N = 1500
    analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))

    # choose an initial state
    # x0 = np.array([-0.3*np.pi, 0.2*np.pi, 0.35*np.pi, 0.5*np.pi], dtype=np.float32)
    print('Making train...')
    x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
    t = np.arange(N, dtype=np.float32) # time steps 0 to N
    x_train = jax.device_get(solve_analytical(x0, t)) # dynamics for first N time steps
    xt_train = jax.device_get(jax.vmap(f_analytical)(x_train)) # time derivatives of each state
    y_train = jax.device_get(analytical_step(x_train)) # analytical next step

    print('Making test...')
    noise = np.random.RandomState(0).randn(x0.size)
    t_test = np.arange(N, 2*N, dtype=np.float32) # time steps N to 2N
    x_test = jax.device_get(solve_analytical(x0, t_test)) # dynamics for next N time steps
    xt_test = jax.device_get(jax.vmap(f_analytical)(x_test)) # time derivatives of each state
    y_test = jax.device_get(analytical_step(x_test)) # analytical next step

    train_arr = [x_train, xt_train, y_train]
    test_arr = [x_test, xt_test, y_test]

    print('Saving data...')
    path_to_save = './data'
    os.makedirs(path_to_save, exist_ok=True)

    with open(f'{path_to_save}/train_data.pickle', 'wb') as handle:
        pickle.dump(train_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'{path_to_save}/test_data.pickle', 'wb') as handle:
        pickle.dump(test_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    visualize_data(x_train, x_test)


if __name__ == '__main__':
    main()