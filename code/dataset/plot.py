import matplotlib.pyplot as plt
import numpy as np
import jax

import jax.numpy as jnp


def normalize_dp(state):
    """
    wrap generalized coordinates to [-pi, pi]
    """
    return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])


def visualize_data(x_train, x_test):
    print('Preparing for visualization...')
    # preprocess
    train_vis = jax.vmap(normalize_dp)(x_train)
    test_vis = jax.vmap(normalize_dp)(x_test)

    vel_angle = lambda data:  (np.arctan2(data[:,3], data[:,2]) / np.pi + 1) / 2
    vel_color = lambda vangle: np.stack([np.zeros_like(vangle), vangle, 1-vangle]).T
    train_colors = vel_color(vel_angle(train_vis))
    test_colors = vel_color(vel_angle(test_vis))

    print('Making visualization...')
    # plot
    SCALE = 80
    WIDTH = 0.006
    plt.figure(figsize=[8,4], dpi=120)
    plt.subplot(1, 2, 1)
    plt.title("Train data") 
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.quiver(*train_vis.T, color=train_colors, scale=SCALE, width=WIDTH)

    plt.subplot(1, 2, 2)
    plt.title("Test data")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.quiver(*test_vis.T, color=test_colors, scale=SCALE, width=WIDTH)

    plt.tight_layout()
    plt.savefig('./figures/train_test_data_vis.png', dpi=200)
    plt.savefig('./figures/train_test_data_vis.pdf', dpi=200)
