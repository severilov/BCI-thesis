import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import jax
import numpy as np
from functools import partial


def plot_loss(train_losses, test_losses, model_name, log_dir):
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    #plt.yscale('log')
    #plt.ylim(None, 200)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{log_dir}/loss_{model_name}.png', dpi=150)


def compare_prediction(params, x_test, xt_test):
    xt_pred = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(x_test)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
    axes[0].scatter(xt_test[:, 2], xt_pred[:, 2], s=6, alpha=0.2)
    axes[0].set_title('Predicting $\dot q$')
    axes[0].set_xlabel('$\dot q$ actual')
    axes[0].set_ylabel('$\dot q$ predicted')
    axes[1].scatter(xt_test[:, 3], xt_pred[:, 3], s=6, alpha=0.2)
    axes[1].set_title('Predicting $\ddot q$')
    axes[1].set_xlabel('$\ddot q$ actual')
    axes[1].set_ylabel('$\ddot q$ predicted')
    plt.tight_layout()

# def make_plot(i, cart_coords, l1, l2, max_trail=30, trail_segments=20, r = 0.05):
#     # Plot and save an image of the double pendulum configuration for time step i.
#     plt.cla()

#     x1, y1, x2, y2 = cart_coords
#     ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k') # rods
#     c0 = Circle((0, 0), r/2, fc='k', zorder=10) # anchor point
#     c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10) # mass 1
#     c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10) # mass 2
#     ax.add_patch(c0)
#     ax.add_patch(c1)
#     ax.add_patch(c2)

#     # plot the pendulum trail (ns = number of segments)
#     s = max_trail // trail_segments
#     for j in range(trail_segments):
#         imin = i - (trail_segments-j)*s
#         if imin < 0: continue
#         imax = imin + s + 1
#         alpha = (j/trail_segments)**2 # fade the trail into alpha
#         ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
#                 lw=2, alpha=alpha)

#     # Center the image on the fixed anchor point. Make axes equal.
#     ax.set_xlim(-l1-l2-r, l1+l2+r)
#     ax.set_ylim(-l1-l2-r, l1+l2+r)
#     ax.set_aspect('equal', adjustable='box')
#     plt.axis('off')
#     # plt.savefig('./frames/_img{:04d}.png'.format(i//di), dpi=72)

def radial2cartesian(t1, t2, l1, l2):
    # Convert from radial to Cartesian coordinates.
    x1 = l1 * np.sin(t1)
    y1 = -l1 * np.cos(t1)
    x2 = x1 + l2 * np.sin(t2)
    y2 = y1 - l2 * np.cos(t2)
    return x1, y1, x2, y2

def fig2image(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def plot_predicted_trajectory(cart_coords, model_name):
    x1, y1, x2, y2 = cart_coords
    
    length = len(x2)
    t = np.arange(length)
    
    plt.title(f"{model_name} Double Pendulum " + f"Timeseries - {length - 5} timesteps")

    plt.plot(x2, y2, marker='.', color="lightgray", zorder=0)
    plt.scatter(x2, y2, marker='o', c=t[:length], cmap="viridis", s=10, zorder=1)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$t$', rotation=270)

    #plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'../figures/predicted_trajectory_{model_name}.png', dpi=150)
