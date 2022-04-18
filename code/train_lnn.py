import pickle
from datetime import datetime
from functools import partial
import jax
from jax.experimental import stax
from jax.experimental import optimizers
import jax.numpy as jnp

from dataset.make_data import equation_of_motion, rk4_step, solve_lagrangian
from dataset.plot import normalize_dp
from visualization import plot_loss


TRAIN_DATASET_PATH = "../data/train_data.pickle"
TEST_DATASET_PATH = "../data/test_data.pickle"
LOG_DIR = "./logs"

# build a neural network model
init_random_params, nn_forward_fn = stax.serial(
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(128),
        stax.Softplus,
        stax.Dense(1),
    )

# replace the lagrangian with a parameteric model
def learned_lagrangian(params):
    def lagrangian(q, q_t):
        assert q.shape == (2,)
        state = normalize_dp(jnp.concatenate([q, q_t]))
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
    return lagrangian

# define the loss of the model (MSE between predicted q, \dot q and targets)
@jax.jit
def loss(params, batch, time_step=None):
    state, targets = batch
    if time_step is not None:
        f = partial(equation_of_motion, learned_lagrangian(params))
        preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step))(state)
    else:
        preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(state)
    return jnp.mean((preds - targets) ** 2)

######## Define optimization and data

# @jax.jit
# def update_timestep(i, opt_state, batch, opt_update):
#     params = get_params(opt_state)
#     return opt_update(i, jax.grad(loss)(params, batch, time_step), opt_state)

def predict_lnn(params, x, t):
    return jax.device_get(solve_lagrangian(learned_lagrangian(params), x, t=t))


def train(init_random_params, x_train, xt_train, x_test, xt_test, log_dir=None):
    run_name = datetime.now().strftime("%d_%m_%Y.%H_%M")

    rng = jax.random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 4))

    # numbers in comments denote stephan's settings
    batch_size = 100
    test_every = 10
    num_batches = 1500

    train_losses = []
    test_losses = []

    # adam w learn rate decay
    opt_init, opt_update, get_params = optimizers.adam(
        lambda t: jnp.select([t < batch_size*(num_batches//3),
                            t < batch_size*(2*num_batches//3),
                            t > batch_size*(2*num_batches//3)],
                            [1e-3, 3e-4, 1e-4]))
    opt_state = opt_init(init_params)

    @jax.jit
    def update_derivative(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)


    for iteration in range(batch_size*num_batches + 1):
        if iteration % batch_size == 0:
            params = get_params(opt_state)
            train_loss = loss(params, (x_train, xt_train))
            train_losses.append(train_loss)
            test_loss = loss(params, (x_test, xt_test))
            test_losses.append(test_loss)
            if iteration % (batch_size*test_every) == 0:
                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        opt_state = update_derivative(iteration, opt_state, (x_train, xt_train))

    params = get_params(opt_state)

    if log_dir is not None:
        with open(f'{log_dir}/lnn_model_{run_name}.pickle', 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plot_loss(train_losses, test_losses, model_name='lnn', log_dir=log_dir)

def main():
    with open(TRAIN_DATASET_PATH, 'rb') as f:
        train_data = pickle.load(f)
    with open(TEST_DATASET_PATH, 'rb') as f:
        test_data = pickle.load(f)
    
    [x_train, xt_train, y_train] = train_data
    [x_test, xt_test, y_test] = test_data

    x_train = jax.device_put(jax.vmap(normalize_dp)(x_train))
    y_train = jax.device_put(y_train)

    x_test = jax.device_put(jax.vmap(normalize_dp)(x_test))
    y_test = jax.device_put(y_test)

    train(init_random_params, x_train, xt_train, x_test, xt_test, log_dir=LOG_DIR)


if __name__ == '__main__':
    main()