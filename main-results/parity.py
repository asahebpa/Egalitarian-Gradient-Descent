from math import sqrt
from functools import partial
from sklearn.utils import check_random_state
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle


k = 4


if k == 2:
  n = 400 # total input bits
  k = 2   # parity on k secret bits
  hidden_dim = 50  # 2-layer neural network with this number of hidden neurons
  lr = 1e-2 # learning rate
  wd = 1e-3  # weight decay (L2)
  N = 2000
  batch_size = 32
  epochs = 10100 # number of epochs
elif k == 3:
  n = 100 # total input bits
  hidden_dim = 100  # 2-layer neural network with this number of hidden neurons
  lr = 0.0416239 # learning rate
  wd = 1e-2  # weight decay (L2)
  N = 2000
  batch_size = 32
  epochs = 2000 # number of epochs
elif k == 4:
  n = 50 # total input bits
  hidden_dim = 100  # 2-layer neural network with this number of hidden neurons
  lr = 0.0230841 # learning rate 0.0330841 0.0295952 0.0343251
  wd = 1e-2  # weight decay (L2)
  N = 2000
  batch_size = 32
  epochs = 6000 # number of epochs


h = 1 # controls hinge loss (edited)


# ── Data generation ────────────────────────────────────────────────────────────
def make_parity_dataset(n, k, N, random_state=None):

    rng = check_random_state(random_state)

    # uniform {0,1}^n, labels ±1 according to parity on first k bits
    X = rng.choice([-1, 1], size=(N, n)).astype(jnp.float32)
    y = y = X[:, :k].prod(axis=1)
    return X, y

rng = check_random_state(0)
X_tr, y_tr = make_parity_dataset(n, k, N, random_state=rng)
X_te, y_te = make_parity_dataset(n, k, 10000, random_state=rng)

# ── Model init ────────────────────────────────────────────────────────────────
def init_params(n, hidden_size, random_state=None):
    rng = check_random_state(random_state)
    W1 = rng.randn(n, hidden_size) * sqrt(2 / n)
    b1 = rng.randn(hidden_size) * sqrt(2 / n)
    W2 = rng.randn(hidden_size) * sqrt(2 / hidden_size)
    return W1, b1, W2

# ── Forward & loss ────────────────────────────────────────────────────────────
def forward(params, X):
    W1, b1, W2 = params
    h = jnp.maximum(0, X@W1 + b1)  # ReLU
    logits = h @ W2
    return logits

def hinge_loss(params, X, y):
    X = jnp.atleast_2d(X)
    y = jnp.atleast_1d(y)
    logits = forward(params, X)
    margins = y * logits
    loss = jnp.mean((logits - y) ** 2)

    # weight decay on W1 and W2
    W1, b1, W2 = params
    l2_reg = jnp.sum(W1 ** 2) + jnp.sum(W2 ** 2)
    return loss + 0.5 * wd * l2_reg

@jax.jit
def accuracy(params, X, y):
    preds = forward(params, X)
    margins = y * preds
    return jnp.mean(margins > 0)



@jax.jit
def update(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)

    return tuple(p - lr * g for p,g in zip(params, grads)), grads

@jax.jit
def col_norm(params, X, y):
    gW1, gb1, gW2 = jax.grad(hinge_loss)(params, X, y)
    col_scales = jnp.linalg.norm(gW1, axis=0)
    col_scales = jnp.maximum(col_scales, 0.01)
    dW1_scaled = gW1 / col_scales[None, :]
    W1, b1, W2 = params
    new_params = (
        W1 - lr * dW1_scaled,
        b1 - lr * gb1,
        W2 - lr * gW2,
    )
    return new_params, (gW1, gb1, gW2)


@jax.jit
def egd_update(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    print("Computing SVD...")
    U, s = jnp.linalg.svd(G1, full_matrices=False)[:2]
    s = s.ravel()
    aux = (U / s) @ U.T
    G1 = aux@G1
    grads[0] = G1
    return tuple(p - lr * g for p,g in zip(params, grads)), grads

def randomized_svd_jax(G, rank, n_iter=2, key=jax.random.PRNGKey(0)):
    m, n = G.shape

    # Random gaussian matrix 
    Q = jax.random.normal(key, (n, rank), dtype=G.dtype)

    # Power iterations
    for _ in range(n_iter):
        Q = jnp.linalg.qr(G @ Q, mode='reduced')[0]
        Q = jnp.linalg.qr(G.T @ Q, mode='reduced')[0]

    # Small matrix
    B = G @ Q

    # SVD on B
    U_hat, S, V_hat_t = jnp.linalg.svd(B, full_matrices=False)

    return U_hat, S

@jax.jit
def egd_rsvd_update_40(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s= randomized_svd_jax(G1, 40)
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned
    return tuple(p - lr * g for p, g in zip(params, grads))

@jax.jit
def egd_rsvd_update_30(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s= randomized_svd_jax(G1, 30) 
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned
    return tuple(p - lr * g for p, g in zip(params, grads))

@jax.jit
def egd_rsvd_update_20(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s = randomized_svd_jax(G1, 20)
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned

    return tuple(p - lr * g for p, g in zip(params, grads))

@jax.jit
def egd_rsvd_update_10(params, X, y):
    grads = jax.grad(hinge_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s = randomized_svd_jax(G1, 10)
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned

    return tuple(p - lr * g for p, g in zip(params, grads))

# ── Training loop ─────────────────────────────────────────────────────────────

num_batches = X_tr.shape[0] // batch_size
artifacts = {}
methods = ["vanilla", "egd" ,"egd_rsvd10", "egd_rsvd20", "egd_rsvd30", "egd_rsvd40"]

for method in methods:
    history = {}
    train_acc = []
    test_acc  = []

    params = init_params(n, hidden_dim, random_state=42)
    history = [param.copy() for param in params]
    with open(f"init_parity_k{k}_{method}.pkl", "wb") as f:
      pickle.dump(history, f)
    reach_95 = None
    te = 0.0
    avg_epoch_cost = 0.0
    for epoch in range(epochs):
        # shuffle
        perm = rng.permutation(X_tr.shape[0])
        X_shuf = X_tr[perm]; y_shuf = y_tr[perm]

        # run mini‐batch GD (i.e SGD)
        for i in range(num_batches):
            xb = X_shuf[i*batch_size:(i+1)*batch_size]
            yb = y_shuf[i*batch_size:(i+1)*batch_size]
            if (te > .95) and (reach_95 is None):
                print(epoch)
                reach_95 = epoch
            if method == "vanilla" or (epoch > 0 and te > .95):
                params, grads = update(params, xb, yb)
            elif method == "col_norm":
                params, grads = col_norm(params, xb, yb)
            elif method == "egd":
                params, grads = egd_update(params, xb, yb)
            elif method == "egd_rsvd10":
                params = smartplus_update_10(params, xb, yb)
            elif method == "egd_rsvd20":
                params = smartplus_update_20(params, xb, yb)
            elif method == "egd_rsvd30":
                params = smartplus_update_30(params, xb, yb)
            elif method == "egd_rsvd40":
                params = smartplus_update_40(params, xb, yb)

        ta = accuracy(params, X_tr, y_tr)
        te = accuracy(params, X_te, y_te)
        train_acc.append(float(ta))
        test_acc.append(float(te))
        
        if epoch % 10 == 0:
            print(f"{method:s}: Epoch {epoch:03d}: train_acc={ta:.4f}, test_acc={te:.4f}")
            history[epoch] = [param.copy() for param in params]
    artifacts[method] = train_acc, test_acc, history

