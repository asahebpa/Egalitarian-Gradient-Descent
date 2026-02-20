from math import sqrt
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ── Data generation (disjoint train/test) ─────────────────────────────────────
def make_disjoint_mod_datasets(key, p, train_fraction=0.25, shuffle_seed=None):
    """
    Create disjoint train/test datasets for modular multiplication.
    Train is a fraction of all possible samples.
    Test is the remaining samples (no overlap).
    """
    total_examples = p * p
    all_a = jnp.repeat(jnp.arange(p), p)
    all_b = jnp.tile(jnp.arange(p), p)
    y = (all_a * all_b) % p

    # Shuffle indices once
    if shuffle_seed is not None:
        perm_key = random.PRNGKey(shuffle_seed)
    else:
        perm_key = key
    perm = random.permutation(perm_key, total_examples)

    # Train/test split
    train_size = int(total_examples * train_fraction)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    a_train, b_train, y_train = all_a[train_idx], all_b[train_idx], y[train_idx]
    a_test, b_test, y_test = all_a[test_idx], all_b[test_idx], y[test_idx]

    def encode(a, b, y):
        N = a.shape[0]
        X = jnp.zeros((N, 2 * p), dtype=jnp.float32)
        X = X.at[jnp.arange(N), a].set(1.0)
        X = X.at[jnp.arange(N), p + b].set(1.0)
        y_onehot = jax.nn.one_hot(y, p)
        return X, y_onehot

    X_tr, y_tr = encode(a_train, b_train, y_train)
    X_te, y_te = encode(a_test, b_test, y_test)

    return (X_tr, y_tr), (X_te, y_te)


# ── GROKKING HYPERPARAMETERS ────────────────────────────────────────────────
# p = 79
# p = 97
p = 127
train_fraction = 0.5

(X_tr, y_tr), (X_te, y_te) = make_disjoint_mod_datasets(
    random.PRNGKey(0), p, train_fraction=train_fraction, shuffle_seed=42
)

print(f"Training examples: {X_tr.shape[0]} ({train_fraction:.1%} of {p*p} total)")
print(f"Test examples: {X_te.shape[0]} ({1-train_fraction:.1%} of {p*p} total)")

n = X_tr.shape[1]
hidden_dim = 512
lr = 7e-1
wd = 0.0001
batch_size = 512
epochs = 15000

print(f"Key grokking parameters:")
print(f"  Weight decay: {wd}")
print(f"  Learning rate: {lr}")
print(f"  Batch size: {batch_size}")
print(f"  Hidden dim: {hidden_dim}")


# ── Model init ───────────────────────────────────────────────────────────────
def init_params(key, n, hidden_size, out_dim):
    k1, k2, k3 = random.split(key, 3)
    # Xavier/Glorot initialization
    W1 = random.normal(k1, (n, hidden_size)) * sqrt(2 / n)
    b1 = jnp.zeros((hidden_size,))
    W2 = random.normal(k3, (hidden_size, out_dim)) * sqrt(2 / hidden_size)
    return (W1, b1, W2)


# ── Forward & loss ───────────────────────────────────────────────────────────
def forward(params, X):
    W1, b1, W2 = params
    h = jnp.maximum(0, X @ W1 + b1)  # ReLU
    return h @ W2


def cross_entropy_loss(params, X, y):
    logits = forward(params, X)
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    loss = -jnp.mean(jnp.sum(y * log_probs, axis=1))

    # STRONG weight decay
    W1, b1, W2 = params
    l2_reg = jnp.sum(W1**2) + jnp.sum(W2**2)
    return loss + 0.5 * wd * l2_reg


@jax.jit
def accuracy(params, X, y):
    preds = jnp.argmax(forward(params, X), axis=1)
    labels = jnp.argmax(y, axis=1)
    return jnp.mean(preds == labels)


def weight_norm(params):
    W1, b1, W2 = params
    return jnp.sqrt(jnp.sum(W1**2) + jnp.sum(W2**2))


# ── Training step variations ─────────────────────────────────────────────────
@jax.jit
def update(params, X, y):
    grads = jax.grad(cross_entropy_loss)(params, X, y)
    return tuple(p - lr * g for p, g in zip(params, grads))


@jax.jit
def col_norm_update(params, X, y):
    gW1, gb1, gW2 = jax.grad(cross_entropy_loss)(params, X, y)

    col_scales = jnp.linalg.norm(gW1, axis=0)
    col_scales = jnp.maximum(col_scales, 0.01)
    dW1_scaled = gW1 / col_scales[None, :]

    W1, b1, W2 = params
    new_params = (
        W1 - lr * dW1_scaled,
        b1 - lr * gb1,
        W2 - lr * gW2,
    )
    return new_params


def randomized_svd_jax(G, rank, n_iter=2, key=jax.random.PRNGKey(0)):
    m, n = G.shape

    # Random gaussian matrix (same as torch.randn)
    Q = jax.random.normal(key, (n, rank), dtype=G.dtype)

    # Power iterations (QR on G Q, then on Gᵀ Q)
    for _ in range(n_iter):
        Q = jnp.linalg.qr(G @ Q, mode='reduced')[0]
        Q = jnp.linalg.qr(G.T @ Q, mode='reduced')[0]

    # Small matrix
    B = G @ Q

    # SVD on B
    U_hat, S, V_hat_t = jnp.linalg.svd(B, full_matrices=False)

    return U_hat, S

@jax.jit
def egd_rsvd_update_128(params, X, y):
    grads = jax.grad(cross_entropy_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s = randomized_svd_jax(G1, 128) 
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned
    return tuple(p - lr * g for p, g in zip(params, grads))

@jax.jit
def egd_rsvd_update_64(params, X, y):
    grads = jax.grad(cross_entropy_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s = randomized_svd_jax(G1, 64) 
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned

    return tuple(p - lr * g for p, g in zip(params, grads))

@jax.jit
def egd_update(params, X, y):
    grads = jax.grad(cross_entropy_loss)(params, X, y)
    grads = list(grads)
    G1 = grads[0]
    U, s,_ = jnp.linalg.svd(G1, full_matrices=False)
    s = jnp.maximum(s, 1e-6)
    s_inv = 1.0 / s
    aux = (U * s_inv[None, :]) @ U.T
    G1_preconditioned = aux @ G1
    grads[0] = G1_preconditioned
    return tuple(p - lr * g for p, g in zip(params, grads))


# ── Training loop ────────────────────────────────────────────────────────────
num_batches = max(1, X_tr.shape[0] // batch_size)
artifacts = {}
methods = ["vanilla", "col_norm", "egd", "egd_rsvd128", "egd_rsvd64"]
for method in methods:
    print(f"\n{'='*30}")
    print(f"Training with method: {method}")
    print(f"{'='*30}")

    params = init_params(random.PRNGKey(42), n, hidden_dim, p)
    train_acc, test_acc, weight_norms = [], [], []
    eval_epochs = []
    reach_95 = None
    start = time.time()
    te = 0.0
    for epoch in range(epochs):
        epoch_start = time.time()
        perm = random.permutation(random.PRNGKey(epoch), X_tr.shape[0])
        X_shuf, y_shuf = X_tr[perm], y_tr[perm]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, X_tr.shape[0])
            xb, yb = X_shuf[start_idx:end_idx], y_shuf[start_idx:end_idx]
            if (te > .95) and (reach_95 is None):
                reach_95 = epoch

            if method == "vanilla" or (epoch > 0 and te > .95):
                params = update(params, xb, yb)
            elif method == "col_norm":
                params = col_norm_update(params, xb, yb)
            elif method == "egd":
                params = egd_update(params, xb, yb)
            elif method == "egd_rsvd128":
                params = egd_rsvd_update_128(params, xb, yb)
            elif method == "egd_rsvd64":
                params = egd_rsvd_update_64(params, xb, yb)
        if epoch < 1000:
            eval_freq = 50
        elif epoch < 5000:
            eval_freq = 100
        else:
            eval_freq = 200
        epoch_time = time.time() - epoch_start
        if epoch % 1 == 0 or epoch == epochs - 1:
            ta = accuracy(params, X_tr, y_tr)
            te = accuracy(params, X_te, y_te)
            wn = weight_norm(params)

            train_acc.append(float(ta))
            test_acc.append(float(te))
            weight_norms.append(float(wn))
            eval_epochs.append(epoch)

            print(f"{method:>10s}: Epoch {epoch:05d}: train_acc={ta:.4f}, test_acc={te:.4f}, weight_norm={wn:.3f}")

    artifacts[method] = (train_acc, test_acc, weight_norms, eval_epochs)




