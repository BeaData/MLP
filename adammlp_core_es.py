#!/usr/bin/python3

try:
    import numpy as np
except ImportError:
    np = None

#
# ACTIVATION FUNCTIONS
# ####################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a):
    return a * (1 - a)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


#
# LOSS
# ####

def categorical_cross_entropy(y_true, y_pred, eps=1e-12):
    """
    Softmax + one-hot targets
    """
    y_pred = np.clip(y_pred, eps, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


#
# INITIALIZATION
# ##############

def xavier_uniform(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(
        -limit,
        limit,
        (fan_in, fan_out)
    )


def init_weights(layer_sizes, seed=None):

    if seed is not None:
        np.random.seed(seed)

    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):

        w = xavier_uniform(
            layer_sizes[i],
            layer_sizes[i + 1]
        )

        b = np.zeros(
            (1, layer_sizes[i + 1])
        )

        weights.append(w)
        biases.append(b)

    return weights, biases


def init_adam(weights, biases):

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    return (
        m_w,
        v_w,
        m_b,
        v_b
    )


#
# FORWARD
# #######

def forward(X, weights, biases):

    activations = [X]
    zs = []
    a = X

    for i in range(len(weights)):

        z = np.dot(a, weights[i]) + biases[i]
        zs.append(z)

        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = sigmoid(z)
        activations.append(a)

    return activations, zs


#
# BACKPROPAGATION
# ###############

def backward(
    activations,
    weights,
    biases,
    y_true
):

    m = y_true.shape[0]

    grad_w = [np.zeros_like(w) for w in weights]
    grad_b = [np.zeros_like(b) for b in biases]

    # Softmax + Cross Entropy
    delta = activations[-1] - y_true

    for layer in reversed(range(len(weights))):

        grad_w[layer] = (
            np.dot(
                activations[layer].T,
                delta
            ) / m
        )

        grad_b[layer] = (
            np.sum(
                delta,
                axis=0,
                keepdims=True
            ) / m
        )

        if layer > 0:

            delta = (
                np.dot(delta, weights[layer].T)
                * sigmoid_derivative(activations[layer])
            )

    return grad_w, grad_b


#
# UPDATE
# ######

def update_weights(
    weights,
    biases,
    grad_w,
    grad_b,
    learning_rate
):

    for i in range(len(weights)):

        weights[i] -= learning_rate * grad_w[i]
        biases[i] -= learning_rate * grad_b[i]


def update_weights_adam(
    weights,
    biases,
    grad_w,
    grad_b,
    m_w,
    v_w,
    m_b,
    v_b,
    learning_rate,
    t,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):

    for i in range(len(weights)):

        m_w[i] = (beta1 * m_w[i] + (1 - beta1) * grad_w[i])
        v_w[i] = (beta2 * v_w[i] + (1 - beta2) * (grad_w[i] ** 2))
        m_b[i] = (beta1 * m_b[i] + (1 - beta1) * grad_b[i])
        v_b[i] = (beta2 * v_b[i] + (1 - beta2) * (grad_b[i] ** 2))

        m_w_hat = (m_w[i] / (1 - beta1 ** t))
        v_w_hat = (v_w[i] / (1 - beta2 ** t))
        m_b_hat = (m_b[i] / (1 - beta1 ** t))
        v_b_hat = (v_b[i] / (1 - beta2 ** t))

        weights[i] -= (
            learning_rate * m_w_hat
            / (np.sqrt(v_w_hat) + eps)
        )

        biases[i] -= (
            learning_rate * m_b_hat
            / (np.sqrt(v_b_hat) + eps)
        )

#
# TRAIN
# #####


def train(
    X_train,
    y_train,
    X_valid,
    y_valid,
    layer_sizes,
    epochs,
    batch_size,
    learning_rate,
    seed=None,
    patience=10,
    min_delta=1e-4,
    verbose=True
):

    weights, biases = init_weights(
        layer_sizes,
        seed
    )

    m_w, v_w, m_b, v_b = init_adam(
        weights,
        biases
    )

    t = 0

    n_samples = X_train.shape[0]

    history = {
        "loss": [],
        "val_loss": [],
        "acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    best_weights = [
        w.copy()
        for w in weights
    ]

    best_biases = [
        b.copy()
        for b in biases
    ]

    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        indices = np.random.permutation(
            n_samples
        )

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(
            0,
            n_samples,
            batch_size
        ):

            end = min(
                start + batch_size,
                n_samples
            )

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            activations, _ = forward(
                X_batch,
                weights,
                biases
            )

            grad_w, grad_b = backward(
                activations,
                weights,
                biases,
                y_batch
            )

            t += 1

            update_weights_adam(
                weights,
                biases,
                grad_w,
                grad_b,
                m_w,
                v_w,
                m_b,
                v_b,
                learning_rate,
                t
            )

        # -------------
        # TRAIN METRICS
        # -------------

        train_act, _ = forward(
            X_train,
            weights,
            biases
        )

        train_probs = train_act[-1]

        loss = categorical_cross_entropy(
            y_train,
            train_probs
        )

        train_pred = np.argmax(
            train_probs,
            axis=1
        )

        train_true = np.argmax(
            y_train,
            axis=1
        )

        acc = np.mean(
            train_pred == train_true
        )

        # -------------
        # VALID METRICS
        # -------------

        valid_act, _ = forward(
            X_valid,
            weights,
            biases
        )

        valid_probs = valid_act[-1]

        val_loss = categorical_cross_entropy(
            y_valid,
            valid_probs
        )

        valid_pred = np.argmax(
            valid_probs,
            axis=1
        )

        valid_true = np.argmax(
            y_valid,
            axis=1
        )

        val_acc = np.mean(
            valid_pred == valid_true
        )

        history["loss"].append(loss)
        history["val_loss"].append(val_loss)

        history["acc"].append(acc)
        history["val_acc"].append(val_acc)

        if verbose:

            print(
                f"epoch {epoch:02d}/{epochs}"
                f" - loss: {loss:.4f}"
                f" - val_loss: {val_loss:.4f}"
                f" - acc: {acc:.4f}"
                f" - val_acc: {val_acc:.4f}"
            )
        # ------------------
        # EARLY STOPPING
        # ------------------

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss

            best_weights = [
                w.copy()
                for w in weights
            ]

            best_biases = [
                b.copy()
                for b in biases
            ]

            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )

            print(
                f"Best epoch: {best_epoch}"
            )

            break


    weights = best_weights
    biases = best_biases

    return (
        weights,
        biases,
        history
    )


#
# PREDICT
# #######

def predict(X, weights, biases):

    activations, _ = forward(
        X,
        weights,
        biases
    )

    probs = activations[-1]

    preds = np.argmax(
        probs,
        axis=1
    )

    return preds, probs


#
# MODEL IO
# ########

def save_model(
    filepath,
    weights,
    biases,
    layer_sizes,
    mean,
    std
):

    model = {
        "weights": weights,
        "biases": biases,
        "layer_sizes": layer_sizes,
        "mean": mean,
        "std": std
    }

    np.save(filepath, model)

    print(f"> saving model '{filepath}' to disk...")


def load_model(filepath):

    model = np.load(
        filepath,
        allow_pickle=True
    ).item()

    return (
        model["weights"],
        model["biases"],
        model["layer_sizes"],
        model["mean"],
        model["std"]
    )
