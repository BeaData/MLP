#!/usr/bin/python3

import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ==========================================================
# CSV LOADING
# ==========================================================

def load(path):

    if not isinstance(path, str):
        print("Error: path must be a string.")
        return None

    if not os.path.isfile(path):
        print(f"Error: file '{path}' does not exist.")
        return None

    if not path.lower().endswith(".csv"):
        print("Error: file is not a CSV.")
        return None

    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            dtype=str
        )
    except Exception:
        print("Error: unable to read CSV.")
        return None

    if data.size == 0:
        print("Error: empty dataset.")
        return None

    return data


# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare(data):

    y_raw = data[:, 1]

    X = data[:, 2:].astype(np.float64)

    y = np.zeros(
        (len(y_raw), 2),
        dtype=np.float64
    )

    y[y_raw == "M", 0] = 1.0
    y[y_raw == "B", 1] = 1.0

    return X, y, y_raw


# ==========================================================
# NORMALIZATION
# ==========================================================

def fit_normalization(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    return mean, std


def apply_normalization(X, mean, std):

    return (X - mean) / std


# ==========================================================
# DATASET LOADING
# ==========================================================

def load_dataset(path):

    data = load(path)

    if data is None:
        return None, None, None

    return prepare(data)


# ==========================================================
# TRAIN / VALID SPLIT
# ==========================================================

def split_data(
    filepath,
    val_ratio=0.2,
    seed=None
):

    if not 0 < val_ratio < 1:
        print(
            "Error: val_ratio must be between 0 and 1."
        )
        return

    if seed is not None:
        np.random.seed(seed)

    data = load(filepath)

    if data is None:
        return

    # ----------------------------------
    # Stratification on diagnosis column
    # ----------------------------------

    m_idx = np.where(data[:, 1] == "M")[0]
    b_idx = np.where(data[:, 1] == "B")[0]

    np.random.shuffle(m_idx)
    np.random.shuffle(b_idx)

    m_split = int(len(m_idx) * (1 - val_ratio))

    b_split = int(len(b_idx) * (1 - val_ratio))

    train_idx = np.concatenate(
        (
            m_idx[:m_split],
            b_idx[:b_split]
        )
    )

    valid_idx = np.concatenate(
        (
            m_idx[m_split:],
            b_idx[b_split:]
        )
    )

    # Final mix

    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)

    train_data = data[train_idx]
    valid_data = data[valid_idx]

    base = os.path.splitext(filepath)[0]

    train_path = (f"{base}_training.csv")

    valid_path = (f"{base}_validation.csv")

    np.savetxt(
        train_path,
        train_data,
        delimiter=",",
        fmt="%s"
    )

    np.savetxt(
        valid_path,
        valid_data,
        delimiter=",",
        fmt="%s"
    )

    # Check

    train_m = np.sum(train_data[:, 1] == "M")
    train_b = np.sum(train_data[:, 1] == "B")

    valid_m = np.sum(valid_data[:, 1] == "M")
    valid_b = np.sum(valid_data[:, 1] == "B")

    print(f"Train shape : {train_data.shape}")
    print(f"Valid shape : {valid_data.shape}")

    print(
        f"Train -> M:{train_m} "
        f"B:{train_b}"
    )

    print(
        f"Valid -> M:{valid_m} "
        f"B:{valid_b}"
    )

    print(
        f"Saved files : "
        f"{train_path}, "
        f"{valid_path}"
    )


#
# LEARNING CURVES
# ###############

def plot_learning_curves(history):

    if plt is None:
        print(
            "matplotlib not available."
        )
        return

    plt.figure(figsize=(12, 4))

    # Loss

    plt.subplot(1, 2, 1)

    plt.plot(
        history["loss"],
        label="Train"
    )

    plt.plot(
        history["val_loss"],
        label="Validation"
    )

    plt.title("Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )
    # grid(color='Gainsboro', linestyle=':', linewidth=0.5)

    # Accuracy

    plt.subplot(1, 2, 2)

    plt.plot(
        history["acc"],
        label="Train"
    )

    plt.plot(
        history["val_acc"],
        label="Validation"
    )

    plt.title("Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.savefig("adam_learning_curves_es.png", dpi=150)
    plt.show()

    print(
        "> learning curves saved as 'adam_learning_curves_es.png'"
    )

#
# CONFUSION MATRIX
# ################


def plot_confusion_matrix(tp, fp, fn, tn):

    if plt is None:
        print("matplotlib not available.")
        return

    matrix = np.array([
        [tp, fn],
        [fp, tn]
    ])

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(matrix)

    ax.set_title("Adam - Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred M", "Pred B"])
    ax.set_yticklabels(["True M", "True B"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    plt.savefig(
        "adam_confusion_matrix_es.png",
        dpi=150
    )

    plt.show()

    print(
        "> confusion matrix saved as "
        "'adam_confusion_matrix_es.png'"
    )
