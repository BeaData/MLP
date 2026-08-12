#!/usr/bin/python3

"""
python mlp.py \
    --split \
    --dataset data.csv \
    --seed 42
python mlp.py \
    --train \
    --dataset data_training.csv \
    --valid data_validation.csv
"""

import argparse
import sys
import numpy as np

from utils import (
    load,
    prepare,
    fit_normalization,
    apply_normalization,
    split_data,
    plot_learning_curves
)

from mlp_core import (
    train,
    predict,
    save_model,
    load_model,
    categorical_cross_entropy,
    forward
)


def main():

    parser = argparse.ArgumentParser(
        description="MLP Breast Cancer"
    )

    # phases
    parser.add_argument(
        "--split",
        action="store_true"
    )

    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--predict",
        action="store_true"
    )

    # files
    parser.add_argument(
        "--dataset",
        type=str
    )

    parser.add_argument(
        "--valid",
        type=str
    )

    parser.add_argument(
        "--model",
        type=str,
        default="saved_model.npy"
    )

    # network
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[24, 24]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        # default=70
        default=90
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr",
        type=float,
        # default=0.0314
        default=0.01
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2
    )

    args = parser.parse_args()

    # ======================================================
    # SPLIT
    # ======================================================

    if args.split:

        if not args.dataset:
            print(
                "Error: --dataset required."
            )
            sys.exit(1)

        split_data(
            args.dataset,
            args.val_ratio,
            args.seed
        )

        sys.exit(0)

    # ======================================================
    # TRAIN
    # ======================================================

    if args.train:

        if not args.dataset:
            print(
                "Error: training dataset missing."
            )
            sys.exit(1)

        if not args.valid:
            print(
                "Error: validation dataset missing."
            )
            sys.exit(1)

        train_data = load(args.dataset)

        if train_data is None:
            sys.exit(1)

        valid_data = load(args.valid)

        if valid_data is None:
            sys.exit(1)

        X_train, y_train, _ = prepare(
            train_data
        )

        X_valid, y_valid, _ = prepare(
            valid_data
        )

        mean, std = fit_normalization(
            X_train
        )

        X_train = apply_normalization(
            X_train,
            mean,
            std
        )

        X_valid = apply_normalization(
            X_valid,
            mean,
            std
        )

        print(
            f"x_train shape : {X_train.shape}"
        )

        print(
            f"x_valid shape : {X_valid.shape}"
        )

        input_size = X_train.shape[1]

        layer_sizes = (
            [input_size]
            + args.layers
            + [2]
        )

        weights, biases, history = train(
            X_train,
            y_train,
            X_valid,
            y_valid,
            layer_sizes,
            args.epochs,
            args.batch_size,
            args.lr,
            args.seed
        )

        save_model(
            args.model,
            weights,
            biases,
            layer_sizes,
            mean,
            std
        )

        plot_learning_curves(
            history
        )

        activations, _ = forward(
            X_valid,
            weights,
            biases
        )

        probs = activations[-1]

        loss = categorical_cross_entropy(
            y_valid,
            probs
        )

        preds = np.argmax(
            probs,
            axis=1
        )

        truth = np.argmax(
            y_valid,
            axis=1
        )

        acc = np.mean(
            preds == truth
        )

        print(
            f"Final results "
            f"- loss: {loss:.4f} "
            f"- acc: {acc:.4f}"
        )

        sys.exit(0)

    # ======================================================
    # PREDICT
    # ======================================================

    if args.predict:

        if not args.dataset:
            print(
                "Error: dataset missing."
            )
            sys.exit(1)

        data = load(args.dataset)

        if data is None:
            sys.exit(1)

        X, y, y_raw = prepare(
            data
        )

        (
            weights,
            biases,
            layer_sizes,
            mean,
            std
        ) = load_model(args.model)

        X = apply_normalization(
            X,
            mean,
            std
        )

        preds, probs = predict(
            X,
            weights,
            biases
        )

        loss = categorical_cross_entropy(
            y,
            probs
        )

        truth = np.argmax(
            y,
            axis=1
        )

        acc = np.mean(
            preds == truth
        )

        print(
            f"Loss: {loss:.4f}"
        )

        print(
            f"Accuracy: {acc:.4f}"
        )

        for i in range(
            min(10, len(preds))
        ):

            pred_label = (
                "M"
                if preds[i] == 0
                else "B"
            )

            print(
                f"{i:03d} "
                f"pred={pred_label} "
                f"true={y_raw[i]}"
            )

        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
