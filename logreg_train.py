#!/usr/bin/env python3

import json
import sys

import polars as pl
import numpy as np
import numpy.typing as npt
# from sklearn.metrics import accuracy_score

from describe import ft_count, ft_arithmetic_mean, ft_std


def load_dataset(path: str) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    df = pl.read_csv(path)

    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Int64) and c not in ["Index"]
    ]
    print("Rows before drop:", df.height)

    numeric_cols = [
        c
        for c in numeric_cols
        if c
        not in (
            "Arithmancy",
            "Care of Magical Creatures",  # homogene histograms, low anova f-scores
            "Defense Against the Dark Arts",  # perfect correlation with Astronomy
        )
    ]
    # df = df.fill_null(strategy="mean")
    df = df.drop_nulls(subset=numeric_cols)

    print("Rows after drop:", df.height)
    print("Features kept:", numeric_cols)

    X = df.select(numeric_cols).to_numpy()
    y = df["Hogwarts House"].to_numpy()
    return X, y, numeric_cols


def train_test_stratified_split(
    X: npt.NDArray,
    y: npt.NDArray,
    ratio: float = 0.8,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    train_indices = []
    val_indices = []

    houses = np.unique(y)

    for h in houses:
        h_indices = np.where(y == h)[0]
        np.random.shuffle(h_indices)

        split = int(len(h_indices) * ratio)

        train_indices.extend(h_indices[:split])
        val_indices.extend(h_indices[split:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return (
        X[train_indices],
        X[val_indices],
        y[train_indices],
        y[val_indices],
    )


def get_all_subjects_means_stds(X: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    means: list = []
    stds: list = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_count = ft_count(col)
        m = ft_arithmetic_mean(col, col_count)
        s = ft_std(col, col_count, m)
        means.append(m)
        stds.append(s)
    return np.array(means), np.array(stds)


def apply_standardization(
    X: npt.NDArray, means: npt.NDArray, stds: npt.NDArray
) -> npt.NDArray:

    return (X - means) / stds


def logistic_function(z: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-z))


def compute_loss(
    y: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_binary(
    X: npt.NDArray[np.float64],  # shape (m, n)
    y_binary: npt.NDArray[np.float64],  # shape (m, 1)
    alpha: float = 0.005,  # learning rate
    epochs: int = 10_000,
) -> tuple[npt.NDArray[np.float64], np.float64]:
    """
    Long docstring here. Fold it by clicking on the little down arrow just between the
    line number and the triple quote (on VSCode).
    Will probaby write a PDF/markdown instead :p

    Here are some links between the suject's formulas, and the following code,
    which performs the gradient descent.
    Here in the code, all calculation are made with matrices
    (over all the student and subject, at the same time), so no sum needed.
    Remind well this sentence, as it will be summoned several times during these explanations.

    Notations, matches between subject's formulas and code :
    * m:
        number of students in the clean train dataset.
        Here in the code: `m`
    * j (lowercase):
        number of subjects (kept ; thanks to dataviz/analytics enlightenment).
        Here in the code: `nb_subjects`, and `n` in the shape comments
    * θ:
        is the nb_subjects sized vector for the weights.
        Updated during each logistic regression gradient descent loop iteration
        (see the related line explanation below).
        Here in the code: `weights`
    * y_i:
        written in the subject in a sum formula, as i is the index incremented.
        Here in the code: `y_binary`
            is then a m sized vector, filled with 0 or 1, according to
            student house membership, depending on the current house's model training.
    * x (also written x^i, or even (x_j)^i in the subject):
        x^i (in the loss function):
            subjects' marks for the i-th student.
        x (in the h function definition):
            matrix with m lines, j columns.
            All students' marks, on all subjects.
        (x_j)^i (in the partial derivative definition):
            The i-th student's mark, in the j-th subject.
        Here in the code:
            `X` (uppercase) is defined and used as a (m, n) shaped matrix,
            with m lines (number of students), and n columns (number of subjects)
    * ^T (the T exponent):
        Is not a power, but states instead for "transpose"
        (see there: https://en.wikipedia.org/wiki/Transpose).
        If a matrix M is (m, n) shaped, then M^T is (n, m) shaped.
        If a vector V is (m, 1) shaped (a column matrix, with m lines),
        then V^T is (1, m) shaped (a line matrix, with m columns).
        To understand what is coming next, you may also read how matrices are multiplied here:
        https://en.wikipedia.org/wiki/Matrix_multiplication.
        Here in the code, matrix multiplication symbol is `@`
        Here is, shortly, the logic of how the result's shape, according the two matrices' shapes:
        Let's consider (in this example symbols such as X or Y are not related to the suject's symbols):
            * X, a (m, n) shaped matrix ; m lines, n columns
            * Y, a (n, p) shaped matrix ; n lines, p columns
            Thus, X.Y
                (where . is the matrix multiplication
                    (which is not commutative: X.Y ≠ Y.X ; instead of real multiplication a*b = b*a
                    X.Y can be equal to Y.X under specific circumstances,
                    but equality is not always true (provided that multiplication is possible
                    (see below about multiplication possibility)
                    )
                )
            gives a result Z (m, p) shaped.
            Important:
                X.Y is possible only if the X's number of columns
                    (`n` in this example)
                is equal to the Y's number of lines
                    (`n` (same symbol)).
                If X was (m, n) shaped, Y (p, q) shaped, and n ≠ q,
                thus X.Y does not make any sense, and cannot be calculated.

        More specifically, let's consider:
            * X, a (n, 1) shaped matrix ; also called a "column matrix" (n lines, 1 column),
            or a "vector".
            * Y, a same as X shaped matrix: (n, 1)
            According to how matrix multiplication is possible, X @ Y does not make any sense:
            1 ≠ n
            However, transposition allows to multiply !
            If X is (n, 1) shaped, thus X^T is (1, n) shaped.
            Thus:
                X^T . Y is now possible, and gives as a result just a number
                (or a (1, 1) shaped matrix we could say.)
            In this case, X^T . Y = Y^T . X = Y . X^T = Y^T . X

        Here in the code, transposition of a matrix X is symbolized as such:
            `X.T`
    * h_θ:
        Defined in the subject as such:
            g(θ^T * x)
            Read carefully the next section about the score calculation.
        * g, the logistic/sigmoid function is also defined just below in the subject as such:
            1/(1+e^-z)
        Here in the code:
            `prediction`: npt.NDArray[np.float64] = logistic_function(score)


    Current function code instructions, related to subject's formulas
    * score: npt.NDArray[np.float64] = X @ weights + intercept
        This score calculation is related to θ^T * x inside the g function in the subject.
        "Where hθ(x) is defined in the following way : hθ(x) = g(θ^T x)
        In the subject, it is given for one student.
        As:
        * θ is a (n, 1) shaped vector (the weights for each subject)
        * x (which should be written x_i in fact) is also a (n, 1) shaped vector.
        Then, multiply them requires to transpose one them,
        otherwise, multiplication is not possible.
        As saw just above, with only two same shaped vectors:
            no matter which one is transposed, and then the order of multiplication,
            the result is the same (one number) as long as
            the number of columns of the first equals the number of lines of the second.
        Here in the code (important):
        * As said before:
            "all calculation are made with matrices"!
            So here, `X` is a (m, n) shaped matrix (as defined above).
        * θ remains a (n, 1) vector (as defined above): the `weights`
        * Thus, there is only one way to multiply them:
            X @ θ ; Reminder: @ is the symbol to multiply two matrices in the code.
            (And not θ^T @ X, as it is written in the subject.)
            No need to transpose, just adapt the order, and it provides the proper result:
                a (m, 1) shaped vector: one score per student.
    *   prediction: npt.NDArray[np.float64] = logistic_function(score)  # shape (m,)
        Related to z in the subject's formula.
        Once the logistic function is applied, score is shaped into a number between 0 and 1,
        as a probability/prediction should be.
    *   error: npt.NDArray[np.float64] = prediction - y_binary  # shape (m,)
        Is related to the last subject's formula: the cost function partial derivative.
        Still, as "all calculation are made with matrices", no need to care about indices,
        or sum symbols.
        These aspects are instead handled with matrices calculations (multiplications).
        Reminder:
        * `prediction` is related to h_θ in the subject's formulas
        * `y_binary` is related to y^i
    *   grad_weights: npt.NDArray[np.float64] = (X.T @ error) / sample_size  # shape (n,)
        In the partial derivative subject's formula, a multiplication by (x_j)^i is written.
        Reminder: this partial derivative formula is given for one subject.
        Still, as "all calculation are made with matrices", no sum is needed.
        However, matrices multiplications implies to deal properly to make them possible.
        Let's study matrices shapes, to explain the order:
        * `X` is (m, n) shaped
        * `error` is (m, 1) shaped
        Thus:
            The only way to make this matrices multiplication possible is:
            `X` transposed, (n, m) shaped
            multiplied by
            `error` (m, 1) shaped
            which provides a (n, 1) result:
                a column matrix, with n lines, and 1 column.
                One weight gradient per subject.
        Also, 1/m is out of the sum in the partial derivative formula, is still has to be done,
        by dividing by `sample_size`.

    """
    sample_size: int
    nb_subjects: int
    sample_size, nb_subjects = X.shape
    weights: npt.NDArray[np.float64] = np.zeros(
        nb_subjects, dtype=np.float64
    )  # shape (n, 1)
    intercept: np.float64 = np.float64(0.0)

    for epoch in range(epochs):
        score: npt.NDArray[np.float64] = X @ weights + intercept  # shape (m, 1)
        prediction: npt.NDArray[np.float64] = logistic_function(score)  # shape (m, 1)
        error: npt.NDArray[np.float64] = prediction - y_binary  # shape (m, 1)
        grad_weights: npt.NDArray[np.float64] = (
            X.T @ error
        ) / sample_size  # shape (n, 1)
        grad_intercept: np.float64 = np.float64(np.mean(error))  # real

        weights -= alpha * grad_weights  # shape (n, 1)
        intercept -= alpha * grad_intercept  # real

        if epoch % 1_000 == 0:
            loss = compute_loss(y_binary, prediction)
            formatted = ", ".join(f"{w:.6f}" for w in grad_weights)
            print(
                f"Epoch {epoch} | Loss {loss:.4f} | intercept={intercept} | "
                f"Weights gradient {formatted}"
            )

    return weights, intercept


def logreg_one_vs_rest(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> dict[str, tuple[npt.NDArray[np.float64], np.float64]]:
    houses = np.unique(y)
    models: dict = dict()

    for house in houses:
        print("\nTraining:", house)
        y_binary = (y == house).astype(int)
        house_subjects_weights, intercept = train_binary(X, y_binary)
        models[house] = (house_subjects_weights, intercept)

    return models


def predict_ovr(
    models: dict[str, tuple[npt.NDArray[np.float64], np.float64]],
    X: npt.NDArray[np.float64],  # shape (m, n)
) -> npt.NDArray[np.str_]:

    classes: list[str] = list(models.keys())  # k classes ; here: 4
    weight_matrix = np.vstack([models[c][0] for c in classes])  # shape (k, n)
    intercept_vector = np.array([models[c][1] for c in classes])  # shape (k, 1)
    scores = X @ weight_matrix.T + intercept_vector.T  # (k, m)
    best_indices = np.argmax(scores, axis=1)

    return np.array([classes[i] for i in best_indices])


def accuracy(
    y_true: npt.NDArray[np.int16], y_pred: npt.NDArray[np.str_]
) -> np.floating:
    return np.mean(y_true == y_pred)


def save_model(
    models: dict[str, tuple[npt.NDArray[np.float64], np.float64]],
    means: npt.NDArray[np.float64],
    stds: npt.NDArray[np.float64],
    feature_names: list[str],
    output_path: str = "model.json",
) -> None:

    model_data: dict = {
        "metadata": {
            "model": "logistic_regression_ovr",
            "n_features": len(feature_names),
        },
        "features": feature_names,
        "standardization": {
            "means": means.tolist(),
            "stds": stds.tolist(),
        },
        "classes": {},
    }

    for house, (weights, intercept) in models.items():
        model_data["classes"][house] = {
            "weights": weights.tolist(),
            "intercept": float(intercept),
        }

    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=4)

    print(f"\nModel saved to {output_path}")


def main(train_file_path: str) -> None:
    X, y, feature_names = load_dataset(train_file_path)
    X_train, X_val, y_train, y_val = train_test_stratified_split(X, y)
    means, stds = get_all_subjects_means_stds(X_train)
    X_train = apply_standardization(X_train, means, stds)
    X_val = apply_standardization(X_val, means, stds)
    models = logreg_one_vs_rest(X_train, y_train)
    save_model(models, means, stds, feature_names)
    preds = predict_ovr(models, X_val)
    acc = accuracy(y_val, preds)
    print(f"\nAccuracy: {acc}")
    # print(f"skluracy: {accuracy_score(y_val, preds)}")  # sklearn accuracy comparison


if __name__ == "__main__":
    if len(sys.argv) != 2 and sys.argv[1] != "dataset_train.csv".strip():
        print("Usage: ./logreg_train.py dataset_train.csv")
        exit(1)
    main(sys.argv[1])
