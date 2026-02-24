#!.venv/bin/python

import sys

import polars as pl
import numpy as np
import numpy.typing as npt
# from sklearn.metrics import accuracy_score

from describe import ft_count, ft_arithmetic_mean, ft_std


def load_dataset(
    path: str
) -> tuple[npt.NDArray, npt.NDArray]:
    df = pl.read_csv(path)

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Float64, pl.Int64)
        and c not in ["Index"]
    ]
    print("Rows before drop:", df.height)

    numeric_cols = [
        c for c in numeric_cols
        if c not in (
            "Arithmancy", "Care of Magical Creatures",  # homogene histograms
            "Defense Against the Dark Arts",  # perfect correlation with Astronomy
        )
    ]
    df = df.drop_nulls(subset=numeric_cols + ["Hogwarts House"])

    print("Rows after drop:", df.height)
    print("Features kept:", numeric_cols)

    X = df.select(numeric_cols).to_numpy()
    y = df["Hogwarts House"].to_numpy()
    return X, y


def train_val_split(
    X: npt.NDArray,
    y: npt.NDArray,
    ratio: float = 0.85,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    n: int = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    split = int(n * ratio)
    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def get_all_subjects_means_stds(
    X: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    means:list = []
    stds:list = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_count = ft_count(col)
        m = ft_arithmetic_mean(col, col_count)
        s = ft_std(col, col_count, m)
        means.append(m)
        stds.append(s)
    return np.array(means), np.array(stds)


def apply_standardization(
    X: npt.NDArray,
    means: npt.NDArray,
    stds: npt.NDArray
) -> npt.NDArray:
    X_std: npt.NDArray = np.zeros_like(X)

    for j in range(X.shape[1]):
        X_std[:, j] = (X[:, j] - means[j]) / stds[j]

    return X_std


def logistic_function(z: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-z))


def compute_loss(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_binary(
    X: npt.NDArray[np.float64],          # shape (m, n)
    y_binary: npt.NDArray[np.float64],   # shape (m,)
    alpha: float = 0.001,
    epochs: int = 700,
) -> tuple[npt.NDArray[np.float64], np.float64]:
    """
    Long docstring here. Fold it by clicking on the little down arrow just beside the
    number (on VSCode).
    Will probaby write a PDF instead :p

    Here are some links between the suject's formulas, and the following code,
    which performs the gradient descent.
    Here in the code, all calculation are made with matrices
    (over all the student and subject, at the same time), so no sum needed.
    
    Notations, matches between subject's formulas and code :
    * m:
        number of students in the clean train dataset.
        Here in the code: `m`
    * j (lowercase):
        number of subjects (kept ; thanks to dataviz/analytics enlightenment).
        Here in the code: `nb_subjects`
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
    * x (also written x^i, or even x^i_j in the subject):
        x^i (in the loss function):
            subjects' marks for the i-th student.
        x (in the h function definition):
            matrix with m lines, j columns.
            All students' marks, on all subjects.
        x^i_j (in the partial derivative definition):
            The i-th student's mark, in the j subject.
        Here in the code: `X` (uppercase)
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
                    (which is not commutative: X.Y ≠ Y.X ;
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
            * X, a (j, 1) shaped matrix ; also called a "column matrix" (j lines, 1 column),
            or a "vector".
            * Y, a same as X shaped matrix: (j, 1)
            According to how matrix multiplication is possible, X @ Y does not make any sense:
            1 ≠ j
            However, transposition allows to multiply !
            If X is (j, 1) shaped, thus X^T is (1, j) shaped.
            Thus:
                X^T . Y is now possible, and gives as a result just a number
                (or a (1, 1) shaped matrix we could say.)
            In this case, X^T . Y = Y^T . X = Y . X^T = Y^T . X

        Here in the code, transposition of a matrix X is symbolized as such:
            `X.T`
    * h_θ:
        The "score" calculation.
        In the subject, it is given for one student.
        As:
        * θ is a (j, 1) shaped vector
        * x (which should be written x_i in fact) is also a (j, 1) shaped vector.
        Then, getting the dot product of them requires to
    * z: npt.NDArray[np.float64] = X @ weights + intercept
        computes θ^T * x, according to the subject's formula stated as such:
        "Where hθ(x) is defined in the following way : hθ(x) = g(θ^T x)
        
    
    """
    sample_size: int
    nb_subjects: int
    sample_size, nb_subjects = X.shape
    weights: npt.NDArray[np.float64] = np.zeros(nb_subjects, dtype=np.float64)  # shape (n, )
    intercept: np.float64 = np.float64(0.0)

    print(X.shape)
    print(weights.shape)

    for epoch in range(epochs):
        z: npt.NDArray[np.float64] = weights @ X + intercept  # shape (m,)
        p: npt.NDArray[np.float64] = logistic_function(z)  # shape (m,)
        error: npt.NDArray[np.float64] = p - y_binary  # shape (m,)
        grad_weights: npt.NDArray[np.float64] = (X.T @ error) / sample_size  # shape (n,)
        grad_intercept: np.float64 = np.float64(np.mean(error))

        weights -= alpha * grad_weights
        intercept -= alpha * grad_intercept

        if epoch % 100 == 0:
            loss: np.float64 = compute_loss(y_binary, p)
            print(
                f"Epoch {epoch} | Loss {loss:.4f} "
                f"| error.shape={error.shape} "
                f"| z.shape={z.shape} "
                f"| intercept={intercept}"
            )

    return weights, intercept


def logreg_one_vs_rest(X: npt.NDArray[np.float64], y):
    houses = np.unique(y)
    models: dict = dict()

    for house in houses:
        print("\nTraining:", house)
        y_binary = (y == house).astype(int)
        house_subjects_weights, intercept = train_binary(X, y_binary)
        models[house] = (house_subjects_weights, intercept)

    #TODO
    print("########### MUST GENERATE WEIGHTS FILE FOR PREDICT #################")
    print(models)
    print("########### MUST GENERATE WEIGHTS FILE FOR PREDICT #################")

    return models


def predict_ovr(models, X):
    scores = []

    for house in models:
        house_subject_weights, intercept = models[house]
        scores.append(logistic_function(X @ house_subject_weights + intercept))

    scores = np.array(scores)
    preds = np.argmax(scores, axis=0)

    classes = list(models.keys())

    return np.array([classes[i] for i in preds])


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main(train_file_path: str) -> None:
    X, y = load_dataset(train_file_path)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    means, stds = get_all_subjects_means_stds(X_train)
    X_train = apply_standardization(X_train, means, stds)
    X_val = apply_standardization(X_val, means, stds)

    models = logreg_one_vs_rest(X_train, y_train)

    preds = predict_ovr(models, X_val)

    acc = accuracy(y_val, preds)
    print("Accuracy:", acc)

    # print(accuracy_score(y_val, preds))  # sklearn usage (remind to uncomment the proper import)


if __name__ == "__main__":
    if len(sys.argv) != 2 and sys.argv[1] != "dataset_train.csv".strip():
        print("Usage: ./logreg_train.py dataset_train.csv")
        exit(1)
    main(sys.argv[1])