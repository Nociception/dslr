import numpy as np
from typing import List
import json

from minibatch_picker import minibatch_picker
from OnlineStandardizer import OnlineStandardizer
from logistic_functions import logistic_function_vector
from logreg_train import compute_loss


class LogisticRegressionOVRMinibatch:
    def __init__(self, classes: List[str], nb_features: int, alpha: float = 0.01):
        self.classes = classes
        self.alpha = alpha
        self.weights = np.zeros((len(classes), nb_features))
        self.bias = np.zeros(len(classes))

    def gradients_batch_update(self, X: np.ndarray, y: np.ndarray):
        for idx, class_name in enumerate(self.classes):
            y_binary = (y == class_name).astype(np.float64)
            scores = X @ self.weights[idx] + self.bias[idx]
            predictions = logistic_function_vector(scores)
            error = predictions - y_binary

            grad_weights = X.T @ error / len(X)
            self.weights[idx] -= self.alpha * grad_weights

            grad_intercept = np.mean(error)
            self.bias[idx] -= self.alpha * grad_intercept

            print(
                f"Class '{class_name}': Loss = {compute_loss(y_binary, predictions):.4f}",
                end=" | ",
            )
        print()
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.weights.T + self.bias
        best_indices = np.argmax(scores, axis=1)
        return np.array([self.classes[i] for i in best_indices])


def train_big_data_logreg(
    file_path: str,
    features_kept: List[str],
    target: str,
    classes: List[str],
    batch_size: int = 128,
    epochs: int = 3,
    alpha: float = 0.01,
):
    nb_features = len(features_kept)

    standardizer = OnlineStandardizer(nb_features)
    for X_batch, _ in minibatch_picker(
        file_path,
        features_kept,
        target,
        batch_size,
        mode="train",
        shuffle_within_batch=False,
    ):
        standardizer.batch_update(X_batch)
    standardizer.finalize()

    model = LogisticRegressionOVRMinibatch(classes, nb_features, alpha)
    for epoch in range(epochs):
        for X_batch, y_batch in minibatch_picker(
            file_path,
            features_kept,
            target,
            batch_size,
            mode="train",
            shuffle_within_batch=True,
        ):
            X_batch = standardizer.standardize(X_batch)
            model.gradients_batch_update(X_batch, y_batch)

        print(f"Epoch {epoch+1}/{epochs}")

    return model, standardizer


def save_model(
    filepath: str,
    features: List[str],
    means: np.ndarray,
    stds: np.ndarray,
    weights_dict: dict,
    intercepts_dict: dict,
) -> None:
    model_dict = {
        "metadata": {"model": "logistic_regression_ovr", "nb_features": len(features)},
        "features": features,
        "standardization": {"means": means.tolist(), "stds": stds.tolist()},
        "classes": {},
    }

    for class_name in weights_dict:
        model_dict["classes"][class_name] = {
            "weights": weights_dict[class_name].tolist(),
            "intercept": float(intercepts_dict[class_name]),
        }

    with open(filepath, "w") as f:
        json.dump(model_dict, f, indent=4)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    features_kept = [
        "Astronomy",
        "Herbology",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
        "Potions",
        "Charms",
        "Flying",
    ]
    model, standardizer = train_big_data_logreg(
        file_path="datasets/dataset_train.csv",
        features_kept=features_kept,
        target="Hogwarts House",
        classes=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
        batch_size=32,
        epochs=3,
        alpha=0.01,
    )

    save_model(
        filepath="model.json",
        features=features_kept,
        means=standardizer.mean,
        stds=standardizer.std,
        weights_dict={
            class_name: model.weights[idx]
            for idx, class_name in enumerate(model.classes)
        },
        intercepts_dict={
            class_name: model.bias[idx] for idx, class_name in enumerate(model.classes)
        },
    )
