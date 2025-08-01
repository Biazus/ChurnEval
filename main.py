from typing import Any, Dict, List
import numpy as np
import logging
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

HYPERPARAMS = {
    "kernel_size": 2,
    "epochs": 10,
    "activation": "relu",
    "optimizer": 'adam',
    "loss": 'binary_crossentropy',
    "metrics": ['accuracy']
}

def build_data_vectors(
    filename: str,
    remove_header: bool = True,
    remove_first_feature: bool = True
) -> 'np.ndarray':
    """
    Loads data from a CSV file and optionally removes the header row and the first feature column.
    """
    try:
        data = np.genfromtxt(filename, delimiter=',', dtype=object)
    except OSError as e:
        logging.error(f"File not found or inaccessible: {filename}")
        raise
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise
    if data is None or data.size == 0:
        raise ValueError(f"No data loaded from file: {filename}")
    if remove_header:
        data = data[1:]
    if remove_first_feature:
        data = data[:, 1:]
    return data


def normalize_data(data, label_column):
    """
    Normalizes the input data by applying min-max normalization to numerical columns and label encoding to categorical columns.
    Separates the label column and returns the processed feature data and labels.
    """
    data_copy = data.copy()
    for k in range(len(data_copy[0])):
        try:
            float(data_copy[0][k])
            apply_min_max_normalization(data_copy, k)
        except ValueError:
            apply_label_encoding(data_copy, k)
    labels = data_copy[:, label_column]
    data_copy = np.delete(data_copy, label_column, axis=1)
    return data_copy, labels

def apply_label_encoding(arr: 'np.ndarray', col: int) -> None:
    """
    Encodes categorical columns in a NumPy array using label encoding.

    Args:
        arr (np.ndarray): Input array with categorical data.
        columns (list): List of column indices to encode.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    # Find unique values and create a mapping to integer codes
    unique_values, encoded = np.unique(arr[:, col], return_inverse=True)
    arr[:, col] = encoded.astype(str)


def safe_convert(value):
    """Converts the input value to a float, returning np.nan if conversion fails."""
    try:
        return float(value)
    except ValueError:
        return np.nan


def apply_min_max_normalization(arr: np.ndarray, col: int) -> None:
    """
    Applies min-max normalization to specified columns of a NumPy array.

    Args:
        arr (np.ndarray): Input array with numerical or convertible data.
        columns (list): List of column indices to normalize.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    # Convert column to float using safe_convert
    arr[:, col] = np.vectorize(safe_convert)(arr[:, col]).astype(float)
    col_vals = arr[:, col].astype(float)
    min_val = np.nanmin(col_vals)
    max_val = np.nanmax(col_vals)
    # Avoid division by zero
    if max_val != min_val:
        arr[:, col] = (col_vals - min_val) / (max_val - min_val)
    else:
        arr[:, col] = 0.0



def split_dataset(
    dataset: List[Any],
    labels: List[Any],
    training: float = 0.7,
    validation: float = 0.15,
    test: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    Splits a dataset and corresponding labels into training, validation, and test subsets.

    Args:
        dataset (list): The input data to be split.
        labels (list): The labels corresponding to the dataset.
        training (float, optional): Proportion of data for training. Defaults to 0.7.
        validation (float, optional): Proportion of data for validation. Defaults to 0.15.
        test (float, optional): Proportion of data for testing. Defaults to 0.15.

    Returns:
        dict: Dictionary containing reshaped train, validation, and test sets with their labels.

    Raises:
        ValueError: If the sum of training, validation, and test proportions is not 1.
    """
    total = training + validation + test
    if not np.isclose(total, 1.0):
        raise ValueError("Sum of subsets should be 1")

    dataset = np.asarray(dataset)
    labels = np.asarray(labels)

    n_samples = len(dataset)
    idx_train = int(training * n_samples)
    idx_val = int((training + validation) * n_samples)

    x_train = dataset[:idx_train].reshape(-1, dataset.shape[1], 1)
    y_train = labels[:idx_train]
    x_val = dataset[idx_train:idx_val].reshape(-1, dataset.shape[1], 1)
    y_val = labels[idx_train:idx_val]
    x_test = dataset[idx_val:].reshape(-1, dataset.shape[1], 1)
    y_test = labels[idx_val:]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test
    }


def build_model(num_features):
    """
    Builds and compiles a 1D convolutional neural network model for binary classification using the specified number of input features.

    Args:
        num_features (int): Number of features in the input data.
    Returns:
        keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Conv1D(
        filters=32, kernel_size=HYPERPARAMS["kernel_size"], activation=HYPERPARAMS["activation"], input_shape=(num_features, 1))
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=HYPERPARAMS['optimizer'],
        loss=HYPERPARAMS['loss'],
        metrics=HYPERPARAMS['metrics']
    )
    model.summary()
    return model

def train_model(
    m,
    vectors: dict
) -> "keras.callbacks.History":
    """
    Trains the given model using provided training and validation data.

    Args:
        m: A compiled Keras model to be trained.
        vectors (dict): Dictionary containing 'x_train', 'y_train', 'x_val', and 'y_val' arrays.

    Returns:
        History object containing training metrics.
    """
    return m.fit(
        vectors["x_train"], vectors["y_train"],
        batch_size=16,
        epochs=HYPERPARAMS['epochs'],
        validation_data=(vectors["x_val"], vectors["y_val"])
    )

def evaluate_model(m, vectors):
    """
    Evaluates the given model on test data and logs the loss and accuracy.

    Args:
        m: Trained model to be evaluated.
        vectors (dict): Dictionary containing `x_test` and `y_test` data.

    Returns:
        tuple: (`loss`, `accuracy`) of the evaluated model.
    """
    if not isinstance(vectors, dict) or "x_test" not in vectors or "y_test" not in vectors:
        raise ValueError("`vectors` must be a dict containing 'x_test' and 'y_test' keys.")
    if vectors["x_test"] is None or vectors["y_test"] is None:
        raise ValueError("`x_test` and `y_test` in `vectors` must not be None.")
    loss, accuracy = m.evaluate(vectors["x_test"], vectors["y_test"], verbose=0)
    logging.info('Test loss: %s', loss)
    logging.info('Test accuracy: %s', accuracy)
    # TODO Remove prints and use logs only
    print("Test loss: {}".format(loss))
    print("Test accuracy: {}".format(accuracy))
    return loss, accuracy


if __name__ == "__main__":
    label_column = 19  # TODO remove hardcoded val
    data = build_data_vectors(filename="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data, labels = normalize_data(data, label_column)

    num_classes = len(set(labels))
    num_features = data.shape[1]

    v = split_dataset(data, labels)
    model = build_model(num_features)
    history = train_model(model, v)

    print(history.history)

    evaluate_model(model, v)