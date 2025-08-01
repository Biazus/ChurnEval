import numpy as np

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

def build_data_vectors(filename, remove_header=True, remove_first_feature=True):
    """
    Loads data from a CSV file and optionally removes the header row and the first feature column.
    Args:
        filename (str): Path to the CSV file.
        remove_header (bool, optional): If True, removes the first row (header). Defaults to True.
        remove_first_feature (bool, optional): If True, removes the first column (typically an ID). Defaults to True.

    Returns:
        np.ndarray: The processed data array.
    """
    data = np.genfromtxt(filename, delimiter=',', dtype=object)
    if remove_header:
        data = data[1:]
    if remove_first_feature:
        # usually first column is the ID so we can ignore it
        data = np.delete(data, 0, axis=1)  # Remove first column (customer id)
    return data


def normalize_data(data, label_column):
    """
    Normalizes the input data by applying min-max normalization to numerical columns and label encoding to categorical columns.
    Separates the label column and returns the processed feature data and labels.

    Args:
        data (np.ndarray): Input data array.
        label_column (int): Index of the label column.

    Returns:
        tuple: A tuple containing the normalized data array (with the label column removed) and the labels array.
    """
    for k in range(len(data[0])):
        try:
            float(data[0][k])
            apply_min_max_normalization(data, k)
        except ValueError:
            apply_label_encoding(data, k)
    labels = data[:, label_column]  # Labels is a binary array
    data = np.delete(data, label_column, axis=1)  # Remove last column (label)
    return data, labels

def apply_label_encoding(arr, col):
    """
    Encodes categorical columns in a NumPy array using label encoding.

    Args:
        arr (np.ndarray): Input array with categorical data.
        columns (list): List of column indices to encode.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    unique_values = np.unique(arr[:, col])
    encoder = {label: idx for idx, label in enumerate(unique_values)}
    arr[:, col] = [encoder[val] for val in arr[:, col]]


def safe_convert(value):
    """Converts the input value to a float, returning np.nan if conversion fails."""
    try:
        return float(value)
    except ValueError:
        return np.nan


def apply_min_max_normalization(arr, col):
    """
    Applies min-max normalization to specified columns of a NumPy array.

    Args:
        arr (np.ndarray): Input array with numerical or convertible data.
        columns (list): List of column indices to normalize.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    vectorized_conversion = np.vectorize(safe_convert)(arr[:, col])
    converted_array = np.array(vectorized_conversion, dtype=float)
    arr[:, col] = converted_array
    min_val = np.min(arr[:, col].astype(float))
    max_val = np.nanmax(arr[:, col].astype(float))
    arr[:, col] = (arr[:, col] - min_val) / (max_val - min_val)


def split_dataset(
        dataset: list,
        labels: list,
        training: float = 0.7,
        validation: float = 0.15,
        test: float = 0.15
):
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
    if sum([training, validation, test]) != 1:
        raise ValueError("Sum of subsets should be 1")

    idx_val = int(training * len(dataset))
    idx_test = int((training + validation) * len(dataset))
    train = dataset[:idx_val]
    val = dataset[idx_val:idx_test]
    test = dataset[idx_test:]

    return {
        "x_train": train.reshape((train.shape[0], train.shape[1], 1)),
        "y_train": labels[:idx_val],
        "x_val": val.reshape((val.shape[0], val.shape[1], 1)),
        "y_val": labels[idx_val:idx_test],
        "x_test": test.reshape((test.shape[0], test.shape[1], 1)),
        "y_test": labels[idx_test:]
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
    # model.add(MaxPooling1D(pool_size=2))
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

def train_model(m, vectors):
    """
    Trains the given model using provided training and validation data.

    Args:
        m: A compiled Keras model to be trained.
        vectors (dict): Dictionary containing 'x_train', 'y_train', 'x_val', and 'y_val' arrays.

    Returns:
        History object containing training metrics.
    """
    h = m.fit(
        vectors["x_train"], vectors["y_train"],
        batch_size=16,
        epochs=10,
        validation_data=(vectors["x_val"], vectors["y_val"])
    )
    return h


def evaluate_model(m, vectors):
    """
    Evaluates the given model on test data and prints the loss and accuracy.

    Args:
        m: Trained model to be evaluated.
        vectors (dict): Dictionary containing 'x_test' and 'y_test' data.
    """
    loss, accuracy = m.evaluate(vectors["x_test"], vectors["y_test"], verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)


if __name__ == "__main__":
    label_column = 19
    data = build_data_vectors(filename="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data, labels = normalize_data(data, label_column)

    num_classes = len(set(labels))
    num_features = data.shape[1]

    v = split_dataset(data, labels)
    model = build_model(num_features)
    history = train_model(model, v)

    print(history.history)

    evaluate_model(model, v)
