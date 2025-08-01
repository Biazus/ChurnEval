import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def apply_label_encoding(arr, columns):
    """
    Encodes categorical columns in a NumPy array using label encoding.

    Args:
        arr (np.ndarray): Input array with categorical data.
        columns (list): List of column indices to encode.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    for i in columns:
        unique_values = np.unique(arr[:, i])
        encoder = {label: idx for idx, label in enumerate(unique_values)}
        arr[:, i] = [encoder[val] for val in arr[:, i]]

def safe_convert(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

def apply_min_max_normalization(arr, columns):
    """
    Applies min-max normalization to specified columns of a NumPy array.

    Args:
        arr (np.ndarray): Input array with numerical or convertible data.
        columns (list): List of column indices to normalize.

    Returns:
        None: The input array is modified in place with normalized values in the specified columns.
    """
    for i in columns:
        vectorized_conversion = np.vectorize(safe_convert)(arr[:, i])
        converted_array = np.array(vectorized_conversion, dtype=float)
        arr[:, i] = converted_array
        min_val = np.min(arr[:, i].astype(float))
        max_val = np.nanmax(arr[:, i].astype(float))
        arr[:, i] = (arr[:, i] - min_val) / (max_val - min_val)

def split_dataset(dataset: list,
            labels: list,
            num_classes: int,
            training: float = 0.7,
            validation: float = 0.15,
            test: float = 0.15):
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

def build_model(dataset: list, num_classes: int,):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(19, 1)))
    # model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def compile_model(m):
    m.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return m

def train_model(m, vectors):
    h = m.fit(
        vectors["x_train"], vectors["y_train"],
        batch_size=16,
        epochs=10,
        validation_data=(vectors["x_val"], vectors["y_val"])
    )
    return h

def evaluate_model(m, vectors):
    loss, accuracy = m.evaluate(vectors["x_test"], vectors["y_test"], verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

if __name__ == "__main__":
    data = np.genfromtxt("WA_Fn-UseC_-Telco-Customer-Churn.csv", delimiter=',', dtype=object)

    # Normalizing Data
    data = data[1:]  # Remove header
    data = np.delete(data, 0, axis=1)  # Remove first column (customer id)
    apply_label_encoding(data, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19])
    apply_min_max_normalization(data, [4, 17, 18])
    labels = data[:, 19]  # Labels is a 0/1 array
    data = np.delete(data, 19, axis=1)  # Remove last column (label)

    v = split_dataset(data, labels, 2)


    model = build_model(v["x_train"], 2)
    model = compile_model(model)
    history = train_model(model, v)
    print(history.history)

    evaluate_model(model, v)