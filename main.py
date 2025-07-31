import numpy as np

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
        return np.nan  # Tratar strings não numéricas

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

if __name__ == "__main__":
    data = np.genfromtxt("WA_Fn-UseC_-Telco-Customer-Churn.csv", delimiter=',', dtype=object)

    # Normalizing Data
    data = data[1:]  # Remove header
    data = np.delete(data, 0, axis=1)  # Remove first column (customer id)
    apply_label_encoding(data, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19])
    apply_min_max_normalization(data, [4, 17, 18])
    labels = data[:, 19]  # Labels is a 0/1 array
    data = np.delete(data, 19, axis=1)  # Remove last column (label)
    print(data)
    print(labels)

    # TODO Splitting data between sets of training / val/ test
    vectors = {
        "x_train": [],
        "x_val": [],
        "x_test": [],
    }

    # TODO keep working

