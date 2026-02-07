"""Module for loading and preparing data for taxi fare prediction."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid CSV.
    
    Examples
    --------
    >>> data = load_data("data/uber.csv")
    >>> print(data.shape)
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV file: {file_path}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {e}") from e


def split_features_target(
    data: pd.DataFrame,
    target_column: str = "fare_amount",
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple:
    """
    Split data into features, target and train/test sets.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with features and target.
    target_column : str, default="fare_amount"
        Name of the target column.
    test_size : float, default=0.2
        Proportion of test data (0.0 to 1.0).
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    
    Raises
    ------
    ValueError
        If target_column is not in data columns or data is empty.
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = split_features_target(data)
    >>> print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    """
    # Validation checks
    if data.empty:
        raise ValueError("Data is empty")
    
    if target_column not in data.columns:
        available_cols = list(data.columns)
        raise ValueError(
            f"Target column '{target_column}' not found in data. "
            f"Available columns: {available_cols}"
        )
    
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Split features and target
    features = data.drop(columns=[target_column])
    target = data[target_column]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    
    # Logging information
    print(f"Data split completed:")
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features: {features.shape[1]} columns")
    print(f"  Target column: '{target_column}'")
    
    return X_train, X_test, y_train, y_test


def validate_data(data: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate data for required columns and basic integrity.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate.
    required_columns : list, optional
        List of columns that must be present in the data.
    
    Returns
    -------
    bool
        True if data is valid.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    if data.empty:
        raise ValueError("Data is empty")
    
    if data.isnull().all().any():
        raise ValueError("Some columns contain only null values")
    
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    
    # Check for duplicate columns
    if len(data.columns) != len(set(data.columns)):
        duplicates = [col for col in data.columns if list(data.columns).count(col) > 1]
        raise ValueError(f"Duplicate column names found: {duplicates}")
    
    return True


def get_data_info(data: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    
    Returns
    -------
    dict
        Dictionary with data information.
    """
    info = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "missing_values": int(data.isnull().sum().sum()),
        "missing_per_column": data.isnull().sum().to_dict(),
        "numeric_columns": list(data.select_dtypes(include=['number']).columns),
        "categorical_columns": list(data.select_dtypes(include=['object']).columns),
        "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    return info


# Alias for backward compatibility
split_data = split_features_target

