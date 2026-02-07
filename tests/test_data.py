"""Test data loading functionality."""

import pandas as pd
from src.data import load_data


def test_load_data():
    """
    Test that load_data function loads the dataset correctly.
    
    Verifies that:
    1. Data is loaded as a pandas DataFrame
    2. The DataFrame is not empty
    3. Contains expected columns
    """
    # Load the data
    df = load_data("data/uber.csv")
    
    # Verify it's a DataFrame
    assert isinstance(df, pd.DataFrame), \
        "load_data should return a pandas DataFrame"
    
    # Verify it's not empty
    assert not df.empty, \
        "Loaded DataFrame should not be empty"
    
    # Verify it has expected structure
    assert len(df.shape) == 2, \
        "DataFrame should have 2 dimensions (rows and columns)"
    
    # Verify it has data
    assert df.shape[0] > 0, \
        "DataFrame should have at least one row"
    
    assert df.shape[1] > 0, \
        "DataFrame should have at least one column"
    
    # Verify column names are strings
    for column in df.columns:
        assert isinstance(column, str), \
            f"Column name '{column}' should be a string"
    
    # Print some info for debugging
    print(f"Data loaded successfully:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First few rows:")
    print(df.head(2))