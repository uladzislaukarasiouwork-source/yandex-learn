"""Module for feature engineering and preprocessing."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class FeatureProcessor:
    """Processor for taxi fare prediction features."""
    
    # Columns to always remove
    COLUMNS_TO_DROP = ['key', 'Unnamed: 0']
    
    # Date/time columns to process
    DATETIME_COLUMNS = ['pickup_datetime', 'dropoff_datetime']
    
    # Columns to extract from datetime
    DATETIME_FEATURES = ['hour', 'day_of_week', 'month', 'day', 'year']
    
    def __init__(
        self,
        drop_columns: Optional[List[str]] = None,
        handle_missing: str = 'drop',  # 'drop', 'fill', or 'error'
        fill_strategy: str = 'median',  # 'mean', 'median', or 'constant'
        fill_value: Optional[float] = None,
    ):
        """
        Initialize feature processor.
        
        Parameters
        ----------
        drop_columns : list, optional
            Additional columns to drop.
        handle_missing : str, default='drop'
            Strategy for handling missing values.
            Options: 'drop', 'fill', 'error'
        fill_strategy : str, default='median'
            Strategy for filling missing values.
            Options: 'mean', 'median', 'constant'
        fill_value : float, optional
            Value to use when fill_strategy='constant'.
        """
        self.drop_columns = drop_columns or []
        self.handle_missing = handle_missing
        self.fill_strategy = fill_strategy
        self.fill_value = fill_value
        
        self._validate_parameters()
        self._fitted = False
        self._fill_values = {}
        self._processed_columns = []
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        valid_handling = ['drop', 'fill', 'error']
        if self.handle_missing not in valid_handling:
            raise ValueError(
                f"handle_missing must be one of {valid_handling}, "
                f"got {self.handle_missing}"
            )
        
        valid_fill = ['mean', 'median', 'constant']
        if self.fill_strategy not in valid_fill:
            raise ValueError(
                f"fill_strategy must be one of {valid_fill}, "
                f"got {self.fill_strategy}"
            )
        
        if self.fill_strategy == 'constant' and self.fill_value is None:
            raise ValueError(
                "fill_value must be provided when fill_strategy='constant'"
            )
    
    def fit(self, data: pd.DataFrame) -> 'FeatureProcessor':
        """
        Fit processor on training data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        
        Returns
        -------
        self
            Fitted processor.
        """
        # Store column information
        self._original_columns = data.columns.tolist()
        self._original_shape = data.shape
        
        # Calculate fill values if needed
        if self.handle_missing == 'fill':
            self._calculate_fill_values(data)
        
        self._fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted processor.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        if not self._fitted:
            raise RuntimeError("Processor must be fitted before transform")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Step 1: Remove unwanted columns
        df = self._drop_columns(df)
        
        # Step 2: Process datetime columns
        df = self._process_datetime_columns(df)
        
        # Step 3: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 4: Store processed columns
        self._processed_columns = df.columns.tolist()
        
        return df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit processor and transform data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to fit and transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(data).transform(data)
    
    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove specified columns."""
        columns_to_drop = self.COLUMNS_TO_DROP + self.drop_columns
        columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
        
        return data
    
    def _process_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        for col in self.DATETIME_COLUMNS:
            if col in data.columns:
                data = self._extract_datetime_features(data, col)
        
        return data
    
    def _extract_datetime_features(
        self, 
        data: pd.DataFrame, 
        column: str
    ) -> pd.DataFrame:
        """Extract features from a datetime column."""
        try:
            # Convert to datetime
            data[column] = pd.to_datetime(data[column])
            
            # Extract features
            if 'hour' in self.DATETIME_FEATURES:
                data[f'{column}_hour'] = data[column].dt.hour
            
            if 'day_of_week' in self.DATETIME_FEATURES:
                data[f'{column}_dayofweek'] = data[column].dt.dayofweek
            
            if 'month' in self.DATETIME_FEATURES:
                data[f'{column}_month'] = data[column].dt.month
            
            if 'day' in self.DATETIME_FEATURES:
                data[f'{column}_day'] = data[column].dt.day
            
            if 'year' in self.DATETIME_FEATURES:
                data[f'{column}_year'] = data[column].dt.year
            
            # Drop original datetime column
            data = data.drop(columns=[column])
            
        except Exception as e:
            raise ValueError(f"Failed to process datetime column '{column}': {e}")
        
        return data
    
    def _calculate_fill_values(self, data: pd.DataFrame) -> None:
        """Calculate values to fill missing data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.fill_strategy == 'mean':
                self._fill_values[col] = data[col].mean()
            elif self.fill_strategy == 'median':
                self._fill_values[col] = data[col].median()
            elif self.fill_strategy == 'constant':
                self._fill_values[col] = self.fill_value
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategy."""
        missing_count = data.isnull().sum().sum()
        
        if missing_count == 0:
            return data
        
        if self.handle_missing == 'error':
            raise ValueError(f"Data contains {missing_count} missing values")
        
        elif self.handle_missing == 'drop':
            initial_rows = len(data)
            data = data.dropna()
            removed_rows = initial_rows - len(data)
            print(f"Dropped {removed_rows} rows with missing values")
        
        elif self.handle_missing == 'fill':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in self._fill_values:
                    fill_val = self._fill_values[col]
                    nan_count = data[col].isnull().sum()
                    if nan_count > 0:
                        data[col] = data[col].fillna(fill_val)
                        print(f"Filled {nan_count} missing values in '{col}' "
                              f"with {fill_val:.2f}")
        
        return data
    
    def get_processing_info(self) -> dict:
        """Get information about the processing."""
        return {
            'original_columns': self._original_columns,
            'original_shape': self._original_shape,
            'processed_columns': self._processed_columns,
            'handle_missing': self.handle_missing,
            'fill_strategy': self.fill_strategy,
        }


# Legacy function for backward compatibility
def add_time_features(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Legacy function for adding time features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    drop_columns : list, optional
        Additional columns to drop.
    
    Returns
    -------
    pd.DataFrame
        Processed data with time features.
    """
    processor = FeatureProcessor(
        drop_columns=drop_columns,
        handle_missing='drop',
        fill_strategy='median',
    )
    
    return processor.fit_transform(df)


if __name__ == "__main__":
    """Test the feature processor."""
    
    # Create test data
    test_data = pd.DataFrame({
        'key': ['test1', 'test2', 'test3'],
        'pickup_datetime': [
            '2023-01-01 08:30:00',
            '2023-01-02 14:45:00',
            '2023-01-03 20:15:00'
        ],
        'fare_amount': [10.5, 15.3, 12.7],
        'distance': [2.5, 3.8, 2.9],
        'passenger_count': [1, 2, 1]
    })
    
    print("Testing FeatureProcessor...")
    
    # Test the processor
    processor = FeatureProcessor(handle_missing='drop')
    processed = processor.fit_transform(test_data)
    
    print(f"\nOriginal data shape: {test_data.shape}")
    print(f"Processed data shape: {processed.shape}")
    print(f"\nProcessed columns: {list(processed.columns)}")
    print(f"\nProcessed data head:\n{processed.head()}")
    
    # Test legacy function
    print("\n" + "="*50)
    print("Testing legacy add_time_features...")
    
    legacy_processed = add_time_features(test_data)
    print(f"Legacy processed shape: {legacy_processed.shape}")
