"""Taxi fare prediction model."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TaxiFareModel(BaseEstimator, RegressorMixin):
    """
    Taxi fare prediction model using Gradient Boosting.
    
    This model wraps sklearn's GradientBoostingRegressor with additional
    functionality for evaluation, saving/loading, and hyperparameter tuning.
    
    Attributes
    ----------
    model : GradientBoostingRegressor
        The underlying sklearn model.
    is_fitted : bool
        Whether the model has been fitted.
    feature_names_ : list or None
        Names of features used during training.
    
    Examples
    --------
    >>> model = TaxiFareModel()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> score = model.score(X_test, y_test)
    """
    
    # Default hyperparameters for Gradient Boosting
    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
    }
    
    # Parameter grid for hyperparameter tuning
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
    }
    
    def __init__(
        self,
        use_pipeline: bool = False,
        use_scaling: bool = False,
        **kwargs,
    ):
        """
        Initialize the taxi fare prediction model.
        
        Parameters
        ----------
        use_pipeline : bool, default=False
            Whether to use a sklearn Pipeline with preprocessing.
        use_scaling : bool, default=False
            Whether to include StandardScaler in the pipeline.
            Only used if use_pipeline=True.
        **kwargs
            Additional parameters for GradientBoostingRegressor.
            See sklearn documentation for available parameters.
        
        Examples
        --------
        >>> # Basic model
        >>> model = TaxiFareModel()
        >>> 
        >>> # Model with custom parameters
        >>> model = TaxiFareModel(n_estimators=200, max_depth=5)
        >>> 
        >>> # Model with preprocessing pipeline
        >>> model = TaxiFareModel(use_pipeline=True, use_scaling=True)
        """
        # Merge default parameters with user-provided ones
        params = self.DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        # Initialize model or pipeline
        if use_pipeline:
            steps = []
            if use_scaling:
                steps.append(('scaler', StandardScaler()))
            
            steps.append(('gb', GradientBoostingRegressor(**params)))
            self.model = Pipeline(steps)
        else:
            self.model = GradientBoostingRegressor(**params)
        
        self.use_pipeline = use_pipeline
        self.is_fitted = False
        self.feature_names_ = None
        self.training_shape_ = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs,
    ) -> 'TaxiFareModel':
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_kwargs
            Additional parameters to pass to the underlying fit method.
        
        Returns
        -------
        self
            Fitted model.
        
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        # Validate input
        self._validate_input(X, y)
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        elif isinstance(X, np.ndarray):
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Store training shape
        self.training_shape_ = (X.shape[0], X.shape[1])
        
        # Fit the model
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        
        print(f"Model fitted successfully.")
        print(f"  Training samples: {self.training_shape_[0]}")
        print(f"  Features: {self.training_shape_[1]}")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict taxi fares for given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        
        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If input data has wrong number of features.
        """
        self._check_is_fitted()
        self._validate_predict_input(X)
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True target values.
        metrics : list, optional
            List of metrics to compute. Default: ['mse', 'mae', 'r2']
            Available: 'mse', 'rmse', 'mae', 'r2'
        
        Returns
        -------
        dict
            Dictionary with metric names and values.
        """
        self._check_is_fitted()
        
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']
        
        predictions = self.predict(X)
        results = {}
        
        for metric in metrics:
            if metric == 'mse':
                results['mse'] = mean_squared_error(y, predictions)
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            elif metric == 'mae':
                results['mae'] = mean_absolute_error(y, predictions)
            elif metric == 'r2':
                results['r2'] = r2_score(y, predictions)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = 'r2',
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, default='r2'
            Scoring metric.
        n_jobs : int, default=-1
            Number of jobs to run in parallel.
        
        Returns
        -------
        dict
            Cross-validation results.
        """
        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'cv': cv,
            'scoring': scoring,
        }
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> 'TaxiFareModel':
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        param_grid : dict, optional
            Parameter grid for tuning. Uses default if None.
        cv : int, default=3
            Number of cross-validation folds.
        scoring : str, default='r2'
            Scoring metric.
        n_jobs : int, default=-1
            Number of jobs to run in parallel.
        verbose : int, default=0
            Verbosity level.
        
        Returns
        -------
        self
            Model with optimized hyperparameters.
        """
        if param_grid is None:
            param_grid = self.PARAM_GRID
        
        # Create base estimator without pipeline for tuning
        if self.use_pipeline:
            gb_params = {f'gb__{k}': v for k, v in param_grid.items()}
            grid_search = GridSearchCV(
                self.model,
                gb_params,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        
        # Perform grid search
        print(f"Starting hyperparameter tuning with {cv}-fold CV...")
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")
        
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        
        Raises
        ------
        NotFittedError
            If model is not fitted.
        AttributeError
            If model doesn't support feature importance.
        """
        self._check_is_fitted()
        
        # Get the underlying model if using pipeline
        if self.use_pipeline:
            model = self.model.named_steps['gb']
        else:
            model = self.model
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if self.feature_names_ is not None:
                df = pd.DataFrame({
                    'feature': self.feature_names_,
                    'importance': importance,
                })
            else:
                df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importance))],
                    'importance': importance,
                })
            
            return df.sort_values('importance', ascending=False)
        else:
            raise AttributeError("Model does not support feature importance")
    
    def _validate_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> None:
        """Validate input data for fitting."""
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        
        if len(y) == 0:
            raise ValueError("y cannot be empty")
    
    def _validate_predict_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Validate input data for prediction."""
        expected_features = self.training_shape_[1]
        
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"got {X.shape[1]}"
            )
    
    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise NotFittedError(
                "This TaxiFareModel instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        
        if self.is_fitted:
            return f"{class_name}({fitted_status}, n_features={self.training_shape_[1]})"
        else:
            return f"{class_name}({fitted_status})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


# For backward compatibility
def create_default_model() -> TaxiFareModel:
    """
    Create a default taxi fare model.
    
    Returns
    -------
    TaxiFareModel
        Default model instance.
    """
    return TaxiFareModel()


if __name__ == "__main__":
    """Test the model class."""
    
    print("Testing TaxiFareModel class...")
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randn(20)
    
    # Test basic functionality
    print("\n1. Testing basic model...")
    model = TaxiFareModel(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"  Predictions shape: {predictions.shape}")
    
    # Test evaluation
    print("\n2. Testing evaluation...")
    metrics = model.evaluate(X_test, y_test)
    print(f"  Metrics: {metrics}")
    
    # Test with pipeline
    print("\n3. Testing model with pipeline...")
    model_pipe = TaxiFareModel(use_pipeline=True, use_scaling=True)
    model_pipe.fit(X_train, y_train)
    print(f"  Pipeline model fitted: {model_pipe.is_fitted}")
    
    # Test feature importance
    print("\n4. Testing feature importance...")
    try:
        importance = model.get_feature_importance()
        print(f"  Feature importance computed. Top features:")
        print(importance.head())
    except AttributeError:
        print("  Feature importance not available for this configuration")
    
    print("\nAll tests completed!")