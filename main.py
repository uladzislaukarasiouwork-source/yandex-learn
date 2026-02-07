#!/usr/bin/env python3
"""
Main script for taxi fare prediction model training.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.data import load_data, split_features_target
from src.features import add_time_features
from src.model import TaxiFareModel


# Constants
DATA_PATH = Path("data/uber.csv")
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def train_model(
    data_path: Path,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple:
    """
    Train taxi fare prediction model.
    
    Parameters
    ----------
    data_path : Path
        Path to the data file.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (model, score) - trained model and its R² score.
    
    Raises
    ------
    FileNotFoundError
        If data file doesn't exist.
    ValueError
        If data is invalid or model training fails.
    """
    logger = logging.getLogger(__name__)
    
    # Check if data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Starting model training with data from {data_path}")
    
    try:
        # Load data
        logger.info("Loading data...")
        raw_data = load_data(str(data_path))
        
        # Process features
        logger.info("Processing features...")
        processed_data = add_time_features(raw_data)
        
        # Split data
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = split_features_target(
            data=processed_data,
            target_column="fare_amount",
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        
        # Train model
        logger.info("Training model...")
        model = TaxiFareModel()
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        score = model.model.score(X_test, y_test)
        
        logger.info(f"Model trained successfully. R² score: {score:.4f}")
        
        return model, score
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def save_results(model, score: float, output_dir: Path = Path("results")):
    """
    Save model training results.
    
    Parameters
    ----------
    model : TaxiFareModel
        Trained model.
    score : float
        Model R² score.
    output_dir : Path, default=Path("results")
        Directory to save results.
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to file
    results_file = output_dir / "training_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Model Training Results\n")
        f.write(f"=" * 30 + "\n")
        f.write(f"R² Score: {score:.4f}\n")
        f.write(f"Model Type: {type(model.model).__name__}\n")
        f.write(f"Features used: {model.model.n_features_in_ if hasattr(model.model, 'n_features_in_') else 'Unknown'}\n")
    
    logger.info(f"Results saved to {results_file}")


def main():
    """Main execution function."""
    logger = setup_logging()
    
    try:
        # Train model
        model, score = train_model(
            data_path=DATA_PATH,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
        )
        
        # Save results
        save_results(model, score)
        
        # Final message
        logger.info("=" * 50)
        logger.info(f"Training completed successfully!")
        logger.info(f"Final R² score: {score:.4f}")
        logger.info("=" * 50)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Data error: {e}")
        logger.info("Please make sure the data file exists at 'data/uber.csv'")
        return 1
        
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())