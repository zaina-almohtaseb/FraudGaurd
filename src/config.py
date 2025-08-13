import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DB_PATH = os.path.join(BASE_DIR, "data", "fraud.db")
CSV_DATA_PATH = os.path.join(BASE_DIR, "data", "fraud - fraud.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "gb_model.pkl")

# Model hyperparameters
GB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.2,
    'max_depth': 5,
    'subsample': 0.8,
    'max_features': 'sqrt',
    'random_state': 42
}

# Train-test split settings
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Retraining threshold
RETRAIN_THRESHOLD = 1000
