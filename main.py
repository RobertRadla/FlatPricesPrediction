from preprocessing import prepare_data
from training import train_and_save_models
from evaluation import evaluate_all_models
from tuning import run_grid_search

# === PREPROCESSING ===
df, X_train, X_test, y_train, y_test = prepare_data()

# === TRAINING ===
models = train_and_save_models(X_train, y_train)

# === EVALUATION ===
evaluate_all_models(models, X_test, y_test)

# === TUNING ===
run_grid_search(X_train, y_train, X_test, y_test)
