from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# === STEP 6: EVALUATION ===
# Function for evaluating the model - MAE, MSE, R²
# Evaluation of all three models
# Comparison in the form of a table


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print(f"MAE: {mae:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"R²: {r2:.4f}")
    return {"Model": name, "MAE": mae, "MSE": mse, "R2": r2}

def evaluate_all_models(models, X_test, y_test):
    results = []
    results.append(evaluate_model("Linear Regression", models["Linear Regression"], X_test, y_test))
    results.append(evaluate_model("Decision Tree", models["Decision Tree"], X_test, y_test))
    results.append(evaluate_model("Random Forest", models["Random Forest"], X_test, y_test))

    results_df = pd.DataFrame(results)
    print("\nSrovnání všech modelů:")
    print(results_df.sort_values(by="R2", ascending=False))
    return results_df
