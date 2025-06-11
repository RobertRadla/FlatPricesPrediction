from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# === STEP 7: HYPERPARAMETER TUNING ===
# The function run_grid_search is used for tuning the hyperparameters of the Random Forest model
# definition of the range of values for tuning
# creation of the GridSearch object
# execution of the tuning process
# display of the best found parameters
# evaluation on the TEST dataset
# saving the best model to a file
# saving the scaler and names of columns after one-hot encoding

def run_grid_search(X_train, y_train, X_test, y_test):
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,                  
        scoring='r2',          
        verbose=2,             
        n_jobs=-1              
    )

    print("------------------- Spouštím Grid Search pro Random Forest -------------------")
    grid_search.fit(X_train, y_train)
    print("Grid Search hotov!")

    print("\nNejlepší kombinace hyperparametrů:")
    print(grid_search.best_params_)
    print("\nNejlepší průměrné R² skóre při validaci:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)

    print("\nVýsledky nejlepšího modelu na TEST datech:")
    print("MAE:", mean_absolute_error(y_test, y_pred_best))
    print("MSE:", mean_squared_error(y_test, y_pred_best))
    print("R² :", r2_score(y_test, y_pred_best))




    joblib.dump(best_model, "models/final_random_forest_model.pkl")

    num_cols = ["Obytná plocha", "Počet místností", "Podlaží"]
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])
    joblib.dump(scaler, "models/scaler.pkl")

    misto_columns = [col for col in X_train.columns if col.startswith("Místo_")]
    lokalita_columns = [col for col in X_train.columns if col.startswith("Lokalita_")]
    joblib.dump(misto_columns, "models/misto_columns.pkl")
    joblib.dump(lokalita_columns, "models/lokalita_columns.pkl")