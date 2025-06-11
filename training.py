from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib



# === STEP 5: MODEL TRAINING ===
# linear regression, decision tree, random forest
# initialization of models
# training the models on training data
# saving the trained models to files

def train_and_save_models(X_train, y_train):

    linear_model = LinearRegression()
    tree_model = DecisionTreeRegressor(random_state=42)
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

    print("-------------------------Trénuji Linear Regression... -------------------------")
    linear_model.fit(X_train, y_train)
    print("Linear Regression hotov.")

    print("-------------------------Trénuji Decision Tree...--------------------------")
    tree_model.fit(X_train, y_train)
    print("Decision Tree hotov.")

    print("--------------------------Trénuji Random Forest...--------------------------")
    forest_model.fit(X_train, y_train)
    print("Random Forest hotov.")

    joblib.dump(linear_model, "models/linear_model.pkl")
    joblib.dump(tree_model, "models/tree_model.pkl")
    joblib.dump(forest_model, "models/forest_model.pkl")

    return {
        "Linear Regression": linear_model,
        "Decision Tree": tree_model,
        "Random Forest": forest_model
    }
