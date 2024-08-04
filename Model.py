# Imported necessary libraries
import optuna  # Library for hyperparameter optimization
from sklearn.datasets import load_iris  # Function to load the Iris dataset
from sklearn.model_selection import train_test_split  # Function to split data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier model
from sklearn.metrics import accuracy_score  # Function to calculate accuracy
import joblib # for serilalizing the model

# Loaded the Iris dataset
# X contained the features (sepal length, sepal width, petal length, petal width)
# y contained the target labels (species of iris flowers)
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
# 80% of the data was used for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defined the objective function for Optuna
# This function was called multiple times with different hyperparameters
def objective(trial):
    # Defined the hyperparameter search space
    # trial.suggest_int defined integer parameters (range from 50 to 200 for n_estimators)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 10, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    # trial.suggest_categorical defined categorical parameters
    max_features = trial.suggest_categorical('max_features', [ 'sqrt', 'log2'])

    # Created and trained the Random Forest model with the suggested hyperparameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42  # Ensured reproducibility
    )
    clf.fit(X_train, y_train)  # Fitted the model to the training data
    
    # Predicted the labels for the test set
    y_pred = clf.predict(X_test)
    # Calculated the accuracy of the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    
    # Returned the accuracy as the objective value to be maximized
    return accuracy

# Created a study object to manage the optimization process
# direction='maximize' indicated that we wanted to maximize the accuracy
study = optuna.create_study(direction='maximize')
# Ran the optimization process for a specified number of trials
study.optimize(objective, n_trials=100)  # Number of trials could be adjusted based on computational resources

# Extracted the best hyperparameters and the corresponding accuracy from the study
best_params = study.best_params
best_accuracy = study.best_value


print("\n\n---------------------Best identified parameters from Optuna are------------\n\n")
# Printed the best parameters found and the best accuracy achieved



print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_accuracy}")



# building the model using above hyperparameters

# Create the Random forest classifier using the identified hyper params
rfc = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                             max_depth=best_params['max_depth'],
                             min_samples_split=best_params['min_samples_split'],
                             min_samples_leaf=best_params['min_samples_leaf'],
                             max_features=best_params['min_samples_leaf']
)

# Train the classifier
rfc.fit(X_train, y_train)
print('Training completed')
print("Save model")

#serialize the model (store it in a binary format). Since its KNN, data will also be part of the model
joblib.dump(rfc,'group63_mlops_M3_optuna_model.joblib')