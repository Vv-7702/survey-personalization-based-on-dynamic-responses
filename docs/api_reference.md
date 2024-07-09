# API Reference

## Functions

### train_and_evaluate_model
Trains and evaluates a given model using the provided training and test data.

**Parameters:**
- `model`: The machine learning model to train and evaluate.
- `X_train`: Training feature data.
- `y_train`: Training target data.
- `X_test`: Test feature data.
- `y_test`: Test target data.

### grid_search
Performs grid search hyperparameter tuning on a given model.

**Parameters:**
- `model`: The machine learning model to tune.
- `param_grid`: The hyperparameter grid to search over.
- `X_train`: Training feature data.
- `y_train`: Training target data.

**Returns:**
- The best estimator from the grid search.

### personalize_survey
Personalizes survey responses using a trained model and label encoders.

**Parameters:**
- `responses`: A dictionary of survey responses.
- `label_encoders`: A dictionary of label encoders used for preprocessing.
- `model`: The trained model to use for prediction.

**Returns:**
- The predicted next question for the survey.
