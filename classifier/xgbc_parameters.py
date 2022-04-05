# Parameters for data reading function when using XGBClassifier
dataset_parameters = {
    'par_count': None,
    'val_percent': 0.1,
    'test_percent': 0.1,
    'leave_out': [],
    'withhold_moves': [],
    'normalize': False,
    'one_hot': False,
    'flatten': True,
    'verbose': True
}

# Parameters for sklearn parameter searches
model_search_parameters = {
    'learning_rate': [0.1],
    # 'gamma': [0, 0.5, 1],
    'subsample': [1],
    'max_depth': [2,4,6,8,10,12,14,16,18,20],
    'min_child_weight': [1],
    'n_estimators': [150]
}

# Parameters for random search
random_grid_parameters = {
    'rand_param_comb': 150,
    'folds': 5
}

# Parameters for grid search
grid_parameters = {
    'folds': 5
}

# Parameters for XGBClassifier fitter method
fitter_parameters = {
    'early_stopping_rounds': 5,
    'eval_set': [],
    'eval_metric': ['merror', 'mlogloss'],
    'verbose': False
}

# Parameters for creating XGBClassifier
model_parameters = {
    'learning_rate': 0.1,
    # 'gamma': 1,
    'subsample': 1,
    'max_depth': 2,
    'min_child_weight': 1,
    'n_estimators': 150
}


# Helper function to correctly set eval_set in fitter_parameters
def set_fitter_eval_xgbc(fitter_params, x_train, y_train, x_val, y_val):
    fitter_params['eval_set'] = [(x_train, y_train), (x_val, y_val)]
