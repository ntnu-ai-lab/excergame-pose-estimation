import xgbc_parameters as xparams
import cnn_parameters as cparams
from read_data import get_dataset
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import PATH_CONSTANTS as PATHS


def random_grid(model, model_search_params, fitter_params, rand_param_comb=5, folds=5):
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(**xparams.dataset_parameters)
    xparams.set_fitter_eval_xgbc(fitter_params, x_train, y_train, x_val, y_val)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=None)
    # f1 = f1_score(y_test, predictions, average='macro')
    scorer = make_scorer(f1_score, **{'average': 'macro'})

    grid_search_random = RandomizedSearchCV(model, param_distributions=model_search_params, n_iter=rand_param_comb,
                                            scoring=scorer, cv=skf.split(x_train, y_train), verbose=3)

    grid_search_random.fit(x_train, y_train, **fitter_params)

    print('\n All results:')
    print(grid_search_random.cv_results_)
    print('\n Best estimator:')
    print(grid_search_random.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, rand_param_comb))
    print(grid_search_random.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(grid_search_random.best_params_)
    results = pd.DataFrame(grid_search_random.cv_results_)
    results.to_csv(PATHS.FIGURES_FOLDER + 'xgb-random-grid-search-results-latest.csv', index=False)


def grid_search(model, model_search_params, fitter_params, folds=5):
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(**xparams.dataset_parameters)
    xparams.set_fitter_eval_xgbc(fitter_params, x_train, y_train, x_val, y_val)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=None)
    # f1 = f1_score(y_test, predictions, average='macro')
    scorer = make_scorer(f1_score, **{'average': 'macro'})

    search_results = GridSearchCV(model, param_grid=model_search_params, scoring=scorer, cv=skf.split(x_train, y_train), verbose=3)

    search_results.fit(x_train, y_train, **fitter_params)

    print('\n All results:')
    print(search_results.cv_results_)
    print('\n Best estimator:')
    print(search_results.best_estimator_)
    print('\n Best normalized gini score for %d-fold search:' % folds)
    print(search_results.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(search_results.best_params_)
    results = pd.DataFrame(search_results.cv_results_)
    results.to_csv(PATHS.FIGURES_FOLDER + 'xgb-grid-search-results-latest.csv', index=False)
