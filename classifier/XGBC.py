import warnings
from xgbc_parameters import dataset_parameters, set_fitter_eval_xgbc
from misc import timer, evaluations, xgbc_plots, get_f1, get_accuracy
from training import train, leave_one_out
from parameter_search import random_grid, grid_search
from xgboost import XGBClassifier
import numpy as np
from read_data import get_dataset, get_pars
from sklearn.feature_selection import SelectFromModel


warnings.filterwarnings("ignore", category=UserWarning)


def make_xgbc(model_params={}):
    model = XGBClassifier(tree_method='gpu_hist', verbosity=0, **model_params)
    return model


def train_and_select(model_params, fitter_params, feature_selection=False):
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(**dataset_parameters)
    set_fitter_eval_xgbc(fitter_params, x_train, y_train, x_val, y_val)

    timer_1 = timer(None)

    model, predictions, _ = train(make_xgbc, model_params, fitter_params, 'xgbc', x_train, y_train, x_test)

    print('Base Model Training', end=' ')
    timer(timer_1)

    # feature_names = get_header()*100
    # model.get_booster().feature_names = feature_names

    # make plots
    xgbc_plots(model)

    if feature_selection:
        feature_select(model, model_params, fitter_params, x_train, y_train, x_test, y_test)


def feature_select(model, model_params, fitter_params, x_train, y_train, x_test, y_test):
    print('Feature selecting')
    old_features = len(model.feature_importances_)

    threshold = np.sort(model.feature_importances_)[-50:]
    # print(np.sort(model.feature_importances_)[-1])
    # print(threshold)
    selection = SelectFromModel(model, threshold=threshold[0], prefit=True)

    # Feature select
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    # Fix fitter parameters to fit new model
    old_eval_set = fitter_params['eval_set']
    select_x_val = selection.transform(old_eval_set[1][0])
    set_fitter_eval_xgbc(fitter_params, select_x_train, y_train, select_x_val, old_eval_set[1][1])

    timer_2 = timer(None)
    selection_model, predictions, _ = train(make_xgbc, model_params, fitter_params, 'xgbc', select_x_train, y_train, select_x_test)
    print('Feature Selection Model Training', end=' ')
    timer(timer_2)

    new_features = len(selection_model.feature_importances_)
    print('New features: %i, down from %i' % (new_features, old_features))

    # Make plots
    xgbc_plots(selection_model, 'selection')

    # Leave one out + feature selection
    timer_3 = timer(None)
    leave_one_out(make_xgbc, model_params, fitter_params, dataset_parameters, 'xgbc', selection)
    print('Feature Selection Leave-One-Out', end=' ')
    timer(timer_3)
