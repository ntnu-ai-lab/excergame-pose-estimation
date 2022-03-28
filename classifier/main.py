import xgbc_parameters as xp
import XGBC as xgbc
import cnn_parameters as cp
import CNN as cnn
from training import leave_one_out
from parameter_search import *
from misc import timer


def do_random_grid_search():
    model = xgbc.make_xgbc()
    random_grid(model, xp.model_search_parameters, xp.fitter_parameters, **xp.random_grid_parameters)


def do_grid_search():
    model = xgbc.make_xgbc()
    grid_search(model, xp.model_search_parameters, xp.fitter_parameters, **xp.grid_parameters)


def do_leave_one_out(model_type='xgbc'):
    if model_type == 'xgbc':
        leave_one_out(xgbc.make_xgbc, xp.model_parameters, xp.fitter_parameters, xp.dataset_parameters, 'xgbc')
    elif model_type == 'cnn':
        leave_one_out(cnn.make_model, cp.model_parameters, cp.fitter_parameters, cp.dataset_parameters, 'cnn')
    else:
        print('Bad mode')


def do_feature_select():
    xgbc.train_and_select(xp.model_parameters, xp.fitter_parameters, feature_selection=True)


def do_train_cnn():
    cnn.train_and_evaluate(cp.model_parameters, cp.fitter_parameters)


#do_feature_select()
#do_leave_one_out('xgbc')
do_leave_one_out('cnn')
#do_train_cnn()
