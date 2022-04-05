from xgbc_parameters import set_fitter_eval_xgbc
from cnn_parameters import set_fitter_val_cnn
from read_data import get_dataset, get_pars
import numpy as np
from misc import evaluations, get_f1, get_accuracy


# Turns prediction into one-hot through argmax
def clean_predictions(predictions):
    cleaned_predictions = []
    length = len(predictions[0])
    for prediction in predictions:
        max_index = np.argmax(prediction)
        t = np.array([0 if i != max_index else 1 for i in range(length)])
        cleaned_predictions.append(t)
    return np.array(cleaned_predictions)


# Turns one-hot into integer
def compact(one_hot_vector):
    temp = []
    for e in one_hot_vector:
        temp.append(np.argmax(e)+1)
    return np.array(temp)


# Trains a model. Returns the trained model, predictions and a history
def train(model_constructor, model_params, fitter_params, model_type, x_train, y_train, x_test):
    if model_type == 'xgbc':
        model = model_constructor(model_params)
    elif model_type == 'cnn':
        model = model_constructor(**model_params)
    else:
        raise Exception('No model specified')

    hist = model.fit(x_train, y_train, **fitter_params)
    predictions = model.predict(x_test)

    if model_type == 'xgbc':
        predictions = [round(i) for i in predictions]
    elif model_type == 'cnn':
        predictions = clean_predictions(predictions)
        predictions = compact(predictions)

    return model, predictions, hist


# Does leave-one-group-out cross-validation. At each iteration one participant is left out.
def leave_one_out(model_constructor, model_params, fitter_params, dataset_parameters, model_type, selection=None):
    data_params = dataset_parameters.copy()
    data_params['verbose'] = False
    always_leave_out = data_params['leave_out'][:]

    total_y_test = []
    total_predictions = []

    acc_scores = []
    f1_scores = []

    pars = [par for par in get_pars() if par not in always_leave_out]
    print(pars)
    print(always_leave_out)

    for par in pars:
        print('Leaving out', par)
        leave_out = always_leave_out[:]  # Leave out a participant
        leave_out.append(par)
        data_params['leave_out'] = leave_out

        # Read data and adjust to fit the model type
        x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(**data_params)
        if model_type == 'xgbc':
            if selection is not None:  # Select features if applicable
                x_train = selection.transform(x_train)
                x_test = selection.transform(x_test)
                x_val = selection.transform(x_val)
            set_fitter_eval_xgbc(fitter_params, x_train, y_train, x_val, y_val)
        elif model_type == 'cnn':
            set_fitter_val_cnn(fitter_params, x_val, y_val)
            y_test = compact(y_test)
        else:
            raise Exception('No model specified')

        # Train the model
        model, predictions, _ = train(model_constructor, model_params, fitter_params, model_type, x_train, y_train, x_test)

        # Save results
        total_y_test.extend(y_test)
        total_predictions.extend(predictions)

        acc_scores.append(get_accuracy(y_test, predictions))
        f1_scores.append(get_f1(y_test, predictions))

    # Print total results
    print(np.mean(acc_scores), np.std(acc_scores), acc_scores)
    print(np.mean(f1_scores), np.std(f1_scores), f1_scores)

    # Perform model evaluations
    if model_type == 'xgbc':
        if selection is not None:
            evaluations(total_y_test, total_predictions, 'XGBC', 'xgbc_selected')
        else:
            evaluations(total_y_test, total_predictions, 'XGBC', 'xgbc')
    if model_type == 'cnn':
        evaluations(total_y_test, total_predictions, 'CNN', 'cnn')
