from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score
from xgboost import plot_importance, XGBClassifier
from keras.utils.vis_utils import plot_model
import PATH_CONSTANTS as PATHS


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def get_accuracy(y_test, predictions):
    return accuracy_score(y_test, predictions)


def get_f1(y_test, predictions, average='macro'):
    return f1_score(y_test, predictions, average=average)


def evaluations(y_test, predictions, title='', filename=''):
    if filename != '':
        filename = '_' + filename

    # evaluate predictions
    accuracy = get_accuracy(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    f1 = get_f1(y_test, predictions, 'macro')
    print("F1: %.2f" % f1)

    # create confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, normalize='true')
    plt.rc('axes', titlesize=40, labelsize=30)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax)
    disp.ax_.set_title(title)
    plt.savefig(PATHS.FIGURES_FOLDER + 'confusion_matrix' + filename + '.png')
    plt.clf()

    # Reset font sizes to default
    plt.rc('axes', titlesize=10, labelsize=10)


def xgbc_plots(model, filename_tail=''):
    if filename_tail != '':
        filename_tail = '_' + filename_tail

    # plot feature importance
    plot_importance(model, max_num_features=30)
    plt.savefig(PATHS.FIGURES_FOLDER + 'feature_importance_xgbc' + filename_tail + '.png')
    plt.clf()

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.savefig(PATHS.FIGURES_FOLDER + 'xgbc_logloss' + filename_tail + '.png')
    plt.clf()

    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig(PATHS.FIGURES_FOLDER + 'xgbc_class_error' + filename_tail + '.png')
    plt.clf()


def cnn_plots(model, history, filename_tail=''):
    # retrieve performance metrics
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'], label='accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(PATHS.FIGURES_FOLDER + 'cnn_accuracy' + filename_tail + '.png')
    plt.clf()

    # plot loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(PATHS.FIGURES_FOLDER + 'cnn_loss' + filename_tail + '.png')

    # Create model visualization
    # If show_shapes and show_layer_activations are both set to true, plot_model() will throw errors due to a bug
    plot_model(model, to_file=PATHS.FIGURES_FOLDER + 'cnn_visual' + filename_tail + '.png', show_shapes=True, show_layer_names=False, show_layer_activations=False)
