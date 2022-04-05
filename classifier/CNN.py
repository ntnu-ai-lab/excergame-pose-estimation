from cnn_parameters import *
from training import compact, train
from misc import timer, cnn_plots, get_accuracy
import tensorflow as tf
from tensorflow.keras import layers, models
from read_data import get_dataset

print("TensorFlow version:", tf.__version__)


# Create CNN model
def make_model(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model = models.Sequential()
    model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(100, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(100, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(150, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(200, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(17))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


# Train a model according to parameters in cnn_parameters.py, create some plots and calculate accuracy
def train_and_evaluate(model_params, fitter_params):
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(**dataset_parameters)
    set_fitter_val_cnn(fitter_params, x_val, y_val)

    timer_1 = timer(None)
    model, predictions, history = train(make_model, model_params, fitter_params, 'cnn', x_train, y_train, x_test)
    print('Base Model Training', end='')
    timer(timer_1)

    # Evaluate model
    cnn_plots(model, history)

    return get_accuracy(compact(y_test), predictions)
