from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import optimizers

dataset_parameters = {
    'par_count': None,
    'val_percent': 0.1,
    'test_percent': 0.1,
    'leave_out': [],
    'withhold_moves': [],
    'normalize': True,
    'one_hot': True,
    'flatten': False,
    'verbose': True
}

optimizer_parameters = {
    'learning_rate': 0.001
}

model_parameters = {
    'optimizer': optimizers.Adam(**optimizer_parameters),
    'loss': CategoricalCrossentropy(from_logits=True),
    'metrics': ['categorical_accuracy', 'categorical_crossentropy']
}

fitter_parameters = {
    'epochs': 15,
    'validation_data': (),
    # 'validation_split': 0.1,
    'verbose': 2,
}


def set_fitter_val_cnn(fitter_params, x_val, y_val):
    fitter_params['validation_data'] = (x_val, y_val)
