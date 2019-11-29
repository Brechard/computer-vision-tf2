import subprocess

from tensorflow.keras.callbacks import (
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping
)

import constants
import helpers
import models.custom_callbacks as custom_callbacks
from visualization.visualize import plot_history


def train(model, epochs, train_data, val_data, save_freq, initial_lr, train_info, use_fit_generator, use_cosine_lr):
    """
    Standar method to train the model received.
    :param model: Model to train already compiled.
    :param epochs: Number of epochs to train the model.
    :param train_data: Train data as tf.data.Dataset or keras ImageDataGenerator if use_fit_generator.
    :param val_data: Validation data as tf.data.Dataset or keras ImageDataGenerator if use_fit_generator.
    :param save_freq: Checkpoints frequency.
    :param initial_lr: Initial learning rate.
    :param train_info: Training information to save in text fail and print.
    :param use_fit_generator: Flag to use the function fit_generator instead of fit.
    :param use_cosine_lr: Flag to use the cosine decay scheduler.
    :return: history from calling fit(_generator) function, history from custom_callbacks.
    """
    with open(model.checkpoints_path + 'train_info.txt', 'w') as t:
        if use_cosine_lr:
            train_info += "    - Use cosine decay scheduler. Initial LR = " + str(initial_lr)
        else:
            train_info += "    - Use constant LR = " + str(initial_lr)
        t.write(train_info)
        last_commit = subprocess.check_output(['git', 'describe', '--always']).strip().decode('UTF-8')
        t.write('\nLast commit: ' + last_commit + '\n')

    print(constants.C_WARNING, train_info, constants.C_ENDC)

    history_callback = custom_callbacks.History()
    callbacks = [
        EarlyStopping(patience=4, verbose=1),
        custom_callbacks.ModelCheckpoint(
            model.checkpoints_path + model.model_name + '_{epoch}-loss-{loss:.5f}.ckpt',
            monitor='loss', verbose=1, save_weights_only=True, save_freq=save_freq,
            save_best_only=True),
        history_callback,
        custom_callbacks.CustomSaveHistory(model.logs_path + 'train_history.p'),
        TensorBoard(log_dir=model.logs_path, update_freq='epoch')
    ]

    if use_cosine_lr:
        cosine_lr = custom_callbacks.CosineDecayScheduler(
            initial_lr=initial_lr,
            epochs=epochs,
            epochs_hold_initial_lr=int(epochs / 20), verbose=0)
        callbacks.append(cosine_lr)
    else:
        cosine_lr = None
        callbacks.append(ReduceLROnPlateau(verbose=1, min_lr=1e-5, patience=2))

    if use_fit_generator:
        history = model.train_model.fit_generator(train_data,
                                                  epochs=epochs,
                                                  callbacks=callbacks,
                                                  validation_data=val_data)
    else:
        history = model.train_model.fit(train_data,
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        validation_data=val_data,
                                        use_multiprocessing=True,
                                        workers=8)

    model.train_model.save_weights(model.checkpoints_path + 'weights.ckpt')
    if use_cosine_lr:
        history.history['learning_rates'] = cosine_lr.learning_rates

    plot_history(history.history, model.figs_path)
    print(constants.C_OKBLUE, "Model finished training. Took in total",
          helpers.display_time(history_callback.total_time, 4),
          constants.C_ENDC)

    for epoch_history in history_callback.history:
        print(epoch_history)

    with open(model.checkpoints_path + 'train_info.txt', 'a') as t:
        t.write('Model finished training. History:\n')
        for epoch_history in history_callback.history:
            t.write('    ' + epoch_history + '\n')

    return history, history_callback
