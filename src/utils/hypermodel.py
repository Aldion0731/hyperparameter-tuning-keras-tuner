import keras_tuner as kt
from tensorflow import keras


def model_builder(hp: kt.HyperParameters) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation="relu"))

    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation="softmax"))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
