import time

from tensorflow import keras


class TimeRecorder(keras.callbacks.Callback):
    def on_train_begin(self, logs={}) -> None:  # type: ignore
        self.times: list[float] = []

    def on_epoch_begin(self, batch, logs={}) -> None:  # type: ignore
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}) -> None:  # type: ignore
        self.times.append(time.time() - self.epoch_time_start)
