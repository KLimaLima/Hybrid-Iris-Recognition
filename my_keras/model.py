import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import datetime

class kmodel():

    def __init__(self, model_name = None):
        self.model = None
        self.model_name = model_name

        if model_name is None:
            now = datetime.datetime.now()
            self.model_name = f'{now.month : 03d}M-{now.day :02d}D-{now.hour :02d}h-{now.minute :02d}m-{now.second :02d}s'
        else:
            self.model = keras.models.load_model(f"keras/{self.model_name}.keras")

    def make_model(self, gabor_shape, fd_shape, output_size,
                my_gabor1, my_gabor2, my_gabor3, my_gabor4, my_fd, my_output):
        
        if self.model is not None:
            print('This model has already been made\nmethod terminated')
            return

        gabor_input1 = Input(shape=gabor_shape, name="gabor_input_1")
        gabor_input2 = Input(shape=gabor_shape, name="gabor_input_2")
        gabor_input3 = Input(shape=gabor_shape, name="gabor_input_3")
        gabor_input4 = Input(shape=gabor_shape, name="gabor_input_4")

        x1 = Flatten()(gabor_input1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = Dense(128, activation='relu')(x1)
        x1 = BatchNormalization()(x1)

        x2 = Flatten()(gabor_input2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D()(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dense(256, activation='relu')(x2)
        x2 = Dense(128, activation='relu')(x2)
        x2 = BatchNormalization()(x2)

        x3 = Flatten()(gabor_input3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D()(x3)
        x3 = BatchNormalization()(x3)
        x3 = Dense(256, activation='relu')(x3)
        x3 = Dense(128, activation='relu')(x3)
        x3 = BatchNormalization()(x3)

        x4 = Flatten()(gabor_input4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D()(x4)
        x4 = BatchNormalization()(x4)
        x4 = Dense(256, activation='relu')(x4)
        x4 = Dense(128, activation='relu')(x4)
        x4 = BatchNormalization()(x4)

        # --- Structured data branch ---
        fd_input = Input(shape=fd_shape, name="fd_input")

        y = BatchNormalization()(fd_input)
        y = Dense(64, activation='relu')(y)
        y = Dense(32, activation='relu')(y)
        y = BatchNormalization()(y)

        # --- Concatenate ---
        combined = Concatenate()([x1,x2,x3,x4, y])

        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.5)(z)
        output = Dense(output_size, activation='softmax', name="my_output")(z)  # change depending on task

        # --- Model ---
        self.model = Model(inputs=[gabor_input1, gabor_input2, gabor_input3, gabor_input4, fd_input], outputs=output)

        keras.utils.plot_model(self.model, f"my_keras/{self.model_name}.png", show_shapes=True)

        self.model.compile(
            optimizer='adam',
            loss='kl_divergence',
            metrics=[
                'accuracy'
                # keras.metrics.Precision(),
                # keras.metrics.Recall()
                # keras.metrics.F1Score()
                ]
        )

        self.model.summary()

        csv_logger = CSVLogger(f'my_keras/{self.model_name}.log', append=False)

        early_stopper = EarlyStopping(monitor= 'val_loss', min_delta=1, patience=10, verbose=1, mode='min', start_from_epoch=75)

        history = self.model.fit(
            {"gabor_input_1": my_gabor1, "gabor_input_2": my_gabor2, "gabor_input_3": my_gabor3, "gabor_input_4": my_gabor4, "fd_input": my_fd},
            # {"my_output": my_output},
            my_output,
            epochs=200,
            batch_size=16,
            callbacks=[csv_logger],
            verbose=2,
            validation_split=0.2,
            shuffle=True
        )

        self.model.save(f'my_keras/{self.model_name}.keras')

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        print(f'Finished training model {self.model_name}')

    def inference(self, X_gabor1, X_gabor2, X_gabor3, X_gabor4, X_fd, y_label):
        # NOTE: make it so that it can compare
        # maybe change this to take one input
        # and make new method for validation of multiple inputs
        my_pred = self.model.predict({"gabor_input_1": X_gabor1, "gabor_input_2": X_gabor2, "gabor_input_3": X_gabor3, "gabor_input_4": X_gabor4, "fd_input": X_fd})

        for idx, row in enumerate(my_pred):
            print(f'For {y_label}:')
            print(np.argmax(row))
            print(np.max(row))

class amodel(kmodel):

    def __init__(self):
        super().__init__()

    def make_model(self, gabor_shape, fd_shape, output_size, my_gabor1, my_gabor2, my_gabor3, my_gabor4, my_fd, my_output):

        if self.model is not None:
            print('This model has already been made\nmethod terminated')
            return

        gabor_input1 = Input(shape=gabor_shape, name="gabor_input_1")
        gabor_input2 = Input(shape=gabor_shape, name="gabor_input_2")
        gabor_input3 = Input(shape=gabor_shape, name="gabor_input_3")
        gabor_input4 = Input(shape=gabor_shape, name="gabor_input_4")

        x1 = Flatten()(gabor_input1)
        x1 = BatchNormalization()(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = Dense(128, activation='relu')(x1)

        x2 = Flatten()(gabor_input2)
        x2 = BatchNormalization()(x2)
        x2 = Dense(256, activation='relu')(x2)
        x2 = Dense(128, activation='relu')(x2)

        x3 = Flatten()(gabor_input3)
        x3 = BatchNormalization()(x3)
        x3 = Dense(256, activation='relu')(x3)
        x3 = Dense(128, activation='relu')(x3)

        x4 = Flatten()(gabor_input4)
        x4 = BatchNormalization()(x4)
        x4 = Dense(256, activation='relu')(x4)
        x4 = Dense(128, activation='relu')(x4)

        # --- Structured data branch ---
        fd_input = Input(shape=fd_shape, name="fd_input")

        y = BatchNormalization()(fd_input)
        y = Dense(64, activation='relu')(y)
        y = Dense(32, activation='relu')(y)

        # --- Concatenate ---
        combined = Concatenate()([x1,x2,x3,x4, y])

        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.5)(z)
        output = Dense(output_size, activation='softmax', name="my_output")(z)  # change depending on task

        # --- Model ---
        self.model = Model(inputs=[gabor_input1, gabor_input2, gabor_input3, gabor_input4, fd_input], outputs=output)

        keras.utils.plot_model(self.model, f"my_keras/{self.model_name}.png", show_shapes=True)

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy'
                # keras.metrics.Precision(),
                # keras.metrics.Recall()
                # keras.metrics.F1Score()
                ]
        )

        self.model.summary()

        csv_logger = CSVLogger(f'my_keras/{self.model_name}.log', append=False)

        early_stopper = EarlyStopping(monitor= 'val_loss', min_delta=1, patience=10, verbose=1, mode='min', start_from_epoch=75)

        self.model.fit(
            {"gabor_input_1": my_gabor1, "gabor_input_2": my_gabor2, "gabor_input_3": my_gabor3, "gabor_input_4": my_gabor4, "fd_input": my_fd},
            # {"my_output": my_output},
            my_output,
            epochs=100,
            batch_size=32,
            callbacks=[csv_logger],
            verbose=2,
            validation_split=0.2,
            shuffle=True
        )

        self.model.save(f'my_keras/{self.model_name}.keras')
        print(f'Finished training model {self.model_name}')

class bmodel(kmodel):

    def __init__(self):
        super().__init__()

    def make_model(self, gabor_shape, fd_shape, output_size, my_gabor1, my_gabor2, my_gabor3, my_gabor4, my_fd, my_output):
        if self.model is not None:
            print('This model has already been made\nmethod terminated')
            return

        gabor_input1 = Input(shape=gabor_shape, name="gabor_input_1")
        gabor_input2 = Input(shape=gabor_shape, name="gabor_input_2")
        gabor_input3 = Input(shape=gabor_shape, name="gabor_input_3")
        gabor_input4 = Input(shape=gabor_shape, name="gabor_input_4")

        x1 = Conv2D(32, (3,3), activation='relu')(gabor_input1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D((2,2))(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D((2,2))(x1)
        x1 = Flatten()(x1)

        x2 = Conv2D(32, (3,3), activation='relu')(gabor_input2)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D((2,2))(x2)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D((2,2))(x2)
        x2 = Flatten()(x2)

        x3 = Conv2D(32, (3,3), activation='relu')(gabor_input3)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D((2,2))(x3)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D((2,2))(x3)
        x3 = Flatten()(x3)

        x4 = Conv2D(32, (3,3), activation='relu')(gabor_input4)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D((2,2))(x4)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D((2,2))(x4)
        x4 = Flatten()(x4)

        # --- Structured data branch ---
        fd_input = Input(shape=fd_shape, name="fd_input")

        y = BatchNormalization()(fd_input)
        y = Dense(64, activation='relu')(y)
        y = Dense(32, activation='relu')(y)

        # --- Concatenate ---
        combined = Concatenate()([x1,x2,x3,x4, y])

        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.5)(z)
        output = Dense(output_size, activation='softmax', name="my_output")(z)  # change depending on task

        # --- Model ---
        self.model = Model(inputs=[gabor_input1, gabor_input2, gabor_input3, gabor_input4, fd_input], outputs=output)

        keras.utils.plot_model(self.model, f"my_keras/{self.model_name}.png", show_shapes=True)

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'accuracy'
                # keras.metrics.Precision(),
                # keras.metrics.Recall()
                # keras.metrics.F1Score()
                ]
        )

        self.model.summary()

        csv_logger = CSVLogger(f'my_keras/{self.model_name}.log', append=False)

        early_stopper = EarlyStopping(monitor= 'val_loss', min_delta=1, patience=10, verbose=1, mode='min', start_from_epoch=75)

        history = self.model.fit(
            {"gabor_input_1": my_gabor1, "gabor_input_2": my_gabor2, "gabor_input_3": my_gabor3, "gabor_input_4": my_gabor4, "fd_input": my_fd},
            # {"my_output": my_output},
            my_output,
            epochs=200,
            batch_size=16,
            callbacks=[csv_logger],
            verbose=2,
            validation_split=0.2,
            shuffle=True
        )

        self.model.save(f'my_keras/{self.model_name}.keras')

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        print(f'Finished training model {self.model_name}')

if __name__ == "__main__":
    pass