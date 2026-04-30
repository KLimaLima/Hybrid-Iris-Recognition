import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow import keras

import numpy as np

import datetime

class kmodel:

    def __init__(self, model_name = None):
        self.model = None
        self.model_name = model_name

        if model_name is None:
            now = datetime.datetime.now()
            self.model_name = f'{now.day :03d}'
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
            metrics=['accuracy']
        )

        self.model.summary()

        self.model.fit(
            {"gabor_input_1": my_gabor1, "gabor_input_2": my_gabor2, "gabor_input_3": my_gabor3, "gabor_input_4": my_gabor4, "fd_input": my_fd},
            # {"my_output": my_output},
            my_output,
            epochs=101,
            batch_size=32,
        )

        self.model.save(f'my_keras/{self.model_name}.keras')

    def inference(self, X_gabor1, X_gabor2, X_gabor3, X_gabor4, X_fd):
        my_pred = self.model.predict({"gabor_input_1": X_gabor1, "gabor_input_2": X_gabor2, "gabor_input_3": X_gabor3, "gabor_input_4": X_gabor4, "fd_input": X_fd})

        
        print(np.argmax(my_pred[0]))
        print(my_pred[0].max())
if __name__ == "__main__":
    pass