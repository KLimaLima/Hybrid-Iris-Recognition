import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow import keras

def make_model(gabor_shape, fd_shape, output_size,
               my_gabor1, my_gabor2, my_gabor3, my_gabor4, my_fd, my_output):

    # use output.size for output_size
    
    # --- Image branch (no CNN, just flatten) ---
    gabor_input1 = Input(shape=gabor_shape, name="gabor_input_1")  # adjust size
    gabor_input2 = Input(shape=gabor_shape, name="gabor_input_2")  # adjust size
    gabor_input3 = Input(shape=gabor_shape, name="gabor_input_3")  # adjust size
    gabor_input4 = Input(shape=gabor_shape, name="gabor_input_4")  # adjust size

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
    fd_input = Input(shape=fd_shape, name="fd_input")  # adjust feature size

    y = BatchNormalization()(fd_input)
    y = Dense(64, activation='relu')(y)
    y = Dense(32, activation='relu')(y)

    # --- Concatenate ---
    combined = Concatenate()([x1,x2,x3,x4, y])

    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(output_size, activation='softmax', name="my_output")(z)  # change depending on task

    # --- Model ---
    model = Model(inputs=[gabor_input1, gabor_input2, gabor_input3, gabor_input4, fd_input], outputs=output)

    keras.utils.plot_model(model, "my_keras/model.png", show_shapes=True)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        {"gabor_input_1": my_gabor1, "gabor_input_2": my_gabor2, "gabor_input_3": my_gabor3, "gabor_input_4": my_gabor4, "fd_input": my_fd},
        # {"my_output": my_output},
        my_output,
        epochs=101,
        batch_size=32,
    )

    model.save('my_model.keras')

if __name__ == "__main__":
    pass