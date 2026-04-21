import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras

# --- Image branch (no CNN, just flatten) ---
image_input = Input(shape=(150, 300), name="image_input")  # adjust size

x = Flatten()(image_input)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

# --- Structured data branch ---
numeric_input = Input(shape=(10,), name="numeric_input")  # adjust feature size

y = Dense(64, activation='relu')(numeric_input)
y = Dense(32, activation='relu')(y)

# --- Concatenate ---
combined = Concatenate()([x, y])

z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(1, activation='sigmoid')(z)  # change depending on task

# --- Model ---
model = Model(inputs=[image_input, numeric_input], outputs=output)

keras.utils.plot_model(model, "keras/model.png", show_shapes=True)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()