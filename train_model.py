import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# LOAD DATASET

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# PADDING

max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)

X_test = pad_sequences(X_test, maxlen=max_length)

# BUILD MODEL

model = Sequential(
    [
        Embedding(input_dim=10000, output_dim=32, input_length=max_length),
        SimpleRNN(32, activation="tanh"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

# COMPILE

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# EARLY STOPPING

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# TRAIN

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
)

# TEST

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

# SAVE MODEL

model.save("models/rnn_model.keras")

print("Model Saved Successfully")
