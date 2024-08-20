import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data from CSV files
def load_data_from_csv(csv_file_name):
    data_df = pd.read_csv(csv_file_name, header=None)
    return data_df.values.astype(np.float32)  # Convert the data to float32

# Load X_train, Y_train, X_cv, Y_cv from CSV files
X_train = load_data_from_csv('X_train.csv')
Y_train = load_data_from_csv('Y_train.csv')
X_cv = load_data_from_csv('X_cv.csv')
Y_cv = load_data_from_csv('Y_cv.csv')

# Subtract 1 from the target labels to convert them to 0-based indices
Y_train = Y_train.astype(np.int32) - 1
Y_cv = Y_cv.astype(np.int32) - 1

# One-hot encode the target labels
num_classes = 29
Y_train_encoded = tf.keras.utils.to_categorical(Y_train, num_classes)
Y_cv_encoded = tf.keras.utils.to_categorical(Y_cv, num_classes)

# Reshape the data for the model
X_train = X_train.reshape((-1, 200, 200, 1))
X_cv = X_cv.reshape((-1, 200, 200, 1))

# Define and compile the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using Y_train_encoded as the target labels
model.fit(X_train, Y_train_encoded, epochs=20, batch_size=32)

# Make predictions
predictions = model.predict(X_cv)

# Calculate accuracy
correct_predictions = np.argmax(predictions, axis=1) == np.argmax(Y_cv_encoded, axis=1)
accuracy = np.mean(correct_predictions)
print("Accuracy on the cross-validation set:", accuracy)
