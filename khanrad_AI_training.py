import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('cancer.csv')

# x is all the data minus the independent variable; y is the independent variable only
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Divide the dataset into a training portion and the testing portion (test_size)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()

# Define the neural network
model.add(tf.keras.layers.Dense(32, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Generate the neural network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=100)

# Test model
model.evaluate(x_test, y_test)
