import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art

# Generate ASCII art with the text "RouletteAi"
ascii_art = text2art("RouletteAi")

print("============================================================")
print("RouletteAi")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Print the generated ASCII art
print(ascii_art)
print("Roulette prediction artificial intelligence")

# Load data from file, ignoring white spaces and accepting unlimited length numbers
data = np.genfromtxt('data.txt', delimiter='\n', dtype=int)

# Filter out numbers that are not between 0 and 36 (inclusive)
data = data[(data >= 0) & (data <= 36)]

# Define the length of the input sequences
sequence_length = 10

# Create sequences of fixed length from the data
sequences = np.array([data[i:i+sequence_length] for i in range(len(data)-sequence_length)])

# Create target values which are the next number after each sequence
targets = data[sequence_length:]

# Split the data into training and validation sets
train_data = sequences[:int(0.8*len(sequences))]
train_targets = targets[:int(0.8*len(targets))]
val_data = sequences[int(0.8*len(sequences)):]
val_targets = targets[int(0.8*len(targets)):]

# Get the maximum value in the data
max_value = np.max(data)

# Set the number of features to 1
num_features = 1

model = keras.Sequential()
model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
model.add(layers.LSTM(256))
model.add(layers.Dense(num_features, activation='softmax'))  # Set the number of units to match the number of features

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=100)

predictions = model.predict(val_data)

indices = np.argsort(predictions, axis=1)[:, -num_features:]
predicted_numbers = np.take_along_axis(val_data, indices, axis=1)

print("============================================================")
print("Predicted Number:")
for numbers in predicted_numbers[:1]:
    print(', '.join(map(str, numbers)))

print("============================================================")
print("If you won buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Prevent the window from closing immediately
input('Press ENTER to exit')
