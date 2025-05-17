# Data Handling
import numpy as np
# Building and training the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
# Data Normalization
from sklearn.preprocessing import MinMaxScaler
# Saving the scaler (reusabled component)
import joblib
# Reusing the preprocessing and evaluting model function
from preprocess_data import preprocess_data
from evaluate_model import evaluate_model


def train_model():

    # Load the preprocessed data
    x, y = preprocess_data()

    # Normalize the input data using MinMaxScaler
    scaler = MinMaxScaler() # Scales the data to the range [0, 1] which helps neural networks converge faster
    x_scaled = scaler.fit_transform(x) # Flattens before scaling (Scales x values between 0 and 1)
    x_scaled = x_scaled.reshape((x_scaled.shape[0], x_scaled.shape[1], 1)) # Reshapes to 3D for LSTM (Long Short-Term Memory) input

    # Normalize the target data
    y = y.reshape(-1, 1) # Reshape y to 2D for scaling
    y_scaled = scaler.fit_transform(y) # Scales y values between 0 and 1

    # Save the scaler for inference later
    joblib.dump(scaler, './scaler/scaler.save') # Save the scaler to a file

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42) # 80% train, 20% test, random state for reproducibility (42 is arbitrary)

    # Build the model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(x.shape[1], 1)), #64 units in the LSTM layer, return_sequences=False means only the last output is returned, higher the number, more capacity to learn patterns
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse') # Adam optimizer adjusts model weights efficiently using gradients, Mean Squared Error is a standard for regression, penalizes larger errors more than smaller ones
    model.summary()

    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test)) # epoch is one full pass through the training data, batch size is the number of samples processed before the model is updated, validation_data is used to evaluate the model during training

    # Save the model
    model.save("model/stock_model.keras")  
    print("Model trained and saved as stock_model.h5")

    # Call the evaluate_model function to evaluate the model after training
    evaluate_model()

if __name__ == "__main__":
    train_model()
