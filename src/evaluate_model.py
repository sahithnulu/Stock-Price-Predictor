import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess_data import preprocess_data
from sklearn.model_selection import train_test_split

def evaluate_model():
    # Load model and scaler
    model = load_model("model/stock_model.keras")
    scaler = joblib.load("./scaler/scaler.save")

    # Load and preprocess data
    x, y = preprocess_data()
    x_scaled = scaler.fit_transform(x) # Flattens before scaling (Scales x values between 0 and 1)
    x_scaled = x_scaled.reshape((x_scaled.shape[0], x_scaled.shape[1], 1)) # Reshapes to 3D for LSTM (Long Short-Term Memory) input

    # Normalize the target data
    y = y.reshape(-1, 1) # Reshape y to 2D for scaling
    y_scaled = scaler.fit_transform(y) # Scales y values between 0 and 1

    # Split test data
    _, x_test, _, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

    # Predict on test data
    y_pred_scaled = model.predict(x_test)

    # Inverse transform predictions
    y_test_unscaled = scaler.inverse_transform(y_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

    # Compute metrics
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

    print(f"\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled, label='Actual Prices', color='blue')
    plt.plot(y_pred_unscaled, label='Predicted Prices', color='red')
    plt.title('Stock Price Prediction - Test Data')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
