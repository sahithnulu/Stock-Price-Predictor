# Stock Price Predictor
 
1) Command to activate python virtual environment
.\venv\Scripts\Activate.ps1

2) To deactivate virtual environment
deactivate

3) To run docker container
docker run --rm stock-trainer

# Notes

1) We use LSTM (Long Short-Term Memory) neural network since stock prices are a sequence of data over time. A time step in this scenario would be each day. This is because we need to remember patterns across sequences

2) We use MinMax Scaler to normalize the data. Neural networks perform best when input data is between 0 and 1. We reshape x into 3D for the LSTM Model (samples, time steps, features) because the LSTM model expects this shape since it reads sequences of values over time. We then need to scale y so that it is on the same scale as the input. We reverse this scaling during prediction to get the real value

3) LSTM(64) - number of units in the LSTM layer, each unit can store temporal patterns, higher the number more capacity to learn patterns (but more computationally expensive)

4) Dense(32) - number of neurons in the fully connected layer, each neuron receives input from all outputs of the previous layer and computes a weighted sum followed by an activation.

5) Dense(1) - outputs a single predicted price



