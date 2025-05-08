import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from preprocess_data import preprocess_data

def train_model():

    # Load the preprocessed data
    x, y = preprocess_data()

    # Normalize the input data
