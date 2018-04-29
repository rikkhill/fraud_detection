from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import os


# Set filepath for model
filepath = "../models/anomaly_autoencoder.h5"

# Read in the data
df_fraud = pd.read_csv("../data/fraud.csv")
df_validate = pd.read_csv("../data/not_fraud_validation.csv")
components = ["V%d" % n for n in range(1, 29)]
fraud_components = df_fraud[components].as_matrix()
validation_components = df_validate[components].as_matrix()

# Define and load the model

features = Input(shape=(28, ))

encoded = Dense(28, activation='relu')(features)
encoded = Dense(12, activation='relu')(encoded)
encoded = Dense(6, activation='relu')(encoded)
decoded = Dense(12, activation='relu')(encoded)
decoded = Dense(26, activation='relu')(decoded)
decoded = Dense(28, activation='tanh')(decoded)

autoencoder = Model(features, decoded)

autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

# Load model if it exists
if os.path.isfile(filepath):
    autoencoder.load_weights(filepath)
