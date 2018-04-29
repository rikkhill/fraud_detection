from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os


# Set filepath for model
filepath = "../models/anomaly_autoencoder_2.h5"
checkpoint_path = "../models/anomaly_autoencoder_checkpoint.h5"

# Read in the data
df_train = pd.read_csv("../data/not_fraud.csv")
components = ["V%d" % n for n in range(1, 29)]
just_components = df_train[components].as_matrix()

# Training set and test set

train = just_components[:-2000]
test = just_components[-2000:]

# Define and fit the model

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

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

callbacks_list = [checkpoint]

autoencoder.fit(train, train,
                epochs=20,
                batch_size=16,
                shuffle=True,
                validation_data=(test, test),
                verbose=1,
                callbacks=callbacks_list
                )

autoencoder.save(filepath)
