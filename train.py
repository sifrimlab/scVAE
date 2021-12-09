import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sklearn as sk
import seaborn as sns
import tensorflow as tf
# import tensorflow_probability as tfp
import umap

import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scVAE.py import *

# Read in and get the data in the correct formatting
raw_data = pd.read_csv("./Example Data/synthetic_2K_features.csv")
raw_data = raw_data.transpose()
raw_data.columns = raw_data.iloc[0]
raw_data = raw_data.iloc[1:] #drop the first row since it is just the same as the columns names, no extra info
# raw_data.shape is 10000 x 2000

# Read in the clusters (label information)
clusters = pd.read_csv("./Example Data/synthetic_clusters_processed.csv")

# Log transform the data
log_data = raw_data.astype(np.float32).apply(lambda x: np.log2(x+1), axis=1)

# Standardize data
standardizer = StandardScaler()
standard_data = standardizer.fit_transform(log_data)
standard_df = pd.DataFrame(standard_data)
standard_df.columns = raw_data.columns
scaled_df=standard_df
del standard_df

# Define and train model
encoder_hidden_structure = [1500, 1000, 1000, 500, 500, 250, 250, 32, 16, 8, 2]
name = "PReLU-BothNonLinear-Weight2"

decoder_hidden_structure = encoder_hidden_structure[::-1][1:] # Inverse of encoder
epochs = 1000
batch_size = 252
max_value_kl_weight = 2
warmup_epochs = 500
function = "linear"

# Set (adaptive) lr
lr = 0.0001
adaptive_lr = LearningRateScheduler(step_lr, verbose=1)

vae = VAE()
vae.set_encoder(n_input=scaled_df.shape[1], hidden_structure=encoder_hidden_structure)
vae.set_decoder(n_output=scaled_df.shape[1], hidden_structure=decoder_hidden_structure, n_latent=encoder_hidden_structure[-1])

# vae.set_linear_encoder(n_input=scaled_df.shape[1], hidden_structure=encoder_hidden_structure)
# vae.set_linear_decoder(n_output=scaled_df.shape[1], hidden_structure=decoder_hidden_structure, n_latent=encoder_hidden_structure[-1])

vae.compile(optimizer=Adam(learning_rate=lr))

initial_time = datetime.datetime.now()
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=400)
warmingup = WarmingupKL(vae, warmup_epochs, max_value_kl_weight, function)
accuracy = accuracy_per_epoch(model=vae, data=scaled_df, clusters=clusters, name=name)

history = vae.fit(scaled_df.to_numpy().astype(np.float32), scaled_df.to_numpy().astype(np.float32), epochs=epochs,
                  batch_size=batch_size,callbacks=[warmingup, accuracy]) #adaptive_lr, callback])

print("It took {} min to train the model.".format(datetime.datetime.now()-initial_time))

# Show accuracy plot and metrics
accuracy.show_accuracy_plot()
print(accuracy.metrics_report())

# Plot losses
plt.plot(history.history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(name)
plt.show()

plt.plot(history.history["reconstruction_loss"], 'g')
plt.xlabel("Epoch")
plt.ylabel("Reconstruction loss")
plt.title(name)
plt.show()

kl_loss_plot = plt.plot(history.history["kl_loss"], "r")
plt.xlabel("Epoch")
plt.ylabel("KL loss")
plt.title(name)
plt.show()

# Get the latent space and visualize it
_, _, latent = vae.get_encoder().predict(scaled_df.to_numpy().astype(np.float32))
plt.scatter(latent[:,0],latent[:,1], alpha=.4, c=clusters["Cluster Nr"], cmap="tab10")
plt.title(name)
plt.show()

# Get reconstructions
reconstructed = vae.predict(scaled_df.to_numpy().astype(np.float32)).numpy()
reconstructed = pd.DataFrame(reconstructed, columns=raw_data.columns, index=raw_data.index)
# reconstructed.to_csv("./reconstructed_normalized.csv")
