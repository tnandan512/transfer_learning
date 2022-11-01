import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import models, layers
import seaborn as sns 
import matplotlib.pyplot as plt

train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
pretrain_features = pd.read_csv('pretrain_features.csv')
pretrain_labels = pd.read_csv('pretrain_labels.csv')

# ------------ NN TRAINED ON PRE-TRAIN DATASET ---------------

X_ptrain = pretrain_features.iloc[:,2:] 
y_ptrain = pretrain_labels.iloc[:,1] 
X_train, X_test, y_train, y_test = train_test_split(X_ptrain, y_ptrain, test_size=0.3, random_state=1)
n_input = X_ptrain.shape[1]  

inp = keras.Input(shape=(n_input)) 
h1 = layers.Dense(name="h1", input_dim = n_input, units=int(round((n_input)/2)),
            activation='relu')(inp)
h2 = layers.Dense(name="h2", units=int(round((n_input)/4)), activation='relu')(h1)
h3 = layers.Dense(name="h3", units=int(round((n_input)/6)), activation='relu')(h2)
h4 = layers.Dense(name="h4", units=int(round((n_input)/8)), activation='relu')(h3)
output = layers.Dense(name="output", units=1, activation='linear')(h4)

model = keras.Model(inp, output)
model.compile(loss='mean_squared_error', optimizer='adam') 
print('About to train model...')
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 75, batch_size=1000, verbose = 1)

hist_df = pd.DataFrame.from_dict(hist.history)
plt.plot(hist_df['val_loss'])
plt.xlabel("epochs")
plt.ylabel("Validation loss") 

# ------------- TRANSFER LEARNING -------------

Xtrain = train_features.iloc[:,2:] 
ytrain = train_labels.iloc[:,1] 

X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.3, random_state=1)

encoder = keras.Model(inp, h4) 
encoder.trainable = False 

inputs = keras.Input(shape = (n_input))
x = encoder(inputs, training=False) 
pre_output = keras.layers.Dense(250, activation = 'relu')(x) 
pre_output = keras.layers.Dense(400, activation = 'relu')(pre_output)
output = keras.layers.Dense(1, activation = 'linear')(pre_output) 

tl_model = keras.Model(inputs, output)
tl_model.compile(optimizer='adam', loss='mean_squared_error')
hist_tl = tl_model.fit(X_train, y_train, 
                       validation_data=(X_test, y_test), 
                       epochs = 200, batch_size=10, verbose = 1) 

# hist_tl_df = pd.DataFrame.from_dict(hist_tl.history)
# plt.plot(hist_tl_df['val_loss'])
# plt.xlabel("epochs")
# plt.ylabel("Validation loss") 

# Fine tuning

encoder.trainable = True
tl_model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss='mean_squared_error')

hist_tl = tl_model.fit(X_train, y_train, 
                       validation_data=(X_test, y_test), 
                       epochs = 50, batch_size=10, verbose = 1) 

hist_tl_df2 = pd.DataFrame.from_dict(hist_tl.history)
plt.plot(hist_tl_df2['val_loss'])
plt.xlabel("epochs")
plt.ylabel("Validation loss") 

# ------------ PREDICT ON TEST DATA PROVIDED -------------

test_features = pd.read_csv('test_features.csv') 
Xtest = test_features.iloc[:,2:] 
values = tl_model.predict(Xtest) 
vals = []
for val in values:
    vals.append(float(val))
task4 = {}
task4['Id'] = test_features['Id']
task4['y'] = vals 
data = pd.DataFrame.from_dict(task4, orient='columns')
data.to_csv('homo_lumo.csv', index=False) 
