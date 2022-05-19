from locale import normalize
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

# xData = pd.read_csv("X_train.csv")
# yData = pd.read_csv("y_train.csv")

# xTrain = xData.values
# yTrain = yData.values

# np.random.seed(6969)  
# np.random.shuffle(xTrain)
# np.random.seed(6969)  
# np.random.shuffle(yTrain)  

# # normalize data (0 - 255 -> 0 - 1)
# xTrain = tf.keras.utils.normalize(xTrain, axis = 1)

# xTrain = xTrain.reshape(len(xTrain), 32, 32, 1)

# #def build_model(hp):
# model = tf.keras.Sequential()

# #hp1 = Int('units', min_value=16, max_value=128, step=16)
# #hp2 = hp.Int('units', min_value=16, max_value=128, step=16)

# model.add(tf.keras.Input(shape=(32, 32, 1) ,name='input_data'))
# model.add(tf.keras.layers.Conv2D(64, 3,activation='relu',padding='same',kernel_initializer='he_normal'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# model.add(tf.keras.layers.Dropout(0.1))
# model.add(tf.keras.layers.Conv2D(64, 3,activation='relu',padding='same',kernel_initializer='he_normal'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
# model.add(tf.keras.layers.Dropout(0.1))

# model.add(tf.keras.layers.Dense(64, activation='relu'))

# model.add(tf.keras.layers.Dropout(0.1))

# model.add(tf.keras.layers.Reshape((64, 64)))

# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True,dropout=0.2)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True,dropout=0.25)))

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(64,activation='relu'))
# model.add(tf.keras.layers.Dense(30,activation='softmax',kernel_initializer='he_normal'))

# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
#     #return model
# model.summary()

# # train the model
# model.fit(xTrain, yTrain, epochs = 10, verbose = 1, shuffle=True, validation_split = 0.1)

# # save the model
# model.save("handwriting_arabic_5")


xData = pd.read_csv("X_test.csv")
yData = pd.read_csv("y_test.csv")

xTest = xData.values
yTest = yData.values

# normalize data (0 - 255 -> 0 - 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)

xTest = xTest.reshape(len(xTest), 32, 32, 1)

model = tf.keras.models.load_model("handwriting_arabic_5")

model.summary()

loss, accuracy = model.evaluate(xTest, yTest)

print(loss)
print(accuracy)

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# tuner = kt.Hyperband(build_model,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='intro_to_kt')

# tuner.search(xTrain, yTrain, epochs=5, validation_split=0.2, callbacks=[stop_early])

# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]