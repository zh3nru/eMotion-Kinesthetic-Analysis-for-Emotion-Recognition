import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping



def load_npy(directory):

    npy_file = []

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            path = os.path.join(directory, filename)
            npy_file.append(np.load(path))

    npy_array = np.array(npy_file)

    return npy_array

directory = {'happy': r"E:\emotion_joints\Happiness", 'sad': r"E:\emotion_joints\Sadness",
             'surprise': r"E:\emotion_joints\Surprise", 'anger': r"E:\emotion_joints\Anger",
             'aversion': r"E:\emotion_joints\Aversion", 'peace': r"E:\emotion_joints\Peace",
             'fear': r"E:\emotion_joints\Fear"}

npy_happy = load_npy(directory['happy'])
npy_sad = load_npy(directory['sad'])
npy_surprise = load_npy(directory['surprise'])
npy_anger = load_npy(directory['anger'])
npy_aversion = load_npy(directory['aversion'])
npy_peace = load_npy(directory['peace'])
npy_fear = load_npy(directory['fear'])

x = []
y = []
no_of_timesteps = 20

for npy_array in npy_happy:
    datasets = npy_array[:, 2:]
    n_samples = len(datasets)
    for i in range(no_of_timesteps, n_samples):
        x.append(datasets[i-no_of_timesteps:i, :])
        y.append(0)

for npy_array in npy_sad:
    datasets = npy_array[:, 2:]
    n_samples = len(datasets)
    for i in range(no_of_timesteps, n_samples):
        x.append(datasets[i-no_of_timesteps:i, :])
        y.append(1)


x, y = np.array(x), np.array(y)
print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))  

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save("happy_model.h5")

