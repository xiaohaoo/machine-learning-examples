import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from tensorflow.python.keras import layers, models, callbacks

data = np.loadtxt('datasets/sequential_data.csv', skiprows=1, delimiter=',')
data[np.isnan(data)] = np.mean(data[~np.isnan(data)])
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data = min_max_scaler.fit_transform(data)

# 1075×1×6
x = data[:, :-1].reshape(-1, 1, 6)

# 1075×1
y = data[:, -1:].reshape(-1)


# 绘制数据集折线图
def plot_train_data():
    plt.title('oil price since 1997.01.17')
    plt.plot(y)
    plt.show()


def narx_model():
    input = layers.Input(shape=(1, 6))
    x = layers.LSTM(16, return_sequences=True, activation='tanh')(input)
    x = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs=input, outputs=x)


def train_modal():
    model = narx_model()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    early_stopping = callbacks.EarlyStopping(monitor='mse', min_delta=1e-5, mode='auto', patience=12)
    history = model.fit(x, y, epochs=500, batch_size=64, shuffle=True, validation_split=0.01, callbacks=[early_stopping])
    model.save('model.h5')
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'], color='g')
    plt.title('model loss and accuracy')
    plt.yticks(np.arange(0, 0.05, step=0.002))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss', 'train_acc', 'test_acc'], loc='upper right')
    plt.show()
    return model


def load_model():
    model = models.load_model('model.h5')
    return model


def predict(x):
    model = load_model()
    y = model.predict(x)
    return y


def predict_and_plot_data():
    Y = predict(x[::35])
    plt.title('model predict and actual')
    plt.plot(Y.reshape(-1))
    plt.plot(y[::35].reshape(-1))
    plt.show()
    plt.show()
