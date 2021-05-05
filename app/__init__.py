from app.main.upbit import Upbit

### correlation and flask setting
from flask import Flask, send_file, render_template, make_response

from io import BytesIO
from functools import wraps, update_wrapper
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

### forecast library

import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

#################

app = Flask(__name__)
upbit_object = Upbit()


@app.route('/')
def normal():
    return render_template("index.html", width=800,
                           height=600)


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.route('/fig/<bit1_bit2>')
@nocache
def fig(bit1_bit2):  # TODO: fig에서 분석한 데이터를 normal로 보내서 render_templates에 뿌려야한다.
    bit1, bit2 = bit1_bit2.split("_")
    bit1_candles = upbit_object.get_days_candles(bit1)
    bit2_candles = upbit_object.get_days_candles(bit2)

    bit1_xs, bit1_dataset = get_rate_data(bit1_candles)
    bit2_xs, bit2_dataset = get_rate_data(bit2_candles)

    plt.plot(bit1_xs, bit1_dataset, label=bit1)
    plt.plot(bit2_xs, bit2_dataset, label=bit2)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, "correlation result: {0:0.3f}".format(np.corrcoef(bit1_dataset, bit2_dataset)[0, 1]),
             fontsize=10,
             horizontalalignment='left', verticalalignment='bottom', bbox=props)

    plt.legend()

    np.corrcoef(bit1_dataset, bit2_dataset)

    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    return send_file(img, mimetype='image/png')


def get_corr(bit1, bit2):
    return np.corrcoef(bit1, bit2)


def get_rate_data(bit):
    i = 1
    bit_xs = []
    bit_dataset = []
    for candle in bit:
        bit_xs.append(i)
        bit_dataset.append(candle['change_rate'])
        i += 1

    return bit_xs, bit_dataset


def get_price_data(bit):
    i = 1
    bit_xs = []
    bit_dataset = []
    for candle in bit:
        bit_xs.append(i)
        bit_dataset.append(candle['change_rate'])
        i += 1

    return bit_dataset


@app.route('/forecast/<bit>')
def forecast(bit):  # TODO: 비트코인 주가 예측
    candles = upbit_object.get_days_many_candles(bit)
    data = pd.DataFrame(candles)

    high_prices = data['high_price'].values
    low_prices = data['low_price'].values
    mid_prices = (high_prices + low_prices) / 2

    seq_len = 51

    result = []
    for index in range(len(mid_prices) - seq_len):
        result.append(mid_prices[index: index + seq_len])

    result = normalize_windows(result)

    # split train and test data
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()

    start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=10,
              epochs=20,
              callbacks=[
                  TensorBoard(log_dir='logs/%s' % (start_time)),
                  ModelCheckpoint('./models/%s_eth.h5' % (start_time), monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='auto'),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
              ])

    pred = model.predict(x_test)

    fig = plt.figure(facecolor='white', figsize=(50, 10))
    ax = fig.add_subplot(111)
    ax.plot(y_test, label='True')
    ax.plot(pred, label='Prediction')
    ax.legend()

    plt.xticks(np.arange(0, 100, step=5), ["x_{:0<2d}".format(x) for x in np.arange(0, 100, step=5)])
    plt.show()

    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
    return send_file(img, mimetype='image/png')


def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

# def main():
#     app.run(debug=True)

# if __name__ == '__main__':
#     main()
