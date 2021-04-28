import logging
from upbit import Upbit

from flask import Flask, send_file, render_template, make_response, request

from io import BytesIO, StringIO
import numpy as np

## macOS의 경우 아래 순서에 따라서 library를 import해줘야 에러없이 잘 됩니다.
import matplotlib

# import tensorflow.compat.v1 as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt

#################

app = Flask(__name__)
upbit = Upbit()
upbit.get_hour_candles('KRW-BTC')


@app.route('/')
def normal():
    return render_template("index.html", width=800, height=600)


@app.route('/fig/<market>')  # TODO: xs, ys 구해서 그래프로 만들어야함
def fig(market):
    if market is None or market == '':
        return 'No market parameter'

    candles = upbit.get_hour_candles(market)
    if candles is None:
        return 'invalid market: {}'.format(market)

    xs = []
    i = 1
    dataset = []

    for candle in candles:
        xs.append(i)
        dataset.append(candle['trade_price'])
        i += 1

    plt.plot(xs, dataset, 'r-', label=market)
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
    return send_file(img, mimetype='image/png')


# plt.savefig(img, format='svg')
# return send_file(img, mimetype='image/svg')


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
