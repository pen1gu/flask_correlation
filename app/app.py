import logging
from upbit import Upbit

from flask import Flask, send_file, render_template, make_response, request

from io import BytesIO, StringIO
from functools import wraps, update_wrapper
from datetime import datetime
## macOS의 경우 아래 순서에 따라서 library를 import해줘야 에러없이 잘 됩니다.
import matplotlib

# import tensorflow.compat.v1 as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#################

app = Flask(__name__)
upbit = Upbit()

result_corr = 0.0


@app.route('/')
def normal():  # TODO: 아래 corr 변수에 상관도 분석 값을 넣어야한다.
    return render_template("index.html", corr=result_corr, width=800, height=600)


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


@app.route('/fig/<btc_eth>')
@nocache
def fig(btc_eth):  # TODO: fig에서 분석한 데이터를 normal로 보내서 render_templates에 뿌려야한다.
    btc, eth = btc_eth.split("_")
    btc_candles = upbit.get_days_candles(btc)
    eth_candles = upbit.get_days_candles(eth)

    if btc_candles is None:
        return 'invalid market: {}'.format(btc_candles)
    elif eth_candles is None:
        return 'invalid market: {}'.format(eth_candles)

    i = 1
    btc_xs = []
    btc_dataset = []
    for candle in btc_candles:
        btc_xs.append(i)
        btc_dataset.append(candle['change_rate'])
        i += 1
    plt.plot(btc_xs, btc_dataset, 'b-', label=btc)

    i = 1
    eth_xs = []
    eth_dataset = []
    for candle in eth_candles:
        eth_xs.append(i)
        eth_dataset.append(candle['change_rate'])
        i += 1

    plt.plot(eth_xs, eth_dataset, 'r-', label=eth)
    plt.legend()

    result_corr = np.corrcoef(btc_dataset, eth_dataset)

    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
    return send_file(img, mimetype='image/png')


@app.route('/corr')
def corr():  # TODO: 상관도 분석 완료. 값만 화면 딴에 전달하면 끝
    plt.plot(result_corr, 'r-', label='')

    img = BytesIO()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
    return send_file(img, mimetype='image/png')


# TODO: 상관도 분석 결과 -> 전체 1 나왔음 : 에러


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()
