# from upbit import Upbit
#
# from flask import Flask, send_file, render_template, make_response, request
#
# from io import BytesIO, StringIO
# from functools import wraps, update_wrapper
# from datetime import datetime
# import matplotlib
#
# # import tensorflow.compat.v1 as tf
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# #################
#
# app = Flask(__name__)
# upbit = Upbit()
#
# result_corr = 0.0
#
#
# @app.route('/')
# def normal():
#     return render_template("index.html", corr=result_corr, width=800, height=600)
#
#
# def nocache(view):
#     @wraps(view)
#     def no_cache(*args, **kwargs):
#         response = make_response(view(*args, **kwargs))
#         response.headers['Last-Modified'] = datetime.now()
#         response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#         response.headers['Pragma'] = 'no-cache'
#         response.headers['Expires'] = '-1'
#         return response
#
#     return update_wrapper(no_cache, view)
#
#
# @app.route('/fig/<bit1_bit2>')
# @nocache
# def fig(bit1_bit2):  # TODO: fig에서 분석한 데이터를 normal로 보내서 render_templates에 뿌려야한다.
#     bit1, bit2 = bit1_bit2.split("_")
#     bit1_candles = upbit.get_days_candles(bit1)
#     bit2_candles = upbit.get_days_candles(bit2)
#
#     if bit1_candles is None:
#         return 'invalid market: {}'.format(bit1_candles)
#     elif bit2_candles is None:
#         return 'invalid market: {}'.format(bit2_candles)
#
#     bit1_xs, bit1_dataset = get_bit_data(bit1_candles)
#     bit2_xs, bit2_dataset = get_bit_data(bit2_candles)
#
#     plt.plot(bit1_xs, bit1_dataset, label=bit1)
#     plt.plot(bit2_xs, bit2_dataset, label=bit2)
#
#     # TODO : 원하는 코인 수 만큼 화면에 띄울 수 있도록 유지보수 할 것
#     plt.legend()
#
#     data_frame1 = data = [[], [bit1_dataset], []]
#     data_frame2 = data = [[], [bit1_dataset], []]
#
#     data1 = pd.DataFrame(data_frame1, columns=["high", "Low"], dtype=np.int8)
#     data2 = pd.DataFrame(data_frame2, dtype=np.int8)
#
#     # result_corr = np.corrcoef(bit1_dataset, bit2_dataset)
#
#     img = BytesIO()
#     plt.savefig(img, format='png', dpi=300)
#     img.seek(0)  ## object를 읽었기 때문에 처음으로 돌아가줌
#     return send_file(img, mimetype='image/png')
#
#
# def get_bit_data(bit):
#     i = 1
#     bit_xs = []
#     bit_dataset = []
#     for candle in bit:
#         bit_xs.append(i)
#         bit_dataset.append(candle['change_rate'])
#         i += 1
#
#     return bit_xs, bit_dataset
#
#
# # @app.route('/forecast')
# def forecast(data):  # TODO: 비트코인 주가 예측
#     high_prices = data['High'].values
#     low_prices = data['Low'].values
#     mid_prices = (high_prices + low_prices) / 2
#     pass
#
#
# # def main():
# #     app.run(debug=True)
#
# # if __name__ == '__main__':
# #     main()
