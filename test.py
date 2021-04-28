import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

#W는 Weight(가중치). -100에서 100사이의 랜덤값  b는 bias y절편
W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))

#placeholder라는 하나의 틀
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#직선이라는 가설식
H = W*X + b

#비용함수 코스트는 (이상치-실제값)의 제곱의 평균
cost = tf.reduce_mean(tf.square(H-Y))
#(제일중요)경사 하강 알고리즘에서 한번에 얼만큼 점프할건지. step의 크기.
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)

#비용함수를 가장 적게 만드는 방법으로 학습.
train = optimizer.minimize(cost)

#변수 초기화
init = tf.global_variables_initializer()

#세션
sess = tf.Session()
sess.run(init)

#여기가 실제로 학습이 일어나는 부분
for i in range(5001):
    sess.run(train, feed_dict = {X : xData, Y : yData})
    # if i & 500 ==0:
        # print(i, sess.run(cost, feed_dict = {X:xData, Y:yData}), sess.run(W),sess.run(b))


print (sess.run(H, feed_dict={X : [8]}))

