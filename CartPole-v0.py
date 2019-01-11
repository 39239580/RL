# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:45:34 2019

@author: Administrator
"""

import numpy as np
import tensorflow as tf
import gym

# 创建CartPole问题的环境env
env = gym.make('CartPole-v0')

# 先测试在CartPole环境中使用随机Action的表现，作为接下来对比的baseline。
env.reset() # 初始化环境
random_episodes = 0
reward_sum = 0 #奖励和
while random_episodes < 10:
    env.render() # 将CartPole问题的图像渲染出来
    observation, reward, done, _ = env.step(np.random.randint(0, 2)) #执行0,1
    reward_sum += reward  #总得奖励
    if done: #若完成10次
        random_episodes += 1 #次数+1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()

# 我们的策略网络使用简单的带有一个隐含层的MLP。
H = 50 # 隐含层节点数
batch_size = 25
learning_rate = 1e-1
D = 4 # 环境信息observation的维度为4，小车位置，速度，角速度。
gamma = 0.99 # reward的discount比例

observations = tf.placeholder(tf.float32, [None, D], name='input_x')
w1 = tf.get_variable('w1', shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, w1))
w2 = tf.get_variable('w2', shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, w2)
probability = tf.nn.sigmoid(score)

# 这里模型的优化器使用Adam算法。我们分别设置两层神经网络参数的梯度：W1Grad和W2Grad，并使用
# adam.apply_gradients定义我们更新模型参数的操作updateGrads。之后计算参数的梯度，当累积到一
# 定样本量的梯度，就传入W1Grad和W2Grad，并执行updateGrads更新模型参数。我们不逐个样本地更新
# 参数，而是累计一个batch_size的样本的梯度再更新参数，防止单一样本随机扰动的噪声对模型带来
# 不良影响
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name='batch_grad1')
W2Grad = tf.placeholder(tf.float32, name='batch_grad2')
tvars = tf.trainable_variables()
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# 下面定义discount_rewards,用来估算每一个Action对应的潜在价值discount_r
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 我们定义人工设置的虚拟label的placeholder---input_y,以及每个Action的潜在价值的placeholder---
# advangtages.

input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal") #累积奖励
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))#概率的对数。
loss = -tf.reduce_mean(loglik * advantages) #目标函数
newGrads = tf.gradients(loss, tvars)

xs, ys, drs = [], [], [] #环境列表， label列表，每个action 的奖励
reward_sum = 0 #累计额奖励
episode_number = 1 #初始化实验次数
total_episodes = 10000 #总得实验次数

with tf.Session() as sess:
    rendering = False #不开渲染。
    init = tf.global_variables_initializer() #初始化全局变量
    sess.run(init) #运行初始化
    obervation = env.reset() #初始化环境

    gradBuffer = sess.run(tvars) #计算参数的梯度
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0#参数全部出事为0.

    while episode_number <= total_episodes:
        if reward_sum / batch_size > 100 or rendering == True: #平均奖励大于100对环境进行渲染
            env.render()
            rendering = True
        x = np.reshape(observation, [1, D]) #将环境支撑，1，d矩阵，

        tfprob = sess.run(probability, feed_dict={observations: x}) #获取网络输出概率。输出action为1的概率
        action = 1 if np.random.uniform() < tfprob else 0# 在(0,1)之间随机抽样，若小于这个概率，取1，若大于取0
        xs.append(x)
        y = 1 - action
        ys.append(y) #产生lable表，

        observation, reward, done, info = env.step(action) #执行动作产生四个表，
        reward_sum += reward #计算累计额奖励
        drs.append(reward)
        if done: #若结束
            episode_number += 1 #游戏次数加1
            epx = np.vstack(xs) #堆栈数据  一次实验获取的所有列表
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []  #清空列表

            discounted_epr = discount_rewards(epr)  #计算每一步潜在价值
            discounted_epr -= np.mean(discounted_epr) #计算均值
            discounted_epr /= np.std(discounted_epr) #计算表准差

            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy,
                                                  advantages: discounted_epr})  #获取梯度
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0: #每25次计算一次梯度梯度，
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Average reward for episode %d: %f.' % (episode_number, reward_sum / batch_size))

                if reward_sum / batch_size > 200:
                    print('Task solved in', episode_number, 'episodes!')
                    break

                reward_sum = 0
            observation = env.reset()
