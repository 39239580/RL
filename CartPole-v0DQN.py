# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:50:33 2019

@author: Administrator
"""

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
# 超参数
GAMMA = 0.9 # discount factor for target Q 折扣系数
INITIAL_EPSILON = 0.5 # starting value of epsilon  初始化贪婪系数
FINAL_EPSILON = 0.01 # final value of epsilon  最终的贪婪系数
REPLAY_SIZE = 10000 # experience replay buffer size   重放尺寸
BATCH_SIZE = 32 # size of minibatch 尺寸大小

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON #初始贪婪系数
    self.state_dim = env.observation_space.shape[0]  #state维度
    self.action_dim = env.action_space.n #action_space空间维度

    self.create_Q_network()  #创建Q网络
    self.create_training_method() #创建训练方法

    # Init session
    self.session = tf.InteractiveSession() #初始化
    self.session.run(tf.initialize_all_variables()) #初始化权重

  def create_Q_network(self): #创建Q网络
    # network weights  MLP 
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])  #输入为状态空间
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2 #输出为Q值 动作对应的价值

  def create_training_method(self):  #训练方法
  # y_input是TargetQ
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action)) #均方根误差
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):# S A R S DONE存储样本的
    #动作格式的转换，如[0,1]，输出动作为1，若[1,0],输出动作为0.
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))#将样本存储到重放缓存区域内
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft() #进行样本更新

    if len(self.replay_buffer) > BATCH_SIZE: #若重放区域缓存大小超过batch_size
      self.train_Q_network() #进行训练

  def train_Q_network(self):#训练网络
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    # 从重放缓存区域内获取 随机批量
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch] #状态
    action_batch = [data[1] for data in minibatch] #动作
    reward_batch = [data[2] for data in minibatch] #单步奖励
    next_state_batch = [data[3] for data in minibatch] #下个动作

    # Step 2: calculate y
    #计算y，计算出每个状态下的状态值
    y_batch = [] #计算积累奖励
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch}) #输出为动作的Q值。
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])  #若为终止状态
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i])) #寻找到最大的Q值

    self.optimizer.run(feed_dict={  #优化目标，Y,A.S
      self.y_input:y_batch, #traget_network
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  def egreedy_action(self,state):#贪婪动作
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      return np.argmax(Q_value)

    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

  def action(self,state): #输出动作
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode 限制每轮优秀的步长
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME) #初始化环境
  agent = DQN(env)

  for episode in range(EPISODE):
    # initialize task
    state = env.reset()  #初始化状态
    # Train
    for step in range(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)  #Env.step(action)    
      #获得next_state， reward，done
      # Define reward for agent
      reward = -1 if done else 0.1 #若结束为-1，否则为0.1  每一步的奖励
      agent.perceive(state,action,reward,next_state,done) #存放样本。
      state = next_state #状态更新
      if done: #若终止，跳出循环，进入下一步
        break
    # Test every 100 episodes
    if episode % 100 == 0: #每100次游戏
      total_reward = 0
      for i in range(TEST):#每100次训练进行一次测试
        state = env.reset() #重置状态
        for j in range(STEP): #
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      if ave_reward >= 200:
        break

if __name__ == '__main__':
  main()