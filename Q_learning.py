# coding=utf-8
import gym
import numpy as np

env = gym.make('FrozenLake-v0') #加载实验环境
print("Agent所处的环境")
env.render()  #输出4*4网络表格
rList = [] #记录奖励
lr = 0.85
y =  0.99     #权重
num_episodes = 5000  #训练次数
Q = np.zeros([env.observation_space.n,env.action_space.n]) #初始化Q表

def test(i):
    d1 = False
    j1 = 0
    start = 0
    r_sum = 0
    while d1 == False:
          j1 +=1
          a = np.argmax(Q[start,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
          s1,r,d1,_ = env.step(a)
          start = s1
          r_sum +=r
    if(r_sum == 1.0):
       print('达到终点，移动的次数',j1)
    else:
       print('未达到终点')


def Q_learning(env,lr,y):
    for i in range(num_episodes):
        s = env.reset()  #重置环境
        rAll = 0
        d = False
        j = 0

        # if (i % 1000) ==0:  #测试Q表
        #    test(i)

        while j < 99:  #Q-Table算法
              j+=1
              a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) #基于贪心法选择action
              s1,r,d,_ = env.step(a)  #获取新状态和奖励
              Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])  #更新Q表
              rAll += r
              s = s1 #更新状态
              if d == True:  #如果到达目标格
                 break

        rList.append(rAll)
    return (sum(rList) / num_episodes * 100)


Q_learning(env,lr,y)
print("Q-learning")
print("正确率: " +  str(sum(rList)/num_episodes*100) + "%")
# print("得到的Q-Table")
# print(Q)  #输出Q-Table