import os
import numpy as np
import random
from tqdm import tqdm
from env import Environment

total_reward = []
times=5000
decay=-4/times
#self.count or total_count?
class Agent():
    def __init__(self, env, epsilon=1, learning_rate=0.7, gamma=0.75):
        self.env=env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.qtable = np.zeros((env.observation_space, env.observation_space, env.n+1))

    def choose_action(self, state):
        if random.random()<self.epsilon:
            action=random.randint(0,self.env.n)
        else:
            state=tuple(state)
            action=np.argmax(self.qtable[state])
        return action

    def learn(self, state, action, reward, next_state):
        state=tuple(state)
        next_state=tuple(next_state)
        self.qtable[state][action]=(1-self.learning_rate)*self.qtable[state][action]+self.learning_rate*(reward+self.gamma*max(self.qtable[next_state]))

def train(env):
    training_agent=[]
    for i in range(env.n):
        training_agent.append(Agent(env))
    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        env.reset()
        count = 0
        done_cnt=0
        # print()
        # print(env.x) 
        while True:
            p=np.random.randint(0,env.n)
            state=env.extract_state(p)
            # print(p)
            training_agent[p].epsilon=0.05+(0.95)*np.exp(decay*done_cnt)
            action = training_agent[p].choose_action(state)
            next_state, reward= env.step(action,p)
            # print(action)
            # print(next_state)
            # print(reward)
            training_agent[p].learn(state, action, reward, next_state)
            state = next_state
            count += reward
            done_cnt+=1
            if done_cnt>times:
                # print(env.x)
                # os.system("pause")
                break
        rewards.append(count/times)

    for i in range(env.n):
        np.save("./Tables/table"+str(i)+".npy",training_agent[i].qtable)
    total_reward.append(rewards)

def test(env):
    testing_agent=[]
    for i in range(env.n):
        testing_agent.append(Agent(env))
    for i in range(env.n):
        testing_agent[i].qtable = np.load("D:\Tables/table"+str(0)+".npy")
    rewards = []
    
    for _ in range(100):
        env.reset()
        count = 0
        done_cnt=0
        # print("origin")
        # print(env.x)
        while True:
            p=np.random.randint(0,env.n)
            state=env.extract_state(p)

            state=tuple(state)
            action = np.argmax(testing_agent[p].qtable[state])
            next_state, reward= env.step(action,p)
            state = next_state
            count += reward
            done_cnt+=1
            if done_cnt>times:
                # print("after")
                print(env.uti())
                break
        rewards.append(count/times)

    print(f"average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    SEED = 20

    env = Environment()
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
        
    #train(env)   
    test(env)
        
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/rewards.npy", np.array(total_reward))