import os
import numpy as np
import random
from tqdm import tqdm
from env import Environment

total_reward = []
times=1000

class Co():
    def _init_(self,epsilon=1):
        self.epsilon = epsilon

class Agent():
    def __init__(self, env, learning_rate=0.7, gamma=0.75):

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.qtable = np.zeros((env.action_space, env.action_space))

    def choose_action(self, state,epsilon):
        if random.random()<epsilon:
            action=random.randint(0,2047)
        else:
            action=np.argmax(self.qtable[state])
        return action

    def learn(self, state, action, reward, next_state):
        self.qtable[state][action]=(1-self.learning_rate)*self.qtable[state][action]+self.learning_rate*(reward+self.gamma*max(self.qtable[next_state]))

def train(env):
    coo=Co()
    training_agent = [Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env)]    
    #training_agent = Agent(env)

    episode = 3000
    rewards = []
    for ep in tqdm(range(episode)):
        state, p = env.reset()
        count = 0
        cnt=0
        # print()
        # print(env.x) 
        while True:
            coo.epsilon=0.05+(0.95)*np.exp(-0.004*cnt)
            if cnt>0:
                p=np.random.randint(0,env.n)
                state=env.extract_state(p)
            # print(p)
            action = training_agent[p].choose_action(state,coo.epsilon)
            next_state, reward= env.step(action,p)
            # print(action)
            # print(next_state)
            # print(reward)
            training_agent[p].learn(state, action, reward, next_state)
            count += reward

            state = next_state

            cnt+=1
            if cnt>times:
                # print(env.x)
                # os.system("pause")
                break
        rewards.append(count)

    np.save("D:/Tables/taxi_table0.npy", training_agent[0].qtable)
    np.save("D:/Tables/taxi_table1.npy", training_agent[1].qtable)
    np.save("D:/Tables/taxi_table2.npy", training_agent[2].qtable)
    np.save("D:/Tables/taxi_table3.npy", training_agent[3].qtable)
    np.save("D:/Tables/taxi_table4.npy", training_agent[4].qtable)
    np.save("D:/Tables/taxi_table5.npy", training_agent[5].qtable)
    np.save("D:/Tables/taxi_table6.npy", training_agent[6].qtable)
    np.save("D:/Tables/taxi_table7.npy", training_agent[7].qtable)
    np.save("D:/Tables/taxi_table8.npy", training_agent[8].qtable)
    np.save("D:/Tables/taxi_table9.npy", training_agent[9].qtable)
    np.save("D:/Tables/taxi_table10.npy", training_agent[10].qtable)
    total_reward.append(rewards)

def test(env):
    coo=Co()
    #testing_agent = Agent(env)
    testing_agent = [Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env),Agent(env)]
    testing_agent[0].qtable = np.load("D:/Tables/taxi_table0.npy")
    testing_agent[1].qtable = np.load("D:/Tables/taxi_table1.npy")
    testing_agent[2].qtable = np.load("D:/Tables/taxi_table2.npy")
    testing_agent[3].qtable = np.load("D:/Tables/taxi_table3.npy")
    testing_agent[4].qtable = np.load("D:/Tables/taxi_table4.npy")
    testing_agent[5].qtable = np.load("D:/Tables/taxi_table5.npy")
    testing_agent[6].qtable = np.load("D:/Tables/taxi_table6.npy")
    testing_agent[7].qtable = np.load("D:/Tables/taxi_table7.npy")
    testing_agent[8].qtable = np.load("D:/Tables/taxi_table8.npy")
    testing_agent[9].qtable = np.load("D:/Tables/taxi_table9.npy")
    testing_agent[10].qtable = np.load("D:/Tables/taxi_table10.npy")
    rewards = []
    
    for _ in range(100):
        state, p = env.reset()
        count = 0
        cnt=0
        print("origin")
        print(env.x)
        while True:
            if cnt>0:
                p=np.random.randint(0,env.n)
                state=env.extract_state(p)
            action = np.argmax(testing_agent[p].qtable[state])
            next_state, reward= env.step(action,p)
            count += reward
            state = next_state

            cnt+=1
            if cnt>times:
                print("after")
                print(env.x)
                break
        rewards.append(count)

    print(f"average reward: {np.mean(rewards)}")


if __name__ == "__main__":
    SEED = 20

    env = Environment()
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    if not os.path.exists("D:/Tables"):
        os.mkdir("D:/Tables")
        
    train(env)   
    test(env)
        
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/taxi_rewards.npy", np.array(total_reward))

    env.close()