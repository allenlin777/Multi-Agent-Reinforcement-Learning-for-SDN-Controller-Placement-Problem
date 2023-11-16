import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    Q_learning_Rewards = np.load("./Rewards/rewards.npy").transpose()
    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    plt.figure(figsize=(10, 5))
    plt.title('MARL')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
    plt.plot([i for i in range(3000)], Q_learning_avg,label='marl')
    plt.legend(loc="best")
    plt.savefig("./Graphs/marl.png")
    plt.show()
    plt.close()