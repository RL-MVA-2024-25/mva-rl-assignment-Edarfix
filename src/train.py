from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import gzip
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
    #env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.env = env
        self.nb_actions = 4
        self.path = './Q.gz'
        self.Q = None
        self.Q_not_fitted = True
        if os.path.exists(self.path):
            self.Q_not_fitted = False
        else:
            self.Q = RandomForestRegressor()
            self.train(epoch = 10, nb_iter = 12, horizon = 1000,gamma = 0.9,eps = 0.1)
        
    def act(self, observation, use_random=False):
        if use_random or self.Q_not_fitted:
            return self.env.action_space.sample()
        else:
            Qsa = []
            for a in range(self.nb_actions):
                sa = np.append(observation,a).reshape(1, -1)
                Qsa.append(self.Q.predict(sa)[0])
            return np.argmax(Qsa)
    
    def save(self,path=None):
        if path is None:
            path = self.path
        try:
            with gzip.open(path, 'wb') as file:  # Utilisation de gzip pour la compression
                pickle.dump(self.Q, file)
            print(f"Q values successfully saved to {path}")
        except Exception as e:
            print(f"An error occurred while saving Q values: {e}")

    def load(self):
        try:
            with gzip.open(self.path, 'rb') as file:  # Utilisation de gzip pour décompresser
                self.Q = pickle.load(file)
            print(f"Q values successfully loaded from {self.path}")
        except FileNotFoundError:
            print(f"No file found at {self.path}. Unable to load Q values.")
        except Exception as e:
            print(f"An error occurred while loading Q values: {e}")

    def collect_samples(self,env, horizon = 200, eps = 0.1):
        s, _ = env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in range(horizon):
            a = self.act(observation=s, use_random = (np.random.random() < eps))
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def train(self, epoch = 3, nb_iter = 10, horizon= 200,gamma = 0.9,eps = 0.1):
        self.Q = RandomForestRegressor()
        for e in range(epoch):
            if e > 5:
                eps = 0.05
                gamma = 0.95
            print(f"Epoch {e+1}/{epoch}")
            S, A, R, S2, D = self.collect_samples(self.env, horizon = horizon,eps = 0.2)
            SA = np.concatenate((S,A),axis=1)
            # now we add new datetime
            for nb in tqdm(range(nb_iter)):
                S_prime,A_prime,R_prime,S2_prime,D_prime = self.collect_samples(env, horizon = horizon,eps = eps)
                SA_prime = np.concatenate((S2_prime,A_prime),axis=1)
                S , A , R , S2 , D = np.vstack((S,S_prime)), np.vstack((A,A_prime)), np.hstack((R,R_prime)), np.vstack((S2,S2_prime)), np.hstack((D,D_prime))
                SA = np.append(S,A,axis=1)
                if nb == 0:
                    value = R.copy()
                else:
                    Q2 = np.zeros((S.shape[0],self.nb_actions))
                    for a2 in range(self.nb_actions):
                        A2 = a2*np.ones((S.shape[0],1))
                        S2A2 = np.append(S2,A2,axis=1)
                        Q2[:,a2] = self.Q.predict(S2A2)
                    max_Q2 = np.max(Q2,axis=1)
                    value = R + gamma*(1-D)*max_Q2
                    
                self.Q.fit(SA,value)
                self.Q_not_fitted = False
            self.save(f"./Q_epoch{e+1}.gz")
        print("Training completed")
        self.save()