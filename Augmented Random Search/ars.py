#Import libraries.
import os
import numpy as np
import gym
from gym import wrappers

#Create a constant with the name of the enviroment to easily replace it if we would like to use another env.
ENV_NAME = 'BipedalWalker-v2'

#Create a class with all the hyperparameters to instanciate an object later.
class Hp():
    # Hyperparameters that the object will have if we dont define any new.
    def __init__(self,
                 nb_steps=1000,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env_name='BipedalWalker-v2',
                 record_every=50):

        #Number of training loops.
        self.nb_steps = nb_steps
        #Number of steps the agent can perform in a loop.
        self.episode_length = episode_length
        #Hyperparameter 'alpha', refers to how frequently we update the weights in each iterations.
        self.learning_rate = learning_rate
        #Hyperparameter 'delta', refers to the change from one value to antoher, also called increment or decrement; useful to know if algorithm is converging.
        self.num_deltas = num_deltas
        #The best 'delta'
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        #This is the strenght of the noise variation.
        self.noise = noise
        #The function random generates a number based on another one called a 'seed', we used this parameter to generate random numbers.
        self.seed = seed
        #The enviroment name from OpenAI's gym.
        self.env_name = env_name
        #If we actually want to record every training loop we'll take much more time to train, we record every set number of steps so we
        # can have less time of training but still see our algorithm converge. 
        self.record_every = record_every


#Class to instancitate an object that normalizes our inputs to get us and output between 0 and 1, which speeds up convergence.
class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        #Total number of states.
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        #Variance, a math formula.
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


#This is a class to instanciate a policy object that we need for our agent to converge.
# This happens at the moment of explorations of the enviroment by selecting some actions in states
# and keep selecting the actions that output the best reward(exploitation).
class Policy():
    def __init__(self, input_size, output_size, hp):
        #Matrix of weights.
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step


#This class is the core of the program, this is the class that creates an object to train the agent.
class ArsTrainer():
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 monitor_dir=None):

        #Sets all the parameters to the respective values typed, or difines them with objects.
        self.hp = hp or Hp()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.env.spec.timestep_limit or self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        self.record_video = False

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.hp.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self):
        for step in range(self.hp.nb_steps):
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.hp.num_deltas
            negative_rewards = [0] * self.hp.num_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(self.hp.num_deltas):
                positive_rewards[k] = self.explore(direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(direction="-", delta=deltas[k])
                
            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.hp.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, sigma_rewards)

            # Only record video during evaluation, every n steps
            if step % self.hp.record_every == 0:
                self.record_video = True
                np.savetxt("weights.csv", self.policy.theta, delimiter=",")
            # Play an episode with the new weights and print the score
            reward_evaluation = self.explore()
            print('Step: ', step, 'Reward: ', reward_evaluation)
            self.record_video = False

#Function for returning a PATH in the computer's file system
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Main code
if __name__ == '__main__':
    #This two lines are 
    videos_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(videos_dir, ENV_NAME)
    hp = Hp(seed=1946, env_name=ENV_NAME)
    trainer = ArsTrainer(hp=hp, monitor_dir=monitor_dir)
    trainer.train()

