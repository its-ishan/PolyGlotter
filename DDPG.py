import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import gym

env = gym.make('Pendulum-v1').unwrapped
state_shape = env.observation_space.shape[0]
print(f'state_shape: {state_shape}')

action_shape = env.action_space.shape[0]
print(f'action_shape: {action_shape}')

action_bound = [env.action_space.low, env.action_space.high]
print(f'action_bound: {action_bound}')

# Defining environment specific variables

gamma = 0.9
tau = 0.001
replay_buffer = 10000
batch_size=32

# Defining DDPG class

class DDPG(object):
    def __init__(self, state_shape, action_shape, high_action_value,):
        self.replay_buffer = np.zeros((replay_buffer, state_shape * 2 + action_shape + 1), dtype=np.float32)
        self.num_transition = 0
        self.sess = tf.compat.v1.Session()
        self.noise = 3.0
        self.state_shape, self.action_shape, self.high_action_value = state_shape, action_shape, high_action_value
        self.state = tf.compat.v1.placeholder(tf.float32, [None, state_shape], 'state')
        self.next_state = tf.compat.v1.placeholder(tf.float32, [None, state_shape], 'next_state')
        self.reward = tf.compat.v1.placeholder(tf.float32, [None, 1], 'reward')
        with tf.compat.v1.variable_scope('Actor'):
            self.actor = self.build_actor_network(self.state, scope='main', trainable=True)
            target_actor = self.build_actor_network(self.next_state, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            critic = self.build_critic_network(self.state, self.actor, scope='main', trainable=True)
            target_critic = self.build_critic_network(self.next_state, target_actor, scope='target', trainable=False)
        self.main_actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/main') #phi
        self.target_actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/target') #phi-dash
        self.main_critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/main') #theta
        self.target_critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/target') #theta-dash

        self.soft_replacement = [[tf.compat.v1.assign(phi_, tau*phi+(1-tau)*phi_), tf.compat.v1.assign(theta_, tau*theta+(1-tau)*theta_)] for phi, phi_, theta, theta_ in zip(self.main_actor_params, self.target_actor_params, self.main_critic_params, self.target_critic_params)]
        y = self.reward + gamma * target_critic
        MSE = tf.compat.v1.losses.mean_squared_error(labels=y, predictions=critic)
        self.train_critic = tf.compat.v1.train.AdamOptimizer(0.001).minimize(MSE, name='adam-ink', var_list=self.main_critic_params)
        actor_loss = -tf.reduce_mean(critic)
        self.train_actor = tf.compat.v1.train.AdamOptimizer(0.001).minimize(actor_loss, var_list = self.main_actor_params)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def select_action(self, state):
        action = self.sess.run(self.actor, {self.state: state[np.newaxis, :]})[0]
        action = np.random.normal(action, self.noise)
        action = np.clip(action, action_bound[0], action_bound[1])
        return action

    def train(self):
        self.sess.run(self.soft_replacement)
        indices = np.random.choice(replay_buffer, size=batch_size)
        batch_transition = self.replay_buffer[indices,:]
        batch_states = batch_transition[:, :self.state_shape]
        batch_actions = batch_transition[:, self.state_shape: self.state_shape + self.action_shape]
        batch_rewards = batch_transition[:, -self.state_shape -1: -self.state_shape]
        batch_next_state = batch_transition[:, -self.state_shape,:]

        self.sess.run(self.train_actor, {self.state: batch_states})
        self.sess.run(self.train_critic, {self.state: batch_states, self.actor: batch_actions, self.reward: batch_rewards, self.next_state: batch_next_state})

    def store_transition(self, state, actor, reward, next_state):
        trans = np.hstack((state, actor,[reward], next_state))
        index = self.num_transition % replay_buffer
        self.replay_buffer[index,:] = trans
        self.num_tansition +=1
        if self.num_transition > replay_buffer:
            self.noise *= 0.9995
            self.train()

    def build_actor_network(self, state, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            layer_1 = tf.compat.v1.layers.dense(state, 30, activation=tf.nn.tanh, name='layer_1', trainable=trainable)
            actor = tf.compat.v1.layers.dense(layer_1, self.action_shape, activation=tf.nn.tanh, name='actor', trainable=trainable)
            return tf.multiply(actor, self.high_action_value, name="scaled_a")

    def build_critic_network(self, state, actor, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            w1_s = tf.compat.v1.get_variable('w1_s', [self.state_shape, 30], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.action_shape, 30], trainable=trainable)

            b1 = tf.compat.v1.get_variable('b1', [1, 30], trainable=trainable)
            net = tf.nn.tanh(tf.matmul(state, w1_s)+ tf.matmul(actor, w1_a) + b1)

            critic = tf.compat.v1.layers.dense(net, 1, trainable=trainable)
            return critic


ddpg = DDPG(state_shape, action_shape, action_bound[1])
num_episodes = 300
num_timestamps = 500
for i in range(num_episodes):
    state = env.reset()
    Return = 0
    for t in range(num_timestamps):
        env.render()
        action = ddpg.select_action(state)
        next_state, reward, done, info = env.step(action)
        ddpg.store_transition(state, action, reward, next_state)
        Return +=reward
        if done:
            break

        state= next_state

        if i%10 ==0:
            print("Episode: {}, Return: {}".format(i, Return))




