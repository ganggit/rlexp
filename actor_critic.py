import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
import math

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0, steps=[1, 1, 1, 1]):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    loops = int(num_episodes/len(steps))
    index = 0
    step = 10
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        next_state = state # initialization
        action = 0
        episode = []
        if((i_episode)%loops==0):
           # get the step
           step = steps[index]
           print("the current index and stepsize  is %d and %d \n" % (index, steps[index]))
           index = index+1
        
        env.render()
        # One step in the environment
        for t in itertools.count():
            count = 0
            totalR = 0
            subepisode = []
            while(True):
                local_state = next_state
                # Take a step
                action_probs = estimator_policy.predict(local_state)
                
                # using the e-greedy algorithm
                nA = len(action_probs)
                epsilon = 0.1
                aProb = np.ones(nA, dtype=float) * epsilon / nA
                best_action = np.argmax(action_probs)
                aProb[best_action] += (1.0 - epsilon)
                # decide to use e-greedy or not
                next_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)   
                #next_action = np.random.choice(np.arange(len(action_probs)), p=aProb)    
                if(count ==0):
                    action = next_action
              
                next_state, reward, done, _ = env.step(next_action)
                totalR = totalR*discount_factor+reward
                # save the data
                subepisode.append(Transition(
                  state=local_state, action=next_action, reward=reward, next_state=next_state, done=action_probs))
                
                if done or count>=step:
                    break  
                count = count + 1

            # the out loop to keep the state    
            episode.append(Transition(
              state=state, action=action, reward=totalR, next_state=next_state, done=done))
            
            
            if(step>1 and len(subepisode)>0): 
              tmp, td_error, targets =  zoomUpdate(estimator_value, estimator_policy, subepisode, discount_factor, 0.5)
              idx = np.argmax(tmp)
              
              estimator_value.update(subepisode[idx][0], targets[idx])
              estimator_policy.update(subepisode[idx][0], td_error[idx], subepisode[idx][1])
            
            # Keep track of the transition
            # Update statistics
            stats.episode_rewards[i_episode] += totalR
            stats.episode_lengths[i_episode] = t
            
            '''
            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = totalR + (discount_factor**count) * value_next
            td_error = td_target - estimator_value.predict(state)
            
            # Update the value estimator
            estimator_value.update(state, td_target)
            
            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)
            '''
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break
            state = next_state

    return stats

def zoomUpdate(estimator_value, estimator_policy, subepisode, discount_factor, mix):
    num = len(subepisode)
    objval = np.zeros( num)
    tdiff = np.zeros(num)
    targets = np.zeros(num)
    diff = []
    esum = []
    for i in range(num):
        trans = subepisode[i]
        value = estimator_value.predict(trans[0])
        value_next = estimator_value.predict(trans[3])
        rc = value - discount_factor*value_next
        re = trans[2]
        diff.append( (re-rc)**2 )
        prob = trans[4]
        esum.append(entropy(prob))
        objval[i] = (re-rc)**2 + mix*(entropy(prob))
        target = re + discount_factor*value_next
        td = target - value
        tdiff[i] = td
        targets[i] = target
    return objval, tdiff, targets    
    

def entropy(prob):
    e = 0
    for p in prob:
        e = e - p*math.log(p, 2)
    return e    

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()


steps = [6, 6, 6, 3]

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, policy_estimator, value_estimator, 300, 1, steps)


plotting.plot_episode_stats(stats, smoothing_window=10)   
