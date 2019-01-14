# %matplotlib inline
# from lib import plotting
# from lib.envs.cliff_walking import CliffWalkingEnv
import gym
import itertools
import matplotlib
import numpy as np
import math
import sys
import tensorflow as tf
import collections
import gym
import pickle

import plotting

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

if "../" not in sys.path:
    sys.path.append("../")

# matplotlib.style.use('ggplot')

env = gym.make('MountainCar-v0')
# model initialization
D = 2  # input dimensionality
C = 3  # class number

env.observation_space.sample()
observation_examples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform(state)
    featurized = featurizer.transform(scaled)
    return featurized  # featurized[0]


class PolicyEstimator:
    """
    Policy Function approximator.
    """

    def __init__(self, state_dim=4, action_dim=2, learning_rate=0.001, tau=0.001):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.state, self.action_probs = self.create_actor()
        self.network_params = tf.trainable_variables()

        self.target_state, self.target_action_probs = self.create_actor()
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # self.action_probs = tf.nn.softmax(self.output_layer)

        self.action = tf.placeholder(
            tf.int32, [None, 1],  name="action")
        self.picked_action_prob = tf.batch_gather(
            self.action_probs, self.action)

        self.target = tf.placeholder(
            tf.float32, [None, 1], name="target")
        # Loss and train op
        self.loss1 = tf.reduce_mean(-
                                    tf.log(tf.clip_by_value(self.picked_action_prob, 1e-10, 1.0)) * self.target)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        self.train_op1 = self.optimizer.minimize(
            self.loss1, global_step=tf.contrib.framework.get_global_step())

        self.tderror = tf.placeholder(
            tf.float32, [None, 2], name="target")
        self.loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.action_probs, labels=self.tderror))
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

        # specify the gradient by my self
        grads = tf.gradients(self.loss1, self.network_params)
        max_grad_norm = 0.5
        alpha = 0.99
        epsilon = 1e-5
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, self.network_params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate, decay=alpha, epsilon=epsilon)
        self.gradtrainer = trainer.apply_gradients(grads)

    def create_actor(self):
        state = tf.placeholder(
            tf.float32, [None, self.state_dim], "state")
        # This is just table lookup estimator
        x = tf.contrib.layers.fully_connected(
            inputs=state,
            num_outputs=24,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.uniform_unit_scaling_initializer(-0.05, 0.05))

        output_layer = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=env.action_space.n,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.uniform_unit_scaling_initializer(-0.05, 0.05))
        probs = tf.clip_by_value(output_layer, 1e-10, 1.0)
        return state, probs

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state,
                     self.target: target, self.action: action}
        _, loss = sess.run([self.train_op1, self.loss1], feed_dict)
        return loss

    def gradupdate(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state,
                     self.target: target, self.action: action}
        _, loss = sess.run([self.gradtrainer, self.loss1], feed_dict)
        return loss

    def update2(self, state, tderror, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.tderror: tderror}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def predict_target(self, state, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.target_action_probs, feed_dict={
            self.target_state: state
        })

    def update_target_network(self, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, state_dim=4, action_dim=2, learning_rate=0.005, tau=0.001, num_actor_vars=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        # Create the critic network
        self.state,  self.out = self.create_critic(
            "s1", "t1")
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_state,  self.target_out = self.create_critic(
            "s2", "t2")

        self.target_network_params = tf.trainable_variables(
        )[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.target = tf.placeholder(
            tf.float32, [None, 1], name="target_name")
        # self.loss = tf.reduce_mean(tf.squared_difference(
        #    self.value_estimate, self.target))
        # self.loss = tf.reduce_mean(
        #    tf.square(tf.subtract(self.target, self.out)))
        self.loss = tf.nn.l2_loss(self.target - self.out)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

    def create_critic(self, state_name, target_name):
        state = tf.placeholder(
            tf.float32, [None, self.state_dim], name=state_name)

        # This is just table lookup estimator
        x = tf.contrib.layers.fully_connected(
            inputs=state,
            num_outputs=24,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer(0, 0.1))

        output_layer = tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(0, 0.1))

        return state, output_layer

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.out, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = {self.state: state, self.target: target}
        _, loss, v = sess.run([self.train_op, self.loss, self.out], feed_dict)
        return loss, v

    def predict_target(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.target_out, feed_dict={
            self.target_state: state
        })

    def update_target_network(self, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_target_network_params)


def zoomUpdate(estimator_value, estimator_policy, subepisode, discount_factor, mix):

    num = len(subepisode)
    objval = np.zeros(num)
    tdiff = np.zeros(num)
    targets = np.zeros(num)
    diff = []
    esum = []

    for i in range(num):
        trans = subepisode[i]
        value = estimator_value.predict(trans[0])
        value_next = estimator_value.predict(trans[3])
        # if trans[4]:
        #    value_next = 0
        rc = value - discount_factor*value_next
        re = trans[2]
        diff.append((re-rc)**2)
        prob = trans[5]
        esum.append(entropy(prob))
        objval[i] = (re-rc)**2 + mix*(entropy(prob))
        target = re + discount_factor*value_next
        td = target - value
        tdiff[i] = td
        targets[i] = target
    return objval, tdiff, targets


def computeReward(rewards, discount):
    val = 0
    factor = 1
    for r in rewards:
        r = r * factor
        val = val + r
        factor = factor * discount
    return val


def entropy(prob):
    e = 0

    for p in prob:
        e = e - p*math.log(p, 2)
    '''
    try:
        for p in prob:
            e = e - p*math.log(p, 2)
    except:
        print("except happen: {}".format(prob))
    '''
    return e


def actor_critic(env, estimator_policy, estimator_value, num_episodes, steps=[1, 1, 1, 1], discount_factor=1.0):
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

    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done", "probs", "index"])

    StateInfo = collections.namedtuple(
        "Transition", ["state", "action", "reward",  "index",  "next_state"])
    loops = int(num_episodes/len(steps))
    index = 0
    step = 10

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        state = np.reshape(state, (1, D))
        next_state = state  # initialization
        action = 0
        episode = []

        if((i_episode) % loops == 0):
            # get the step
            step = steps[index]
            print("the current index and stepsize  is %d and %d \n" %
                  (index, steps[index]))
            index = index+1

        # env.render()
        rewardList = []
        replay = []
        gid = 0
        # One step in the environment
        for t in itertools.count():
            count = 1
            R = []
            subepisode = []

            while(True):
                local_state = next_state
                # Take a step
                action_probs = estimator_policy.predict(local_state)

                # decide to use e-greedy or not
                '''
                eps = 0.9
                a_max = np.argmax(action_probs[0])
                policy = np.ones(estimator_policy.action_dim) * \
                    eps / estimator_policy.action_dim
                policy[a_max] += 1. - eps
                next_action = np.random.choice(
                    np.arange(len(action_probs[0])), p=policy)
                '''
                next_action = np.random.choice(
                    np.arange(len(action_probs[0])), p=action_probs[0])
                # next_action = np.random.choice(np.arange(len(action_probs)), p=aProb)
                next_state, reward, done, _ = env.step(next_action)
                next_state = np.reshape(next_state, (1, D))
                if(count == 1):
                    action = next_action
                    replay.append(StateInfo(
                        state=state, action=action, reward=reward, index=gid, next_state=next_state))
                # if done and i_episode<400:
                #   reward = -20
                R.append(reward)
                rewardList.append(reward)
                # save the data
                subepisode.append(Transition(
                    state=local_state, action=next_action, reward=reward, next_state=next_state, done=done, probs=action_probs[0], index=gid))

                gid = gid + 1
                if done or count >= step:
                    break

                count = count + 1

            totalR = computeReward(R, discount_factor)
            # the out loop to keep the state
            # episode.append(Transition(
            #    state=state, action=action, reward=totalR, next_state=next_state, done=done, probs=action_probs[0], index=count))
            idx = 0
            if(step > 1 and len(subepisode) > 0):
                tmp, td_error, targets = zoomUpdate(
                    estimator_value, estimator_policy, subepisode, discount_factor, 0.5)
                idx = np.argmax(tmp)

                estimator_value.update(
                    subepisode[idx][0], np.reshape(targets[idx], (-1, 1)))
                estimator_policy.gradupdate(subepisode[idx][0], np.reshape(td_error[idx], (-1, 1)), np.reshape(
                    subepisode[idx][1], (-1, 1)))

                # Keep track of the transition
                replay.append(StateInfo(
                    state=subepisode[idx][0], action=subepisode[idx][1], reward=subepisode[idx][2], index=subepisode[idx][6], next_state=subepisode[idx][3]))
                subid = len(subepisode)-1
                if subepisode[subid][4] and idx != subid:
                    replay.append(StateInfo(
                        state=subepisode[subid][0], action=subepisode[subid][1], reward=subepisode[subid][2], index=subepisode[subid][6], next_state=subepisode[subid][3]))
            # Update statistics
            stats.episode_rewards[i_episode] += sum(R)
            stats.episode_lengths[i_episode] = gid

            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            if done:
                value_next = 0
            td_target = totalR + (discount_factor**count) * value_next
            td_error = td_target - estimator_value.predict(state)

            # Update the value estimator
            #estimator_value.update(state, np.reshape(td_target, (-1, 1)))
            # Update the policy estimator
            # using the td error as our advantage estimate
            # estimator_policy.update(state, np.reshape(
            #    td_error, (-1, 1)), np.reshape(action, (-1, 1)))

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

        # for the replay buffer
        advantages = []
        update_vals = []
        states = np.zeros((len(replay), D))
        actions = []
        num = len(replay)
        for i in range(0, len(replay)):
            state, action, reward, cid, _ = replay[i]
            lastid = replay[num-1][3]+1
            # calculate discounted monte-carlo return
            future_reward = 0
            factor = 1
            for ii in range(cid, lastid):
                future_reward = future_reward + rewardList[ii]*factor
                factor = factor*discount_factor

            currentval = estimator_value.predict(state)
            # advantage: how much better was this action than normal
            advantages.append(future_reward - currentval)
            # update the value function towards new return
            update_vals.append(future_reward)
            actions.append(action)
            states[i] = state
        # Update the value estimator
        estimator_value.update(states, np.reshape(
            update_vals, (num, 1)))
        # Update the policy estimator
        estimator_policy.update(states, np.reshape(
            advantages, (num, 1)), np.reshape(actions, (num, 1)))

    return stats


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(400, 3, 0.01, 0.001)
value_estimator = ValueEstimator(
    400, 3, 0.1, 0.001, policy_estimator.get_num_trainable_vars())
steps = [20, 20, 10, 10]
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, policy_estimator,
                         value_estimator, 900, steps, 0.99)

plotting.plot_episode_stats(stats, smoothing_window=10)
with open("ac_activelearning_mountaincar.pickle", 'wb') as handle:
    pickle.dump([stats[0], stats[1], stats.episode_lengths,
                 stats.episode_rewards], handle)
