import gym, tensorflow as tf, numpy as np, os
env = gym.make("CartPole-v1")

#Broom-v1

class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        out = tf.layers.dense(hidden_layer2, num_actions, activation=None)
        self.output = tf.nn.softmax(out)
        self.choice = tf.argmax(self.output, axis=1)
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)
        one_hot_actions = tf.one_hot(self.actions, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
        self.loss = tf.reduce_mean(cross_entropy*self.rewards)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())
        self.gradients2apply = []
        for idx, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients2apply.append(gradient_placeholder)
        optimizer = tf.train.AdamOptimizer(learning_rate=2e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients2apply, tf.trainable_variables()))
discount_rate = .95
def discount_normalize_rewards(rewards):
    discount_rewards = np.zeros_like(rewards)
    total_rewards = 0
    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards* discount_rate + rewards[i]
        discount_rewards[i] = total_rewards

    discount_rewards -= np.mean(discount_rewards)
    discount_rewards /= np.std(discount_rewards)

    return discount_rewards

#TODO Create training loop
tf.reset_default_graph()

num_actions = 2
state_size = 4

path = "./RoombaStick-pg"

training_eps = 5000
max_steps_per_episode = 10000
episode_batch_size = 5

agent = Agent(num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)

    total_episode_rewards = []

    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for ep in range(training_eps):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):
            if ep%100 ==0:
                env.render()
            action_probabilities = sess.run(agent.output, feed_dict={agent.input_layer: [state]})
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])
            state_next, reward, done, _ = env.step(action_choice)
            episode_history.append([state, action_choice, reward, state_next])
            state = state_next
            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = discount_normalize_rewards(episode_history[:, 2])

                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:, 1],
                                                                    agent.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient

                break
        if ep % episode_batch_size == 0:

            feed_dict_gradients = dict(zip(agent.gradients2apply, gradient_buffer))

            sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)

            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0

            if ep % 100 == 0:
                saver.save(sess, path + "pg-checkpoint", ep)
                print("Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))

# # TODO Create the testing loop
# testing_episodes = 5
#
# with tf.Session() as sess:
#     checkpoint = tf.train.get_checkpoint_state(path)
#     saver.restore(sess,checkpoint.model_checkpoint_path)
#
#     for episode in range(testing_episodes):
#
#             state = env.reset()
#
#             episode_rewards = 0
#
#             for step in range(max_steps_per_episode):
#
#                 env.render()
#
#                 # Get Action
#                 action_argmax = sess.run(agent.choice, feed_dict={agent.input_layer: [state]})
#                 action_choice = action_argmax[0]
#
#                 state_next, reward, done, _ = env.step(action_choice)
#                 state = state_next
#
#                 episode_rewards += reward
#
#                 if done or step + 1 == max_steps_per_episode:
#                    print("Rewards for episode " + str(episode) + ": " + str(episode_rewards))
#                    break