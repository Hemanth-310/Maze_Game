import logging
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

from environment import Status
from models import AbstractModel

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(2)


class PPOModel(AbstractModel):
    """ Proximal Policy Optimization (PPO) prediction model.
    
    PPO is a policy gradient method that uses a clipped surrogate objective to ensure
    stable policy updates. This implementation includes:
    - Actor network (policy) that outputs action probabilities
    - Critic network (value function) that estimates state values
    - Clipped surrogate loss for policy updates
    - Generalized Advantage Estimation (GAE) for advantage calculation
    """
    
    default_check_convergence_every = 5
    
    def __init__(self, game, **kwargs):
        """ Create a new PPO prediction model for 'game'.
        
        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="PPOModel", **kwargs)
        
        # PPO hyperparameters
        self.clip_ratio = kwargs.get("clip_ratio", 0.2)
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        
        # Network architecture parameters
        self.hidden_size = kwargs.get("hidden_size", 128)
        
        # Initialize networks
        self._build_actor_critic()
        
        # Storage for experience collection
        self.reset_episode_data()
        
    def _build_actor_critic(self):
        """ Build the actor (policy) and critic (value) networks. """
        # Input layer - state is 2D (x, y position)
        state_input = Input(shape=(2,), name='state_input')
        
        # Shared hidden layers
        shared = Dense(self.hidden_size, activation='relu', name='shared1')(state_input)
        shared = Dense(self.hidden_size, activation='relu', name='shared2')(shared)
        
        # Actor network (policy)
        actor_output = Dense(len(self.environment.actions), activation='softmax', name='actor_output')(shared)
        self.actor = Model(inputs=state_input, outputs=actor_output, name='actor')
        
        # Critic network (value function)
        critic_output = Dense(1, activation='linear', name='critic_output')(shared)
        self.critic = Model(inputs=state_input, outputs=critic_output, name='critic')
        
        # Compile networks
        self.actor_optimizer = Adam(learning_rate=3e-4)
        self.critic_optimizer = Adam(learning_rate=1e-3)
        
        logging.info(f"Actor network built with {len(self.environment.actions)} output actions")
        logging.info(f"Critic network built with 1 output (value)")
        
    def reset_episode_data(self):
        """ Reset data storage for new episode. """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def collect_experience(self, state, action, reward, done):
        """ Collect experience data for PPO training. """
        state_array = np.array(state).flatten()
        
        # Get action probabilities and value estimate
        action_probs = self.actor.predict(state_array.reshape(1, -1), verbose=0)[0]
        value = self.critic.predict(state_array.reshape(1, -1), verbose=0)[0][0]
        
        # Calculate log probability of taken action
        log_prob = np.log(action_probs[action] + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Store experience
        self.states.append(state_array)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def compute_advantages_and_returns(self, next_value=0.0):
        """ Compute advantages and returns using GAE. """
        advantages = []
        returns = []
        
        # Add next value for bootstrap
        values = self.values + [next_value]
        
        # Compute advantages using GAE
        advantage = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                # Terminal state
                delta = self.rewards[t] + 0.0 - values[t]  # No bootstrap for terminal
            else:
                # Non-terminal state
                delta = self.rewards[t] + 0.99 * values[t + 1] - values[t]  # Assuming gamma=0.99
            
            advantage = delta + 0.99 * self.gae_lambda * advantage * (1 - self.dones[t])
            advantages.insert(0, advantage)
        
        # Compute returns
        for t in range(len(self.rewards)):
            returns.append(advantages[t] + values[t])
            
        return np.array(advantages), np.array(returns)
    
    def train_on_batch(self, states, actions, old_log_probs, advantages, returns):
        """ Train the actor and critic networks on a batch of experience. """
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train actor network
        with tf.GradientTape() as tape:
            # Get current policy
            action_probs = self.actor(states)
            log_probs = tf.reduce_sum(tf.math.log(action_probs + 1e-8) * tf.one_hot(actions, len(self.environment.actions)), axis=1)
            
            # Compute policy ratio
            ratio = tf.exp(log_probs - old_log_probs)
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Add entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
            actor_loss -= self.entropy_coef * entropy
            
        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Train critic network
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.critic(states))
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
        # Update critic
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return actor_loss.numpy(), critic_loss.numpy()
    
    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the PPO model.
        
        :param stop_at_convergence: stop training as soon as convergence is reached
        
        Hyperparameters:
        :keyword float discount: (gamma) preference for future rewards
        :keyword float exploration_rate: not used in PPO (exploration via policy entropy)
        :keyword float learning_rate: learning rate for networks
        :keyword int episodes: number of training games to play
        :keyword int update_frequency: how often to update networks
        :return list, list, int, datetime: cumulative rewards, win rates, episodes, time spent
        """
        discount = kwargs.get("discount", 0.99)
        episodes = max(kwargs.get("episodes", 1000), 1)
        update_frequency = kwargs.get("update_frequency", 10)  # Update every N episodes
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)
        
        # Variables for reporting
        cumulative_reward_history = []
        win_history = []
        
        start_list = []
        start_time = datetime.now()
        
        logging.info(f"Starting PPO training for {episodes} episodes")
        
        for episode in range(1, episodes + 1):
            # Reset starting cells if exhausted
            if not start_list:
                start_list = self.environment.empty.copy()
            
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)
            
            # Reset environment and episode data
            state = self.environment.reset(start_cell)
            self.reset_episode_data()
            
            episode_reward = 0
            
            # Collect experience for one episode
            while True:
                # Get action from current policy
                action = self.predict(state)
                
                # Take action
                next_state, reward, status = self.environment.step(action)
                episode_reward += reward
                
                # Store experience
                done = status in (Status.WIN, Status.LOSE)
                self.collect_experience(state, action, reward, done)
                
                # Update state
                state = next_state
                
                # Render Q values if in training mode
                self.environment.render_q(self)
                
                # Check if episode is done
                if done:
                    break
            
            # Update networks if we have enough episodes
            if episode % update_frequency == 0:
                # Get final value for bootstrap
                final_state = state.flatten() if len(state.shape) > 1 else state
                final_value = self.critic.predict(final_state.reshape(1, -1), verbose=0)[0][0]
                
                # Compute advantages and returns
                advantages, returns = self.compute_advantages_and_returns(final_value)
                
                # Train on collected experience
                actor_loss, critic_loss = self.train_on_batch(
                    self.states, self.actions, self.log_probs, advantages, returns
                )
                
                logging.info(f"Episode {episode}: Updated networks - Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
            cumulative_reward_history.append(episode_reward)
            
            # Check convergence
            if episode % check_convergence_every == 0:
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                
                logging.info(f"Episode {episode}: Win Rate: {win_rate:.3f}, Avg Reward: {np.mean(cumulative_reward_history[-check_convergence_every:]):.2f}")
                
                if w_all and stop_at_convergence:
                    logging.info("Won from all start cells, stopping training")
                    break
        
        training_time = datetime.now() - start_time
        logging.info(f"PPO training completed: {episode} episodes in {training_time}")
        
        return cumulative_reward_history, win_history, episode, training_time
    
    def q(self, state):
        """ Return action probabilities (Q-like values) for state.
        
        In PPO, this returns the policy probabilities rather than Q-values.
        """
        if isinstance(state, tuple):
            state = np.array(state).flatten()
        elif len(state.shape) > 1:
            state = state.flatten()
        
        action_probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        return action_probs
    
    def predict(self, state):
        """ Predict action based on current policy.
        
        :param np.ndarray state: game state
        :return int: selected action
        """
        action_probs = self.q(state)
        
        # Sample action from policy (stochastic)
        action = np.random.choice(len(action_probs), p=action_probs)
        
        logging.debug(f"Action probabilities: {action_probs}, Selected action: {action}")
        return action
    
    def save(self, filename):
        """ Save the actor and critic networks. """
        self.actor.save(f"{filename}_actor.h5")
        self.critic.save(f"{filename}_critic.h5")
        logging.info(f"Saved PPO model to {filename}_actor.h5 and {filename}_critic.h5")
    
    def load(self, filename):
        """ Load the actor and critic networks. """
        self.actor = tf.keras.models.load_model(f"{filename}_actor.h5")
        self.critic = tf.keras.models.load_model(f"{filename}_critic.h5")
        logging.info(f"Loaded PPO model from {filename}_actor.h5 and {filename}_critic.h5")
