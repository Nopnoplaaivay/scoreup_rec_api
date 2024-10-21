import warnings
import os
import keras
import warnings
from keras.api.layers import Dense
from keras.api.models import Sequential

from src.utils.logger import Logger

logger = Logger().get_logger()
warnings.filterwarnings("ignore")


class ActorCriticNetwork(keras.Model):
    def __init__(
        self,
        n_actions=81,
        cur_chapter="chuong-1",
        name="nmcntt",
        trainable=True,
        dtype=None,
        checkpoint_dir="tmp/checkpoints",
        **kwargs,
    ):
        super(ActorCriticNetwork, self).__init__(
            trainable=trainable, dtype=dtype, **kwargs
        )
        self.trainable = trainable
        self.model_name = name
        self.cur_chapter = cur_chapter
        self.checkpoint_dir = checkpoint_dir
        self.osn = 8
        self.n_actions = n_actions

        # Create a checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_c{self.cur_chapter.split('-')[-1]}.weights.h5" 
        )


        self.model = Sequential(
            [
                Dense(32, input_dim=self.osn, activation="relu"),
                Dense(16, activation="relu"),
            ]
        )

        self.v = Dense(1, activation=None)
        self.pi = Dense(self.n_actions, activation="softmax")

    def call(self, state):

        value = self.model(state)
        v = self.v(value)
        pi = self.pi(value)

        return v, pi
    

    def save_weights(self, filepath=None, overwrite=True):
        if filepath is None:
            filepath = self.checkpoint_file
        try:
            super().save_weights(filepath, overwrite=overwrite)
            # logger.info(f"Weights saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")

    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            filepath = self.checkpoint_file
        if not os.path.exists(filepath):
            logger.warning(f"Warning: Weights file not found at {filepath}. Skipping weight loading.")
            return
        try:
            super().load_weights(filepath, by_name=by_name)
            logger.info(f"Weights loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")

    def get_config(self):
        # Provide the configuration needed to recreate the model
        config = super(ActorCriticNetwork, self).get_config()
        config.update(
            {
                "n_actions": self.n_actions,
                "cur_chapter": self.cur_chapter,
                "name": self.model_name,
                "checkpoint_dir": self.checkpoint_dir,
                "trainable": self.trainable,  # Add trainable to config
                "dtype": self.dtype_policy.name,  # Include dtype in the config
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Create an instance of ActorCriticNetwork from the config
        return cls(**config)


# actor_critic = ActorCriticNetwork(n_actions=485)
# state = [0, 1, 0, 0, 0, 0, 0, 1]
# next_state = [0, 1, 0, 0, 0, 0, 0, 1]
# state = tf.convert_to_tensor([state], dtype=tf.float32)
# next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
# state_value, probs = actor_critic(state)

# print(f"state_value: {state_value}")
# print(f"probs: {probs}")
# state_value = tf.squeeze(state_value)
# print(f"state_value: {state_value}")

# action_probs = tfp.distributions.Categorical(probs=probs)
# print(f"action_probs: {action_probs}")
# action = action_probs.sample()
# print(f"action: {action}")
# action_prob = action_probs.prob(action)
# print(f"action_prob: {action_prob}")
# log_prob = action_probs.log_prob(action)
# print(f"log_prob: {log_prob}")
