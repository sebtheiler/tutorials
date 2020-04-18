### The Config ###

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'BreakoutDeterministic-v4'

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = None
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
# Since Breakout is a simple game, I wouldn't recommend using it here.
USE_PER = False

PRIORITY_SCALE = 0.7              # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
CLIP_REWARD = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged


TOTAL_FRAMES = 30000000           # Total number of frames to train for
MAX_EPISODE_LENGTH = 18000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
FRAMES_BETWEEN_EVAL = 100000      # Number of frames between evaluations
EVAL_LENGTH = 10000               # Number of frames to evaluate for
UPDATE_FREQ = 10000               # Number of actions chosen between updating the target network

DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 1000000                # The maximum size of the replay buffer

MAX_NOOP_STEPS = 20               # Randomly perform this number of actions before every evaluation to give it an element of randomness
UPDATE_FREQ = 4                   # Number of actions between gradient descent steps

INPUT_SHAPE = (84, 84)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                   # Number of samples the agent learns from at once
LEARNING_RATE = 0.00001
