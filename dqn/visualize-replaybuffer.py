import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from sklearn.decomposition import PCA

from config import (BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR, ENV_NAME,
                    EVAL_LENGTH, FRAMES_BETWEEN_EVAL, INPUT_SHAPE,
                    LEARNING_RATE, LOAD_FROM, MAX_EPISODE_LENGTH,
                    MAX_NOOP_STEPS, MEM_SIZE, MIN_REPLAY_BUFFER_SIZE,
                    PRIORITY_SCALE, SAVE_PATH, TOTAL_FRAMES, UPDATE_FREQ,
                    WRITE_TENSORBOARD)
from train_dqn import (Agent, GameWrapper, ReplayBuffer, build_q_network,
                       process_frame)

# This will usually fix any issues involving the GPU and cuDNN
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

# Change this to the path of the model you would like to visualize
RESTORE_PATH = None

if RESTORE_PATH is None:
    raise UserWarning('Please change the variable `RESTORE_PATH` to where you would like to load the model from. If you haven\'t trained a model, try \'example-save\'')

ENV_NAME = 'BreakoutDeterministic-v4'
FRAMES_TO_VISUALIZE = 750
FRAMES_TO_ANNOTATE = 4

# Create environment
game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

# Create agent
MAIN_DQN = build_q_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

print('Loading agent...')
agent.load(RESTORE_PATH)

print('Generating embeddings...')
embeddings = []
values = []

frame_indices = np.random.choice(agent.replay_buffer.count, size=FRAMES_TO_VISUALIZE)

for frame in agent.replay_buffer.frames[frame_indices]:
    # TODO: combine things into one
    embeddings.append(agent.get_intermediate_representation(frame, 'flatten_1')[0])
    values.append(agent.get_intermediate_representation(frame, 'dense')[0])

print('Fitting PCA...')
pca = PCA(2)
pca_embeddings = pca.fit_transform(embeddings)

print('Displaying...')
fig, ax = plt.subplots()
indices = np.random.choice(100, FRAMES_TO_ANNOTATE)
for i, frame in enumerate(agent.replay_buffer.frames[frame_indices]):
    if i in indices:
        im = OffsetImage(frame, zoom=2, cmap='gray')
        im.image.axes = ax
        ab = AnnotationBbox(im, pca_embeddings[i],
                            xybox=(-120., 120.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.3,
                            arrowprops=dict(arrowstyle="->"))

        ax.add_artist(ab)

plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=values, cmap='jet', norm=matplotlib.colors.LogNorm(vmin=min(values), vmax=max(values)))
plt.colorbar().set_label('Q-Value')
plt.show()
