import os
import tarfile

# Set matplotlib to jshtml so animations work with colab.
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics
DATASET_FOLDER = '/media/ke/Windows/waymo_data/validation'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'validation.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test.tfrecord*')
# Define the dataset from the TFRecords.
filenames = tf.io.matching_files(VALIDATION_FILES)
dataset = tf.data.TFRecordDataset(filenames)
# Since these are raw Scenario protos, we need to parse them in eager mode.
dataset_iterator = dataset.as_numpy_iterator()
bytes_example = next(dataset_iterator)
scenario = scenario_pb2.Scenario.FromString(bytes_example)
print(f'Checking type: {type(scenario)}')
print(f'Loaded scenario with ID: {scenario.scenario_id}')

challenge_type = submission_specs.ChallengeType.SIM_AGENTS
submission_config = submission_specs.get_submission_config(challenge_type)

print(f'Simulation length, in steps: {submission_config.n_simulation_steps}')
print(
    'Duration of a step, in seconds:'
    f' {submission_config.step_duration_seconds}s (frequency:'
    f' {1/submission_config.step_duration_seconds}Hz)'
)
print(
    'Number of parallel simulations per Scenario:'
    f' {submission_config.n_rollouts}'
)
challenge_type = submission_specs.ChallengeType.SCENARIO_GEN
submission_config = submission_specs.get_submission_config(challenge_type)

print(f'Generation length, in steps: {submission_config.n_simulation_steps}')
print(
    'Duration of a step, in seconds:'
    f' {submission_config.step_duration_seconds}s (frequency:'
    f' {1 / submission_config.step_duration_seconds}Hz)'
)
print(
    'Number of parallel generation per Scenario:'
    f' {submission_config.n_rollouts}'
)

