# Imports
import collections
import os
import tarfile

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

# Set matplotlib to jshtml so animations work with colab.
# rc('animation', html='jshtml')
# Please edit.

# Replace this path with your own tfrecords./
# This tutorial is based on using data in the Scenario proto format directly,
# so choose the correct dataset version.
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
# Visualize the reference (ground truth) scenario.


# def plot_track_trajectory(track: scenario_pb2.Track) -> None:
#   valids = np.array([state.valid for state in track.states])
#   if np.any(valids):
#     x = np.array([state.center_x for state in track.states])
#     y = np.array([state.center_y for state in track.states])
#     ax.plot(x[valids], y[valids], linewidth=5)


# Plot their tracks.
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# visualizations.add_map(ax, scenario)
#
# for track in scenario.tracks:
#   if track.id in submission_specs.get_sim_agent_ids(
#       scenario, challenge_type=submission_specs.ChallengeType.SCENARIO_GEN
#   ):
#     plot_track_trajectory(track)
# #
# plt.show()

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

TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3

def num_agents_per_type(scenario: scenario_pb2.Scenario) -> dict[int, int]:
  num_agents_by_type = collections.defaultdict(int)

  for track in scenario.tracks:
    if track.id in submission_specs.get_sim_agent_ids(scenario, challenge_type):
      num_agents_by_type[track.object_type] += 1

  return num_agents_by_type


total_agents = len(submission_specs.get_sim_agent_ids(scenario, challenge_type))
print(f'Total agents to generate: {total_agents}')

num_agents_by_type = num_agents_per_type(scenario)
print(f'Number of vehicles to generate, including ADV: {num_agents_by_type[TYPE_VEHICLE]}')
print(f'Number of pedestrians to generate: {num_agents_by_type[TYPE_PEDESTRIAN]}')
print(f'Number of cyclists to generate: {num_agents_by_type[TYPE_CYCLIST]}')

def strip_logged_trajectories(
    logged_trajectories: trajectory_utils.ObjectTrajectories,
    submission_config: submission_specs.SubmissionConfig,
) -> trajectory_utils.ObjectTrajectories:
  """Strips privileged information from `ObjectTrajectories`.

  Args:
    logged_trajectories: an `ObjectTrajectories` containing full trajectories of
      all the objects in a scenario.
    submission_config: Config object holding number of past/current/future
      timesteps.

  Returns:
    An `ObjectTrajectories` with all trajectory information removed except for
    the ADV's history and the object_types and object_ids of all agents.
  """
  x = np.zeros_like(logged_trajectories.x)
  y = np.zeros_like(logged_trajectories.y)
  z = np.zeros_like(logged_trajectories.z)
  heading = np.zeros_like(logged_trajectories.heading)
  length = np.zeros_like(logged_trajectories.length)
  width = np.zeros_like(logged_trajectories.width)
  height = np.zeros_like(logged_trajectories.height)
  valid = np.zeros_like(logged_trajectories.valid, dtype=np.bool)
  # Restore ADV history.
  adv_idx = 0
  current_time_idx = submission_config.current_time_index
  hist_end = current_time_idx + 1
  x[adv_idx, :hist_end] = logged_trajectories.x[adv_idx, :hist_end]
  y[adv_idx, :hist_end] = logged_trajectories.y[adv_idx, :hist_end]
  z[adv_idx, :hist_end] = logged_trajectories.z[adv_idx, :hist_end]
  heading[adv_idx, :hist_end] = logged_trajectories.heading[adv_idx, :hist_end]
  length[adv_idx, :hist_end] = logged_trajectories.length[adv_idx, :hist_end]
  width[adv_idx, :hist_end] = logged_trajectories.width[adv_idx, :hist_end]
  height[adv_idx, :hist_end] = logged_trajectories.height[adv_idx, :hist_end]
  valid[adv_idx, :hist_end] = logged_trajectories.valid[adv_idx, :hist_end]
  # Restore validity of all objects at current timestep.
  valid[:, current_time_idx] = logged_trajectories.valid[:, current_time_idx]
  # Return new object.
  return trajectory_utils.ObjectTrajectories(
      x=tf.convert_to_tensor(x),
      y=tf.convert_to_tensor(y),
      z=tf.convert_to_tensor(z),
      heading=tf.convert_to_tensor(heading),
      length=tf.convert_to_tensor(length),
      width=tf.convert_to_tensor(width),
      height=tf.convert_to_tensor(height),
      valid=tf.convert_to_tensor(valid),
      object_id=logged_trajectories.object_id,
      object_type=logged_trajectories.object_type,
  )


def _generate_trajectories(
        logged_trajectories: trajectory_utils.ObjectTrajectories,
        submission_config: submission_specs.SubmissionConfig,
        print_verbose_comments: bool = True,
) -> tf.Tensor:
  """Generates initial states and trajectories for all required agents."""
  vprint = print if print_verbose_comments else lambda arg: None
  # Extract the ADV's velocity (x/y/z components) at the first timestep.
  adv_idx = 0
  init_adv_states = tf.stack(
    [
      logged_trajectories.x[adv_idx, :2],
      logged_trajectories.y[adv_idx, :2],
      logged_trajectories.z[adv_idx, :2],
      logged_trajectories.heading[adv_idx, :2],
    ],
    axis=-1,
  )
  init_adv_velocity = init_adv_states[-1, :3] - init_adv_states[-2, :3]
  # We also make the heading constant, so concatenate 0.0 as angular velocity.
  init_adv_velocity = tf.concat([init_adv_velocity, tf.zeros([1])], axis=-1)

  # Now we create a simulation. As we discussed, we actually want 32
  # parallel simulations, so we make this batched from the very beginning. We
  # add some random noise on top of our actions to make sure the behaviors are
  # different.
  NOISE_SCALE = 0.05
  # `max_action` shape: (4,).
  max_action = tf.constant([5, 5, 5, 0], dtype=tf.float32)

  init_adv_states = tf.stack(
    [
      logged_trajectories.x[adv_idx, :1],
      logged_trajectories.y[adv_idx, :1],
      logged_trajectories.z[adv_idx, :1],
      logged_trajectories.heading[adv_idx, :1],
    ],
    axis=-1,
  )

  # We create `simulated_states` with shape (n_rollouts, n_objects, n_steps, 4).
  n_objects, n_steps = logged_trajectories.valid.shape
  simulated_states = tf.tile(
    init_adv_states[tf.newaxis, :, tf.newaxis, :],
    [submission_config.n_rollouts, n_objects, 1, 1],
  )
  vprint(f'Initial simulated state shape: {simulated_states.shape}')

  # Set initial agent locations to random locations around the ADV.
  init_state_noise = tf.random.normal(
    simulated_states.shape, mean=0.0, stddev=5
  )
  simulated_states = simulated_states + init_state_noise

  # Rollout trajectories using constant velocity.
  for _ in range(submission_config.n_simulation_steps - 1):
    current_state = simulated_states[:, :, -1, :]
    # Random actions, take a normal and normalize by min/max actions
    action_noise = tf.random.normal(
      current_state.shape, mean=0.0, stddev=NOISE_SCALE
    )
    actions_with_noise = init_adv_velocity[tf.newaxis, tf.newaxis, :] + (
            action_noise * max_action
    )
    next_state = current_state + actions_with_noise
    simulated_states = tf.concat(
      [simulated_states, next_state[:, :, None, :]], axis=2
    )

  vprint(f'Final simulated states shape: {simulated_states.shape}')
  return simulated_states


def _generate_sizes(
        logged_trajectories: trajectory_utils.ObjectTrajectories,
) -> tf.Tensor:
  """Generates agent sizes for all required agents."""
  # For demonstration purposes, we use a simple policy which sets all agents to
  # a fixed size depending on the agent type.
  size_vehicle = tf.constant([4.78, 2.07, 1.53])
  size_pedestrian = tf.constant([0.92, 0.82, 1.52])
  size_cyclist = tf.constant([1.70, 0.82, 1.76])

  is_veh = tf.cast(logged_trajectories.object_type == TYPE_VEHICLE, tf.float32)
  is_ped = tf.cast(
    logged_trajectories.object_type == TYPE_PEDESTRIAN, tf.float32
  )
  is_cyc = tf.cast(logged_trajectories.object_type == TYPE_CYCLIST, tf.float32)

  n_objects, n_steps = logged_trajectories.valid.shape

  agent_size_if_veh = (
          tf.tile(
            size_vehicle[tf.newaxis, tf.newaxis, tf.newaxis, :],
            (submission_config.n_rollouts, n_objects, n_steps, 1),
          )
          * is_veh[tf.newaxis, :, tf.newaxis, tf.newaxis]
  )
  agent_size_if_ped = (
          tf.tile(
            size_pedestrian[tf.newaxis, tf.newaxis, tf.newaxis, :],
            (submission_config.n_rollouts, n_objects, n_steps, 1),
          )
          * is_ped[tf.newaxis, :, tf.newaxis, tf.newaxis]
  )
  agent_size_if_cyc = (
          tf.tile(
            size_cyclist[tf.newaxis, tf.newaxis, tf.newaxis, :],
            (submission_config.n_rollouts, n_objects, n_steps, 1),
          )
          * is_cyc[tf.newaxis, :, tf.newaxis, tf.newaxis]
  )
  # Shape (n_rollouts, n_objects, n_steps, 3)
  simulated_sizes = agent_size_if_veh + agent_size_if_ped + agent_size_if_cyc
  return simulated_sizes


def generate_with_random_policy(
        scenario: scenario_pb2.Scenario, print_verbose_comments: bool = True
) -> tuple[tf.Tensor, trajectory_utils.ObjectTrajectories]:
  vprint = print if print_verbose_comments else lambda arg: None
  full_logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
    scenario
  )
  # Remove all privileged information for the scenario gen challenge.
  logged_trajectories = strip_logged_trajectories(
    full_logged_trajectories, submission_config
  )
  # Select just the objects that we need to simulate.
  vprint(
    'Original shape of tensors containing trajectory data:'
    f' {logged_trajectories.valid.shape} (n_objects, n_steps)'
  )
  logged_trajectories = logged_trajectories.gather_objects_by_id(
    tf.convert_to_tensor(
      submission_specs.get_sim_agent_ids(scenario, challenge_type)
    )
  )
  vprint(
    'Modified shape of tensors containing trajectory data:'
    f' {logged_trajectories.valid.shape} (n_objects, n_steps)'
  )

  # We can verify that all of these objects are valid at the current step.
  current_time_index = submission_config.current_time_index
  all_agents_valid = tf.reduce_all(
    logged_trajectories.valid[:, current_time_index]
  )
  vprint(f'Are all agents valid: {all_agents_valid.numpy()}')

  simulated_states = _generate_trajectories(
    logged_trajectories, submission_config, print_verbose_comments
  )
  simulated_sizes = _generate_sizes(logged_trajectories)
  return logged_trajectories, simulated_states, simulated_sizes


logged_trajectories, simulated_states, simulated_sizes = (
  generate_with_random_policy(scenario, print_verbose_comments=True)
)

# Select which one of the 32 simulations to visualize.
# SAMPLE_INDEX = 0
#
# n_objects, n_steps = logged_trajectories.valid.shape
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# visualizations.get_animated_states(
#   fig,
#   ax,
#   scenario,
#   simulated_states[SAMPLE_INDEX, :, :, 0],
#   simulated_states[SAMPLE_INDEX, :, :, 1],
#   simulated_states[SAMPLE_INDEX, :, :, 3],
#   length=simulated_sizes[SAMPLE_INDEX, :, :, 0],
#   width=simulated_sizes[SAMPLE_INDEX, :, :, 1],
#   color_idx=tf.zeros((n_objects, n_steps), dtype=tf.int32),
# )
#
# plt.show()

def joint_scene_from_states(
    states: tf.Tensor,
    sizes: tf.Tensor,
    object_ids: tf.Tensor,
) -> sim_agents_submission_pb2.JointScene:
  # States shape: (num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  states = states.numpy()
  sizes = sizes.numpy()
  simulated_trajectories = []
  for i_object in range(len(object_ids)):
    simulated_trajectories.append(
        sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i_object, :, 0],
            center_y=states[i_object, :, 1],
            center_z=states[i_object, :, 2],
            heading=states[i_object, :, 3],
            object_id=object_ids[i_object],
            length=sizes[i_object, :, 0],
            width=sizes[i_object, :, 1],
            height=sizes[i_object, :, 2],
        )
    )
  return sim_agents_submission_pb2.JointScene(
      simulated_trajectories=simulated_trajectories
  )


# Package the first simulation into a `JointScene`
joint_scene = joint_scene_from_states(
    simulated_states[0, :, :, :],
    simulated_sizes[0, :, :, :],
    logged_trajectories.object_id,
)
# Validate the joint scene. Should raise an exception if it's invalid.
submission_specs.validate_joint_scene(joint_scene, scenario, challenge_type)

# Now we can replicate this strategy to export all the parallel simulations.
def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario,
    states: tf.Tensor,
    sizes: tf.Tensor,
    object_ids: tf.Tensor,
) -> sim_agents_submission_pb2.ScenarioRollouts:
  # States shape: (num_rollouts, num_objects, num_steps, 4).
  # Objects IDs shape: (num_objects,).
  joint_scenes = []
  for i_rollout in range(states.shape[0]):
    joint_scenes.append(
        joint_scene_from_states(states[i_rollout], sizes[i_rollout], object_ids)
    )
  return sim_agents_submission_pb2.ScenarioRollouts(
      # Note: remember to include the Scenario ID in the proto message.
      joint_scenes=joint_scenes,
      scenario_id=scenario.scenario_id,
  )


scenario_rollouts = scenario_rollouts_from_states(
    scenario, simulated_states, simulated_sizes, logged_trajectories.object_id
)
# As before, we can validate the message we just generated.
submission_specs.validate_scenario_rollouts(
    scenario_rollouts, scenario, challenge_type
)

# Compute the features for a single JointScene.
single_scene_features = metric_features.compute_metric_features(
    scenario, joint_scene, challenge_type
)

# These features will be computed only for the `tracks_to_predict` objects.
print(
    'Evaluated objects:'
    f' {submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type)}'
)
# This will also match single_scene_features.object_ids
print(f'Evaluated objects in features: {single_scene_features.object_id}')

# Features contain a validity flag, which for simulated rollouts must be always
# True, because we are requiring the generated agents to be always valid when
# replaced.
print(f'Are all agents valid: {tf.reduce_all(single_scene_features.valid)}')

# ============ FEATURES ============
# Average displacement feature. This corresponds to ADE in the BP challenges.
# Here it is used just for demonstration (it's not actually included in the
# final score).
# Shape: (1, n_objects).
print(
    f'ADE: {tf.reduce_mean(single_scene_features.average_displacement_error)}'
)
#
# # Kinematic features.
# print('\n============ KINEMATIC FEATURES ============')
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i_object in range(len(single_scene_features.object_id)):
#     _object_id = single_scene_features.object_id[i_object].numpy()
#     axes[0].plot(
#         single_scene_features.linear_speed[0, i_object, :], label=str(_object_id)
#     )
#     axes[1].plot(
#         single_scene_features.linear_acceleration[0, i_object, :],
#         label=str(_object_id),
#     )
#     axes[2].plot(
#         single_scene_features.angular_speed[0, i_object, :], label=str(_object_id)
#     )
#     axes[3].plot(
#         single_scene_features.angular_acceleration[0, i_object, :],
#         label=str(_object_id),
#     )
#
# TITLES = [
#     'linear_speed',
#     'linear_acceleration',
#     'angular_speed',
#     'angular_acceleration',
# ]
# for ax, title in zip(axes, TITLES):
#     ax.legend()
#     ax.set_title(title)
# plt.show()
#
# # Interactive features.
# print('\n============ INTERACTIVE FEATURES ============')
# print(f'Colliding objects: {single_scene_features.collision_per_step[0]}')
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# for i_object in range(len(single_scene_features.object_id)):
#     _object_id = single_scene_features.object_id[i_object].numpy()
#     axes[0].plot(
#         single_scene_features.distance_to_nearest_object[0, i_object, :],
#         label=str(_object_id),
#     )
#     axes[1].plot(
#         single_scene_features.time_to_collision[0, i_object, :],
#         label=str(_object_id),
#     )
#
# TITLES = ['distance to nearest object', 'time to collision']
# for ax, title in zip(axes, TITLES):
#     ax.legend()
#     ax.set_title(title)
# plt.show()
#
# # Map-based features.
# print('\n============ MAP-BASED FEATURES ============')
# print(f'Offroad objects: {single_scene_features.offroad_per_step[0]}')
# fig, axes = plt.subplots(1, 1, figsize=(4, 4))
# for i_object in range(len(single_scene_features.object_id)):
#     _object_id = single_scene_features.object_id[i_object].numpy()
#     axes.plot(
#         single_scene_features.distance_to_road_edge[0, i_object, :],
#         label=str(_object_id),
#     )
# axes.legend()
# axes.set_title('distance to road edge')
#
# plt.show()

# Load the test configuration.
config = metrics.load_metrics_config(challenge_type)

scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
    config, scenario, scenario_rollouts, challenge_type
)
print(scenario_metrics)

# Where results are going to be saved.
OUTPUT_ROOT_DIRECTORY = './waymo_scenario_gen/'
os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
output_filenames = []

# Iterate over shards. This could be parallelized in any custom way, as the
# number of output shards is not required to be the same as the initial dataset.
for shard_filename in tqdm.tqdm(filenames):
  # A shard filename has the structure: `validation.tfrecord-00000-of-00150`.
  # We want to maintain the same shard naming here, for simplicity, so we can
  # extract the suffix.
  shard_suffix = shard_filename.numpy().decode('utf8')[
      -len('-00000-of-00150') :
  ]

  # Now we can iterate over the Scenarios in the shard. To make this faster as
  # part of the tutorial, we will only process 2 Scenarios per shard. Obviously,
  # to create a valid submission, all the scenarios needs to be present.
  shard_dataset = tf.data.TFRecordDataset([shard_filename]).take(2)
  shard_iterator = shard_dataset.as_numpy_iterator()

  scenario_rollouts = []
  for scenario_bytes in shard_iterator:
    scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
    logged_trajectories, simulated_states, simulated_sizes = (
        generate_with_random_policy(scenario, print_verbose_comments=False)
    )
    sr = scenario_rollouts_from_states(
        scenario,
        simulated_states,
        simulated_sizes,
        logged_trajectories.object_id,
    )
    submission_specs.validate_scenario_rollouts(sr, scenario, challenge_type)
    scenario_rollouts.append(sr)

  # Now that we have 2 `ScenarioRollouts` for this shard, we can package them
  # into a `SimAgentsChallengeSubmission`. Remember to populate the metadata
  # for each shard.
  shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
      scenario_rollouts=scenario_rollouts,
      submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
      account_name='your_account@test.com',
      unique_method_name='scenario_gen_tutorial',
      authors=['test'],
      affiliation='waymo',
      description='Submission from the Scenario Gen tutorial',
      method_link='https://waymo.com/open/',
      # New REQUIRED fields.
      uses_lidar_data=False,
      uses_camera_data=False,
      uses_public_model_pretraining=False,
      num_model_parameters='24',
      acknowledge_complies_with_closed_loop_requirement=True,
  )

  # Now we can export this message to a binproto, saved to local storage.
  output_filename = f'submission.binproto{shard_suffix}'
  with open(os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename), 'wb') as f:
    f.write(shard_submission.SerializeToString())
  output_filenames.append(output_filename)

# Once we have created all the shards, we can package them directly into a
# tar.gz archive, ready for submission.
with tarfile.open(
    os.path.join(OUTPUT_ROOT_DIRECTORY, 'submission.tar.gz'), 'w:gz'
) as tar:
  for output_filename in output_filenames:
    tar.add(
        os.path.join(OUTPUT_ROOT_DIRECTORY, output_filename),
        arcname=output_filename,
    )