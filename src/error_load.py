import math
import tarfile

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError, Message
from waymo_open_dataset.protos import sim_agents_submission_pb2










TAR_PATH = (
    "/home/ke/code/sim/src/logs/my/"
    "2026-07-10_17-03-56/wosac_submission.tar.gz"
)

TARGET_SUFFIX = "submission.binproto-00027-of-00147"


def load_target_shard(
    tar_path: str,
    target_suffix: str,
) -> tuple[str, sim_agents_submission_pb2.SimAgentsChallengeSubmission]:
    """
    Find and parse one shard from the tar.gz archive.
    """
    with tarfile.open(tar_path, mode="r:gz") as tar:
        target_member = None

        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(target_suffix):
                target_member = member
                break

        if target_member is None:
            raise FileNotFoundError(
                f"Could not find shard ending with {target_suffix!r}"
            )

        file_obj = tar.extractfile(target_member)
        if file_obj is None:
            raise OSError(f"Could not read {target_member.name}")

        raw_data = file_obj.read()

    submission = (
        sim_agents_submission_pb2.SimAgentsChallengeSubmission()
    )

    try:
        submission.ParseFromString(raw_data)
    except DecodeError as exc:
        raise RuntimeError(
            f"Failed to parse {target_member.name}: {exc}"
        ) from exc

    return target_member.name, submission


member_name, submission = load_target_shard(
    TAR_PATH,
    TARGET_SUFFIX,
)
from pathlib import Path

def save_submission(
    submission,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = submission.SerializeToString()

    with output_path.open("wb") as f:
        f.write(serialized)

    print(
        f"[INFO] Saved submission to: {output_path}\n"
        f"[INFO] File size: {output_path.stat().st_size / 1024**2:.2f} MB",
        flush=True,
    )

    return output_path

output_path = save_submission(
    submission,
    "/home/ke/code/sim/src/debug/"
    "submission.binproto-00027-of-00147",
)

print("Loaded:", member_name)
print("Number of scenarios:", len(submission.scenario_rollouts))
print("Method:", submission.unique_method_name)

def check_target_submission(
    submission: sim_agents_submission_pb2.SimAgentsChallengeSubmission,
    *,
    expected_num_rollouts: int | None = 32,
    expected_num_steps: int | None = 80,
) -> dict:
    num_errors = 0
    num_warnings = 0

    seen_scenario_ids = set()

    def error(message: str) -> None:
        nonlocal num_errors
        num_errors += 1
        print(f"[ERROR] {message}", flush=True)

    def warning(message: str) -> None:
        nonlocal num_warnings
        num_warnings += 1
        print(f"[WARNING] {message}", flush=True)

    for scenario_index, scenario in enumerate(
        submission.scenario_rollouts
    ):
        scenario_context = (
            f"scenario_rollouts[{scenario_index}]"
            f"(scenario_id={scenario.scenario_id!r})"
        )

        if not scenario.scenario_id:
            error(f"{scenario_context}: empty scenario_id")
        elif scenario.scenario_id in seen_scenario_ids:
            error(
                f"{scenario_context}: duplicate scenario_id"
            )
        else:
            seen_scenario_ids.add(scenario.scenario_id)

        if expected_num_rollouts is not None:
            if len(scenario.joint_scenes) != expected_num_rollouts:
                error(
                    f"{scenario_context}: expected "
                    f"{expected_num_rollouts} joint scenes, found "
                    f"{len(scenario.joint_scenes)}"
                )

        reference_object_ids = None

        for rollout_index, joint_scene in enumerate(
            scenario.joint_scenes
        ):
            rollout_context = (
                f"{scenario_context}."
                f"joint_scenes[{rollout_index}]"
            )

            object_ids = []

            if len(joint_scene.simulated_trajectories) == 0:
                error(
                    f"{rollout_context}: no simulated trajectories"
                )
                continue

            for trajectory_index, trajectory in enumerate(
                joint_scene.simulated_trajectories
            ):
                trajectory_context = (
                    f"{rollout_context}."
                    f"simulated_trajectories[{trajectory_index}]"
                    f"(object_id={trajectory.object_id})"
                )

                object_ids.append(int(trajectory.object_id))

                fields = {
                    "center_x": trajectory.center_x,
                    "center_y": trajectory.center_y,
                    "center_z": trajectory.center_z,
                    "heading": trajectory.heading,
                }

                # Add dimensions only when these fields exist in your proto.
                for optional_name in ("length", "width", "height"):
                    if hasattr(trajectory, optional_name):
                        fields[optional_name] = getattr(
                            trajectory,
                            optional_name,
                        )

                lengths = {
                    name: len(values)
                    for name, values in fields.items()
                }

                if len(set(lengths.values())) != 1:
                    error(
                        f"{trajectory_context}: inconsistent field "
                        f"lengths {lengths}"
                    )

                if expected_num_steps is not None:
                    for field_name, field_length in lengths.items():
                        if field_length != expected_num_steps:
                            error(
                                f"{trajectory_context}.{field_name}: "
                                f"expected {expected_num_steps}, "
                                f"found {field_length}"
                            )

                for field_name, values in fields.items():
                    for step_index, value in enumerate(values):
                        value = float(value)

                        if not math.isfinite(value):
                            error(
                                f"{trajectory_context}."
                                f"{field_name}[{step_index}] "
                                f"= {value!r}"
                            )

                for field_name in ("length", "width", "height"):
                    if field_name not in fields:
                        continue

                    for step_index, value in enumerate(
                        fields[field_name]
                    ):
                        value = float(value)

                        if math.isfinite(value) and value <= 0:
                            error(
                                f"{trajectory_context}."
                                f"{field_name}[{step_index}] "
                                f"= {value}: dimension must be positive"
                            )

                for step_index, value in enumerate(
                    trajectory.heading
                ):
                    value = float(value)

                    if (
                        math.isfinite(value)
                        and abs(value) > 4 * math.pi
                    ):
                        warning(
                            f"{trajectory_context}."
                            f"heading[{step_index}]={value}: "
                            f"unusually large heading"
                        )

            sorted_object_ids = tuple(sorted(object_ids))

            if len(object_ids) != len(set(object_ids)):
                error(
                    f"{rollout_context}: duplicate object IDs"
                )

            if reference_object_ids is None:
                reference_object_ids = sorted_object_ids
            elif sorted_object_ids != reference_object_ids:
                error(
                    f"{rollout_context}: object IDs differ "
                    f"from rollout 0"
                )

    result = {
        "valid": num_errors == 0,
        "num_errors": num_errors,
        "num_warnings": num_warnings,
        "num_scenarios": len(submission.scenario_rollouts),
    }

    print("\nResult:", result)
    return result


result = check_target_submission(
    submission,
    expected_num_rollouts=32,
    expected_num_steps=91,
)
