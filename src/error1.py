from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError, Message
from waymo_open_dataset.protos import sim_agents_submission_pb2


# =============================================================================
# Protobuf field types
# =============================================================================

FLOAT_TYPES = {
    FieldDescriptor.TYPE_FLOAT,
    FieldDescriptor.TYPE_DOUBLE,
}


# =============================================================================
# Logging helpers
# =============================================================================

def add_error(errors: list[str], message: str) -> None:
    """
    Store an error and print it immediately.
    """
    errors.append(message)
    print(f"[ERROR] {message}", flush=True)


def add_warning(warnings: list[str], message: str) -> None:
    """
    Store a warning and print it immediately.
    """
    warnings.append(message)
    print(f"[WARNING] {message}", flush=True)


def add_info(message: str) -> None:
    """
    Print informational progress immediately.
    """
    print(f"[INFO] {message}", flush=True)


# =============================================================================
# Protobuf compatibility helpers
# =============================================================================

def is_repeated_field(field: FieldDescriptor) -> bool:
    """
    Check whether a protobuf field is repeated.

    Compatible with both the Python protobuf implementation and the newer
    upb backend.
    """
    repeated = getattr(field, "is_repeated", None)

    if repeated is not None:
        return bool(repeated)

    return field.label == FieldDescriptor.LABEL_REPEATED


def validate_all_numeric_values(
    message: Message,
    errors: list[str],
    context: str,
) -> None:
    """
    Recursively inspect all populated protobuf fields.

    Every float/double field is checked for:
      - NaN
      - positive infinity
      - negative infinity

    This implementation does not use a generator, avoiding the previous
    FieldDescriptor/float unpacking problem.
    """
    for field, value in message.ListFields():
        field_path = f"{context}.{field.name}"
        repeated = is_repeated_field(field)

        # Recursively inspect nested protobuf messages.
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if repeated:
                for index, child_message in enumerate(value):
                    validate_all_numeric_values(
                        message=child_message,
                        errors=errors,
                        context=f"{field_path}[{index}]",
                    )
            else:
                validate_all_numeric_values(
                    message=value,
                    errors=errors,
                    context=field_path,
                )

            continue

        # Only float and double fields need finite-value validation.
        if field.type not in FLOAT_TYPES:
            continue

        if repeated:
            for index, scalar_value in enumerate(value):
                scalar_value = float(scalar_value)

                if not math.isfinite(scalar_value):
                    add_error(
                        errors,
                        f"{field_path}[{index}]: "
                        f"non-finite value {scalar_value!r}",
                    )
        else:
            scalar_value = float(value)

            if not math.isfinite(scalar_value):
                add_error(
                    errors,
                    f"{field_path}: "
                    f"non-finite value {scalar_value!r}",
                )


# =============================================================================
# File loading
# =============================================================================

def load_submission_file(
    submission_path: str | Path,
) -> sim_agents_submission_pb2.SimAgentsChallengeSubmission:
    """
    Load one saved SimAgentsChallengeSubmission .binproto file.
    """
    submission_path = Path(submission_path)

    if not submission_path.is_file():
        raise FileNotFoundError(
            f"Submission file does not exist: {submission_path}"
        )

    serialized_data = submission_path.read_bytes()

    if not serialized_data:
        raise ValueError(
            f"Submission file is empty: {submission_path}"
        )

    submission = (
        sim_agents_submission_pb2.SimAgentsChallengeSubmission()
    )

    submission.ParseFromString(serialized_data)
    add_info(f"Loaded submission: {submission_path}")
    add_info(
        f"File size: "
        f"{submission_path.stat().st_size / (1024 ** 2):.2f} MB"
    )
    add_info(
        f"Number of scenarios: "
        f"{len(submission.scenario_rollouts)}"
    )

    return submission


# =============================================================================
# Metadata validation
# =============================================================================

def validate_submission_metadata(
    submission: sim_agents_submission_pb2.SimAgentsChallengeSubmission,
    context: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Validate submission-level metadata.
    """
    required_string_fields = (
        "account_name",
        "unique_method_name",
        "affiliation",
        "description",
    )

    for field_name in required_string_fields:
        if not hasattr(submission, field_name):
            add_error(
                errors,
                f"{context}: protobuf has no field {field_name!r}",
            )
            continue

        value = getattr(submission, field_name)

        if not str(value).strip():
            add_error(
                errors,
                f"{context}.{field_name}: empty value",
            )

    if hasattr(submission, "authors"):
        if len(submission.authors) == 0:
            add_error(
                errors,
                f"{context}.authors: no authors",
            )

        for author_index, author in enumerate(submission.authors):
            if not str(author).strip():
                add_error(
                    errors,
                    f"{context}.authors[{author_index}]: "
                    f"empty author name",
                )

    if hasattr(submission, "submission_type"):
        expected_type = (
            sim_agents_submission_pb2
            .SimAgentsChallengeSubmission
            .SIM_AGENTS_SUBMISSION
        )

        if submission.submission_type != expected_type:
            add_error(
                errors,
                f"{context}.submission_type: expected "
                f"SIM_AGENTS_SUBMISSION ({expected_type}), found "
                f"{submission.submission_type}",
            )

    closed_loop_field = (
        "acknowledge_complies_with_closed_loop_requirement"
    )

    if hasattr(submission, closed_loop_field):
        if not getattr(submission, closed_loop_field):
            add_warning(
                warnings,
                f"{context}.{closed_loop_field}: False",
            )

    if hasattr(submission, "num_model_parameters"):
        if not str(submission.num_model_parameters).strip():
            add_warning(
                warnings,
                f"{context}.num_model_parameters: empty value",
            )


# =============================================================================
# Trajectory validation
# =============================================================================

def get_available_field_names(message: Message) -> set[str]:
    """
    Return all field names defined for a protobuf message.
    """
    return {
        field.name
        for field in message.DESCRIPTOR.fields
    }


def validate_trajectory(
    trajectory: Message,
    context: str,
    errors: list[str],
    warnings: list[str],
    expected_num_steps: int | None,
    max_abs_position: float,
    max_dimension: float,
) -> None:
    """
    Validate one SimulatedTrajectory.
    """
    available_fields = get_available_field_names(trajectory)

    required_sequence_fields = (
        "center_x",
        "center_y",
        "center_z",
        "heading",
    )

    sequence_fields: dict[str, list[float]] = {}

    for field_name in required_sequence_fields:
        if field_name not in available_fields:
            add_error(
                errors,
                f"{context}: missing protobuf field "
                f"{field_name!r}",
            )
            continue

        values = list(getattr(trajectory, field_name))
        sequence_fields[field_name] = values

        if len(values) == 0:
            add_error(
                errors,
                f"{context}.{field_name}: empty sequence",
            )

    if not sequence_fields:
        add_error(
            errors,
            f"{context}: no trajectory sequence fields available",
        )
        return

    sequence_lengths = {
        field_name: len(values)
        for field_name, values in sequence_fields.items()
    }

    if len(set(sequence_lengths.values())) != 1:
        add_error(
            errors,
            f"{context}: inconsistent trajectory field lengths: "
            f"{sequence_lengths}",
        )

    if expected_num_steps is not None:
        for field_name, field_length in sequence_lengths.items():
            if field_length != expected_num_steps:
                add_error(
                    errors,
                    f"{context}.{field_name}: expected "
                    f"{expected_num_steps} values, found "
                    f"{field_length}",
                )

    # Position sanity checks.
    for field_name in ("center_x", "center_y", "center_z"):
        values = sequence_fields.get(field_name)

        if values is None:
            continue

        for step_index, raw_value in enumerate(values):
            value = float(raw_value)

            # NaN/Inf was already reported by the recursive numeric checker.
            if not math.isfinite(value):
                continue

            if abs(value) > max_abs_position:
                add_warning(
                    warnings,
                    f"{context}.{field_name}[{step_index}]: "
                    f"unusually large position {value}",
                )

    # Heading sanity checks.
    headings = sequence_fields.get("heading")

    if headings is not None:
        for step_index, raw_value in enumerate(headings):
            value = float(raw_value)

            if not math.isfinite(value):
                continue

            if abs(value) > 4.0 * math.pi:
                add_warning(
                    warnings,
                    f"{context}.heading[{step_index}]: "
                    f"heading {value} is outside the usual "
                    f"wrapped range",
                )

    # Optional dimensions for protobuf versions that contain these fields.
    for field_name in ("length", "width", "height"):
        if field_name not in available_fields:
            continue

        field_descriptor = trajectory.DESCRIPTOR.fields_by_name[field_name]

        if is_repeated_field(field_descriptor):
            values = list(getattr(trajectory, field_name))
        else:
            values = [getattr(trajectory, field_name)]

        for value_index, raw_value in enumerate(values):
            value = float(raw_value)

            if not math.isfinite(value):
                continue

            if value <= 0.0:
                add_error(
                    errors,
                    f"{context}.{field_name}[{value_index}]: "
                    f"non-positive dimension {value}",
                )
            elif value > max_dimension:
                add_warning(
                    warnings,
                    f"{context}.{field_name}[{value_index}]: "
                    f"unusually large dimension {value}",
                )


# =============================================================================
# Joint-scene validation
# =============================================================================

def validate_joint_scene(
    joint_scene: Message,
    context: str,
    errors: list[str],
    warnings: list[str],
    expected_num_steps: int | None,
    max_abs_position: float,
    max_dimension: float,
) -> tuple[int, tuple[int, ...]]:
    """
    Validate one JointScene.

    Returns:
        Number of trajectories.
        Sorted object IDs.
    """
    if not hasattr(joint_scene, "simulated_trajectories"):
        add_error(
            errors,
            f"{context}: missing simulated_trajectories field",
        )
        return 0, ()

    trajectories = joint_scene.simulated_trajectories

    if len(trajectories) == 0:
        add_error(
            errors,
            f"{context}: no simulated trajectories",
        )
        return 0, ()

    object_ids: list[int] = []

    for trajectory_index, trajectory in enumerate(trajectories):
        object_id = None

        if hasattr(trajectory, "object_id"):
            object_id = int(trajectory.object_id)
            object_ids.append(object_id)

        trajectory_context = (
            f"{context}."
            f"simulated_trajectories[{trajectory_index}]"
        )

        if object_id is not None:
            trajectory_context += f"(object_id={object_id})"

            if object_id < 0:
                add_error(
                    errors,
                    f"{trajectory_context}: negative object ID",
                )
        else:
            add_error(
                errors,
                f"{trajectory_context}: missing object_id field",
            )

        validate_trajectory(
            trajectory=trajectory,
            context=trajectory_context,
            errors=errors,
            warnings=warnings,
            expected_num_steps=expected_num_steps,
            max_abs_position=max_abs_position,
            max_dimension=max_dimension,
        )

    object_id_counts = Counter(object_ids)

    duplicate_object_ids = [
        object_id
        for object_id, count in object_id_counts.items()
        if count > 1
    ]

    if duplicate_object_ids:
        add_error(
            errors,
            f"{context}: duplicate object IDs: "
            f"{duplicate_object_ids[:20]}",
        )

    return len(trajectories), tuple(sorted(object_ids))


# =============================================================================
# Scenario validation
# =============================================================================

def validate_scenario_rollouts(
    scenario: sim_agents_submission_pb2.ScenarioRollouts,
    context: str,
    errors: list[str],
    warnings: list[str],
    expected_num_rollouts: int | None,
    expected_num_steps: int | None,
    max_abs_position: float,
    max_dimension: float,
) -> tuple[int, int]:
    """
    Validate one ScenarioRollouts message.

    Returns:
        Number of joint scenes.
        Total number of simulated trajectories.
    """
    if not scenario.scenario_id:
        add_error(
            errors,
            f"{context}: empty scenario_id",
        )

    joint_scenes = scenario.joint_scenes

    if len(joint_scenes) == 0:
        add_error(
            errors,
            f"{context}: no joint scenes",
        )
        return 0, 0

    if (
        expected_num_rollouts is not None
        and len(joint_scenes) != expected_num_rollouts
    ):
        add_error(
            errors,
            f"{context}: expected {expected_num_rollouts} "
            f"joint scenes, found {len(joint_scenes)}",
        )

    reference_object_ids: tuple[int, ...] | None = None
    total_trajectories = 0

    for rollout_index, joint_scene in enumerate(joint_scenes):
        rollout_context = (
            f"{context}.joint_scenes[{rollout_index}]"
        )

        num_trajectories, object_ids = validate_joint_scene(
            joint_scene=joint_scene,
            context=rollout_context,
            errors=errors,
            warnings=warnings,
            expected_num_steps=expected_num_steps,
            max_abs_position=max_abs_position,
            max_dimension=max_dimension,
        )

        total_trajectories += num_trajectories

        if reference_object_ids is None:
            reference_object_ids = object_ids
        elif object_ids != reference_object_ids:
            reference_set = set(reference_object_ids)
            current_set = set(object_ids)

            missing_ids = sorted(reference_set - current_set)
            extra_ids = sorted(current_set - reference_set)

            add_error(
                errors,
                f"{rollout_context}: object IDs differ from rollout 0; "
                f"missing={missing_ids[:20]}, "
                f"extra={extra_ids[:20]}",
            )

    return len(joint_scenes), total_trajectories


# =============================================================================
# Complete saved-file validation
# =============================================================================

def validate_wosac_submission_file(
    submission_path: str | Path,
    *,
    expected_num_rollouts: int | None = 32,
    expected_num_steps: int | None = 91,
    max_abs_position: float = 1_000_000.0,
    max_dimension: float = 100.0,
    progress_every: int = 25,
    max_reported_errors: int = 500,
    max_reported_warnings: int = 200,
) -> dict[str, Any]:
    """
    Load and validate one saved WOSAC .binproto submission shard.

    Errors and warnings are printed immediately when detected.
    """
    submission_path = Path(submission_path)

    errors: list[str] = []
    warnings: list[str] = []

    num_scenarios = 0
    num_rollouts = 0
    num_trajectories = 0

    seen_scenario_ids: set[str] = set()
    submission = load_submission_file(submission_path)

    # try:
    #     submission = load_submission_file(submission_path)
    # except (FileNotFoundError, OSError, ValueError) as exc:
    #     add_error(errors, str(exc))
    #
    #     return {
    #         "valid": False,
    #         "submission_path": str(submission_path),
    #         "num_scenarios": 0,
    #         "num_unique_scenario_ids": 0,
    #         "num_rollouts": 0,
    #         "num_trajectories": 0,
    #         "num_errors": len(errors),
    #         "num_warnings": len(warnings),
    #         "errors": errors,
    #         "warnings": warnings,
    #         "errors_truncated": False,
    #         "warnings_truncated": False,
    #     }
    #
    context = submission_path.name

    # Proto2 required-field validation.
    initialization_errors = submission.FindInitializationErrors()

    for initialization_error in initialization_errors:
        add_error(
            errors,
            f"{context}: missing required protobuf field "
            f"{initialization_error}",
        )

    validate_submission_metadata(
        submission=submission,
        context=context,
        errors=errors,
        warnings=warnings,
    )

    # Recursively check every float/double value.
    validate_all_numeric_values(
        message=submission,
        errors=errors,
        context=context,
    )

    total_scenarios = len(submission.scenario_rollouts)

    if total_scenarios == 0:
        add_error(
            errors,
            f"{context}: submission contains no scenarios",
        )

    for scenario_index, scenario in enumerate(
        submission.scenario_rollouts
    ):
        num_scenarios += 1

        if (
            progress_every > 0
            and (
                scenario_index == 0
                or (scenario_index + 1) % progress_every == 0
                or scenario_index + 1 == total_scenarios
            )
        ):
            add_info(
                f"Checking scenario "
                f"{scenario_index + 1}/{total_scenarios}: "
                f"{scenario.scenario_id!r}"
            )

        scenario_context = (
            f"{context}."
            f"scenario_rollouts[{scenario_index}]"
            f"(scenario_id={scenario.scenario_id!r})"
        )

        scenario_id = scenario.scenario_id

        if not scenario_id:
            add_error(
                errors,
                f"{scenario_context}: empty scenario_id",
            )
        elif scenario_id in seen_scenario_ids:
            add_error(
                errors,
                f"{scenario_context}: duplicate scenario_id",
            )
        else:
            seen_scenario_ids.add(scenario_id)

        rollout_count, trajectory_count = (
            validate_scenario_rollouts(
                scenario=scenario,
                context=scenario_context,
                errors=errors,
                warnings=warnings,
                expected_num_rollouts=expected_num_rollouts,
                expected_num_steps=expected_num_steps,
                max_abs_position=max_abs_position,
                max_dimension=max_dimension,
            )
        )

        num_rollouts += rollout_count
        num_trajectories += trajectory_count

    total_errors = len(errors)
    total_warnings = len(warnings)

    return {
        "valid": total_errors == 0,
        "submission_path": str(submission_path),
        "num_scenarios": num_scenarios,
        "num_unique_scenario_ids": len(seen_scenario_ids),
        "num_rollouts": num_rollouts,
        "num_trajectories": num_trajectories,
        "num_errors": total_errors,
        "num_warnings": total_warnings,
        "errors": errors[:max_reported_errors],
        "warnings": warnings[:max_reported_warnings],
        "errors_truncated": (
            total_errors > max_reported_errors
        ),
        "warnings_truncated": (
            total_warnings > max_reported_warnings
        ),
    }


# =============================================================================
# Report printing
# =============================================================================

def print_validation_summary(
    report: dict[str, Any],
) -> None:
    """
    Print the final validation summary.
    """
    print("\n" + "=" * 100)
    print("SAVED WOSAC SUBMISSION VALIDATION SUMMARY")
    print("=" * 100)

    print(
        f"Path:              "
        f"{report.get('submission_path', '')}"
    )
    print(
        f"Valid:             "
        f"{report.get('valid', False)}"
    )
    print(
        f"Scenarios:         "
        f"{report.get('num_scenarios', 0)}"
    )
    print(
        f"Unique scenarios:  "
        f"{report.get('num_unique_scenario_ids', 0)}"
    )
    print(
        f"Joint scenes:      "
        f"{report.get('num_rollouts', 0)}"
    )
    print(
        f"Trajectories:      "
        f"{report.get('num_trajectories', 0)}"
    )
    print(
        f"Errors:            "
        f"{report.get('num_errors', 0)}"
    )
    print(
        f"Warnings:          "
        f"{report.get('num_warnings', 0)}"
    )

    if report.get("errors_truncated", False):
        print(
            "[INFO] Some errors were omitted from the stored report, "
            "but every error was printed when detected."
        )

    if report.get("warnings_truncated", False):
        print(
            "[INFO] Some warnings were omitted from the stored report, "
            "but every warning was printed when detected."
        )

    print("=" * 100)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Change this to the saved shard path.
    submission_path = (
        "/home/ke/code/sim/src/debug/"
        "submission.binproto-00027-of-00147"
    )

    report = validate_wosac_submission_file(
        submission_path=submission_path,

        # WOSAC expected number of rollouts per scenario.
        expected_num_rollouts=32,

        # Your attached script currently uses 91 states.
        expected_num_steps=91,

        # Sanity-check thresholds.
        max_abs_position=1_000_000.0,
        max_dimension=100.0,

        # Print progress every 25 scenarios.
        # Errors and warnings are always printed immediately.
        progress_every=25,

        max_reported_errors=500,
        max_reported_warnings=200,
    )

    print_validation_summary(report)

    if not report["valid"]:
        raise RuntimeError(
            f"Invalid saved submission: "
            f"{report['num_errors']} errors and "
            f"{report['num_warnings']} warnings"
        )