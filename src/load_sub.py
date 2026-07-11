import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError, Message
from waymo_open_dataset.protos import sim_agents_submission_pb2


FLOAT_TYPES = {
    FieldDescriptor.TYPE_FLOAT,
    FieldDescriptor.TYPE_DOUBLE,
}

INTEGER_TYPES = {
    FieldDescriptor.TYPE_INT32,
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT32,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_SINT32,
    FieldDescriptor.TYPE_SINT64,
    FieldDescriptor.TYPE_FIXED32,
    FieldDescriptor.TYPE_FIXED64,
    FieldDescriptor.TYPE_SFIXED32,
    FieldDescriptor.TYPE_SFIXED64,
}


def add_error(errors: list[str], message: str) -> None:
    """Store and immediately print an error."""
    errors.append(message)
    print(f"[ERROR] {message}", flush=True)


def add_warning(warnings: list[str], message: str) -> None:
    """Store and immediately print a warning."""
    warnings.append(message)
    print(f"[WARNING] {message}", flush=True)


def is_repeated_field(field: FieldDescriptor) -> bool:
    repeated = getattr(field, "is_repeated", None)

    if repeated is not None:
        return bool(repeated)

    return field.label == FieldDescriptor.LABEL_REPEATED


def iter_protobuf_values(
    message: Message,
    prefix: str = "",
) -> Iterator[tuple[str, Any, FieldDescriptor]]:
    for field, value in message.ListFields():
        field_path = f"{prefix}.{field.name}" if prefix else field.name

        if is_repeated_field(field):
            for index, item in enumerate(value):
                item_path = f"{field_path}[{index}]"

                if field.type == FieldDescriptor.TYPE_MESSAGE:
                    yield from iter_protobuf_values(item, item_path)
                else:
                    yield item_path, item, field
        else:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                yield from iter_protobuf_values(value, field_path)
            else:
                yield field_path, value, field


def validate_all_numeric_values(
    message: Message,
    errors: list[str],
    context: str,
) -> None:
    for path, value, field in iter_protobuf_values(message):
        if field.type not in FLOAT_TYPES:
            continue

        if not math.isfinite(float(value)):
            add_error(
                errors,
                f"{context}.{path}: non-finite value {value!r}",
            )


def get_repeated_numeric_fields(
    message: Message,
) -> dict[str, list[Any]]:
    result = {}

    for field in message.DESCRIPTOR.fields:
        if not is_repeated_field(field):
            continue

        if field.type not in FLOAT_TYPES | INTEGER_TYPES:
            continue

        result[field.name] = list(getattr(message, field.name))

    return result


def validate_trajectory(
    trajectory: Message,
    context: str,
    errors: list[str],
    warnings: list[str],
    expected_num_steps: int | None,
    max_abs_position: float,
    max_dimension: float,
) -> None:
    numeric_fields = get_repeated_numeric_fields(trajectory)

    sequence_field_names = {
        "center_x",
        "center_y",
        "center_z",
        "heading",
        "length",
        "width",
        "height",
        "valid",
    }

    sequence_fields = {
        name: values
        for name, values in numeric_fields.items()
        if name in sequence_field_names
    }

    nonempty_lengths = {
        name: len(values)
        for name, values in sequence_fields.items()
        if len(values) > 0
    }

    if not nonempty_lengths:
        add_error(
            errors,
            f"{context}: no populated trajectory sequence fields",
        )
        return

    unique_lengths = set(nonempty_lengths.values())

    if len(unique_lengths) != 1:
        add_error(
            errors,
            f"{context}: inconsistent trajectory field lengths: "
            f"{nonempty_lengths}",
        )

    if expected_num_steps is not None:
        for field_name, field_length in nonempty_lengths.items():
            if field_length != expected_num_steps:
                add_error(
                    errors,
                    f"{context}.{field_name}: expected "
                    f"{expected_num_steps} values, found {field_length}",
                )

    for field_name in ("center_x", "center_y", "center_z"):
        values = sequence_fields.get(field_name)

        if values is None:
            continue

        for step_index, value in enumerate(values):
            value = float(value)

            if not math.isfinite(value):
                add_error(
                    errors,
                    f"{context}.{field_name}[{step_index}]: "
                    f"non-finite value {value!r}",
                )
            elif abs(value) > max_abs_position:
                add_warning(
                    warnings,
                    f"{context}.{field_name}[{step_index}]: "
                    f"unusually large position {value}",
                )

    headings = sequence_fields.get("heading")

    if headings is not None:
        for step_index, value in enumerate(headings):
            value = float(value)

            if not math.isfinite(value):
                add_error(
                    errors,
                    f"{context}.heading[{step_index}]: "
                    f"non-finite value {value!r}",
                )
            elif abs(value) > 4.0 * math.pi:
                add_warning(
                    warnings,
                    f"{context}.heading[{step_index}]: "
                    f"heading {value} is far outside the usual range",
                )

    for field_name in ("length", "width", "height"):
        values = sequence_fields.get(field_name)

        if values is None:
            continue

        for step_index, value in enumerate(values):
            value = float(value)

            if not math.isfinite(value):
                add_error(
                    errors,
                    f"{context}.{field_name}[{step_index}]: "
                    f"non-finite value {value!r}",
                )
            elif value <= 0.0:
                add_error(
                    errors,
                    f"{context}.{field_name}[{step_index}]: "
                    f"non-positive dimension {value}",
                )
            elif value > max_dimension:
                add_warning(
                    warnings,
                    f"{context}.{field_name}[{step_index}]: "
                    f"unusually large dimension {value}",
                )


def validate_joint_scene(
    joint_scene: Message,
    context: str,
    errors: list[str],
    warnings: list[str],
    expected_num_steps: int | None,
    max_abs_position: float,
    max_dimension: float,
) -> tuple[int, tuple[int, ...]]:
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

    object_ids = []

    for trajectory_index, trajectory in enumerate(trajectories):
        trajectory_context = (
            f"{context}.simulated_trajectories[{trajectory_index}]"
        )

        if hasattr(trajectory, "object_id"):
            object_id = int(trajectory.object_id)
            object_ids.append(object_id)

            if object_id < 0:
                add_error(
                    errors,
                    f"{trajectory_context}.object_id: "
                    f"negative ID {object_id}",
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

    duplicate_ids = [
        object_id
        for object_id, count in Counter(object_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        add_error(
            errors,
            f"{context}: duplicate object IDs: {duplicate_ids[:20]}",
        )

    return len(trajectories), tuple(sorted(object_ids))


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
    if not scenario.scenario_id:
        add_error(errors, f"{context}: empty scenario_id")

    joint_scenes = scenario.joint_scenes

    if len(joint_scenes) == 0:
        add_error(errors, f"{context}: no joint scenes")
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

    reference_object_ids = None
    total_trajectories = 0

    for rollout_index, joint_scene in enumerate(joint_scenes):
        rollout_context = f"{context}.joint_scenes[{rollout_index}]"

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
            add_error(
                errors,
                f"{rollout_context}: object IDs differ from rollout 0; "
                f"rollout 0 has {len(reference_object_ids)} objects, "
                f"this rollout has {len(object_ids)} objects",
            )

    return len(joint_scenes), total_trajectories


def validate_submission_metadata(
    submission: Message,
    context: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    required_string_fields = (
        "account_name",
        "unique_method_name",
        "affiliation",
        "description",
    )

    for field_name in required_string_fields:
        if not hasattr(submission, field_name):
            continue

        value = getattr(submission, field_name)

        if not str(value).strip():
            add_error(
                errors,
                f"{context}.{field_name}: empty value",
            )

    if hasattr(submission, "authors"):
        if len(submission.authors) == 0:
            add_error(errors, f"{context}.authors: no authors")
        #
        # for author_index, author in enumerate(submission.authors):
        #     print(author_index, author)
        #     if not str(author).strip():
        #         add_error(
        #             errors,
        #             f"{context}.authors[{author_index}]: empty author",
        #         )

    field_name = (
        "acknowledge_complies_with_closed_loop_requirement"
    )

    if hasattr(submission, field_name):
        if not getattr(submission, field_name):
            add_warning(
                warnings,
                f"{context}: closed-loop requirement "
                f"acknowledgement is False",
            )

import re

def validate_wosac_tar(
    tar_path: str | Path,
    *,
    expected_num_rollouts: int | None = 32,
    expected_num_steps: int | None = 80,
    max_abs_position: float = 1_000_000.0,
    max_dimension: float = 100.0,
    max_reported_errors: int = 500,
    max_reported_warnings: int = 200,
) -> dict[str, Any]:
    tar_path = Path(tar_path)

    errors = []
    warnings = []

    num_archive_files = 0
    num_shards = 0
    num_scenarios = 0
    num_rollouts = 0
    num_trajectories = 0

    seen_scenario_ids = set()

    if not tar_path.is_file():
        add_error(errors, f"File does not exist: {tar_path}")

        return {
            "valid": False,
            "tar_path": str(tar_path),
            "num_errors": len(errors),
            "num_warnings": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    try:
        with tarfile.open(tar_path, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                if "submission.binproto" not in member.name:
                    continue

                match = re.search(
                    r"submission\.binproto-(\d+)-of-(\d+)",
                    member.name,
                )

                if match is None:
                    add_warning(
                        warnings,
                        f"{member.name}: cannot determine shard index; skipped",
                    )
                    continue

                shard_index = int(match.group(1))
                total_shards = int(match.group(2))

                if shard_index < 57:
                    continue

                num_archive_files += 1
                num_shards += 1

                print(
                    f"[INFO] Checking shard index {shard_index} "
                    f"({shard_index + 1}/{total_shards}): "
                    f"{member.name}",
                    flush=True,
                )

                file_obj = tar.extractfile(member)

                if file_obj is None:
                    add_error(
                        errors,
                        f"{member.name}: could not open archive member",
                    )
                    continue

                try:
                    serialized_data = file_obj.read()
                except (EOFError, OSError) as exc:
                    add_error(
                        errors,
                        f"{member.name}: failed to read member: {exc}",
                    )
                    continue

                if not serialized_data:
                    add_error(
                        errors,
                        f"{member.name}: empty binproto file",
                    )
                    continue

                submission = (
                    sim_agents_submission_pb2
                    .SimAgentsChallengeSubmission()
                )

                try:
                    submission.ParseFromString(serialized_data)
                except DecodeError as exc:
                    add_error(
                        errors,
                        f"{member.name}: protobuf decode failed: {exc}",
                    )
                    continue

                validate_submission_metadata(
                    submission,
                    member.name,
                    errors,
                    warnings,
                )

                validate_all_numeric_values(
                    submission,
                    errors,
                    member.name,
                )

                if len(submission.scenario_rollouts) == 0:
                    add_error(
                        errors,
                        f"{member.name}: shard contains no scenarios",
                    )
                    continue

                for scenario_index, scenario in enumerate(
                        submission.scenario_rollouts
                ):
                    num_scenarios += 1

                    scenario_context = (
                        f"{member.name}."
                        f"scenario_rollouts[{scenario_index}]"
                    )

                    scenario_id = scenario.scenario_id

                    if scenario_id:
                        if scenario_id in seen_scenario_ids:
                            add_error(
                                errors,
                                f"{scenario_context}: duplicate "
                                f"scenario_id {scenario_id!r}",
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

    except (tarfile.TarError, EOFError, OSError) as exc:
        add_error(
            errors,
            f"Failed to read tar archive: {exc}",
        )

    if num_shards == 0:
        add_error(
            errors,
            "Archive contains no submission.binproto shards",
        )

    return {
        "valid": len(errors) == 0,
        "tar_path": str(tar_path),
        "num_archive_files": num_archive_files,
        "num_shards": num_shards,
        "num_scenarios": num_scenarios,
        "num_unique_scenario_ids": len(seen_scenario_ids),
        "num_rollouts": num_rollouts,
        "num_trajectories": num_trajectories,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors[:max_reported_errors],
        "warnings": warnings[:max_reported_warnings],
        "errors_truncated": len(errors) > max_reported_errors,
        "warnings_truncated": len(warnings) > max_reported_warnings,
    }


if __name__ == "__main__":
    report = validate_wosac_tar(
        "/home/ke/code/sim/src/logs/my/"
        "2026-07-10_17-03-56/"
        "wosac_submission.tar.gz",
        expected_num_rollouts=32,
        expected_num_steps=91,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Valid:             {report['valid']}")
    print(f"Shards:            {report['num_shards']}")
    print(f"Scenarios:         {report['num_scenarios']}")
    print(f"Unique scenarios:  {report['num_unique_scenario_ids']}")
    print(f"Rollouts:          {report['num_rollouts']}")
    print(f"Trajectories:      {report['num_trajectories']}")
    print(f"Errors:            {report['num_errors']}")
    print(f"Warnings:          {report['num_warnings']}")

    if not report["valid"]:
        raise RuntimeError(
            f"Invalid submission: {report['num_errors']} errors found"
        )