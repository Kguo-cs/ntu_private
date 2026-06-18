# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2

from .video_recorder import ImageEncoder


COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_VIOLET = (170, 0, 255)
COLOR_BUTTER = (252, 233, 79)
COLOR_ORANGE = (209, 92, 0)
COLOR_CHOCOLATE = (143, 89, 2)
COLOR_CHAMELEON = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (32, 74, 135)
COLOR_PLUM = (92, 53, 102)
COLOR_SCARLET_RED = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (66, 62, 64)


class VisWaymo:
    def __init__(
        self,
        scenario_path: str,
        save_dir: Path,
        px_per_m: float = 10.0,
        video_size: int = 960,
        n_step: int = 91,
        step_current: int = 10,
        vis_ghost_gt: bool = True,
    ) -> None:
        self.px_per_m = px_per_m
        self.video_size = video_size
        self.n_step = n_step
        self.step_current = step_current
        self.px_agent2bottom = video_size // 2
        self.vis_ghost_gt = vis_ghost_gt

        # colors
        self.lane_style = [
            (COLOR_WHITE, 6),        # FREEWAY = 0
            (COLOR_ALUMINIUM_2, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE, 6),       # STOP_SIGN = 2
            (COLOR_CHOCOLATE, 6),    # BIKE_LANE = 3
            (COLOR_SKY_BLUE_1, 4),   # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM, 4),         # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER, 2),       # BROKEN = 6
            (COLOR_MAGENTA, 2),      # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON, 4),    # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),   # CROSSWALK = 10
        ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0
            COLOR_RED,          # STOP = 1
            COLOR_YELLOW,       # CAUTION = 2
            COLOR_GREEN,        # GO = 3
            COLOR_VIOLET,       # FLASHING = 4
        ]

        # sdc=0, interest=1, predict=2
        self.agent_role_style = [
            COLOR_CYAN,
            COLOR_ALUMINIUM_0,
            COLOR_ALUMINIUM_0,
        ]

        self.agent_cmd_txt = [
            "STATIONARY",
            "STRAIGHT",
            "STRAIGHT_LEFT",
            "STRAIGHT_RIGHT",
            "LEFT_U_TURN",
            "LEFT_TURN",
            "RIGHT_U_TURN",
            "RIGHT_TURN",
        ]

        # load tfrecord scenario
        scenario = scenario_pb2.Scenario()
        for data in tf.data.TFRecordDataset([scenario_path], compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break

        # make output dir
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # draw gt
        mp_xyz, mp_id, mp_type = get_map_features(scenario.map_features)

        tl_lane_state, tl_lane_id = get_traffic_light_features(
            scenario.dynamic_map_states
        )

        ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id = get_agent_features(
            scenario,
            step_current=step_current,
        )

        self.ag_valid = ag_valid
        self.ag_xy = ag_xy
        self.ag_yaw = ag_yaw
        self.ag_size = ag_size
        self.ag_role = ag_role
        self.ag_id = ag_id

        sdc_idx = np.where(ag_role[:, 0])[0]
        if len(sdc_idx) > 0:
            self.ego_current_xy = ag_xy[sdc_idx[0], step_current].copy()
        else:
            self.ego_current_xy = ag_xy[0, step_current].copy()

        self.ag_id2size = dict(zip(ag_id, ag_size))
        self.ag_id2role = dict(zip(ag_id, ag_role))

        raster_map, self.top_left_px = self._register_map(mp_xyz, self.px_per_m)
        self._draw_map(raster_map, mp_xyz, mp_type)

        self.interval = 2

        im_gt_maps = [raster_map.copy() for _ in range(0, n_step, self.interval)]

        self._draw_traffic_lights(
            im_gt_maps,
            tl_lane_state[:: self.interval],
            tl_lane_id[:: self.interval],
            mp_xyz,
            mp_id,
        )

        # save gt video and get paths for wandb logging
        im_gt = deepcopy(im_gt_maps)
        self._draw_agents(
            im_gt,
            ag_valid[:, :: self.interval],
            ag_xy[:, :: self.interval],
            ag_yaw[:, :: self.interval],
            ag_size,
            ag_role,
        )

        gt_video_path = (self.save_dir / "gt.mp4").as_posix()
        self.video_paths = [gt_video_path]

        # prepare images for drawing prediction on top
        self.im_gt_blended = []

        if self.vis_ghost_gt:
            im_gt_agents = [
                np.zeros_like(raster_map) for _ in range(0, n_step, self.interval)
            ]

            self._draw_agents(
                im_gt_agents,
                ag_valid[:, :: self.interval],
                ag_xy[:, :: self.interval],
                ag_yaw[:, :: self.interval],
                ag_size,
                ag_role,
            )

            for i in range(len(im_gt_agents)):
                self.im_gt_blended.append(
                    cv2.addWeighted(im_gt_agents[i], 0.5, im_gt_maps[i], 1, 0)
                )
        else:
            for i in range(len(im_gt_maps)):
                if i <= self.step_current // self.interval:
                    self.im_gt_blended.append(deepcopy(im_gt[i]))
                else:
                    self.im_gt_blended.append(deepcopy(im_gt_maps[i]))

    @staticmethod
    def _register_map(
        mp_xyz: List[np.ndarray],
        px_per_m: float,
        edge_px: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            mp_xyz: len=n_pl, list of np array [n_pl_node, 3]
            px_per_m: float

        Returns:
            raster_map: empty image
            top_left_px
        """
        xmin = min([arr[:, 0].min() for arr in mp_xyz])
        xmax = max([arr[:, 0].max() for arr in mp_xyz])
        ymin = min([arr[:, 1].min() for arr in mp_xyz])
        ymax = max([arr[:, 1].max() for arr in mp_xyz])
        map_boundary = np.array([xmin, xmax, ymin, ymax])

        # y axis is inverted in pixel coordinate
        xmin, xmax, ymax, ymin = (map_boundary * px_per_m).astype(np.int64)

        ymax *= -1
        ymin *= -1

        xmin -= edge_px
        ymin -= edge_px
        xmax += edge_px
        ymax += edge_px

        raster_map = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin], dtype=np.float32)

        return raster_map, top_left_px

    def _draw_map(
        self,
        raster_map: np.ndarray,
        mp_xyz: List[np.ndarray],
        mp_type: np.ndarray,
    ) -> None:
        """
        Args:
            raster_map: image canvas
            mp_xyz: len=n_pl, list of np array [n_pl_node, 3]
            mp_type: [n_pl], int
        """
        for i, _type in enumerate(mp_type):
            color, thickness = self.lane_style[_type]

            cv2.polylines(
                raster_map,
                [self._to_pixel(mp_xyz[i][:, :2])],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_traffic_lights(
        self,
        input_images: List[np.ndarray],
        tl_lane_state: List[np.ndarray],
        tl_lane_id: List[np.ndarray],
        mp_xyz: List[np.ndarray],
        mp_id: np.ndarray,
    ) -> None:
        for step_t, step_image in enumerate(input_images):
            if step_t >= len(tl_lane_state):
                continue

            for i_tl, _state in enumerate(tl_lane_state[step_t]):
                _lane_id = tl_lane_id[step_t][i_tl]

                lane_match = np.argwhere(mp_id == _lane_id)
                if lane_match.size == 0:
                    continue

                _lane_idx = lane_match.item()
                pos = self._to_pixel(mp_xyz[_lane_idx][:, :2])

                cv2.polylines(
                    step_image,
                    [pos],
                    isClosed=False,
                    color=self.tl_style[_state],
                    thickness=8,
                    lineType=cv2.LINE_AA,
                )

                if 1 <= _state <= 3:
                    cv2.drawMarker(
                        step_image,
                        pos[-1],
                        color=self.tl_style[_state],
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=10,
                        thickness=6,
                    )

    def _draw_agents(
        self,
        input_images: List[np.ndarray],
        ag_valid: np.ndarray,  # [n_ag, n_step], bool
        ag_xy: np.ndarray,     # [n_ag, n_step, 2], (x, y)
        ag_yaw: np.ndarray,    # [n_ag, n_step, 1], [-pi, pi]
        ag_size: np.ndarray,   # [n_ag, 3], [length, width, height]
        ag_role: np.ndarray,   # [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
    ) -> None:
        for step_t, step_image in enumerate(input_images):
            if step_t >= ag_valid.shape[1]:
                continue

            _valid = ag_valid[:, step_t]

            if not _valid.any():
                continue

            _pos = ag_xy[:, step_t]
            _yaw = ag_yaw[:, step_t]

            bbox_gt = self._to_pixel(
                self._get_agent_bbox(_valid, _pos, _yaw, ag_size)
            )

            heading_start = self._to_pixel(_pos[_valid])

            _yaw_valid = _yaw[:, 0][_valid]
            heading_end = self._to_pixel(
                _pos[_valid]
                + 1.5
                * np.stack(
                    [np.cos(_yaw_valid), np.sin(_yaw_valid)],
                    axis=-1,
                )
            )

            _role = ag_role[_valid]

            for i in range(_role.shape[0]):
                if not _role[i].any():
                    color = COLOR_ALUMINIUM_0
                else:
                    color = self.agent_role_style[np.where(_role[i])[0].min()]

                cv2.fillConvexPoly(step_image, bbox_gt[i], color=color)

                cv2.arrowedLine(
                    step_image,
                    heading_start[i],
                    heading_end[i],
                    color=COLOR_BLACK,
                    thickness=4,
                    line_type=cv2.LINE_AA,
                    tipLength=0.6,
                )

    def save_video_scenario_rollout(
        self,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        n_vis_rollout: int,
        new_agent=None,
        pred_z_list=None,
        crop_size_m: Optional[float] = 100.0,
        add_subtitle: bool = True,
    ):
        """
        Save one video per rollout.

        If pred_z_list is non-empty, each video contains:

            [initial generation process frames] + [closed-loop rollout frames]

        If pred_z_list is None or empty, the initial generation process is skipped,
        and the video contains only the rollout frames.

        Args:
            scenario_rollout:
                Waymo ScenarioRollouts object.

            n_vis_rollout:
                Number of rollout samples to visualize.

            new_agent:
                Kept for backward compatibility.

            pred_z_list:
                Optional dense generation tensor.

                Supported shapes:
                    [N_agent, N_rollout, N_gen_step, D]
                    [N_agent, N_gen_step, D]

                Expected D layout:
                    [x, y, cos_heading, sin_heading, length, width, vel_x, vel_y]

            crop_size_m:
                If not None, crop every frame to crop_size_m x crop_size_m meters
                centered at the ego current position.

                crop_size_m=100.0 means total width/length is 100 meters.
                Use crop_size_m=200.0 if you want ±100 meters around ego.

            add_subtitle:
                Whether to draw subtitles on frames.
        """
        for i_rollout in range(n_vis_rollout):
            images = []

            # ----------------------------------------------------------
            # 1. Optional initial-generation process frames.
            #    Skip if pred_z_list / scenario_pred_z_list is empty.
            # ----------------------------------------------------------
            gen_z_i = self._select_generation_rollout(
                pred_z_list=pred_z_list,
                i_rollout=i_rollout,
            )

            if gen_z_i is not None:
                gen_images = self._make_generation_images_from_z_list(
                    pred_z_list=gen_z_i,
                    subtitle_prefix="Initial generation",
                    crop_size_m=crop_size_m,
                    add_subtitle=add_subtitle,
                    rollout_idx=i_rollout,
                )
                images.extend(gen_images)

            # ----------------------------------------------------------
            # 2. Existing closed-loop rollout frames.
            # ----------------------------------------------------------
            rollout_images = deepcopy(self.im_gt_blended)

            ag_valid, ag_xy, ag_yaw, ag_size, ag_role = self._get_features_from_trajs(
                scenario_rollout.joint_scenes[i_rollout].simulated_trajectories
            )

            if len(rollout_images) == ag_valid[:, :: self.interval].shape[1]:
                self._draw_agents(
                    rollout_images,
                    ag_valid[:, :: self.interval],
                    ag_xy[:, :: self.interval],
                    ag_yaw[:, :: self.interval],
                    ag_size,
                    ag_role,
                )
            else:
                self._draw_agents(
                    rollout_images[self.step_current // self.interval + 1 :],
                    ag_valid[:, self.interval - 1 :: self.interval],
                    ag_xy[:, self.interval - 1 :: self.interval],
                    ag_yaw[:, self.interval - 1 :: self.interval],
                    ag_size,
                    ag_role,
                )

            processed_rollout_images = []

            for frame_idx, frame in enumerate(rollout_images):
                frame_out = frame

                if crop_size_m is not None:
                    frame_out = self._crop_around_ego(frame_out, crop_size_m)

                if add_subtitle:
                    frame_out = self._add_subtitle(
                        frame_out,
                        f"Closed-loop rollout | rollout {i_rollout:02d} | frame {frame_idx:03d}",
                    )

                processed_rollout_images.append(frame_out)

            images.extend(processed_rollout_images)

            if len(images) == 0:
                continue

            _video_path = (self.save_dir / f"rollout_{i_rollout:02d}.mp4").as_posix()
            self.video_paths.append(_video_path)

            save_images_to_mp4(
                images,
                _video_path,
                fps=10 // self.interval,
            )

    def _select_generation_rollout(self, pred_z_list, i_rollout: int):
        """
        Select one rollout's generation sequence from pred_z_list.

        Returns:
            None if pred_z_list is None or empty.
            np.ndarray [N_agent, N_gen_step, D] otherwise.
        """
        if pred_z_list is None:
            return None

        if hasattr(pred_z_list, "detach"):
            pred_z = pred_z_list.detach().cpu().numpy()
        else:
            pred_z = np.asarray(pred_z_list)

        if pred_z.size == 0:
            return None

        # Empty scenario: [0, ...]
        if pred_z.shape[0] == 0:
            return None

        # [N_agent, N_rollout, N_gen_step, D]
        if pred_z.ndim == 4:
            if pred_z.shape[1] == 0:
                return None

            if i_rollout >= pred_z.shape[1]:
                return None

            gen_z_i = pred_z[:, i_rollout]

        # [N_agent, N_gen_step, D]
        elif pred_z.ndim == 3:
            gen_z_i = pred_z

        else:
            raise ValueError(
                f"pred_z_list must have shape [N,R,G,D] or [N,G,D], "
                f"got {pred_z.shape}"
            )

        if gen_z_i.size == 0:
            return None

        if gen_z_i.shape[0] == 0 or gen_z_i.shape[1] == 0:
            return None

        return gen_z_i

    def _make_generation_images_from_z_list(
        self,
        pred_z_list: np.ndarray,
        subtitle_prefix: str = "Initial generation",
        crop_size_m: Optional[float] = 100.0,
        add_subtitle: bool = True,
        rollout_idx: int = 0,
    ) -> List[np.ndarray]:
        """
        Convert dense generated initial states into video frames.

        Args:
            pred_z_list:
                [N_agent, N_gen_step, D]

            D layout:
                [x, y, cos_heading, sin_heading, length, width, vel_x, vel_y]

        Returns:
            List of rendered frames.
        """
        if pred_z_list is None:
            return []

        pred_z_list = np.asarray(pred_z_list)

        if pred_z_list.size == 0:
            return []

        if pred_z_list.ndim != 3:
            raise ValueError(
                f"pred_z_list must have shape [N_agent, N_gen_step, D], "
                f"got {pred_z_list.shape}"
            )

        n_agent, n_gen_step, state_dim = pred_z_list.shape

        if n_agent == 0 or n_gen_step == 0:
            return []

        if state_dim < 2:
            raise ValueError(
                f"pred_z_list last dimension must be at least 2, got {state_dim}"
            )

        base_idx = min(
            self.step_current // self.interval,
            len(self.im_gt_blended) - 1,
        )
        base_image = self.im_gt_blended[base_idx]

        images = []

        for gen_t in range(n_gen_step):
            step_image = deepcopy(base_image)

            z_t = pred_z_list[:, gen_t]

            valid = np.isfinite(z_t[:, :2]).all(axis=-1)

            xy = z_t[:, None, :2].astype(np.float32)

            # Heading from cos/sin if available.
            if state_dim >= 4:
                yaw = np.arctan2(z_t[:, 3], z_t[:, 2]).astype(np.float32)
            else:
                yaw = np.zeros((n_agent,), dtype=np.float32)

            yaw = yaw[:, None, None]

            size = self._make_generation_agent_size(z_t, n_agent, state_dim)
            role = self._make_generation_agent_role(n_agent)

            self._draw_agents(
                [step_image],
                valid[:, None],
                xy,
                yaw,
                size,
                role,
            )

            if crop_size_m is not None:
                step_image = self._crop_around_ego(step_image, crop_size_m)

            if add_subtitle:
                step_image = self._add_subtitle(
                    step_image,
                    (
                        f"{subtitle_prefix} | rollout {rollout_idx:02d} | "
                        f"step {gen_t + 1:02d}/{n_gen_step:02d}"
                    ),
                )

            images.append(step_image)

        return images

    def _make_generation_agent_size(
        self,
        z_t: np.ndarray,
        n_agent: int,
        state_dim: int,
    ) -> np.ndarray:
        """
        Build agent size array for generation frames.

        Uses generated length/width if available; otherwise falls back to
        scenario sizes. Extra generated agents receive default vehicle size.
        """
        size = np.zeros((n_agent, 3), dtype=np.float32)

        n_existing = min(n_agent, len(self.ag_size))

        if n_existing > 0:
            size[:n_existing] = self.ag_size[:n_existing]

        if n_existing < n_agent:
            size[n_existing:, 0] = 4.5
            size[n_existing:, 1] = 2.0
            size[n_existing:, 2] = 1.5

        if state_dim >= 6:
            gen_length = np.nan_to_num(z_t[:, 4], nan=4.5, posinf=4.5, neginf=4.5)
            gen_width = np.nan_to_num(z_t[:, 5], nan=2.0, posinf=2.0, neginf=2.0)

            size[:, 0] = np.maximum(gen_length, 0.1)
            size[:, 1] = np.maximum(gen_width, 0.1)

            if np.any(size[:, 2] <= 0):
                size[:, 2] = np.maximum(size[:, 2], 1.5)

        return size

    def _make_generation_agent_role(self, n_agent: int) -> np.ndarray:
        """
        Build role array for generation frames.

        Uses scenario roles where available. Extra generated agents are drawn
        as normal non-role agents.
        """
        role = np.zeros((n_agent, 3), dtype=bool)

        n_existing = min(n_agent, len(self.ag_role))

        if n_existing > 0:
            role[:n_existing] = self.ag_role[:n_existing]

        return role

    def _crop_around_ego(
        self,
        image: np.ndarray,
        crop_size_m: float = 100.0,
    ) -> np.ndarray:
        """
        Crop image to crop_size_m x crop_size_m meters centered at ego current position.

        If crop extends outside the raster map, zero-pad the missing area.
        """
        crop_px = int(round(crop_size_m * self.px_per_m))
        crop_px = max(crop_px, 1)

        center_px = self._to_pixel(self.ego_current_xy[None].copy())[0]
        cx, cy = int(center_px[0]), int(center_px[1])

        half = crop_px // 2

        x0 = cx - half
        x1 = x0 + crop_px

        y0 = cy - half
        y1 = y0 + crop_px

        h, w = image.shape[:2]

        src_x0 = max(x0, 0)
        src_x1 = min(x1, w)

        src_y0 = max(y0, 0)
        src_y1 = min(y1, h)

        dst_x0 = src_x0 - x0
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        dst_y0 = src_y0 - y0
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        cropped = np.zeros((crop_px, crop_px, 3), dtype=image.dtype)

        if src_x1 > src_x0 and src_y1 > src_y0:
            cropped[dst_y0:dst_y1, dst_x0:dst_x1] = image[
                src_y0:src_y1,
                src_x0:src_x1,
            ]

        return cropped

    def _add_subtitle(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        Draw a readable subtitle at the bottom of the frame.
        """
        out = image.copy()

        h, w = out.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(w, h) / 900.0)
        thickness = max(1, int(round(font_scale * 2)))

        margin = max(8, int(0.015 * w))
        box_h = max(34, int(0.065 * h))

        # Reduce font scale if text is too wide.
        while font_scale > 0.35:
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            if text_size[0] <= w - 2 * margin:
                break
            font_scale *= 0.9

        y0 = h - box_h
        y1 = h

        overlay = out.copy()

        cv2.rectangle(
            overlay,
            (0, y0),
            (w, y1),
            COLOR_BLACK,
            -1,
        )

        out = cv2.addWeighted(
            overlay,
            0.55,
            out,
            0.45,
            0,
        )

        text_origin = (margin, h - margin)

        cv2.putText(
            out,
            text,
            text_origin,
            font,
            font_scale,
            COLOR_WHITE,
            thickness,
            cv2.LINE_AA,
        )

        return out

    def _get_features_from_trajs(
        self,
        trajs: List[sim_agents_submission_pb2.SimulatedTrajectory],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ag_valid: [n_ag, n_step], bool
        ag_xy: [n_ag, n_step, 2], (x,y)
        ag_yaw: [n_ag, n_step, 1], [-pi, pi]
        ag_size: [n_ag, 3], [length, width, height]
        ag_role: [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
        """
        n_ag = len(trajs)
        n_step = len(trajs[0].center_x)

        ag_valid = np.ones([n_ag, n_step], dtype=bool)
        ag_xy = np.zeros([n_ag, n_step, 2], dtype=np.float32)
        ag_yaw = np.zeros([n_ag, n_step, 1], dtype=np.float32)
        ag_size = np.zeros([n_ag, 3], dtype=np.float32)
        ag_role = np.zeros([n_ag, 3], dtype=bool)

        for i_ag, _traj in enumerate(trajs):
            ag_xy[i_ag] = np.stack(
                [_traj.center_x, _traj.center_y],
                axis=-1,
            )
            ag_yaw[i_ag, :, 0] = _traj.heading

            if len(_traj.length):
                ag_size[i_ag, 0] = _traj.length[0]
                ag_size[i_ag, 1] = _traj.width[0]
                ag_size[i_ag, 2] = _traj.height[0]
            else:
                ag_size[i_ag] = self.ag_id2size[_traj.object_id]

            ag_role[i_ag] = self.ag_id2role[_traj.object_id]

        return ag_valid, ag_xy, ag_yaw, ag_size, ag_role

    def _to_pixel(self, pos: np.ndarray) -> np.ndarray:
        pos = pos * self.px_per_m
        pos[..., 0] = pos[..., 0] - self.top_left_px[0]
        pos[..., 1] = -pos[..., 1] - self.top_left_px[1]
        return np.round(pos).astype(np.int32)

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray,
        agent_pos: np.ndarray,
        agent_yaw: np.ndarray,
        agent_size: np.ndarray,
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # [n, 1]

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset

        return bbox


def save_images_to_mp4(images: List[np.ndarray], out_path: str, fps=20) -> None:
    encoder = ImageEncoder(out_path, images[0].shape, fps, fps)

    for im in images:
        encoder.capture_frame(im)

    encoder.close()
    encoder = None


def get_agent_features(
    scenario: scenario_pb2.Scenario,
    step_current: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ag_valid: [n_ag, n_step], bool
    ag_xy: [n_ag, n_step, 2], (x,y)
    ag_yaw: [n_ag, n_step, 1], [-pi, pi]
    ag_size: [n_ag, 3], [length, width, height]
    ag_role: [n_ag, 3], one_hot [sdc=0, interest=1, predict=2]
    ag_id: [n_ag], int
    """
    tracks = scenario.tracks
    sdc_track_index = scenario.sdc_track_index
    track_index_predict = [i.track_index for i in scenario.tracks_to_predict]
    object_id_interest = [i for i in scenario.objects_of_interest]

    ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id = [], [], [], [], [], []

    for i, _track in enumerate(tracks):
        if not _track.states[step_current].valid:
            continue

        ag_id.append(_track.id)

        step_valid, step_xy, step_yaw = [], [], []

        for s in _track.states:
            step_valid.append(s.valid)
            step_xy.append([s.center_x, s.center_y])
            step_yaw.append([s.heading])

        ag_valid.append(step_valid)
        ag_xy.append(step_xy)
        ag_yaw.append(step_yaw)

        ag_size.append(
            [
                _track.states[step_current].length,
                _track.states[step_current].width,
                _track.states[step_current].height,
            ]
        )

        ag_role.append([False, False, False])

        if i in track_index_predict:
            ag_role[-1][2] = True

        if _track.id in object_id_interest:
            ag_role[-1][1] = True

        if i == sdc_track_index:
            ag_role[-1][0] = True

    ag_valid = np.array(ag_valid)
    ag_xy = np.array(ag_xy)
    ag_yaw = np.array(ag_yaw)
    ag_size = np.array(ag_size)
    ag_role = np.array(ag_role)
    ag_id = np.array(ag_id)

    return ag_valid, ag_xy, ag_yaw, ag_size, ag_role, ag_id


def get_traffic_light_features(
    tl_features,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    n_tl is not constant for each timestep.

    tl_lane_state:
        len=n_step, list of array [n_tl]

    tl_lane_id:
        len=n_step, list of array [n_tl]
    """
    tl_lane_state, tl_lane_id = [], []

    for _step_tl in tl_features:
        step_tl_lane_state, step_tl_lane_id = [], []

        for _tl in _step_tl.lane_states:
            if _tl.state == 0:
                tl_state = 0
            elif _tl.state in [1, 4]:
                tl_state = 1
            elif _tl.state in [2, 5]:
                tl_state = 2
            elif _tl.state in [3, 6]:
                tl_state = 3
            elif _tl.state in [7, 8]:
                tl_state = 4
            else:
                raise ValueError(f"Unknown traffic light state: {_tl.state}")

            step_tl_lane_state.append(tl_state)
            step_tl_lane_id.append(_tl.lane)

        tl_lane_state.append(np.array(step_tl_lane_state))
        tl_lane_id.append(np.array(step_tl_lane_id))

    return tl_lane_state, tl_lane_id


def get_map_features(
    map_features,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    mp_xyz, mp_id, mp_type = [], [], []

    for mf in map_features:
        feature_data_type = mf.WhichOneof("feature_data")

        if feature_data_type is None:
            continue

        feature = getattr(mf, feature_data_type)

        if feature_data_type == "lane":
            if feature.type == 0:
                mp_type.append(1)
            elif feature.type == 1:
                mp_type.append(0)
            elif feature.type == 2:
                mp_type.append(1)
            elif feature.type == 3:
                mp_type.append(3)
            else:
                mp_type.append(1)

            mp_id.append(mf.id)
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])

        elif feature_data_type == "stop_sign":
            for l_id in feature.lane:
                if l_id not in mp_id:
                    continue

                idx_lane = mp_id.index(l_id)

                if mp_type[idx_lane] < 2:
                    mp_type[idx_lane] = 2

        elif feature_data_type == "road_edge":
            assert feature.type > 0

            mp_id.append(mf.id)
            mp_type.append(feature.type + 3)
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])

        elif feature_data_type == "road_line":
            assert feature.type > 0

            if feature.type in [1, 4, 5]:
                feature_type_new = 6
            elif feature.type in [2, 6]:
                feature_type_new = 7
            else:
                feature_type_new = 8

            mp_id.append(mf.id)
            mp_type.append(feature_type_new)
            mp_xyz.append([[p.x, p.y, p.z] for p in feature.polyline][::2])

        elif feature_data_type in ["speed_bump", "driveway", "crosswalk"]:
            xyz = np.array([[p.x, p.y, p.z] for p in feature.polygon])
            polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)

            pl_polygon = _get_polylines_from_polygon(xyz[polygon_idx])

            mp_xyz.extend(pl_polygon)
            mp_id.extend([mf.id] * len(pl_polygon))

            pl_type = 9 if feature_data_type in ["speed_bump", "driveway"] else 10
            mp_type.extend([pl_type] * len(pl_polygon))

        else:
            raise ValueError(f"Unsupported map feature type: {feature_data_type}")

    mp_id = np.array(mp_id)
    mp_type = np.array(mp_type)
    mp_xyz = [np.stack(line) for line in mp_xyz]

    return mp_xyz, mp_id, mp_type


def _get_polylines_from_polygon(polygon: np.ndarray) -> List[List[List]]:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> List[List]:
        length = np.linalg.norm(start - end)

        if length < 1e-6:
            return [[start[0], start[1], start[2]]]

        unit_vec = (end - start) / length
        pl = []

        for i in range(int(length) + 1):
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])

        pl.append([end[0], end[1], end[2]])

        return pl

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])

    return [pl1, pl1[::-1], pl2, pl2[::-1]]