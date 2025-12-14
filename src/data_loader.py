"""BVH file loading and joint position extraction utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class BVHParser:
    """Parser for BVH motion capture files."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.joints: Dict[str, dict] = {}
        self.motion_data: np.ndarray = None
        self.frame_time: float = 0.0
        self.num_frames: int = 0
        self.channel_order: List[str] = []
        self._parse()

    def _parse(self):
        """Parse the BVH file."""
        with open(self.filepath, 'r') as f:
            content = f.read()

        # Split into hierarchy and motion sections
        parts = content.split('MOTION')
        hierarchy = parts[0]
        motion = parts[1] if len(parts) > 1 else ''

        self._parse_hierarchy(hierarchy)
        self._parse_motion(motion)

    def _parse_hierarchy(self, hierarchy: str):
        """Parse the skeleton hierarchy."""
        lines = hierarchy.strip().split('\n')
        joint_stack = []
        current_joint = None

        for line in lines:
            line = line.strip()

            if line.startswith('ROOT') or line.startswith('JOINT'):
                parts = line.split()
                joint_name = parts[1]
                self.joints[joint_name] = {
                    'offset': [0.0, 0.0, 0.0],
                    'channels': [],
                    'parent': joint_stack[-1] if joint_stack else None,
                    'children': []
                }
                if joint_stack:
                    self.joints[joint_stack[-1]]['children'].append(joint_name)
                current_joint = joint_name

            elif line.startswith('End Site'):
                current_joint = None

            elif line.startswith('OFFSET'):
                if current_joint:
                    parts = line.split()
                    self.joints[current_joint]['offset'] = [
                        float(parts[1]), float(parts[2]), float(parts[3])
                    ]

            elif line.startswith('CHANNELS'):
                if current_joint:
                    parts = line.split()
                    num_channels = int(parts[1])
                    channels = parts[2:2+num_channels]
                    self.joints[current_joint]['channels'] = channels
                    for ch in channels:
                        self.channel_order.append((current_joint, ch))

            elif line == '{':
                if current_joint:
                    joint_stack.append(current_joint)

            elif line == '}':
                if joint_stack:
                    joint_stack.pop()

    def _parse_motion(self, motion: str):
        """Parse the motion data section."""
        lines = motion.strip().split('\n')
        data_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('Frames:'):
                self.num_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split(':')[1].strip())
            elif line and not line.startswith('Frames') and not line.startswith('Frame'):
                values = [float(v) for v in line.split()]
                data_lines.append(values)

        self.motion_data = np.array(data_lines)

    def get_joint_positions(self, joint_name: str) -> np.ndarray:
        """
        Calculate world positions for a joint across all frames.

        Returns:
            np.ndarray: Shape (num_frames, 3) with X, Y, Z positions
        """
        if joint_name not in self.joints:
            raise ValueError(f"Joint '{joint_name}' not found. Available: {list(self.joints.keys())}")

        positions = np.zeros((self.num_frames, 3))

        for frame_idx in range(self.num_frames):
            pos = self._calculate_world_position(joint_name, frame_idx)
            positions[frame_idx] = pos

        return positions

    def _calculate_world_position(self, joint_name: str, frame_idx: int) -> np.ndarray:
        """Calculate world position of a joint at a specific frame."""
        # Build chain from root to joint
        chain = []
        current = joint_name
        while current is not None:
            chain.insert(0, current)
            current = self.joints[current]['parent']

        # Start from root position
        position = np.array([0.0, 0.0, 0.0])
        rotation_matrix = np.eye(3)

        for jname in chain:
            joint = self.joints[jname]

            # Apply rotation from parent
            offset = np.array(joint['offset'])
            position = position + rotation_matrix @ offset

            # Get rotation for this joint
            local_rotation = self._get_joint_rotation(jname, frame_idx)
            rotation_matrix = rotation_matrix @ local_rotation

            # Add translation if this joint has position channels (usually root)
            trans = self._get_joint_translation(jname, frame_idx)
            if trans is not None:
                position = position + trans

        return position

    def _get_joint_rotation(self, joint_name: str, frame_idx: int) -> np.ndarray:
        """Get rotation matrix for a joint at a frame."""
        joint = self.joints[joint_name]
        channels = joint['channels']

        # Find channel indices in motion data
        rotation_matrix = np.eye(3)

        rotation_order = []
        for ch in channels:
            if ch in ['Xrotation', 'Yrotation', 'Zrotation']:
                idx = self._get_channel_index(joint_name, ch)
                angle = np.radians(self.motion_data[frame_idx, idx])
                rotation_order.append((ch[0], angle))

        # Apply rotations in order specified
        for axis, angle in rotation_order:
            rotation_matrix = rotation_matrix @ self._rotation_matrix(axis, angle)

        return rotation_matrix

    def _get_joint_translation(self, joint_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get translation for a joint at a frame (usually only root has this)."""
        joint = self.joints[joint_name]
        channels = joint['channels']

        trans = np.array([0.0, 0.0, 0.0])
        has_translation = False

        for ch in channels:
            if ch == 'Xposition':
                idx = self._get_channel_index(joint_name, ch)
                trans[0] = self.motion_data[frame_idx, idx]
                has_translation = True
            elif ch == 'Yposition':
                idx = self._get_channel_index(joint_name, ch)
                trans[1] = self.motion_data[frame_idx, idx]
                has_translation = True
            elif ch == 'Zposition':
                idx = self._get_channel_index(joint_name, ch)
                trans[2] = self.motion_data[frame_idx, idx]
                has_translation = True

        return trans if has_translation else None

    def _get_channel_index(self, joint_name: str, channel: str) -> int:
        """Get the index of a channel in the motion data."""
        for i, (jname, ch) in enumerate(self.channel_order):
            if jname == joint_name and ch == channel:
                return i
        raise ValueError(f"Channel {channel} not found for joint {joint_name}")

    @staticmethod
    def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
        """Create a rotation matrix for rotation around an axis."""
        c, s = np.cos(angle), np.sin(angle)

        if axis == 'X':
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        elif axis == 'Y':
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis == 'Z':
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Unknown axis: {axis}")

    @property
    def fps(self) -> float:
        """Frames per second."""
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.num_frames * self.frame_time

    def get_timestamps(self) -> np.ndarray:
        """Get array of timestamps for each frame."""
        return np.arange(self.num_frames) * self.frame_time

    def list_joints(self) -> List[str]:
        """List all available joints."""
        return list(self.joints.keys())

    def to_dataframe(self, joints: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert motion data to a pandas DataFrame.

        Args:
            joints: List of joint names to include. If None, includes all joints.

        Returns:
            DataFrame with columns for time and each joint's X, Y, Z positions.
        """
        if joints is None:
            joints = self.list_joints()

        data = {'time': self.get_timestamps()}

        for joint in joints:
            if joint in self.joints:
                positions = self.get_joint_positions(joint)
                data[f'{joint}_x'] = positions[:, 0]
                data[f'{joint}_y'] = positions[:, 1]
                data[f'{joint}_z'] = positions[:, 2]

        return pd.DataFrame(data)


def load_bvh(filepath: str) -> BVHParser:
    """Load a BVH file and return a parser object."""
    return BVHParser(filepath)


def get_head_related_joints(parser: BVHParser) -> List[str]:
    """
    Find head-related joints in the skeleton.

    Common naming conventions:
    - Head, Neck, Spine, Shoulder
    """
    head_keywords = ['head', 'neck', 'skull', 'cranium']
    joints = parser.list_joints()

    head_joints = []
    for joint in joints:
        if any(kw in joint.lower() for kw in head_keywords):
            head_joints.append(joint)

    return head_joints
