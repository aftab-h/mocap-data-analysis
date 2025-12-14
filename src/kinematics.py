"""Kinematic calculations for motion capture data analysis."""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Dict, Tuple, Optional


def calculate_velocity(positions: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculate velocity from position data.

    Args:
        positions: Shape (n_frames, 3) or (n_frames,) position data
        fps: Frames per second

    Returns:
        Velocity array (same shape as input, one frame shorter)
    """
    dt = 1.0 / fps
    velocity = np.diff(positions, axis=0) / dt
    return velocity


def calculate_acceleration(velocity: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculate acceleration from velocity data.

    Args:
        velocity: Velocity array
        fps: Frames per second

    Returns:
        Acceleration array (one frame shorter than velocity)
    """
    dt = 1.0 / fps
    acceleration = np.diff(velocity, axis=0) / dt
    return acceleration


def calculate_speed(velocity: np.ndarray) -> np.ndarray:
    """
    Calculate scalar speed from velocity vectors.

    Args:
        velocity: Shape (n_frames, 3) velocity vectors

    Returns:
        Shape (n_frames,) speed values
    """
    if velocity.ndim == 1:
        return np.abs(velocity)
    return np.linalg.norm(velocity, axis=1)


def calculate_jerk(acceleration: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculate jerk (rate of change of acceleration).

    High jerk indicates sudden movements - relevant to headphone stability.

    Args:
        acceleration: Acceleration array
        fps: Frames per second

    Returns:
        Jerk array
    """
    dt = 1.0 / fps
    jerk = np.diff(acceleration, axis=0) / dt
    return jerk


def smooth_signal(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to remove noise.

    Args:
        data: Input signal
        window_size: Size of smoothing window

    Returns:
        Smoothed signal
    """
    if data.ndim == 1:
        return uniform_filter1d(data, size=window_size, mode='nearest')
    else:
        return np.array([
            uniform_filter1d(data[:, i], size=window_size, mode='nearest')
            for i in range(data.shape[1])
        ]).T


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fps: float, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.

    Args:
        data: Input signal
        cutoff: Cutoff frequency in Hz
        fps: Sampling rate (frames per second)
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = fps / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]).T


def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Calculate angle at p2 between vectors p2->p1 and p2->p3.

    Args:
        p1, p2, p3: Position arrays, shape (n_frames, 3) or (3,)

    Returns:
        Angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    if v1.ndim == 1:
        # Single frame
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    else:
        # Multiple frames
        dot_products = np.sum(v1 * v2, axis=1)
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-10
        cos_angles = dot_products / norms
        return np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))


def calculate_neck_flexion(head_pos: np.ndarray, neck_pos: np.ndarray,
                           spine_pos: np.ndarray) -> np.ndarray:
    """
    Calculate neck flexion angle (forward/backward head tilt).

    Args:
        head_pos: Head position, shape (n_frames, 3)
        neck_pos: Neck position, shape (n_frames, 3)
        spine_pos: Upper spine position, shape (n_frames, 3)

    Returns:
        Flexion angles in degrees
    """
    return calculate_angle_3d(head_pos, neck_pos, spine_pos)


def calculate_head_tilt(head_pos: np.ndarray, reference_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate head tilt angles from vertical.

    Args:
        head_pos: Head position, shape (n_frames, 3)
        reference_pos: Neck/shoulder position, shape (n_frames, 3)

    Returns:
        Dictionary with 'pitch', 'roll', 'lateral_tilt' angles
    """
    # Vector from reference to head
    head_vec = head_pos - reference_pos

    # Pitch: forward/backward tilt (in X-Z plane relative to Y)
    pitch = np.degrees(np.arctan2(head_vec[:, 2], head_vec[:, 1]))

    # Roll/lateral tilt: side-to-side (in X-Y plane)
    lateral = np.degrees(np.arctan2(head_vec[:, 0], head_vec[:, 1]))

    return {
        'pitch': pitch,
        'lateral_tilt': lateral
    }


def calculate_range_of_motion(positions: np.ndarray) -> Dict[str, float]:
    """
    Calculate range of motion statistics for a joint.

    Args:
        positions: Shape (n_frames, 3) position data

    Returns:
        Dictionary with ROM metrics
    """
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])

    # Total 3D displacement
    total_displacement = np.sqrt(x_range**2 + y_range**2 + z_range**2)

    return {
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'total_range': total_displacement,
        'x_min': np.min(positions[:, 0]),
        'x_max': np.max(positions[:, 0]),
        'y_min': np.min(positions[:, 1]),
        'y_max': np.max(positions[:, 1]),
        'z_min': np.min(positions[:, 2]),
        'z_max': np.max(positions[:, 2])
    }


def calculate_head_stability_metrics(head_positions: np.ndarray, fps: float) -> Dict[str, float]:
    """
    Calculate comprehensive head stability metrics.

    These metrics are relevant to headphone/earphone stability assessment.

    Args:
        head_positions: Shape (n_frames, 3) head position data
        fps: Frames per second

    Returns:
        Dictionary of stability metrics
    """
    # Filter the data to remove high-frequency noise
    filtered_pos = butter_lowpass_filter(head_positions, cutoff=10, fps=fps)

    # Calculate kinematics
    velocity = calculate_velocity(filtered_pos, fps)
    speed = calculate_speed(velocity)
    acceleration = calculate_acceleration(velocity, fps)
    acc_magnitude = calculate_speed(acceleration)

    # If we have enough data, calculate jerk
    jerk_magnitude = np.array([])
    if len(acceleration) > 1:
        jerk = calculate_jerk(acceleration, fps)
        jerk_magnitude = calculate_speed(jerk)

    # Range of motion
    rom = calculate_range_of_motion(head_positions)

    metrics = {
        # Speed metrics
        'mean_speed': np.mean(speed),
        'max_speed': np.max(speed),
        'std_speed': np.std(speed),
        'median_speed': np.median(speed),

        # Acceleration metrics
        'mean_acceleration': np.mean(acc_magnitude),
        'max_acceleration': np.max(acc_magnitude),
        'std_acceleration': np.std(acc_magnitude),

        # Jerk metrics (smoothness indicator)
        'mean_jerk': np.mean(jerk_magnitude) if len(jerk_magnitude) > 0 else 0,
        'max_jerk': np.max(jerk_magnitude) if len(jerk_magnitude) > 0 else 0,

        # Range of motion
        'total_rom': rom['total_range'],
        'vertical_rom': rom['y_range'],
        'lateral_rom': rom['x_range'],
        'forward_rom': rom['z_range'],

        # Path length (total distance traveled)
        'path_length': np.sum(speed) / fps,

        # Stability index (lower = more stable)
        # Combines variability in position and acceleration
        'stability_index': np.std(speed) + 0.1 * np.std(acc_magnitude)
    }

    return metrics


def detect_high_motion_events(speed: np.ndarray, threshold_percentile: float = 95) -> np.ndarray:
    """
    Detect frames with unusually high motion (potential instability events).

    Args:
        speed: Speed array
        threshold_percentile: Percentile above which motion is considered "high"

    Returns:
        Boolean array indicating high-motion frames
    """
    threshold = np.percentile(speed, threshold_percentile)
    return speed > threshold


def segment_by_activity(speed: np.ndarray, fps: float,
                       low_threshold: float = 0.1,
                       high_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Segment motion data by activity level.

    Args:
        speed: Speed array
        fps: Frames per second
        low_threshold: Speed below this is "stationary"
        high_threshold: Speed above this is "high activity"

    Returns:
        Dictionary with boolean masks for each activity level
    """
    # Smooth speed for segmentation
    smooth_speed = smooth_signal(speed, window_size=int(fps * 0.1))

    return {
        'stationary': smooth_speed < low_threshold,
        'low_activity': (smooth_speed >= low_threshold) & (smooth_speed < high_threshold),
        'high_activity': smooth_speed >= high_threshold
    }


def compare_activities(metrics_list: list, activity_names: list) -> pd.DataFrame:
    """
    Create a comparison DataFrame across multiple activities.

    Args:
        metrics_list: List of metrics dictionaries
        activity_names: List of activity names

    Returns:
        DataFrame comparing metrics across activities
    """
    df = pd.DataFrame(metrics_list)
    df.insert(0, 'activity', activity_names)
    return df
