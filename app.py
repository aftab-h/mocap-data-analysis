"""
Motion Capture Analysis Dashboard for Head Stability Research

A focused single-page Streamlit application for analyzing motion capture data
to understand head movement patterns during different activities.

Author: Built for Apple ARC Biomechanics Role Portfolio
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import BVHParser, load_bvh, get_head_related_joints
from kinematics import (
    calculate_velocity, calculate_speed, calculate_acceleration,
    calculate_head_stability_metrics, calculate_range_of_motion,
    butter_lowpass_filter, detect_high_motion_events
)
from download_data import download_trials, list_available_files, get_trial_info

# Page config
st.set_page_config(
    page_title="MoCap Head Stability Analysis",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- CACHED DATA LOADING ---
@st.cache_data
def load_and_process_file(path: str, filter_cutoff: float) -> dict:
    """Load BVH file and compute metrics. Cached to avoid reprocessing."""
    parser = load_bvh(path)

    # Find head joint
    head_joints = get_head_related_joints(parser)
    head_joint = head_joints[0] if head_joints else parser.list_joints()[0]

    # Get positions and apply filter
    head_pos = parser.get_joint_positions(head_joint)
    filtered_pos = butter_lowpass_filter(head_pos, filter_cutoff, parser.fps)

    # Get head orientation (nose direction)
    head_orientation = parser.get_joint_orientations(head_joint)

    # Calculate metrics
    metrics = calculate_head_stability_metrics(filtered_pos, parser.fps)

    # Calculate time series
    velocity = calculate_velocity(filtered_pos, parser.fps)
    speed = calculate_speed(velocity)

    return {
        'positions': filtered_pos,
        'orientation': head_orientation,
        'speed': speed,
        'timestamps': parser.get_timestamps(),
        'metrics': metrics,
        'fps': parser.fps,
        'duration': parser.duration,
        'joint': head_joint,
        'n_frames': len(filtered_pos)
    }


def main():
    # --- SIDEBAR ---
    st.sidebar.title("Data Controls")

    # Data discovery
    data_dirs = [Path("data/sfu"), Path("data/raw")]
    available_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            available_files.extend(list_available_files(str(data_dir)))

    if not available_files:
        st.sidebar.warning("No data files found.")
        if st.sidebar.button("Download Sample Data", type="primary"):
            with st.spinner("Downloading walking and running trials..."):
                download_trials(['walking', 'running'], str(data_dirs[0]))
            st.rerun()

        st.title("Head Stability Analysis")
        st.info("Click **Download Sample Data** in the sidebar to get started.")
        return

    # File selection grouped by activity
    st.sidebar.subheader("Select Files")
    activities = {}
    for f in available_files:
        act = f['activity']
        if act not in activities:
            activities[act] = []
        activities[act].append(f)

    # Auto-select walking and jumping by default for quick analysis
    auto_select_activities = ['walking', 'jumping']

    selected_files = []
    for activity, files in activities.items():
        with st.sidebar.expander(f"{activity.title()} ({len(files)})", expanded=True):
            for f in files:
                default_selected = activity.lower() in auto_select_activities
                if st.checkbox(f"{f['filename']}", key=f['path'], value=default_selected):
                    selected_files.append(f)

    if not selected_files:
        st.title("Head Stability Analysis")
        st.info("Select files from the sidebar to begin analysis.")
        return

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    filter_cutoff = st.sidebar.slider("Filter cutoff (Hz)", 1, 20, 10)

    # --- LOAD DATA ---
    all_data = []
    all_metrics = []

    for file_info in selected_files:
        try:
            result = load_and_process_file(file_info['path'], filter_cutoff)
            result['file'] = file_info['filename']
            result['activity'] = file_info['activity']
            result['subject'] = file_info['subject']

            metrics = result['metrics'].copy()
            metrics['file'] = file_info['filename']
            metrics['activity'] = file_info['activity']
            metrics['subject'] = file_info['subject']
            metrics['duration'] = result['duration']

            all_data.append(result)
            all_metrics.append(metrics)
        except Exception as e:
            st.warning(f"Error processing {file_info['filename']}: {e}")

    if not all_metrics:
        st.error("No files could be processed.")
        return

    metrics_df = pd.DataFrame(all_metrics)

    # --- MAIN PAGE ---
    import base64
    img_path = Path(__file__).parent / "image_white.png"
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 15px;">'
        f'Head Stability Analysis '
        f'<img src="data:image/png;base64,{img_b64}" width="50" style="vertical-align: middle;">'
        f'</h1>',
        unsafe_allow_html=True
    )

    st.markdown("""
This dashboard analyzes **motion capture data** to understand how the head moves during
different activities like walking, jumping, and running. The goal is to inform the design
of audio wearables (headphones, earbuds) by quantifying head stability — because a device
that stays put during a walk might not survive a jog.

It automates the full pipeline: loading BVH motion capture files, extracting head joint
kinematics, computing stability metrics, and running statistical comparisons between activities.
    """)

    st.markdown("")

    # Color palette for activities (used throughout)
    activity_colors = px.colors.qualitative.Set2
    activity_list = list(metrics_df['activity'].unique())
    color_map = {act: activity_colors[i % len(activity_colors)] for i, act in enumerate(activity_list)}

    # --- SECTION 1: METRICS TABLE ---
    st.markdown("---")
    st.header("Metrics Summary")

    display_cols = ['file', 'activity', 'mean_speed', 'max_speed', 'max_acceleration', 'stability_index']
    display_df = metrics_df[display_cols].copy()
    display_df.columns = ['File', 'Activity', 'Avg Speed (cm/s)', 'Max Speed (cm/s)', 'Max Accel (cm/s²)', 'Stability']
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("What do these metrics mean?"):
        st.markdown("""
| Metric | Unit | Description |
|--------|------|-------------|
| **Avg Speed** | cm/s | Average head velocity magnitude across all frames |
| **Max Speed** | cm/s | Peak instantaneous head velocity |
| **Max Accel** | cm/s² | Peak instantaneous acceleration (rate of velocity change) |
| **Stability** | index | Combined variability measure: `std(speed) + 0.1 × std(acceleration)`. Lower = more stable head. |
        """)

    # --- SECTION 2: DESCRIPTIVE STATISTICS ---
    st.markdown("---")
    st.header("Descriptive Statistics")

    # Store stats for writeup section
    stats_results = None

    if len(metrics_df['activity'].unique()) >= 2:
        activities = metrics_df['activity'].unique()
        if len(activities) == 2:
            group1_name = activities[0]
            group2_name = activities[1]
            group1_data = metrics_df[metrics_df['activity'] == group1_name]['mean_speed']
            group2_data = metrics_df[metrics_df['activity'] == group2_name]['mean_speed']

            if len(group1_data) >= 2 and len(group2_data) >= 2:
                from scipy import stats

                # Welch's t-test (does not assume equal variances)
                t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)

                # Effect size: auto-select Cohen's d vs Hedges' g based on sample size
                n1, n2 = len(group1_data), len(group2_data)
                pooled_std = np.sqrt(((n1-1)*group1_data.std(ddof=1)**2 + (n2-1)*group2_data.std(ddof=1)**2) / (n1+n2-2))

                if pooled_std > 0:
                    d = abs(group1_data.mean() - group2_data.mean()) / pooled_std
                    if n1 < 20 or n2 < 20:
                        correction = 1 - (3 / (4*(n1+n2) - 9))
                        effect_size = d * correction
                        effect_size_name = "Hedges' g"
                    else:
                        effect_size = d
                        effect_size_name = "Cohen's d"
                else:
                    effect_size = 0
                    effect_size_name = "Cohen's d"

                if effect_size < 0.2:
                    effect_label = "negligible"
                elif effect_size < 0.5:
                    effect_label = "small"
                elif effect_size < 0.8:
                    effect_label = "medium"
                else:
                    effect_label = "large"

                if group1_data.mean() > group2_data.mean():
                    higher_name, lower_name = group1_name, group2_name
                    higher_data, lower_data = group1_data, group2_data
                else:
                    higher_name, lower_name = group2_name, group1_name
                    higher_data, lower_data = group2_data, group1_data

                ratio = higher_data.mean() / lower_data.mean()

                stats_results = {
                    'group1_name': group1_name,
                    'group2_name': group2_name,
                    'group1_n': len(group1_data),
                    'group2_n': len(group2_data),
                    'group1_mean': group1_data.mean(),
                    'group2_mean': group2_data.mean(),
                    'group1_std': group1_data.std(),
                    'group2_std': group2_data.std(),
                    'higher_name': higher_name,
                    'lower_name': lower_name,
                    'ratio': ratio,
                    't_stat': t_stat,
                    'p_val': p_val,
                    'effect_size': effect_size,
                    'effect_size_name': effect_size_name,
                    'effect_label': effect_label,
                    'significant': p_val < 0.05
                }

                if p_val < 0.001:
                    p_str = "p < .001"
                else:
                    p_str = f"p = {p_val:.3f}"

                sig_marker = "**" if p_val < 0.05 else ""

                # Descriptive statistics table
                desc_stats = pd.DataFrame({
                    'Group': [group1_name.title(), group2_name.title()],
                    'n': [n1, n2],
                    'Mean': [f"{group1_data.mean():.2f}", f"{group2_data.mean():.2f}"],
                    'SD': [f"{group1_data.std():.2f}", f"{group2_data.std():.2f}"],
                    'Range': [f"{group1_data.min():.1f} – {group1_data.max():.1f}",
                              f"{group2_data.min():.1f} – {group2_data.max():.1f}"]
                })
                st.dataframe(desc_stats, hide_index=True, use_container_width=True)

                # --- ANALYSIS SECTION ---
                st.markdown("---")
                st.header("Analysis")
                st.caption("Welch's t-test")

                # Strip plot
                fig_strip = px.strip(
                    metrics_df,
                    x='activity',
                    y='mean_speed',
                    color='activity',
                    hover_data=['file', 'max_speed', 'stability_index'],
                    labels={'mean_speed': 'Avg Head Speed (cm/s)', 'activity': 'Activity'},
                    color_discrete_map=color_map
                )
                fig_strip.update_traces(marker=dict(size=12, opacity=0.7))

                # Add mean lines for each group
                for i, activity in enumerate([group1_name, group2_name]):
                    group_mean = metrics_df[metrics_df['activity'] == activity]['mean_speed'].mean()
                    fig_strip.add_shape(
                        type="line",
                        x0=i - 0.3, x1=i + 0.3,
                        y0=group_mean, y1=group_mean,
                        line=dict(color=color_map[activity], width=3, dash="solid"),
                    )
                    # Add mean value annotation
                    fig_strip.add_annotation(
                        x=i + 0.35,
                        y=group_mean,
                        text=f"μ={group_mean:.1f}",
                        showarrow=False,
                        font=dict(size=11, color=color_map[activity]),
                        xanchor="left"
                    )

                fig_strip.update_layout(
                    height=350,
                    showlegend=False,
                    margin=dict(t=20, b=30)
                )
                st.plotly_chart(fig_strip, use_container_width=True)

                st.markdown("")

                # Inferential statistics with color coding
                mean_diff = abs(group1_data.mean() - group2_data.mean())

                # Color for p-value: green if significant, red if not
                p_color = "#4ade80" if p_val < 0.05 else "#f87171"

                # Color for effect size: gradient from red (negligible) to green (large)
                effect_colors = {
                    "negligible": "#f87171",  # red
                    "small": "#fb923c",       # orange
                    "medium": "#facc15",      # yellow
                    "large": "#4ade80"        # green
                }
                effect_color = effect_colors.get(effect_label, "#ffffff")

                st.markdown(f"**{group1_name.title()} vs {group2_name.title()}**")
                st.markdown(f"Mean difference: **{mean_diff:.2f} cm/s**")
                st.markdown(
                    f't = {t_stat:.2f}, <span style="color:{p_color}; font-weight:600">{p_str}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'{effect_size_name} = {effect_size:.2f} (<span style="color:{effect_color}; font-weight:600">{effect_label} effect</span>)',
                    unsafe_allow_html=True
                )

                with st.expander("Notes"):
                    st.markdown("""
**Head Speed Data**

Each dot represents one BVH motion capture file. Head speed is calculated by:
1. Extracting the head joint's 3D position from the BVH skeleton hierarchy
2. Applying a low-pass Butterworth filter to remove sensor noise
3. Computing frame-to-frame velocity (position change / time)
4. Taking the magnitude (speed) of the velocity vector
5. Averaging across all frames to get mean head speed for that recording

**Statistical Test:** Welch's t-test (independent samples)

*Why Welch's?* Unlike Student's t-test, Welch's does NOT assume equal variances
between groups. This is more robust when comparing activities that may have
different variability (e.g., jumping is more variable than walking).

**Effect Size:** Cohen's d (n ≥ 20) or Hedges' g (n < 20)

Hedges' g applies a small-sample correction to Cohen's d, reducing bias
when group sizes are under 20. Interpretation:
- < 0.2: negligible
- 0.2–0.5: small
- 0.5–0.8: medium
- \> 0.8: large

A large effect size means the difference is practically meaningful,
not just statistically detectable.
                    """)
    else:
        st.info("Select files from multiple activities to see comparison")

    # --- SECTION 3: AUTO-GENERATED ANALYSIS WRITEUP ---
    if len(metrics_df['activity'].unique()) >= 2 and stats_results is not None:
        st.markdown("---")
        st.header("Analysis Summary")

        # Build the auto-writeup
        sr = stats_results  # shorthand

        # Descriptive stats paragraph
        writeup = f"""
### Methods

Head position data was extracted from BVH motion capture files and low-pass filtered
(Butterworth, {filter_cutoff} Hz cutoff) to remove high-frequency noise. Average head speed
(cm/s) was calculated as the primary outcome measure, representing the average magnitude
of head velocity across all frames.

To compare head stability between activities, we conducted a **Welch's independent
samples t-test**, which does not assume equal variances between groups. Effect size
was quantified using **{sr['effect_size_name']}**.

### Results

**Sample:** {sr['group1_name'].title()} (n={sr['group1_n']}) vs {sr['group2_name'].title()} (n={sr['group2_n']})

| Group | Avg Speed | SD |
|-------|------------|-----|
| {sr['group1_name'].title()} | {sr['group1_mean']:.2f} cm/s | {sr['group1_std']:.2f} |
| {sr['group2_name'].title()} | {sr['group2_mean']:.2f} cm/s | {sr['group2_std']:.2f} |

"""

        if sr['significant']:
            writeup += f"""
**Finding:** {sr['higher_name'].title()} produced significantly higher head speeds than
{sr['lower_name']} (*t* = {sr['t_stat']:.2f}, *p* = {sr['p_val']:.3f}, {sr['effect_size_name']} = {sr['effect_size']:.2f}).

This represents a **{sr['effect_label']} effect**, with {sr['higher_name']} showing
**{sr['ratio']:.1f}x** the head movement of {sr['lower_name']}.
"""
        else:
            writeup += f"""
**Finding:** No statistically significant difference was found between {sr['group1_name']}
and {sr['group2_name']} (*t* = {sr['t_stat']:.2f}, *p* = {sr['p_val']:.3f}, {sr['effect_size_name']} = {sr['effect_size']:.2f}).
"""

        # Add implications
        writeup += f"""
### Implications for Wearable Design

"""
        if sr['significant'] and sr['ratio'] > 1.3:
            writeup += f"""
The {sr['ratio']:.1f}x difference in head speed between {sr['higher_name']} and {sr['lower_name']}
suggests that audio wearables (headphones, earbuds) face substantially different retention
challenges across activities. Devices designed for {sr['lower_name']} stability may not
remain secure during {sr['higher_name']}.

**Recommendation:** Retention mechanisms should be tested under {sr['higher_name']}-like
conditions (peak speeds ~{metrics_df[metrics_df['activity']==sr['higher_name']]['max_speed'].mean():.0f} cm/s)
to ensure real-world stability.
"""
        else:
            writeup += """
The activities tested show similar head movement profiles, suggesting that retention
requirements may be comparable across these use cases.
"""

        # Add limitations
        writeup += f"""
### Limitations

- **Small sample sizes** (n={sr['group1_n']}, n={sr['group2_n']}): Results should be
  confirmed with larger datasets
- **Single metric:** Average speed captures overall movement but may miss important
  temporal patterns (e.g., periodic vs random motion)
- **Lab vs real-world:** Motion capture data may not fully represent natural movement
"""

        st.markdown(writeup)

        # Copy button for the writeup
        st.download_button(
            label="Download Analysis Report",
            data=writeup,
            file_name="head_stability_analysis.md",
            mime="text/markdown"
        )

    # --- SECTION 4: SPEED PROFILE ---
    st.markdown("---")
    st.header("Speed Profile")

    fig_speed = go.Figure()

    for data in all_data:
        timestamps = data['timestamps'][:-1]  # Speed has one less point
        speed = data['speed']

        # Detect high-motion events (95th percentile)
        high_motion = detect_high_motion_events(speed, threshold_percentile=95)

        # Main speed trace
        fig_speed.add_trace(go.Scatter(
            x=timestamps,
            y=speed,
            mode='lines',
            name=f"{data['file']} ({data['activity']})",
            line=dict(color=color_map[data['activity']], width=1.5),
            hovertemplate="Time: %{x:.2f}s<br>Speed: %{y:.2f} cm/s"
        ))

        # Mark high-motion events
        if np.any(high_motion):
            event_times = timestamps[high_motion]
            event_speeds = speed[high_motion]
            fig_speed.add_trace(go.Scatter(
                x=event_times,
                y=event_speeds,
                mode='markers',
                marker=dict(size=6, color='red', symbol='circle', opacity=0.6),
                name=f"High motion ({data['file']})",
                showlegend=False,
                hovertemplate="HIGH MOTION<br>Time: %{x:.2f}s<br>Speed: %{y:.2f} cm/s"
            ))

    fig_speed.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Speed (cm/s)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=50)
    )

    st.plotly_chart(fig_speed, use_container_width=True)
    st.caption("Red markers indicate high-motion events (>95th percentile speed)")

    # --- SECTION 5: 3D TRAJECTORY EXPLORATION ---
    st.markdown("---")
    st.header("Explore 3D Trajectory")

    selected_viz = st.selectbox(
        "Select file to visualize",
        [d['file'] for d in all_data],
        key='viz_select'
    )

    data = next(d for d in all_data if d['file'] == selected_viz)
    pos = data['positions']
    orientation = data['orientation']
    speed = data['speed']
    timestamps = data['timestamps']
    n_frames = len(pos)

    # Head dimensions (in cm)
    head_radius = 10.0  # ~20cm diameter human head
    ear_radius = 2.5    # Small ear spheres
    nose_length = head_radius + 2.0  # Nose extends past head surface

    # Generate sphere/ellipsoid mesh (low-poly for performance)
    def create_sphere_mesh(center, radius, resolution=12, scale=(1, 1, 1)):
        """Create sphere/ellipsoid vertices centered at a point.

        scale: (sx, sy, sz) to create ellipsoid - e.g., (1, 0.3, 1) for flat disc
        """
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)

        x = center[0] + radius * scale[0] * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * scale[1] * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * scale[2] * np.cos(phi)

        return x.flatten(), y.flatten(), z.flatten()

    def get_ear_positions(head_center, orient_idx):
        """Get left and right ear positions based on head orientation."""
        # Right direction is perpendicular to forward (cross product of forward and up)
        forward = orientation[orient_idx]
        up = np.array([0, 1, 0])  # World up
        right = np.cross(forward, up)
        if np.linalg.norm(right) > 0.01:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1, 0, 0])  # Fallback

        ear_offset = head_radius * 0.9  # Ears at edge of head
        left_ear = head_center + right * ear_offset
        right_ear = head_center - right * ear_offset
        return left_ear, right_ear

    # Initial head sphere position (note: y and z swapped for display)
    head_x, head_y, head_z = create_sphere_mesh(
        [pos[0, 0], pos[0, 2], pos[0, 1]], head_radius
    )

    # Initial ear positions (flattened ellipsoids)
    ear_scale = (0.3, 1, 1)  # Flat in x direction (perpendicular to head)
    left_ear, right_ear = get_ear_positions(pos[0], 0)
    left_ear_x, left_ear_y, left_ear_z = create_sphere_mesh(
        [left_ear[0], left_ear[2], left_ear[1]], ear_radius, resolution=8, scale=ear_scale
    )
    right_ear_x, right_ear_y, right_ear_z = create_sphere_mesh(
        [right_ear[0], right_ear[2], right_ear[1]], ear_radius, resolution=8, scale=ear_scale
    )

    # Animation settings
    frame_step = st.select_slider(
        "Frame sampling (lower = smoother)",
        options=[1, 2, 5, 10, 25, 50, 100],
        value=10
    )

    trail_frames = st.slider("Trail length", 5, 100, 30)

    st.markdown("")

    # Downsample for animation performance
    frame_indices = list(range(0, n_frames, frame_step))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    speed_padded = np.concatenate([[speed[0]], speed])

    # Build animated 3D figure
    fig_3d = go.Figure()

    # Invisible anchor trace to lock axis ranges (not animated)
    fig_3d.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 2], z=pos[:, 1],
        mode='markers',
        marker=dict(size=0, opacity=0),
        hoverinfo='skip',
        showlegend=False
    ))

    # Trail (animated)
    fig_3d.add_trace(go.Scatter3d(
        x=pos[:1, 0], y=pos[:1, 2], z=pos[:1, 1],
        mode='lines+markers',
        marker=dict(size=3, color=[speed_padded[0]], colorscale='Viridis',
                   colorbar=dict(title="Speed"), showscale=True, cmin=0, cmax=np.percentile(speed, 95)),
        line=dict(color='rgba(100,100,100,0.7)', width=2),
        name='Trail'
    ))

    # Current head position (scaled 3D mesh sphere)
    fig_3d.add_trace(go.Mesh3d(
        x=head_x, y=head_y, z=head_z,
        alphahull=0,
        color='rgba(100, 200, 255, 0.4)',
        opacity=0.4,
        name='Head'
    ))

    # Left ear
    fig_3d.add_trace(go.Mesh3d(
        x=left_ear_x, y=left_ear_y, z=left_ear_z,
        alphahull=0,
        color='rgba(100, 200, 255, 0.4)',
        opacity=0.4,
        name='Left Ear'
    ))

    # Right ear
    fig_3d.add_trace(go.Mesh3d(
        x=right_ear_x, y=right_ear_y, z=right_ear_z,
        alphahull=0,
        color='rgba(100, 200, 255, 0.4)',
        opacity=0.4,
        name='Right Ear'
    ))

    # Nose direction indicator (line from head center)
    nose_end = pos[0] + orientation[0] * nose_length
    fig_3d.add_trace(go.Scatter3d(
        x=[pos[0, 0], nose_end[0]],
        y=[pos[0, 2], nose_end[2]],
        z=[pos[0, 1], nose_end[1]],
        mode='lines',
        line=dict(color='rgba(255, 100, 100, 0.9)', width=6),
        name='Nose Direction'
    ))

    # Start/End markers
    fig_3d.add_trace(go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 2]], z=[pos[0, 1]],
        mode='markers', marker=dict(size=8, color='green', symbol='diamond'),
        name='Start'
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[pos[-1, 0]], y=[pos[-1, 2]], z=[pos[-1, 1]],
        mode='markers', marker=dict(size=8, color='blue', symbol='diamond'),
        name='End'
    ))

    # Create animation frames
    frames = []
    for idx in frame_indices:
        trail_start = max(0, idx - trail_frames * frame_step)
        t_pos = pos[trail_start:idx+1]
        t_speed = speed_padded[trail_start:idx+1]

        # Calculate nose endpoint for this frame
        nose_end_frame = pos[idx] + orientation[idx] * nose_length

        # Calculate head sphere for this frame
        hx, hy, hz = create_sphere_mesh(
            [pos[idx, 0], pos[idx, 2], pos[idx, 1]], head_radius
        )

        # Calculate ear positions for this frame (flattened)
        l_ear, r_ear = get_ear_positions(pos[idx], idx)
        lex, ley, lez = create_sphere_mesh([l_ear[0], l_ear[2], l_ear[1]], ear_radius, resolution=8, scale=ear_scale)
        rex, rey, rez = create_sphere_mesh([r_ear[0], r_ear[2], r_ear[1]], ear_radius, resolution=8, scale=ear_scale)

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=pos[:, 0], y=pos[:, 2], z=pos[:, 1]),  # Anchor (unchanged)
                go.Scatter3d(x=t_pos[:, 0], y=t_pos[:, 2], z=t_pos[:, 1],
                            marker=dict(size=3, color=t_speed, colorscale='Viridis',
                                       cmin=0, cmax=np.percentile(speed, 95))),
                go.Mesh3d(x=hx, y=hy, z=hz, alphahull=0,
                         color='rgba(100, 200, 255, 0.4)', opacity=0.4),  # Head sphere
                go.Mesh3d(x=lex, y=ley, z=lez, alphahull=0,
                         color='rgba(100, 200, 255, 0.4)', opacity=0.4),  # Left ear
                go.Mesh3d(x=rex, y=rey, z=rez, alphahull=0,
                         color='rgba(100, 200, 255, 0.4)', opacity=0.4),  # Right ear
                go.Scatter3d(x=[pos[idx, 0], nose_end_frame[0]],
                            y=[pos[idx, 2], nose_end_frame[2]],
                            z=[pos[idx, 1], nose_end_frame[1]]),  # Nose direction
                go.Scatter3d(x=[pos[0, 0]], y=[pos[0, 2]], z=[pos[0, 1]]),  # Start
                go.Scatter3d(x=[pos[-1, 0]], y=[pos[-1, 2]], z=[pos[-1, 1]])  # End
            ],
            name=str(idx)
        ))

    fig_3d.frames = frames

    # Calculate real-time frame duration (ms)
    real_time_duration = int(frame_step / data['fps'] * 1000)

    # Calculate fixed axis ranges from full trajectory (with padding for nose)
    padding = nose_length + 0.5  # Extra buffer beyond nose length
    x_range = [pos[:, 0].min() - padding, pos[:, 0].max() + padding]
    y_range = [pos[:, 2].min() - padding, pos[:, 2].max() + padding]  # Z mapped to Y axis
    z_range = [pos[:, 1].min() - padding, pos[:, 1].max() + padding]  # Y mapped to Z axis

    # Layout with Play/Pause and slider
    fig_3d.update_layout(
        title=f"Head Trajectory - {selected_viz} ({data['activity']})",
        scene=dict(
            xaxis=dict(title="X - Lateral (cm)", range=x_range, autorange=False),
            yaxis=dict(title="Z - Forward (cm)", range=y_range, autorange=False),
            zaxis=dict(title="Y - Vertical (cm)", range=z_range, autorange=False),
            aspectmode='cube',
            camera=dict(projection=dict(type='perspective'))
        ),
        uirevision='constant',
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        height=600,
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            y=-0.08, x=0.13, xanchor="right", yanchor="middle",
            bgcolor="rgba(40, 40, 40, 0.9)",
            bordercolor="rgba(60, 60, 60, 0.8)",
            font=dict(color="white", size=14),
            buttons=[
                dict(label=" ▶ ", method="animate",
                     args=[None, {"frame": {"duration": real_time_duration, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label=" ⏸ ", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            y=-0.08, x=0.15, len=0.83,
            currentvalue={"prefix": "Time: ", "suffix": "s", "visible": True},
            steps=[dict(args=[[str(idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                       label=f"{int(round(timestamps[idx]))}" if abs(timestamps[idx] - round(timestamps[idx])) < 0.05 else "",
                       method="animate")
                   for idx in frame_indices]
        )]
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # --- EXPORT ---
    st.markdown("---")
    csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="Export Metrics CSV",
        data=csv,
        file_name="head_stability_metrics.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
