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
    initial_sidebar_state="expanded"
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

    # Calculate metrics
    metrics = calculate_head_stability_metrics(filtered_pos, parser.fps)

    # Calculate time series
    velocity = calculate_velocity(filtered_pos, parser.fps)
    speed = calculate_speed(velocity)

    return {
        'positions': filtered_pos,
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
    st.sidebar.title("Analysis Controls")

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

    selected_files = []
    for activity, files in activities.items():
        with st.sidebar.expander(f"{activity.title()} ({len(files)})", expanded=True):
            for f in files:
                if st.checkbox(f"{f['filename']}", key=f['path']):
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
    st.title("Head Stability Analysis")
    st.caption("Analyzing head movement patterns for audio device stability research")

    # --- SECTION 1: SPEED TIME SERIES WITH EVENT MARKERS ---
    st.header("Speed Profile")

    fig_speed = go.Figure()

    # Color palette for activities
    activity_colors = px.colors.qualitative.Set2
    activity_list = list(metrics_df['activity'].unique())
    color_map = {act: activity_colors[i % len(activity_colors)] for i, act in enumerate(activity_list)}

    for data in all_data:
        timestamps = data['timestamps'][:-1]  # Speed has one less point
        speed = data['speed']

        # Detect high-motion events (95th percentile)
        high_motion = detect_high_motion_events(speed, threshold_percentile=95)
        threshold = np.percentile(speed, 95)

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
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=50, b=50)
    )

    st.plotly_chart(fig_speed, use_container_width=True)
    st.caption("Red markers indicate high-motion events (>95th percentile speed)")

    # --- SECTION 2: METRICS TABLE + ACTIVITY COMPARISON ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Metrics Summary")
        display_cols = ['file', 'activity', 'mean_speed', 'max_speed', 'max_acceleration', 'stability_index']
        display_df = metrics_df[display_cols].copy()
        display_df.columns = ['File', 'Activity', 'Avg Speed (cm/s)', 'Max Speed (cm/s)', 'Max Accel (cm/s²)', 'Stability']
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Metrics explanation
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            | Metric | Unit | Description |
            |--------|------|-------------|
            | **Avg Speed** | cm/s | Average head velocity magnitude across all frames |
            | **Max Speed** | cm/s | Peak instantaneous head velocity |
            | **Max Accel** | cm/s² | Peak instantaneous acceleration (rate of velocity change) |
            | **Stability** | index | Combined variability measure: `std(speed) + 0.1 × std(acceleration)`. Lower = more stable head. |
            """)

    with col2:
        st.subheader("Activity Comparison")

        # Store stats for writeup section
        stats_results = None

        if len(metrics_df['activity'].unique()) >= 2:
            # Strip plot showing individual data points
            fig_strip = px.strip(
                metrics_df,
                x='activity',
                y='mean_speed',
                color='activity',
                hover_data=['file', 'max_speed', 'stability_index'],
                labels={'mean_speed': 'Avg Speed (cm/s)', 'activity': 'Activity'},
                color_discrete_map=color_map
            )
            fig_strip.update_traces(marker=dict(size=12, opacity=0.7))
            fig_strip.update_layout(
                height=300,
                showlegend=False,
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig_strip, use_container_width=True)

            # Statistical comparison
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

                    # Cohen's d effect size
                    pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
                    cohens_d = abs(group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0

                    # Effect size interpretation
                    if cohens_d < 0.2:
                        effect_label = "negligible"
                    elif cohens_d < 0.5:
                        effect_label = "small"
                    elif cohens_d < 0.8:
                        effect_label = "medium"
                    else:
                        effect_label = "large"

                    # Determine which is higher
                    if group1_data.mean() > group2_data.mean():
                        higher_name, lower_name = group1_name, group2_name
                        higher_data, lower_data = group1_data, group2_data
                    else:
                        higher_name, lower_name = group2_name, group1_name
                        higher_data, lower_data = group2_data, group1_data

                    ratio = higher_data.mean() / lower_data.mean()

                    # Store for writeup
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
                        'cohens_d': cohens_d,
                        'effect_label': effect_label,
                        'significant': p_val < 0.05
                    }

                    # Format p-value
                    if p_val < 0.001:
                        p_str = "p < .001"
                    else:
                        p_str = f"p = {p_val:.3f}"

                    sig_marker = "**" if p_val < 0.05 else ""

                    st.markdown(f"""
                    **{group1_name.title()} vs {group2_name.title()}**
                    {sig_marker}t = {t_stat:.2f}, {p_str}{sig_marker}
                    Cohen's d = {cohens_d:.2f} ({effect_label} effect)
                    """)

                    if ratio > 1.1:
                        st.info(f"{higher_name.title()} shows {ratio:.1f}x higher head speed")

            # Method explanation
            with st.expander("How is this calculated?"):
                st.markdown("""
                **Statistical Test:** Welch's t-test (independent samples)

                *Why Welch's?* Unlike Student's t-test, Welch's does NOT assume equal variances
                between groups. This is more robust when comparing activities that may have
                different variability (e.g., running is more variable than walking).

                **Effect Size:** Cohen's d

                Measures the *practical significance* of the difference:
                - d < 0.2: negligible
                - d = 0.2-0.5: small
                - d = 0.5-0.8: medium
                - d > 0.8: large

                A large effect size means the difference is meaningful in practice,
                not just statistically detectable.

                **Strip Plot:** Each dot is one motion capture file. This shows you the
                actual data distribution, not just summary statistics.
                """)
        else:
            st.info("Select files from multiple activities to see comparison")
            stats_results = None

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
was quantified using **Cohen's d**.

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
{sr['lower_name']} (*t* = {sr['t_stat']:.2f}, *p* = {sr['p_val']:.3f}, Cohen's *d* = {sr['cohens_d']:.2f}).

This represents a **{sr['effect_label']} effect**, with {sr['higher_name']} showing
**{sr['ratio']:.1f}x** the head movement of {sr['lower_name']}.
"""
        else:
            writeup += f"""
**Finding:** No statistically significant difference was found between {sr['group1_name']}
and {sr['group2_name']} (*t* = {sr['t_stat']:.2f}, *p* = {sr['p_val']:.3f}, Cohen's *d* = {sr['cohens_d']:.2f}).
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

    # --- SECTION 4: 3D TRAJECTORY EXPLORATION ---
    st.markdown("---")
    st.header("Explore 3D Trajectory")

    selected_viz = st.selectbox(
        "Select file to visualize",
        [d['file'] for d in all_data],
        key='viz_select'
    )

    data = next(d for d in all_data if d['file'] == selected_viz)
    pos = data['positions']
    speed = data['speed']
    timestamps = data['timestamps']
    n_frames = len(pos)

    # Animation settings
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        frame_step = st.select_slider(
            "Frame sampling (lower = smoother)",
            options=[1, 2, 5, 10, 25, 50, 100],
            value=10
        )
    with col_set2:
        trail_frames = st.slider("Trail length", 5, 100, 30)

    # Downsample for animation performance
    frame_indices = list(range(0, n_frames, frame_step))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    speed_padded = np.concatenate([[speed[0]], speed])

    # Build animated 3D figure
    fig_3d = go.Figure()

    # Full path (static background)
    fig_3d.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 2], z=pos[:, 1],
        mode='lines',
        line=dict(color='rgba(150,150,150,0.25)', width=1),
        name='Full Path',
        hoverinfo='skip'
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

    # Current position (animated red marker)
    fig_3d.add_trace(go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 2]], z=[pos[0, 1]],
        mode='markers',
        marker=dict(size=14, color='red', line=dict(color='white', width=2)),
        name='Head Position'
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

        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=pos[:, 0], y=pos[:, 2], z=pos[:, 1]),
                go.Scatter3d(x=t_pos[:, 0], y=t_pos[:, 2], z=t_pos[:, 1],
                            marker=dict(size=3, color=t_speed, colorscale='Viridis',
                                       cmin=0, cmax=np.percentile(speed, 95))),
                go.Scatter3d(x=[pos[idx, 0]], y=[pos[idx, 2]], z=[pos[idx, 1]]),
                go.Scatter3d(x=[pos[0, 0]], y=[pos[0, 2]], z=[pos[0, 1]]),
                go.Scatter3d(x=[pos[-1, 0]], y=[pos[-1, 2]], z=[pos[-1, 1]])
            ],
            name=str(idx)
        ))

    fig_3d.frames = frames

    # Layout with Play/Pause and slider
    fig_3d.update_layout(
        title=f"Head Trajectory - {selected_viz} ({data['activity']})",
        scene=dict(
            xaxis_title="X (Lateral)",
            yaxis_title="Z (Forward)",
            zaxis_title="Y (Vertical)",
            aspectmode='data',
            camera=dict(projection=dict(type='perspective'))
        ),
        uirevision='constant',
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        height=600,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.05, x=0.0, xanchor="left",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            y=-0.02, x=0.15, len=0.8,
            currentvalue={"prefix": "Time: ", "suffix": "s", "visible": True},
            steps=[dict(args=[[str(idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                       label=f"{timestamps[idx]:.1f}", method="animate")
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
