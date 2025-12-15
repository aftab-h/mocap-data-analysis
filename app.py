"""
Motion Capture Analysis Dashboard for Head Stability Research

A Streamlit web application for analyzing motion capture data to understand
head movement patterns during different activities - relevant to headphone/earwear
stability and comfort research.

Author: Built for Apple ARC Biomechanics Role Portfolio
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import time

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.title("📊 Analysis Controls")

    # Data management section
    st.sidebar.header("1. Data Management")

    # Check multiple data directories
    data_dirs = [Path("data/sfu"), Path("data/raw")]
    available_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            available_files.extend(list_available_files(str(data_dir)))

    if not available_files:
        st.sidebar.warning("No data files found.")
        if st.sidebar.button("Download Sample Data", type="primary"):
            with st.spinner("Downloading walking and running trials from CMU..."):
                download_trials(['walking', 'running'], str(data_dir))
            st.rerun()
        st.sidebar.info("Click above to download sample motion capture data from CMU.")

        # Show main area with instructions
        st.markdown('<p class="main-header">Motion Capture Head Stability Analysis</p>',
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analyzing head movement patterns for audio device stability research</p>',
                   unsafe_allow_html=True)

        st.info("""
        ### Getting Started

        1. Click **"Download Sample Data"** in the sidebar to get motion capture files from the CMU database
        2. Once downloaded, select files to analyze
        3. Explore 3D trajectories, velocity profiles, and stability metrics

        ### About This Tool

        This dashboard analyzes motion capture data to understand head movement patterns
        during different activities (walking, running, etc.). These insights are relevant to:

        - **Headphone stability** - When do devices become unstable?
        - **Comfort assessment** - What movement patterns correlate with discomfort?
        - **Design optimization** - What movement thresholds should products accommodate?
        """)
        return

    # File selection
    st.sidebar.subheader("Select Files to Analyze")

    # Group files by activity
    activities = {}
    for f in available_files:
        act = f['activity']
        if act not in activities:
            activities[act] = []
        activities[act].append(f)

    selected_files = []
    for activity, files in activities.items():
        with st.sidebar.expander(f"{activity.title()} ({len(files)} files)", expanded=True):
            for f in files:
                if st.checkbox(f"{f['filename']}", key=f['path']):
                    selected_files.append(f)

    if not selected_files:
        st.markdown('<p class="main-header">Motion Capture Head Stability Analysis</p>',
                   unsafe_allow_html=True)
        st.info("Select one or more files from the sidebar to begin analysis.")
        return

    # Analysis settings
    st.sidebar.header("2. Analysis Settings")

    joint_filter = st.sidebar.selectbox(
        "Focus Joint",
        ["Head", "Neck", "Auto-detect head joints"],
        help="Select which joint to analyze for stability metrics"
    )

    filter_cutoff = st.sidebar.slider(
        "Low-pass filter cutoff (Hz)",
        min_value=1, max_value=20, value=10,
        help="Filter high-frequency noise from position data"
    )

    # Load and process data
    st.markdown('<p class="main-header">Motion Capture Head Stability Analysis</p>',
               unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyzing head movement patterns for audio device stability research</p>',
               unsafe_allow_html=True)

    # Process each selected file
    all_data = []
    all_metrics = []

    progress = st.progress(0)
    for idx, file_info in enumerate(selected_files):
        try:
            parser = load_bvh(file_info['path'])

            # Find head joint
            if joint_filter == "Auto-detect head joints":
                head_joints = get_head_related_joints(parser)
                head_joint = head_joints[0] if head_joints else parser.list_joints()[0]
            else:
                # Try to find the specified joint
                joints = parser.list_joints()
                matching = [j for j in joints if joint_filter.lower() in j.lower()]
                head_joint = matching[0] if matching else joints[0]

            # Get positions
            head_pos = parser.get_joint_positions(head_joint)

            # Apply filter
            filtered_pos = butter_lowpass_filter(head_pos, filter_cutoff, parser.fps)

            # Calculate metrics
            metrics = calculate_head_stability_metrics(filtered_pos, parser.fps)
            metrics['file'] = file_info['filename']
            metrics['activity'] = file_info['activity']
            metrics['subject'] = file_info['subject']
            metrics['duration'] = parser.duration
            metrics['fps'] = parser.fps
            metrics['joint'] = head_joint

            all_metrics.append(metrics)

            # Store time series for visualization
            velocity = calculate_velocity(filtered_pos, parser.fps)
            speed = calculate_speed(velocity)

            all_data.append({
                'file': file_info['filename'],
                'activity': file_info['activity'],
                'subject': file_info['subject'],
                'positions': filtered_pos,
                'speed': speed,
                'timestamps': parser.get_timestamps(),
                'parser': parser,
                'joint': head_joint
            })

        except Exception as e:
            st.warning(f"Error processing {file_info['filename']}: {e}")

        progress.progress((idx + 1) / len(selected_files))

    progress.empty()

    if not all_metrics:
        st.error("No files could be processed successfully.")
        return

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🎯 3D Trajectories",
        "📊 Time Series",
        "📋 Statistical Analysis"
    ])

    # TAB 1: Overview
    with tab1:
        st.header("Head Stability Overview")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Files Analyzed",
                len(all_metrics)
            )

        with col2:
            st.metric(
                "Avg Head Speed",
                f"{metrics_df['mean_speed'].mean():.2f} cm/s"
            )

        with col3:
            st.metric(
                "Max Head Speed",
                f"{metrics_df['max_speed'].max():.2f} cm/s"
            )

        with col4:
            st.metric(
                "Avg Stability Index",
                f"{metrics_df['stability_index'].mean():.2f}"
            )

        st.markdown("---")

        # Comparison by activity
        st.subheader("Comparison by Activity Type")

        if len(metrics_df['activity'].unique()) > 1:
            # Box plot comparison
            fig_box = px.box(
                metrics_df,
                x='activity',
                y='mean_speed',
                color='activity',
                title="Mean Head Speed by Activity",
                labels={'mean_speed': 'Mean Speed (cm/s)', 'activity': 'Activity Type'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Stability index comparison
            col1, col2 = st.columns(2)

            with col1:
                fig_stability = px.box(
                    metrics_df,
                    x='activity',
                    y='stability_index',
                    color='activity',
                    title="Stability Index by Activity",
                    labels={'stability_index': 'Stability Index (lower = more stable)'}
                )
                st.plotly_chart(fig_stability, use_container_width=True)

            with col2:
                fig_rom = px.box(
                    metrics_df,
                    x='activity',
                    y='total_rom',
                    color='activity',
                    title="Total Range of Motion by Activity",
                    labels={'total_rom': 'Range of Motion (cm)'}
                )
                st.plotly_chart(fig_rom, use_container_width=True)

        # Summary table
        st.subheader("Detailed Metrics")

        display_cols = ['file', 'activity', 'subject', 'mean_speed', 'max_speed',
                       'max_acceleration', 'total_rom', 'stability_index', 'duration']
        display_df = metrics_df[display_cols].round(2)
        display_df.columns = ['File', 'Activity', 'Subject', 'Mean Speed', 'Max Speed',
                             'Max Accel', 'ROM', 'Stability', 'Duration (s)']
        st.dataframe(display_df, use_container_width=True)

    # TAB 2: 3D Trajectories
    with tab2:
        st.header("3D Head Trajectories")

        selected_viz = st.selectbox(
            "Select file to visualize",
            [d['file'] for d in all_data]
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
                "Frame sampling (lower = slower/smoother)",
                options=[1, 2, 5, 10, 25, 50, 100],
                value=10,
                help="Sample every Nth frame - lower values = more frames = slower playback"
            )
        with col_set2:
            trail_frames = st.slider(
                "Trail length",
                min_value=5,
                max_value=100,
                value=30,
                help="Past frames shown as trail"
            )

        # Downsample for animation performance
        frame_indices = list(range(0, n_frames, frame_step))
        if frame_indices[-1] != n_frames - 1:
            frame_indices.append(n_frames - 1)

        fps = data['parser'].fps
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

        # Start/End markers (static)
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
                    go.Scatter3d(x=pos[:, 0], y=pos[:, 2], z=pos[:, 1]),  # Full path
                    go.Scatter3d(x=t_pos[:, 0], y=t_pos[:, 2], z=t_pos[:, 1],
                                marker=dict(size=3, color=t_speed, colorscale='Viridis',
                                           cmin=0, cmax=np.percentile(speed, 95))),  # Trail
                    go.Scatter3d(x=[pos[idx, 0]], y=[pos[idx, 2]], z=[pos[idx, 1]]),  # Current
                    go.Scatter3d(x=[pos[0, 0]], y=[pos[0, 2]], z=[pos[0, 1]]),  # Start
                    go.Scatter3d(x=[pos[-1, 0]], y=[pos[-1, 2]], z=[pos[-1, 1]])  # End
                ],
                name=str(idx)
            ))

        fig_3d.frames = frames

        # Layout with Play/Pause and slider
        fig_3d.update_layout(
            title=f"Head Trajectory - {selected_viz} ({data['activity']}) - {len(frame_indices)} frames",
            scene=dict(
                xaxis_title="X (Lateral)",
                yaxis_title="Z (Forward)",
                zaxis_title="Y (Vertical)",
                aspectmode='data',
                camera=dict(projection=dict(type='orthographic'))  # Optional: orthographic view
            ),
            uirevision='constant',  # Preserve camera position during animation
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Default starting view
            height=650,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=-0.05, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 50, "redraw": False},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
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

        st.caption("Use **Play/Pause** buttons and the **time slider** below the plot for smooth animation.")

        # 2D projections
        st.subheader("2D Projections")

        col1, col2 = st.columns(2)

        with col1:
            fig_top = px.scatter(
                x=pos[:, 0], y=pos[:, 2],
                color=speed_padded,
                title="Top View (X-Z plane)",
                labels={'x': 'X (Lateral)', 'y': 'Z (Forward)'},
                color_continuous_scale='Viridis'
            )
            fig_top.update_layout(coloraxis_colorbar_title="Speed")
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            fig_side = px.scatter(
                x=pos[:, 2], y=pos[:, 1],
                color=speed_padded,
                title="Side View (Z-Y plane)",
                labels={'x': 'Z (Forward)', 'y': 'Y (Vertical)'},
                color_continuous_scale='Viridis'
            )
            fig_side.update_layout(coloraxis_colorbar_title="Speed")
            st.plotly_chart(fig_side, use_container_width=True)

    # TAB 3: Time Series
    with tab3:
        st.header("Time Series Analysis")

        # Compare multiple files
        st.subheader("Speed Profiles")

        fig_speed = go.Figure()

        for data in all_data:
            timestamps = data['timestamps'][:-1]  # Speed has one less point
            fig_speed.add_trace(go.Scatter(
                x=timestamps,
                y=data['speed'],
                mode='lines',
                name=f"{data['file']} ({data['activity']})",
                hovertemplate="Time: %{x:.2f}s<br>Speed: %{y:.2f} cm/s"
            ))

        fig_speed.update_layout(
            title="Head Speed Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (cm/s)",
            height=400
        )

        st.plotly_chart(fig_speed, use_container_width=True)

        # Position components
        st.subheader("Position Components")

        selected_ts = st.selectbox(
            "Select file for detailed view",
            [d['file'] for d in all_data],
            key='ts_select'
        )

        data = next(d for d in all_data if d['file'] == selected_ts)
        pos = data['positions']
        timestamps = data['timestamps']

        fig_pos = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=['X (Lateral)', 'Y (Vertical)', 'Z (Forward)'])

        fig_pos.add_trace(go.Scatter(x=timestamps, y=pos[:, 0], mode='lines', name='X'),
                         row=1, col=1)
        fig_pos.add_trace(go.Scatter(x=timestamps, y=pos[:, 1], mode='lines', name='Y'),
                         row=2, col=1)
        fig_pos.add_trace(go.Scatter(x=timestamps, y=pos[:, 2], mode='lines', name='Z'),
                         row=3, col=1)

        fig_pos.update_layout(height=600, title=f"Position Components - {selected_ts}")
        fig_pos.update_xaxes(title_text="Time (s)", row=3, col=1)

        st.plotly_chart(fig_pos, use_container_width=True)

        # High motion events
        st.subheader("High Motion Events")

        speed = data['speed']
        high_motion = detect_high_motion_events(speed, threshold_percentile=95)
        n_events = np.sum(high_motion)
        pct_high = 100 * n_events / len(speed)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("High Motion Frames", f"{n_events} ({pct_high:.1f}%)")
        with col2:
            threshold = np.percentile(speed, 95)
            st.metric("95th Percentile Speed", f"{threshold:.2f} cm/s")

    # TAB 4: Statistical Analysis
    with tab4:
        st.header("Statistical Analysis")

        if len(metrics_df['activity'].unique()) < 2:
            st.warning("Need at least 2 different activity types for statistical comparison. "
                      "Download more data categories.")
        else:
            st.subheader("Activity Comparison Statistics")

            # Group statistics
            group_stats = metrics_df.groupby('activity').agg({
                'mean_speed': ['mean', 'std', 'count'],
                'max_speed': ['mean', 'std'],
                'stability_index': ['mean', 'std'],
                'total_rom': ['mean', 'std']
            }).round(3)

            st.dataframe(group_stats, use_container_width=True)

            # Simple statistical test (if scipy available)
            try:
                from scipy import stats

                st.subheader("Statistical Tests")

                activities = metrics_df['activity'].unique()

                if len(activities) == 2:
                    # Two groups - t-test
                    group1 = metrics_df[metrics_df['activity'] == activities[0]]['mean_speed']
                    group2 = metrics_df[metrics_df['activity'] == activities[1]]['mean_speed']

                    if len(group1) >= 2 and len(group2) >= 2:
                        t_stat, p_val = stats.ttest_ind(group1, group2)
                        cohens_d = (group1.mean() - group2.mean()) / np.sqrt(
                            (group1.std()**2 + group2.std()**2) / 2
                        )

                        st.markdown(f"""
                        **Independent t-test: Mean Speed between {activities[0]} vs {activities[1]}**
                        - t-statistic: {t_stat:.3f}
                        - p-value: {p_val:.4f}
                        - Cohen's d: {cohens_d:.3f}
                        - Interpretation: {'Significant difference' if p_val < 0.05 else 'No significant difference'} (α=0.05)
                        """)

                elif len(activities) > 2:
                    # Multiple groups - ANOVA
                    groups = [metrics_df[metrics_df['activity'] == act]['mean_speed'].values
                             for act in activities]

                    # Only run if we have enough data
                    valid_groups = [g for g in groups if len(g) >= 2]

                    if len(valid_groups) >= 2:
                        f_stat, p_val = stats.f_oneway(*valid_groups)

                        # Calculate eta-squared
                        all_values = np.concatenate(valid_groups)
                        grand_mean = np.mean(all_values)
                        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in valid_groups)
                        ss_total = np.sum((all_values - grand_mean)**2)
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0

                        st.markdown(f"""
                        **One-way ANOVA: Mean Speed across activities**
                        - F-statistic: {f_stat:.3f}
                        - p-value: {p_val:.4f}
                        - η² (eta-squared): {eta_squared:.3f}
                        - Interpretation: {'Significant difference' if p_val < 0.05 else 'No significant difference'} between at least two groups (α=0.05)
                        """)

            except ImportError:
                st.info("Install scipy for statistical tests: pip install scipy")

        # Correlation matrix
        st.subheader("Correlation Matrix")

        corr_cols = ['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration',
                    'stability_index', 'total_rom', 'path_length']
        corr_df = metrics_df[corr_cols].corr()

        fig_corr = px.imshow(
            corr_df,
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="Correlation Between Stability Metrics"
        )
        fig_corr.update_layout(height=500)

        st.plotly_chart(fig_corr, use_container_width=True)

        # Export data
        st.subheader("Export Data")

        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics as CSV",
            data=csv,
            file_name="head_stability_metrics.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
