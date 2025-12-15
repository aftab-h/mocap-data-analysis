# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Motion capture data analysis dashboard for head stability research, specifically for audio device (headphone/earwear) comfort and stability assessment. Analyzes BVH motion capture files to extract kinematic metrics like head velocity, acceleration, and range of motion.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# Download motion capture data
python src/download_data.py --source all           # Download from all sources
python src/download_data.py --source sfu           # Download from SFU database only
python src/download_data.py --list                 # List available files

# Download specific categories
python src/download_data.py --source sfu --categories locomotion dance
```

## Architecture

### Core Modules (src/)

- **data_loader.py**: `BVHParser` class for parsing BVH motion capture files. Extracts skeleton hierarchy, calculates world positions for joints via forward kinematics.

- **kinematics.py**: Kinematic calculations including:
  - `calculate_velocity()`, `calculate_acceleration()`, `calculate_speed()`
  - `butter_lowpass_filter()` for noise filtering
  - `calculate_head_stability_metrics()` - comprehensive metrics dict (mean/max speed, acceleration, jerk, range of motion, stability index)
  - `detect_high_motion_events()` for finding instability frames

- **download_data.py**: Data acquisition from SFU Motion Capture Database. Categories: locomotion, dance, martial_arts, sports, obstacles, bollywood.

### Dashboard (app.py)

Streamlit application with four tabs:
1. **Overview**: Summary metrics and activity type comparisons
2. **3D Trajectories**: Animated head position visualization with Play/Pause
3. **Time Series**: Speed profiles and position components over time
4. **Statistical Analysis**: t-tests, ANOVA, correlation matrices

### Data Flow

1. BVH files loaded via `load_bvh()` -> `BVHParser`
2. Joint positions extracted via `parser.get_joint_positions(joint_name)`
3. Positions filtered with Butterworth low-pass filter
4. Metrics computed via `calculate_head_stability_metrics()`
5. Results displayed in Streamlit tabs

## Data Directories

- `data/sfu/`: SFU Motion Capture Database files (BVH format)
- `data/raw/`: GitHub-sourced BVH files
- `data/processed/`: Cleaned/extracted data
- `results/figures/`: Output visualizations
- `results/tables/`: Statistical results

## Key Patterns

- All kinematic functions accept numpy arrays with shape `(n_frames, 3)` for positions
- FPS extracted from BVH `frame_time` field via `parser.fps` property
- Head-related joints found via `get_head_related_joints()` which searches for 'head', 'neck' in joint names
- Stability index = `std(speed) + 0.1 * std(acceleration)`
