# Product Requirements Document (PRD)
## MoCap Head Stability Analysis Dashboard

**Version:** 2.1
**Last Updated:** December 2024
**Status:** Experimental / Work in Progress

---

## 1. Executive Summary

### 1.1 Product Vision
A research-grade motion capture analysis platform for quantifying head stability during physical activities, specifically designed for audio device (headphone/earwear) comfort and fit assessment. The tool bridges the gap between raw BVH motion capture data and actionable insights for product designers and biomechanics researchers.

### 1.2 Problem Statement
Current tools for analyzing motion capture data are either:
- **Too generic**: General 3D animation tools (Blender, Maya) lack biomechanics-specific metrics
- **Too expensive**: Commercial biomechanics software (Visual3D, OpenSim) costs thousands annually
- **Too complex**: Research tools require extensive setup and programming expertise

Wearables companies and researchers need a focused, accessible tool that answers: *"How stable is the head during activity X, and when/why does instability occur?"*

### 1.3 Target Users
| User Type | Needs | Usage Frequency |
|-----------|-------|-----------------|
| **Product Designers** | Quick stability comparisons across activities | Weekly |
| **Biomechanics Researchers** | Publication-ready statistics, data export | Daily during studies |
| **UX Researchers** | Correlate motion data with comfort surveys | Per study |
| **QA Engineers** | Benchmark device stability thresholds | Per product cycle |

---

## 2. Current State (v2.1)

### 2.1 Implemented Features

#### Data Pipeline
- [x] BVH file parsing with forward kinematics
- [x] Multi-source data acquisition (SFU Motion Capture Database)
- [x] Butterworth low-pass filtering (configurable 1-20 Hz cutoff via sidebar)
- [x] Batch file selection by activity category (expandable sidebar groups)
- [x] Auto-selection of walking/jumping activities by default
- [x] Cached data loading for performance (`@st.cache_data`)

#### Kinematic Metrics
- [x] **Translational**: Position, velocity, acceleration, jerk, speed
- [x] **Summary Stats**: Mean, max, std for speed and acceleration
- [x] **Range of Motion**: Per-axis (X/Y/Z) and total 3D displacement
- [x] **Path Length**: Total distance traveled
- [x] **Stability Index**: `std(speed) + 0.1 * std(acceleration)`
- [x] **Head Orientation**: Nose direction vector extracted from BVH rotations

#### Visualizations
- [x] **3D Animated Trajectory** with:
  - Play/Pause controls with real-time playback speed
  - Translucent head sphere (~20cm diameter)
  - Ear positions (flattened ellipsoids)
  - Nose direction indicator (red line)
  - Speed-colored trail with configurable length
  - Start/End markers
  - Frame sampling slider (1-100)
  - Fixed axis ranges to prevent camera drift
- [x] **Speed Profile** with:
  - Multi-file overlay (multiselect dropdown)
  - Default to single file for cleaner view
  - High-motion event markers (red dots, >95th percentile)
  - Legend grouping (hiding trace also hides its markers)
- [x] **Strip Plot** for activity comparison with mean lines and annotations
- [x] Color-coded activities throughout (consistent palette)

#### Statistical Analysis
- [x] **Welch's t-test** (independent samples, unequal variance)
- [x] **Effect Size**: Auto-selects Cohen's d (n≥20) or Hedges' g (n<20)
- [x] **Effect Size Interpretation**: negligible/small/medium/large with color coding
- [x] **P-value Color Coding**: Green if significant, red if not
- [x] **Descriptive Statistics Table**: n, Mean, SD, Range per group

#### Auto-Generated Reports
- [x] **Analysis Summary** section with:
  - Methods paragraph (filter settings, test used)
  - Results table with means and SDs
  - Statistical finding with effect interpretation
  - Implications for wearable design
  - Limitations section
- [x] **Download Analysis Report** as markdown file

#### Export & Integration
- [x] CSV export of all metrics
- [x] Markdown export of analysis summary

#### UX Polish
- [x] Experimental/WIP disclaimer
- [x] Collapsible Notes sections explaining metrics and statistical tests
- [x] Responsive layout (wide mode)
- [x] Custom header with earbuds icon

### 2.2 Technical Stack
- **Backend**: Python 3.10+, NumPy, SciPy, Pandas
- **Visualization**: Plotly (3D, strip plots, line charts)
- **UI Framework**: Streamlit
- **Statistics**: SciPy (Welch's t-test)

---

## 3. Gap Analysis & Status

### 3.1 Critical Gaps

| Gap | Status | Notes |
|-----|--------|-------|
| **No rotational kinematics** | Partial | Orientation extracted but angular velocity/acceleration not computed |
| **Inappropriate statistical tests** | **Fixed** | Now uses Welch's t-test, auto-selects Hedges' g for small samples |
| **Stability index not validated** | Open | Still using arbitrary formula |
| **No temporal stability view** | Partial | High-motion events shown on speed plot, but no rolling stability |

### 3.2 Significant Gaps

| Gap | Status | Notes |
|-----|--------|-------|
| No head-torso relative motion | Open | Would isolate head movement from locomotion |
| No frequency analysis | Open | Missing FFT-based activity signatures |
| Events not visualized | **Fixed** | Red markers on speed profile |
| No data quality checks | Open | No gap/outlier detection |
| No ANOVA for 3+ groups | Open | Only t-test for 2 groups currently |

### 3.3 Nice-to-Have Gaps

| Gap | Status |
|-----|--------|
| Phase plots (velocity vs position) | Open |
| Ensemble averaging | Open |
| C3D export format | Open |
| Windowed/rolling metrics | Open |
| Batch processing mode | Open |

---

## 4. Remaining Requirements

### 4.1 Rotational Kinematics Module

**Priority:** P0 (Critical)
**Status:** Partial (orientation extracted, calculations pending)

| ID | Requirement | Status |
|----|-------------|--------|
| ROT-1 | Extract Euler angles (pitch/yaw/roll) from BVH | **Done** (orientation vector exists) |
| ROT-2 | Calculate angular velocity (deg/s) | Open |
| ROT-3 | Calculate angular acceleration (deg/s²) | Open |
| ROT-4 | Calculate angular range of motion | Open |
| ROT-5 | Add rotation metrics to stability calculations | Open |
| ROT-6 | Visualize head orientation over time | Open |

### 4.2 Revised Stability Index

**Priority:** P1 (High)

| ID | Requirement | Status |
|----|-------------|--------|
| STB-1 | Replace arbitrary formula with RMS acceleration | Open |
| STB-2 | Add normalized composite index option | Open |
| STB-3 | Include jerk in stability calculation | Open |
| STB-4 | Document formula with literature citations | Partial (Notes section exists) |

### 4.3 Statistical Analysis Improvements

**Priority:** P1 (High)

| ID | Requirement | Status |
|----|-------------|--------|
| STAT-1 | Add normality testing (Shapiro-Wilk) | Open |
| STAT-2 | Add homogeneity of variance test (Levene's) | Open |
| STAT-3 | Auto-switch to non-parametric tests | Open |
| STAT-4 | Add post-hoc tests for ANOVA | Open |
| STAT-5 | Support 3+ group comparisons (ANOVA) | Open |
| STAT-6 | Support repeated measures design | Open |
| STAT-7 | Fix Cohen's d calculation | **Done** (uses Hedges' g for small n) |

### 4.4 Windowed/Temporal Stability

**Priority:** P2 (Medium)

| ID | Requirement | Status |
|----|-------------|--------|
| WIN-1 | Calculate metrics in sliding windows | Open |
| WIN-2 | Identify instability episodes | Partial (event detection exists) |
| WIN-3 | Visualize stability over time | Open |
| WIN-4 | Highlight events on existing plots | **Done** |
| WIN-5 | Report cumulative exposure metrics | Open |

### 4.5 Data Quality Module

**Priority:** P3 (Low)

| ID | Requirement | Status |
|----|-------------|--------|
| QC-1 | Detect marker dropout / gaps | Open |
| QC-2 | Detect outliers (>4 SD from mean) | Open |
| QC-3 | Detect drift over time | Open |
| QC-4 | Compute overall quality score | Open |
| QC-5 | Option to exclude bad segments | Open |

---

## 5. Implementation Roadmap

### Phase 1: Research Credibility - **Mostly Complete**
- [x] Welch's t-test with effect sizes
- [x] Auto-generated analysis report
- [x] High-motion event visualization
- [ ] Add assumption checks (Shapiro-Wilk, Levene's)
- [ ] Support 3+ group ANOVA

### Phase 2: Rotational Kinematics - **In Progress**
- [x] Extract head orientation from BVH
- [x] Visualize nose direction in 3D
- [ ] Calculate angular velocity/acceleration
- [ ] Add rotation metrics to summary table

### Phase 3: Temporal Analysis - **Partial**
- [x] Event detection function exists
- [x] Events visualized on speed plot
- [ ] Rolling stability time series
- [ ] Episode summary statistics

### Phase 4: Polish & Integration - **Future**
- [ ] Data quality checks
- [ ] Head-torso relative motion
- [ ] Frequency domain analysis
- [ ] Batch processing mode
- [ ] C3D export

---

## 6. Success Metrics

### 6.1 Technical Metrics
| Metric | Target | Current Status |
|--------|--------|----------------|
| Statistical test accuracy | Assumption-appropriate | Welch's t-test implemented |
| Rotation extraction accuracy | <0.1 deg error | Orientation vector extracted |
| Processing speed | <5 sec for 10-min trial | Cached loading works well |

### 6.2 User Metrics
| Metric | Target | Current Status |
|--------|--------|----------------|
| Time to first insight | <2 minutes | Auto-loads with defaults |
| Error rate | <5% of sessions | Error handling for file processing |

---

## 7. Open Questions

1. **Stability Index Validation**: Should we conduct a user study correlating metrics with subjective comfort ratings?

2. **Normalization**: When comparing across subjects/trials of different lengths, should we normalize by duration? By body size?

3. **Reference Frame**: Some users may expect world-frame analysis, others body-frame. Should we support both?

4. **Thresholds**: What constitutes "high instability"? Currently using 95th percentile - is this appropriate?

5. **Multi-joint Analysis**: Should we expand beyond head to analyze neck, shoulders, torso?

---

## 8. Appendix

### A. Glossary
| Term | Definition |
|------|------------|
| BVH | Biovision Hierarchy - motion capture file format |
| RMS | Root Mean Square - standard variability measure |
| ROM | Range of Motion - total excursion of a joint |
| Jerk | Rate of change of acceleration (m/s³) |
| Stability Index | Composite metric quantifying motion irregularity |
| Hedges' g | Small-sample corrected effect size (used when n < 20) |

### B. References
- ISO 2631-1: Mechanical vibration and shock - Evaluation of human exposure
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
- Robertson et al. (2013). Research Methods in Biomechanics

### C. Data Sources
- SFU Motion Capture Database (primary)
- CMU Motion Capture Database (supported)

---

*Document Status: Living document, updated as features are implemented*
