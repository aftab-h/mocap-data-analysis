# Product Requirements Document (PRD)
## MoCap Head Stability Analysis Dashboard

**Version:** 2.0
**Last Updated:** December 2024
**Author:** Portfolio Project for Biomechanics Research

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

## 2. Current State (v1.0)

### 2.1 Implemented Features

#### Data Pipeline
- BVH file parsing with forward kinematics
- Multi-source data acquisition (SFU, CMU databases)
- Butterworth low-pass filtering (configurable 1-20 Hz cutoff)
- Support for batch file selection by activity category

#### Kinematic Metrics
- **Translational**: Position, velocity, acceleration, jerk, speed
- **Summary Stats**: Mean, max, std, median for all metrics
- **Range of Motion**: Per-axis (X/Y/Z) and total 3D displacement
- **Path Length**: Total distance traveled
- **Stability Index**: `std(speed) + 0.1 * std(acceleration)`

#### Visualizations
- 3D animated head trajectory with play/pause controls
- Speed-over-time line charts (multi-file overlay)
- Position component breakdown (X, Y, Z vs time)
- Box plots for activity comparisons
- Correlation heatmap for metrics

#### Statistical Analysis
- Independent samples t-test (2 groups)
- One-way ANOVA (3+ groups)
- Cohen's d / eta-squared effect sizes
- CSV export for external analysis

### 2.2 Technical Stack
- **Backend**: Python 3.10+, NumPy, SciPy, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **UI Framework**: Streamlit
- **Statistics**: SciPy, Pingouin, Statsmodels

---

## 3. Gap Analysis & Expert Assessment

Based on consultation with motion capture research domain experts, the following critical gaps were identified:

### 3.1 Critical Gaps (Must Fix)

| Gap | Impact | Current State |
|-----|--------|---------------|
| **No rotational kinematics** | Cannot assess head turns, nodding - primary causes of device slippage | BVH contains rotation data but it's unused |
| **Inappropriate statistical tests** | Results may not survive peer review | Using independent t-test without assumption checking |
| **Stability index not validated** | Arbitrary formula with unit mismatch | `std(speed) + 0.1 * std(acceleration)` |
| **No temporal stability view** | Can't identify *when* instability occurs | Only summary metrics |

### 3.2 Significant Gaps (Should Fix)

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| No head-torso relative motion | Confounds locomotion with head instability | Subtract spine position from head |
| No frequency analysis | Missing activity signatures (walking ~2Hz, running ~3Hz) | Add FFT-based metrics |
| Events not visualized | High-motion detection exists but isn't shown | Highlight on time series |
| No data quality checks | Bad data produces misleading results | Add gap/outlier detection |

### 3.3 Nice-to-Have Gaps

- Phase plots (velocity vs position)
- Ensemble averaging for grouped trials
- C3D export format
- Windowed metrics with rolling stability

---

## 4. Product Requirements (v2.0)

### 4.1 Rotational Kinematics Module

**Priority:** P0 (Critical)
**Effort:** Medium (2-3 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| ROT-1 | Extract Euler angles (pitch/yaw/roll) from BVH | Angles match reference implementation within 0.1 degrees |
| ROT-2 | Calculate angular velocity (deg/s) | Differentiation uses same filtering as position |
| ROT-3 | Calculate angular acceleration (deg/s^2) | Values are physically plausible (<5000 deg/s^2) |
| ROT-4 | Calculate angular range of motion | Per-axis and total spherical range |
| ROT-5 | Add rotation metrics to stability calculations | New metrics appear in metrics_df |
| ROT-6 | Visualize head orientation over time | New subplot in Time Series tab |

#### Proposed Metrics
```python
{
    'pitch_range': float,      # Forward/backward head tilt range (deg)
    'yaw_range': float,        # Left/right head turn range (deg)
    'roll_range': float,       # Side-to-side tilt range (deg)
    'mean_angular_velocity': float,    # deg/s
    'max_angular_velocity': float,     # deg/s
    'angular_stability_index': float   # Rotational equivalent
}
```

#### My Thoughts
This is the single highest-impact improvement. Device displacement during quick head turns is arguably more important than translational motion for earbuds. The BVH parser already extracts rotation channels - we just need to expose them.

---

### 4.2 Revised Stability Index

**Priority:** P0 (Critical)
**Effort:** Low (1 day)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| STB-1 | Replace arbitrary formula with RMS acceleration | Industry-standard metric |
| STB-2 | Add normalized composite index | Z-score normalization for multi-metric |
| STB-3 | Include jerk in stability calculation | Jerk is calculated but currently unused |
| STB-4 | Document formula with literature citations | Methodology section in app |

#### Proposed Formula Options

**Option A: RMS Acceleration (Recommended)**
```python
stability_index = np.sqrt(np.mean(acceleration**2))  # RMS
```
*Rationale: Standard in vibration analysis (ISO 2631), dimensionally consistent, single interpretable unit (cm/s^2)*

**Option B: Normalized Composite**
```python
stability_index = (
    zscore(std_speed) +
    zscore(std_acceleration) +
    zscore(mean_jerk)
) / 3
```
*Rationale: Combines multiple factors, unitless, comparable across studies*

**Option C: Weighted Empirical (requires validation study)**
```python
stability_index = w1*speed_var + w2*acc_var + w3*jerk_mean
# Weights derived from correlation with subjective comfort ratings
```

#### My Thoughts
I recommend **Option A (RMS acceleration)** for the default, with Option B available as an alternative. Option C is ideal but requires a separate validation study. We should also add a tooltip explaining what the index means and how to interpret it.

---

### 4.3 Statistical Analysis Improvements

**Priority:** P0 (Critical)
**Effort:** Low (1-2 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| STAT-1 | Add normality testing (Shapiro-Wilk) | Display p-value and interpretation |
| STAT-2 | Add homogeneity of variance test (Levene's) | Displayed before t-test/ANOVA |
| STAT-3 | Auto-switch to non-parametric tests | Mann-Whitney U when assumptions violated |
| STAT-4 | Add post-hoc tests for ANOVA | Tukey HSD via pingouin |
| STAT-5 | Add multiple comparison correction | Bonferroni or FDR option |
| STAT-6 | Support repeated measures design | Paired t-test, rm-ANOVA for within-subject |
| STAT-7 | Fix Cohen's d calculation | Use Hedges' g for unequal variances |

#### UI Flow
```
[Run Statistical Analysis]
        |
        v
+-------------------+
| Assumption Checks |
| - Normality: p=0.23 (OK)
| - Levene's: p=0.67 (OK)
+-------------------+
        |
        v
+-------------------+
| Test Selection    |
| [x] Parametric (assumptions met)
| [ ] Non-parametric
+-------------------+
        |
        v
+-------------------+
| Results           |
| t(28) = 3.45, p = 0.002
| Hedges' g = 0.89 (large)
| [Post-hoc: Tukey HSD]
+-------------------+
```

#### My Thoughts
This is low-hanging fruit - all the functions exist in scipy/pingouin, we just need to call them. The current t-test implementation would not survive peer review. Adding assumption checking takes maybe 20 lines of code but dramatically improves credibility.

---

### 4.4 Windowed/Temporal Stability Analysis

**Priority:** P1 (High)
**Effort:** Medium (2 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| WIN-1 | Calculate metrics in sliding windows | Configurable window size (0.5-5 sec) |
| WIN-2 | Identify instability episodes | Frames where stability > threshold |
| WIN-3 | Visualize stability over time | New time series subplot |
| WIN-4 | Highlight events on existing plots | Shaded regions on speed plot |
| WIN-5 | Report cumulative exposure metrics | Total time above threshold |

#### Proposed Visualization
```
[Speed over Time]
  ^
  |    ████           ████  <- Shaded = high instability
  |   /    \    /\   /    \
  |  /      \  /  \ /      \
  |_/________\/____\/_______\______> Time

[Windowed Stability Index]
  ^
  |  ----    ----    ----
  |       \/      \/      <- Rolling 1-sec window
  |________________________> Time

[Summary]
- High instability episodes: 3
- Total duration above threshold: 2.4 sec (8% of trial)
- Longest continuous episode: 1.1 sec at t=4.2s
```

#### My Thoughts
Researchers often ask "what moment caused the headphone to slip?" - summary metrics can't answer this. A windowed stability view lets users scrub through the trial and see exactly when problems occur. This directly informs product design decisions.

---

### 4.5 Event Detection & Visualization

**Priority:** P1 (High)
**Effort:** Low (1 day)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| EVT-1 | Detect high-motion events (already implemented) | Expose in UI |
| EVT-2 | Visualize events as markers on time series | Vertical lines or shaded regions |
| EVT-3 | List events with timestamps | Clickable to jump to that time |
| EVT-4 | Configurable detection threshold | Slider: 90th-99th percentile |
| EVT-5 | Export event list | CSV with start/end times, duration, peak value |

#### My Thoughts
The `detect_high_motion_events()` function exists but isn't exposed in the UI. This is pure waste - 30 minutes of work would make it visible and useful.

---

### 4.6 Head-Torso Relative Motion

**Priority:** P2 (Medium)
**Effort:** Low (1 day)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| REL-1 | Calculate head position relative to spine/chest | head_rel = head_pos - spine_pos |
| REL-2 | Compute kinematics on relative motion | Isolates head movement from locomotion |
| REL-3 | Toggle between absolute/relative in UI | Radio button selector |
| REL-4 | Document interpretation differences | Tooltip explanation |

#### Rationale
When a person walks forward, their head has high absolute velocity but may be perfectly stable relative to their torso. Relative motion is more meaningful for device stability - it's the *isolation* of head movement that causes slippage.

---

### 4.7 Data Quality Module

**Priority:** P2 (Medium)
**Effort:** Medium (2 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| QC-1 | Detect marker dropout / gaps | Flag frames with missing data |
| QC-2 | Detect outliers (>4 SD from mean) | Highlight suspicious values |
| QC-3 | Detect drift over time | Linear regression on position |
| QC-4 | Compute overall quality score | 0-100 scale |
| QC-5 | Option to exclude bad segments | User confirmation before exclusion |

#### Quality Score Formula
```python
quality_score = 100 - (
    20 * gap_percentage +
    20 * outlier_percentage +
    20 * (drift_rate / max_acceptable_drift) +
    40 * physiological_implausibility_score
)
```

---

### 4.8 Frequency Domain Analysis

**Priority:** P3 (Low)
**Effort:** Medium (2 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FFT-1 | Compute FFT of position/velocity signals | Using scipy.fft |
| FFT-2 | Identify dominant frequencies | Peak detection |
| FFT-3 | Compute spectral entropy | Randomness/predictability measure |
| FFT-4 | Visualize power spectrum | New tab or subplot |
| FFT-5 | Activity classification by frequency | Walking vs running signature |

#### My Thoughts
This is valuable for research but less critical for product designers. Walking has a characteristic ~2Hz bounce, running is ~3Hz. Knowing this helps design damping systems and set resonance expectations. Lower priority than real-time stability metrics.

---

### 4.9 Export & Integration

**Priority:** P3 (Low)
**Effort:** Medium (2-3 days)

#### Requirements
| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| EXP-1 | Export to C3D format | Compatible with Visual3D, OpenSim |
| EXP-2 | Export to SPSS format (.sav) | For non-Python statisticians |
| EXP-3 | Publication-ready table export | APA formatted tables |
| EXP-4 | Batch processing mode | Process directory, output summary CSV |
| EXP-5 | API endpoint for integration | REST API for pipeline integration |

---

## 5. Implementation Roadmap

### Phase 1: Research Credibility (Week 1-2)
Focus: Make the tool publishable

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Fix stability index (RMS acceleration) | P0 | 2 hrs | None |
| Add statistical assumption checks | P0 | 4 hrs | None |
| Add post-hoc tests | P0 | 2 hrs | Assumption checks |
| Visualize high-motion events | P1 | 2 hrs | None |

**Deliverable:** Statistics tab shows assumption checks, auto-selects appropriate test, displays effect sizes correctly.

### Phase 2: Rotational Kinematics (Week 2-3)
Focus: Close the biggest analysis gap

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Extract rotations from BVH parser | P0 | 4 hrs | None |
| Calculate angular velocity/acceleration | P0 | 3 hrs | Rotation extraction |
| Add rotation metrics to calculate_head_stability_metrics | P0 | 2 hrs | Angular calcs |
| Visualize head orientation over time | P0 | 4 hrs | Rotation extraction |
| Add angular ROM to overview | P0 | 2 hrs | Rotation metrics |

**Deliverable:** New "Rotation" metrics appear throughout app, time series shows pitch/yaw/roll.

### Phase 3: Temporal Analysis (Week 3-4)
Focus: Answer "when does instability occur?"

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Implement windowed metrics calculation | P1 | 4 hrs | None |
| Create rolling stability time series | P1 | 3 hrs | Windowed metrics |
| Add event highlighting to plots | P1 | 3 hrs | Event detection |
| Add episode summary statistics | P1 | 2 hrs | Windowed metrics |

**Deliverable:** Time Series tab shows rolling stability with highlighted instability episodes.

### Phase 4: Polish & Integration (Week 4+)
Focus: Professional research tool

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Add data quality checks | P2 | 8 hrs | None |
| Implement head-torso relative motion | P2 | 4 hrs | None |
| Add frequency domain analysis | P3 | 8 hrs | None |
| Batch processing mode | P3 | 6 hrs | None |
| C3D export | P3 | 8 hrs | None |

---

## 6. Success Metrics

### 6.1 Technical Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Statistical test accuracy | 100% assumption-appropriate | Manual verification |
| Rotation extraction accuracy | <0.1 deg error vs reference | Comparison with known values |
| Processing speed | <5 sec for 10-min trial | Benchmark suite |
| Memory usage | <2 GB for large datasets | Profiling |

### 6.2 User Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first insight | <2 minutes from upload | User testing |
| Analysis completeness | Users export data >80% of sessions | Analytics |
| Error rate | <5% of sessions encounter errors | Error logging |

### 6.3 Research Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Publication readiness | Pass peer review statistics check | Expert review |
| Reproducibility | Same data = same results 100% | Automated tests |
| Citation of methodology | Clear documentation available | PRD + inline docs |

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| BVH rotation conventions vary | High | Medium | Auto-detect or ask user for convention |
| Large files cause memory issues | Medium | High | Implement chunked processing |
| Users misinterpret stability index | High | Medium | Add interpretation guidelines, thresholds |
| Statistical tests misapplied | Medium | High | Auto-check assumptions, warn users |
| Validation data unavailable | Medium | Medium | Use published benchmark datasets |

---

## 8. Open Questions

1. **Stability Index Validation**: Should we conduct a user study correlating metrics with subjective comfort ratings? This would allow empirically-derived weights.

2. **Normalization**: When comparing across subjects/trials of different lengths, should we normalize by duration? By body size?

3. **Reference Frame**: Some users may expect world-frame analysis, others body-frame. Should we support both? Default to which?

4. **Thresholds**: What constitutes "high instability"? Should we provide industry-standard thresholds or let users define their own?

5. **Multi-joint Analysis**: Should we expand beyond head to analyze neck, shoulders, torso? This would enable full upper-body stability assessment.

---

## 9. Appendix

### A. Glossary
| Term | Definition |
|------|------------|
| BVH | Biovision Hierarchy - motion capture file format |
| RMS | Root Mean Square - standard variability measure |
| ROM | Range of Motion - total excursion of a joint |
| Jerk | Rate of change of acceleration (m/s^3) |
| Stability Index | Composite metric quantifying motion irregularity |

### B. References
- ISO 2631-1: Mechanical vibration and shock - Evaluation of human exposure
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
- Robertson et al. (2013). Research Methods in Biomechanics

### C. Related Work
- OpenSim: Open-source musculoskeletal modeling
- Visual3D: Commercial biomechanics analysis
- Mokka: Open-source C3D viewer

---

*Document Status: Draft for Review*
*Next Review: After Phase 1 Implementation*
