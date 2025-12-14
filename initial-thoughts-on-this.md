# Motion Capture Analysis Project for Apple ARC Biomechanics Role

## Project Overview

**Goal:** Develop a motion capture data analysis pipeline to demonstrate competency for Apple's Applied Research & Characterization (ARC) Biomechanics Researcher role focused on audio product comfort and stability.

**Approach:** Analyze existing motion capture data from the CMU Graphics Lab Motion Capture Database to showcase understanding of:
- Biomechanical data processing
- Statistical analysis methods
- Mixed methods research design
- Data visualization for stakeholders
- Translating findings to product design recommendations

---

## Background Context

### The Apple Role

**Position:** Biomechanics Researcher, Applied Research & Characterization (ARC) Team  
**Focus:** Understanding factors that facilitate comfortable and stable experiences with Apple's audio products (AirPods, headphones, etc.)

**Required Skills:**
- Experience with user studies and human research
- Quantitative and qualitative research methods
- Statistical analysis proficiency
- Strong communication and ability to defend results

**Desired Skills (Nice-to-Haves):**
- Motion capture experience (Vicon, OptiTrack, FreeMoCap)
- Experience with sensors and prototype hardware
- Scripting and data visualization (Python, MATLAB, R)

**Relevant Background:** 10+ years in psychoacoustic research at Google (spatial audio R&D for Galaxy XR), Stanford CESTA, Meta Building 8. Strong quantitative analysis, user study design, and audio domain expertise.

---

## How Motion Capture is Used in Headphone Comfort Studies

### Research Objectives

Motion capture in headphone/earwear comfort research quantifies how the device affects:
1. **Posture** - head/neck position over time
2. **Movement patterns** - how much users move/fidget
3. **Contact dynamics** - how device interacts with body during activities
4. **Adjustment behaviors** - when and why users reposition the device

These biomechanical metrics are then correlated with **subjective comfort ratings** to identify design factors that drive discomfort.

### Typical Study Protocol

**Phase 1: Baseline & Setup (15 min)**
- Anthropometric measurements (head circumference, ear dimensions)
- Motion capture calibration
- Participant familiarization with tasks
- Baseline comfort ratings (no device)

**Phase 2: Initial Fit (10 min)**
- Participant dons device and adjusts to preference
- Document initial fit (photos, measurements)
- Initial comfort rating
- Baseline mocap recording (standing, seated postures)

**Phase 3: Task Battery (60-90 min)**
Participants perform standardized activities while wearing device:
- Desk work (typing, reading) - 20 min
- Walking (treadmill or track) - 15 min  
- Head movements (looking around, nodding, shaking) - 10 min
- Reaching tasks (overhead, lateral) - 10 min
- Simulated phone call (talking while moving) - 10 min
- Light exercise (jogging, stairs) - 15 min [optional]

**Phase 4: Comfort Assessments**
- Ratings every 15-30 minutes during tasks
- Post-task questionnaires
- Exit interview

**Phase 5: Comparison** (if testing multiple devices)
- Remove Device A, rest break (5-10 min)
- Repeat protocol with Device B
- Final comparative ratings

### Data Collected

**Quantitative - Motion Capture:**
- 3D position coordinates (X, Y, Z) for tracked joints/markers
- Captured at 60-120 fps
- Typical tracking: head, neck, shoulders, hands, upper torso

**Quantitative - Derived Metrics:**
- Head position and orientation (6DOF: X,Y,Z, pitch, roll, yaw)
- Neck angles: flexion/extension, lateral flexion, rotation
- Movement velocity and acceleration
- Range of motion
- Hand-to-head proximity (adjustment detection)

**Behavioral:**
- Number of adjustments (hand-to-device contact events)
- Time to first adjustment
- Adjustment duration
- Device displacement events (visible slipping)

**Subjective:**
- Overall comfort (1-7 or 1-10 scale)
- Pressure perception
- Pain/discomfort location and intensity
- Stability/security feeling
- Temperature/heat perception

**Physiological** (advanced studies):
- Clamping force (Newtons)
- Contact pressure distribution
- Temperature (IR thermography)
- Muscle activity (EMG)

---

## Understanding Motion Capture Data

### Data Structure

Motion capture produces **time-series data** - 3D coordinates of tracked points over time.

**Example raw data format:**
```
frame | timestamp | head_x | head_y | head_z | hand_x | hand_y | hand_z
0     | 0.000     | 0.15   | 1.65   | 0.10   | 0.45   | 1.20   | 0.30
1     | 0.008     | 0.15   | 1.65   | 0.11   | 0.46   | 1.21   | 0.31
2     | 0.016     | 0.16   | 1.66   | 0.11   | 0.47   | 1.22   | 0.32
...
```

**If tracking:**
- 33 body joints (standard skeletal model)
- At 120 fps
- For 60 seconds

**You get:**
- 7,200 rows (frames)
- 99 position columns (33 joints × 3 coordinates)
- Plus metadata (timestamps, frame numbers, etc.)

### From Positions to Meaningful Metrics

**The core concept:** Calculate change over time

**Velocity = change in position ÷ change in time**
```python
# Simple example
position = [0.0, 0.2, 0.5, 0.9, 1.4]  # meters
time = [0.0, 0.1, 0.2, 0.3, 0.4]      # seconds

# Calculate velocity
velocity = np.diff(position) / np.diff(time)
# Result: [2.0, 3.0, 4.0, 5.0] m/s
```

**Acceleration = change in velocity ÷ change in time**
```python
acceleration = np.diff(velocity) / np.diff(time[:-1])
# Result: [10.0, 10.0, 10.0] m/s²
```

**Joint Angles:**
```python
# Calculate angle between three points (e.g., neck flexion)
shoulder_pos = [x1, y1, z1]
neck_pos = [x2, y2, z2]
head_pos = [x3, y3, z3]

angle = calculate_angle(shoulder_pos, neck_pos, head_pos)
```

**Distance Between Points:**
```python
# Detect when hand approaches head (adjustment behavior)
distance = np.sqrt(
    (hand_x - head_x)**2 + 
    (hand_y - head_y)**2 + 
    (hand_z - head_z)**2
)
```

---

## Project Design

### Research Question Options

Since we're using existing CMU mocap data, we'll adapt our analysis to available movements. Potential research questions:

**Option 1: Activity Intensity and Head Movement**
- "How do different activity types affect head movement patterns?"
- Compare walking vs. running vs. reaching movements
- Hypothesize: Higher-intensity activities produce greater head velocities

**Option 2: Movement Patterns During Reaching**
- "What are natural head compensation patterns during overhead reaching?"
- Analyze neck angles during reaching tasks
- Relevant to: Understanding posture during headphone adjustment gestures

**Option 3: Gait-Related Head Stability**
- "How does head position vary during different walking speeds?"
- Compare slow walk vs. normal walk vs. fast walk
- Relevant to: Predicting when headphones might become unstable

**Option 4: Upper Body Coordination**
- "How do head and hand movements coordinate during reaching tasks?"
- Analyze temporal relationships between head turn and hand trajectory
- Relevant to: Understanding natural adjustment behaviors

### Proposed Analysis Pipeline

**Phase 1: Data Acquisition**
1. Download relevant mocap files from CMU database (http://mocap.cs.cmu.edu/)
2. Focus on activities: walking, running, reaching, general movement
3. Select 3-5 different subjects for comparison

**Phase 2: Data Processing**
1. Load BVH files (standard mocap format)
2. Extract joint positions (head, neck, shoulders, hands)
3. Calculate derived metrics:
   - Velocities (head, hand)
   - Accelerations
   - Neck angles (flexion/extension, lateral, rotation)
   - Hand-head distances

**Phase 3: Event Detection**
1. Identify movement phases (e.g., stance vs. swing in gait)
2. Detect reaching events
3. Segment data by activity type

**Phase 4: Statistical Analysis**
1. Calculate summary statistics per activity/subject
2. Compare across conditions using appropriate tests
3. Correlate movement metrics with activity characteristics

**Phase 5: Visualization**
1. 3D trajectory plots
2. Time-series plots (velocity, angle profiles)
3. Comparative plots (box plots, violin plots)
4. Summary dashboards for stakeholders

**Phase 6: Interpretation**
1. Translate findings to headphone design context
2. Generate design recommendations
3. Identify biomechanical thresholds

---

## Statistical Analysis Framework

### Decision Tree Summary

**Step 1: Define Comparison Type**
- 2 conditions → t-test or Wilcoxon
- 3+ conditions → ANOVA or Kruskal-Wallis
- Continuous predictor → Correlation or Regression

**Step 2: Identify Study Design**
- Same people, multiple conditions → Within-subjects (paired/RM tests)
- Different people → Between-subjects (independent tests)
- Multiple time points → Repeated measures

**Step 3: Check Assumptions**

**For Parametric Tests:**
- **Normality**: Shapiro-Wilk test, Q-Q plots
- **Equal variances** (between-subjects): Levene's test
- **Sphericity** (RM-ANOVA): Mauchly's test
- **No extreme outliers**: Boxplots, Z-scores

**If assumptions violated:**
- Use non-parametric alternatives
- Apply corrections (Greenhouse-Geisser for sphericity)
- Transform data (log, sqrt)

**Step 4: Run Appropriate Test**

**Two Conditions:**
- Within-subjects: Paired t-test
- Between-subjects: Independent t-test
- Non-parametric: Wilcoxon signed-rank or Mann-Whitney U

**Three+ Conditions:**
- Within-subjects: Repeated Measures ANOVA
- Between-subjects: One-way ANOVA
- Non-parametric: Friedman or Kruskal-Wallis

**Multiple Factors:**
- Factorial RM-ANOVA (e.g., Device × Time)
- Test main effects and interactions

**Continuous Relationships:**
- Correlation: Pearson's r or Spearman's ρ
- Prediction: Linear/Multiple/Logistic regression

**Step 5: Post-Hoc Analysis**

**If ANOVA is significant:**
- **Bonferroni correction**: Conservative, planned comparisons
- **Holm-Bonferroni**: Less conservative
- **Tukey HSD**: All pairwise comparisons

**If interaction is significant:**
- **Simple effects analysis**: Test one factor at each level of the other
- Then post-hoc if needed

**Step 6: Report Effect Sizes**

**Always report alongside p-values:**
- **t-test**: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
- **ANOVA**: η² or partial η² (small: 0.01, medium: 0.06, large: 0.14)
- **Correlation**: r² (small: 0.01, medium: 0.09, large: 0.25)

### Common Scenarios

**Scenario 1: Compare two activity types (same subjects)**
```
Walking vs. Running (same people)
↓
Paired t-test
↓
Check normality (Shapiro-Wilk)
↓
If normal: report t, p, Cohen's d
If violated: use Wilcoxon signed-rank
```

**Scenario 2: Compare three activities over time**
```
Activity (Walk, Jog, Run) × Time (0, 30, 60 min)
↓
2-way RM-ANOVA
↓
Check sphericity (Mauchly's test)
If violated: use Greenhouse-Geisser correction
↓
If significant:
- Main effects (Activity? Time?)
- Interaction (Activity×Time?)
↓
Post-hoc: Bonferroni pairwise comparisons
Report: F-stats, p-values, partial η²
```

**Scenario 3: Does head velocity predict movement type?**
```
Continuous predictor (velocity) → Categorical outcome (activity)
↓
Option A: Correlation (if ordinal coding)
Option B: Logistic regression (if binary)
Option C: Discriminant analysis (if multiple categories)
↓
Check assumptions
↓
Report: Coefficients, classification accuracy, effect sizes
```

**Scenario 4: Multiple predictors of head stability**
```
Velocity + Acceleration + Neck Angle → Stability Score
↓
Multiple Linear Regression
↓
Check assumptions:
- Linearity (scatterplots)
- Independence (Durbin-Watson)
- Homoscedasticity (residual plots)
- Normality of residuals (Q-Q plot)
- Multicollinearity (VIF < 10)
↓
Report: R², β coefficients, p-values for each predictor
```

---

## Data Sources

### CMU Graphics Lab Motion Capture Database

**URL:** http://mocap.cs.cmu.edu/

**Description:**
- 2,605 free motion capture trials
- Diverse activities: walking, running, sports, dancing, interactions
- Format: BVH, C3D, AMC/ASF
- License: Free for research and commercial use

**Recommended Trials for This Project:**

**Walking/Running (for gait analysis):**
- Subject 07: Walking trials
- Subject 08: Running trials
- Subject 16: Walking at different speeds

**Reaching/Manipulation:**
- Subject 13: Reaching and grabbing
- Subject 56: Overhead reaching
- Subject 83: Complex movements

**General Movement:**
- Subject 02: Various motions
- Subject 75: Exercise movements
- Subject 86: Daily activities

**Data Format:**
- **BVH** (Biovision Hierarchy) - easiest to work with
- Contains: skeleton hierarchy, joint rotations, translations
- Can be loaded with Python libraries: `bvh`, `pybvh`, `ezc3d`

---

## Technical Implementation

### Python Libraries

**Data Processing:**
```python
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import distance
```

**BVH File Handling:**
```python
# Install: pip install bvh
import bvh
# OR
# pip install pybvh
from pybvh import BvhFile
```

**Statistical Analysis:**
```python
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
```

**Visualization:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
```

### Analysis Workflow Template

```python
# 1. Load Data
mocap_data = load_bvh_file('subject_07_walking.bvh')

# 2. Extract Joint Positions
head_pos = mocap_data.get_joint_position('Head')
neck_pos = mocap_data.get_joint_position('Neck')
hand_pos = mocap_data.get_joint_position('RightHand')

# 3. Calculate Derived Metrics
head_velocity = calculate_velocity(head_pos, fps=120)
head_speed = np.linalg.norm(head_velocity, axis=1)
neck_angle = calculate_joint_angle(shoulder_pos, neck_pos, head_pos)

# 4. Segment by Activity
walking_segments = detect_gait_cycles(foot_position)
standing_segments = detect_standing(movement_threshold=0.01)

# 5. Extract Features
features = {
    'mean_head_speed': np.mean(head_speed),
    'max_head_speed': np.max(head_speed),
    'mean_neck_flexion': np.mean(neck_angle),
    'neck_rom': np.max(neck_angle) - np.min(neck_angle)
}

# 6. Statistical Analysis
activity_comparison = stats.f_oneway(
    walking_speeds,
    running_speeds,
    standing_speeds
)

# 7. Visualization
plot_3d_trajectory(head_pos)
plot_velocity_profile(head_speed, time)
plot_comparative_boxplot(activity_data)
```

---

## Deliverables

### GitHub Repository Structure

```
mocap-headphone-analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Downloaded BVH files
│   ├── processed/        # Cleaned/extracted data
│   └── README.md         # Data source documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_visualization.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # BVH file loading utilities
│   ├── kinematics.py     # Velocity, angle calculations
│   ├── features.py       # Feature extraction functions
│   ├── statistics.py     # Statistical test wrappers
│   └── visualization.py  # Plotting functions
├── results/
│   ├── figures/          # Publication-quality plots
│   ├── tables/           # Statistical results
│   └── report.md         # Final analysis report
└── docs/
    ├── study_protocol.md
    └── analysis_plan.md
```

### Final Report Components

**1. Executive Summary (1 page)**
- Research question
- Key findings
- Design implications

**2. Methods (2-3 pages)**
- Data source description
- Participant/trial selection criteria
- Kinematic metrics extracted
- Statistical approach

**3. Results (3-4 pages)**
- Descriptive statistics
- Statistical test results with tables
- Visualizations (3-5 key figures)
- Effect sizes and confidence intervals

**4. Discussion (2-3 pages)**
- Interpretation of findings
- Connection to headphone design
- Biomechanical thresholds identified
- Limitations and future work

**5. Design Recommendations**
- Specific, actionable suggestions based on findings
- Example: "Design should accommodate head velocities up to 1.5 m/s during running"

### Key Visualizations to Create

**1. 3D Trajectory Plot**
- Head movement path during activity
- Color-coded by velocity or time

**2. Time-Series Comparison**
- Head velocity during walking vs. running
- Neck angle profiles over time

**3. Box Plot Comparisons**
- Peak velocity across activities
- Range of motion by task type

**4. Correlation Matrix**
- Relationships between kinematic variables
- Heat map visualization

**5. Summary Dashboard**
- Multi-panel figure for stakeholders
- Key metrics at-a-glance

---

## Interview Preparation

### Methodology Question Responses

**"What methods do you tend to use?"**

*"My background is in psychoacoustic research, so I primarily use **mixed methods combining objective behavioral measures with subjective ratings**. For study design, I typically use **within-subjects approaches** to reduce individual variability. For statistical analysis, I rely on **repeated measures techniques** - paired t-tests for two-condition comparisons, RM-ANOVA for multiple conditions or time points, and correlation/regression for continuous relationships. I always check assumptions first and use non-parametric alternatives when needed. I work in **Python** (pandas, scipy, statsmodels) and **R** for complex mixed models."*

*"For motion capture specifically, I'd extract kinematic metrics like velocity and joint angles from position time-series data, then correlate these with subjective outcomes using the same analytical framework I've applied to sensor data in my audio research."*

**"How do you handle missing data?"**

*"Depends on the pattern and extent. For completely random missingness under 5%, listwise deletion is often acceptable. For more extensive or patterned missing data, I prefer **multiple imputation** or **mixed-effects models** which handle missingness more gracefully. For time-series specifically, I've used interpolation or forward-fill when appropriate, but always document it as a preprocessing step and test sensitivity to the imputation method."*

**"Walk me through a user study you designed."**

*"At Google, I designed studies evaluating spatial audio HRTF performance. I used **within-subjects design** with 12 participants experiencing multiple HRTF conditions. The protocol included a **task battery** of localization trials across different azimuth and elevation angles, with participants indicating perceived sound location. I collected both **objective accuracy metrics** - azimuth error, elevation error, front-back confusion rate - and **subjective ratings** of externalization and naturalness at regular intervals."*

*"For statistical analysis, I used **repeated measures ANOVA** to compare HRTF conditions, with **Bonferroni post-hoc tests** for pairwise comparisons. I correlated objective performance with subjective ratings to understand the relationship between technical accuracy and perceived quality. Key finding: HRTFs with <5° localization error didn't necessarily receive higher subjective ratings - externalization perception mattered more."*

*"For a headphone comfort study, I'd apply the same framework: within-subjects design, task battery representing realistic use cases, quantitative mocap metrics paired with subjective comfort ratings, RM-ANOVA for comparison, correlation to identify biomechanical predictors of discomfort."*

**"How do you ensure validity and reliability?"**

*"**Internal validity**: Careful experimental design - counterbalancing condition order to control for learning effects, randomizing stimulus presentation, including attention checks. **External validity**: Recruiting diverse participants, using realistic tasks, testing in conditions that approximate real-world use."*

*"**Reliability**: Multiple trials per condition to assess within-participant consistency, inter-rater reliability for qualitative coding, test-retest reliability where feasible. I also pre-register hypotheses and analysis plans when possible to prevent p-hacking."*

**"Tell me about defending results when challenged."**

*"At Google, engineering teams sometimes questioned whether perceptual differences we found were large enough to matter for product. I learned to always lead with **effect sizes** alongside p-values - 'yes it's statistically significant, AND the effect accounts for 23% of variance, which translates to X improvement in user experience.'"*

*"I also present **alternative explanations** proactively. For example, if we found HRTF A outperformed HRTF B, I'd discuss whether the difference could be due to familiarity bias, test order effects, or individual anatomy differences - and show what controls we had in place or what follow-up analyses addressed those concerns."*

*"The key is being **empirically rigorous but intellectually humble** - confident in the data but open to legitimate methodological critiques."*

### Technical Depth Questions

**"What assumptions does RM-ANOVA require?"**

*"Three main assumptions: **sphericity** - the variances of differences between all pairs of conditions are equal, tested via Mauchly's test; **normality** of the dependent variable within each condition, checked with Shapiro-Wilk or Q-Q plots; and **no extreme outliers**. If sphericity is violated, I apply **Greenhouse-Geisser** or **Huynh-Feldt corrections**. If normality is severely violated, I might use **Friedman test** as a non-parametric alternative."*

**"How would you analyze mocap data with high missingness?"**

*"First, I'd investigate the **missingness mechanism**. If markers are occluded during specific movements (not random), that's informative - those movements might be problematic for the headphone design. For random technical dropout, options include: **interpolation** for brief gaps (<100ms), **mixed-effects models** which handle irregular time points naturally, or **excluding severely incomplete trials** if they don't introduce systematic bias. I'd also document the pattern and test whether excluding missing data changes conclusions."*

**"What's the difference between repeated measures and within-subjects?"**

*"**Within-subjects** refers to the experimental design - same participants experience multiple conditions. **Repeated measures** describes the data structure - multiple observations from the same participants, which could be across conditions OR over time. All within-subjects designs involve repeated measures, but repeated measures can also describe longitudinal tracking of a single condition. Statistically, both require accounting for non-independence through RM-ANOVA, mixed models, or paired tests."*

---

## Next Steps

### Immediate Actions

1. **Download CMU mocap data** - Select 3-5 trials covering different activities
2. **Set up Python environment** - Install BVH processing libraries
3. **Create GitHub repository** - Initialize structure
4. **Load and explore data** - Understand format, visualize sample trajectories
5. **Define specific research question** - Based on available data

### Development Priorities

**Week 1:**
- Data acquisition and exploration
- Pipeline development (loading, extracting positions)
- Basic visualizations (3D trajectories)

**Week 2:**
- Feature extraction (velocities, angles, distances)
- Event detection (reaching phases, gait cycles)
- Descriptive statistics

**Week 3:**
- Statistical analysis implementation
- Comparative tests across activities
- Effect size calculations

**Week 4:**
- Visualization refinement
- Results interpretation
- Documentation and report writing

---

## References & Resources

### Motion Capture Tools
- **FreeMoCap**: https://github.com/freemocap/freemocap (open-source markerless mocap)
- **CMU Mocap Database**: http://mocap.cs.cmu.edu/
- **BVH Python Library**: https://pypi.org/project/bvh/

### Statistical Resources
- **Scipy Stats Documentation**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **Statsmodels**: https://www.statsmodels.org/stable/index.html
- **G*Power** (power analysis): https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower

### Biomechanics Literature
- Ergonomics and comfort assessment methodologies
- Wearable device biomechanics studies
- Gait analysis fundamentals
- Upper extremity kinematics

### Domain Knowledge
- Apple product documentation (AirPods, spatial audio)
- Headphone comfort factor research
- Psychoacoustics and spatial audio principles

---

## Notes & Constraints

**What You Have:**
- ✅ Strong quantitative analysis background
- ✅ User study design experience
- ✅ Statistical expertise (repeated measures, mixed methods)
- ✅ Python/R proficiency
- ✅ Domain knowledge in audio products
- ✅ Sensor data experience (IMU from VR)

**What You're Building:**
- ✅ Mocap data analysis pipeline
- ✅ Biomechanical metrics extraction
- ✅ Statistical comparison framework
- ✅ Stakeholder-appropriate visualizations
- ✅ Design recommendation methodology

**What You Don't Have (Yet):**
- ❌ Hands-on experience with commercial mocap systems (Vicon, OptiTrack)
- ❌ Real participant data collection with mocap
- ❌ Formal biomechanics training

**Interview Strategy:**
- Lead with strengths (analytical rigor, study design, domain knowledge)
- Frame gaps as technical (system-specific) not conceptual
- Demonstrate proactive learning (FreeMoCap exploration, this project)
- Connect existing skills to new domain explicitly
- Show enthusiasm for learning commercial systems

---

**Document Version:** 1.0  
**Last Updated:** 2024-12-14  
**Project Status:** Design Phase → Implementation Pending