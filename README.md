# 🛸 Vision-Guided Precision Docking System
> **Autonomous Drone Docking via Real-Time Spatial Tracking & Markov Decision Process (MDP) Bellman Oscillation Optimization Engine**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Control Theory](https://img.shields.io/badge/Control-MDP%20%26%20Bellman-purple.svg)

---

## 📌 Executive Overview
Autonomous drone landing and precision docking systems frequently suffer from **spatial drift** caused by aerodynamic turbulence, ground effects, and camera latency. Traditional PID controllers often induce high-frequency oscillations during rapid real-time corrections.

This project implements a **Two-Layer Hybrid Control & Perception Architecture**:
1. **Perception Layer (`spatial_tracker.py`)**: Real-time OpenCV feature detection and geometric spatial error extraction.
2. **Optimization & Control Layer (`mdp_solver.py`)**: A Markov Decision Process (MDP) powered by Bellman Dynamic Programming to penalize position error and oscillation velocity dynamically.

---

## 🏗️ System Architecture & Mathematical Pipeline
+------------------+     +------------------------+     +----------------------------+
|  Camera Feed     | --> | Perception Layer       | --> | MDP Decision Engine        |
|  (Raw Frame)     |     | (Spatial Offset Tracker)|     | (Bellman Optimisation)     |
+------------------+     +------------------------+     +----------------------------+
|
v
+----------------------------+
| Real-Time Control Action   |
| & Streamlit Dashboard HUD  |
+----------------------------+
### 1. Spatial State Extraction (Perception)
The tracker evaluates the target centroid relative to the drone frame center to generate a 4D state vector $S_t$:

$$S_t = [e_x, e_y, \dot{e}_x, \dot{e}_y]$$

Where:
* $e_x, e_y$: Horizontal and vertical spatial offset pixels from the docking point center.
* $\dot{e}_x, \dot{e}_y$: First-order discrete time derivative (velocity of drift deviation).

### 2. MDP Bellman Value Optimization (Decision Engine)
The MDP solver minimizes a dual-penalty reward function $R(S_t, A_t)$:

$$R(S_t, A_t) = -\left( \alpha \cdot \sqrt{e_x^2 + e_y^2} + \beta \cdot \sqrt{\dot{e}_x^2 + \dot{e}_y^2} \right)$$

* $\alpha$ **(Position Error Weight)**: Penalizes distance away from the target centroid.
* $\beta$ **(Oscillation Weight)**: Penalizes high-frequency corrective motion to eliminate overshoot and control flutter.

The value state is dynamically solved via the Bellman Expectation Equation:

$$V(S_t) = \max_{A_t} \left[ R(S_t, A_t) + \gamma \cdot V(S_{t+1}) \right]$$

Where $\gamma \in [0, 1)$ represents the discount factor for long-term trajectory stabilization.

---

## 📂 Repository Structure
vision-guided-precision-docking/
│
├── webdash.py          # Streamlit UI Dashboard & HUD Telemetry Visualizer
├── spatial_tracker.py  # Computer Vision Perception Module (OpenCV)
├── mdp_solver.py       # Dynamic Programming MDP & Bellman Optimization Engine
├── .streamlit/
│   └── config.toml     # Streamlit Custom Theme Configuration (Royal Blue Theme)
└── README.md           # Project Documentation & Research Overview
---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)minahiliftikhar14/vision-guided-precision-docking.git
cd vision-guided-precision-docking
## 2.Install Dependencies
pip install streamlit opencv-python numpy matplotlib
## 3. Run the Control Dashboard
streamlit run webdash.py

Perception HUD: Live bounding boxes, drift vector arrows, and alignment target reticle overlays.State Vector Monitor: Dynamic calculation of $e_x, e_y$ positional offsets in real time.Live Convergence Curves: Dynamic Matplotlib plots showing Bellman value reward convergence and drift reduction over time.🌟 Key Research ContributionsReduced Oscillation Rate: Eliminates rapid target overshoot by factoring drift acceleration ($\beta$) directly into the Bellman reward function.Lightweight Real-Time Control: Achieves low-latency decision cycles suitable for onboard edge devices (Raspberry Pi / Jetson Nano).
