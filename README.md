# 🛸 Vision-Guided Precision Docking System

> Autonomous Drone Docking via Real-Time Spatial Tracking & Markov Decision Process (MDP) Bellman Oscillation Optimization Engine

---

## 📌 Executive Overview

Autonomous drone landing and precision docking systems frequently suffer from spatial drift caused by aerodynamic turbulence, ground effects, and camera latency. Traditional PID controllers often induce high-frequency oscillations during rapid real-time corrections.

This project implements a Two-Layer Hybrid Control & Perception Architecture:
1. Perception Layer (spatial_tracker.py): Real-time OpenCV feature detection and geometric spatial error extraction.
2. Optimization & Control Layer (mdp_solver.py): A Markov Decision Process (MDP) powered by Bellman Dynamic Programming to penalize position error and oscillation velocity dynamically.

---

## 🏗️ System Architecture & Mathematical Pipeline

### 1. Spatial State Extraction (Perception)
The tracker evaluates the target centroid relative to the drone frame center to generate a 4D state vector S_t:

S_t = [e_x, e_y, e_dot_x, e_dot_y]

Where:
- e_x, e_y: Horizontal and vertical spatial offset pixels from docking point center.
- e_dot_x, e_dot_y: First-order discrete time derivative (velocity of drift deviation).

### 2. MDP Bellman Value Optimization (Decision Engine)
The MDP solver minimizes a dual-penalty reward function R(S_t, A_t):

R(S_t, A_t) = - ( alpha * sqrt(e_x^2 + e_y^2) + beta * sqrt(e_dot_x^2 + e_dot_y^2) )

- alpha (Position Error Weight): Penalizes distance away from the target centroid.
- beta (Oscillation Weight): Penalizes high-frequency corrective motion to eliminate overshoot and control flutter.

---

## 📂 Repository Structure

- webdash.py : Streamlit UI Dashboard & HUD Telemetry Visualizer
- spatial_tracker.py : Computer Vision Perception Module (OpenCV)
- mdp_solver.py : Dynamic Programming MDP & Bellman Optimization Engine
- .streamlit/config.toml : Streamlit Custom Theme Configuration (Royal Blue)
- README.md : Project Documentation & Research Overview

---

## 🚀 Quick Start & Installation

1. Clone the Repository:
git clone https://github.com/minahiliftikhar14/vision-guided-precision-docking.git

2. Navigate to Directory:
cd vision-guided-precision-docking

3. Install Dependencies:
pip install streamlit opencv-python numpy matplotlib

4. Run Dashboard:
streamlit run webdash.py

---

## 🌟 Key Research Contributions

- Reduced Oscillation Rate: Eliminates rapid target overshoot by factoring drift acceleration directly into the Bellman reward function.
- Lightweight Real-Time Control: Achieves low-latency decision cycles suitable for onboard edge devices (Raspberry Pi / Jetson Nano).
