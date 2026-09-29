import streamlit as st
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Import custom modules
from spatial_tracker import SpatialTracker
from mdp_solver import MDPOscillationSolver

# Page Setup
st.set_page_config(
    page_title="Vision-Guided Precision Docking System",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STABLE ROYAL BLUE & WHITE STYLING
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* HEADER BANNER */
    .main-header {
        background: linear-gradient(135deg, #0F172A, #1E3A8A);
        padding: 24px 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    .main-header h1 { color: #FFFFFF !important; font-size: 26px; font-weight: 700; margin: 0; }
    .main-header p { color: #93C5FD !important; font-size: 14px; margin-top: 5px; }

    /* SIDEBAR TEXT & BACKGROUND FIX */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }

    /* UPLOAD DROPZONE */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #1E3A8A !important;
        border: 2px dashed #3B82F6 !important;
        border-radius: 14px !important;
    }
    section[data-testid="stFileUploadDropzone"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* BUTTONS */
    button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* METRICS */
    [data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #2563EB !important;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        padding: 16px !important;
        border-radius: 16px !important;
    }

    /* BADGES */
    .status-badge {
        padding: 10px 16px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        text-align: center;
        margin-top: 10px;
    }
    .status-aligned { background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
    .status-drifting { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .status-danger { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    </style>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
    <div class="main-header">
        <h1>🛸 Vision-Guided Precision Docking System</h1>
        <p>Markov Decision Process (MDP) Bellman Oscillation Optimization Engine</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🕹️ Video Input & MDP Controls")

input_mode = st.sidebar.radio("Select Video Input Mode", ["Preset Video", "Upload Custom MP4"])

video_path = None

if input_mode == "Preset Video":
    preset_videos = {
        "Sample 1 (Standard Track)": "sample.mp4",
        "Sample 2 (Top-Down Drone Landing)": "drone_landing.mp4"
    }
    selected_preset = st.sidebar.selectbox("Choose Preset Video", list(preset_videos.keys()))
    video_path = preset_videos.get(selected_preset)
else:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video File", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Bellman Parameters")
discount_gamma = st.sidebar.slider("Discount Factor (γ)", 0.50, 0.99, 0.95, 0.01)
error_penalty_alpha = st.sidebar.slider("Position Error Weight (α)", 0.1, 3.0, 0.5, 0.1)
oscillation_penalty_beta = st.sidebar.slider("Oscillation Rate Weight (β)", 0.1, 3.0, 1.2, 0.1)

run_simulation = st.sidebar.button("▶ Run Docking Optimization", type="primary")

# Modules Initialization
tracker = SpatialTracker()
mdp_solver = MDPOscillationSolver(gamma=discount_gamma, alpha=error_penalty_alpha, beta=oscillation_penalty_beta)

# Layout
col_video, col_telemetry = st.columns([2, 1])

with col_telemetry:
    st.subheader("📊 Live Telemetry HUD")
    metric_state = st.empty()
    metric_action = st.empty()
    metric_reward = st.empty()
    risk_badge = st.empty()

with col_video:
    st.subheader("👁️ Visual Perception Overlay")
    video_placeholder = st.empty()

st.markdown("---")
st.subheader("📈 Real-Time Optimization Curves")
col_chart1, col_chart2 = st.columns(2)
chart_reward_place = col_chart1.empty()
chart_error_place = col_chart2.empty()

reward_history = []
error_history = []

if run_simulation:
    if not video_path or not os.path.exists(video_path):
        st.error("⚠️ Video file not found! Please check file path or upload video.")
    else:
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            annotated_frame, state_vector = tracker.extract_spatial_telemetry(frame)
            ex, ey, ex_dot, ey_dot = state_vector
            error_magnitude = np.sqrt(ex**2 + ey**2)

            action_vec, action_label, reward, q_val = mdp_solver.solve_bellman_action(state_vector)

            reward_history.append(reward)
            error_history.append(error_magnitude)
            if len(reward_history) > 40:
                reward_history.pop(0)
                error_history.pop(0)

            metric_state.metric("State Vector (e_x, e_y)", f"X: {ex:.1f} | Y: {ey:.1f}")
            metric_action.metric("Optimal Action (A_t)", f"{action_label}")
            metric_reward.metric("Bellman Reward (R_t)", f"{reward:.2f}")

            if error_magnitude < 15.0 and np.abs(ex_dot) < 3.0:
                risk_badge.markdown('<div class="status-badge status-aligned">🟢 STABLE / DOCKED</div>', unsafe_allow_html=True)
            elif error_magnitude < 45.0:
                risk_badge.markdown('<div class="status-badge status-drifting">🟡 RE-ALIGNING DRIFT</div>', unsafe_allow_html=True)
            else:
                risk_badge.markdown('<div class="status-badge status-danger">🔴 HIGH OSCILLATION</div>', unsafe_allow_html=True)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            fig1, ax1 = plt.subplots(figsize=(4, 2.2))
            fig1.patch.set_facecolor('#FFFFFF')
            ax1.set_facecolor('#F8FAFC')
            ax1.plot(reward_history, color='#2563EB', linewidth=2)
            ax1.set_title("Bellman Convergence", fontsize=10, color='#0F172A', fontweight='bold')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            chart_reward_place.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(4, 2.2))
            fig2.patch.set_facecolor('#FFFFFF')
            ax2.set_facecolor('#F8FAFC')
            ax2.plot(error_history, color='#EF4444', linewidth=2)
            ax2.set_title("Drift Displacement (Error px)", fontsize=10, color='#0F172A', fontweight='bold')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            chart_error_place.pyplot(fig2)
            plt.close(fig2)

            time.sleep(0.02)

        cap.release()
else:
    st.info("Upload an MP4 video or select a preset from the sidebar, then click 'Run Docking Optimization'.")