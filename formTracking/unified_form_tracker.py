import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from datetime import datetime


# ---------- Helper Functions ----------

def inject_base_styles():
    """Inject shared styling for the Streamlit experience."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #0b0f1c;
            --panel: rgba(12, 15, 30, 0.72);
            --outline: rgba(255, 255, 255, 0.12);
            --accent: linear-gradient(120deg, #7f5dff, #58d6ff);
            --muted: rgba(245, 247, 252, 0.7);
            --text-strong: #f7f8fc;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 20% 20%, rgba(127, 93, 255, 0.18), transparent 35%),
                        radial-gradient(circle at 80% 0%, rgba(88, 214, 255, 0.16), transparent 40%),
                        linear-gradient(135deg, #050915 0%, #0b1224 70%, #050915 100%);
        }

        .block-container {
            padding: 32px 28px 48px 28px;
            max-width: 1100px;
        }

        h1, h2, h3, h4, h5 {
            color: var(--text-strong);
        }

        p, li, label {
            color: var(--muted);
        }

        .hero-card, .glass-card, .feature-card-v2, .contact-card, .flow-card {
            background: var(--panel);
            border: 1px solid var(--outline);
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 22px 50px rgba(5, 8, 25, 0.35);
            backdrop-filter: blur(16px);
        }

        .hero-card {
            padding: 34px 32px;
        }

        .eyebrow {
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: rgba(247, 248, 252, 0.6);
            font-size: 0.85rem;
            margin-bottom: 6px;
        }

        .hero-title {
            font-size: clamp(2.2rem, 5vw, 3.4rem);
            margin: 0 0 12px 0;
            line-height: 1.1;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.6;
            margin: 0;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            border-radius: 999px;
            border: 1px solid var(--outline);
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-strong);
            font-weight: 600;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }

        .feature-card-v2 h3 {
            margin-top: 6px;
            margin-bottom: 10px;
        }

        .contact-card {
            background: linear-gradient(135deg, rgba(127, 93, 255, 0.12), rgba(88, 214, 255, 0.08));
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        .flow-card {
            border-left: 4px solid rgba(127, 93, 255, 0.7);
        }

        .flow-card.secondary {
            border-left-color: rgba(88, 214, 255, 0.8);
        }

        .stat-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }

        .stat {
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 14px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stat strong {
            font-size: 1.4rem;
            color: #7fd4ff;
        }

        .stButton>button, .stDownloadButton>button {
            background: var(--accent);
            color: #050610;
            border: none;
            padding: 14px 18px;
            border-radius: 12px;
            font-weight: 700;
            width: 100%;
            box-shadow: 0 12px 30px rgba(94, 130, 255, 0.35);
        }

        .stButton>button:hover {
            transform: translateY(-1px);
        }

        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px dashed rgba(255, 255, 255, 0.18);
        }

        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid var(--outline);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-strong);
            font-weight: 600;
        }

        @media (max-width: 640px) {
            .block-container {
                padding: 20px 16px 40px 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b (vertex) formed by a and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# ---------- Pull-up Functions ----------

def get_pose_details_pullup(results):
    """
    Extracts keypoints for pull-up analysis.
    Returns: head_y, shoulder_y, keypoints dict
    """
    if results is None or len(results) == 0:
        return None, None, None

    r = results[0]
    if r.keypoints is None or r.keypoints.xy is None:
        return None, None, None

    kpts = r.keypoints.xy
    if kpts.shape[0] == 0:
        return None, None, None

    # Get main person
    dets = r.boxes
    if dets is None or len(dets) == 0:
        return None, None, None

    scores = dets.conf.cpu().numpy()
    main_idx = int(scores.argmax())
    person_kpts = kpts[main_idx].cpu().numpy()

    def get_pt(idx):
        if idx < len(person_kpts) and person_kpts[idx][0] != 0 and person_kpts[idx][1] != 0:
            return person_kpts[idx]
        return None

    # COCO keypoint indices: 0=nose, 5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow, 11=l_hip, 12=r_hip
    keypoints = {
        "nose": get_pt(0),
        "l_sh": get_pt(5), "r_sh": get_pt(6),
        "l_el": get_pt(7), "r_el": get_pt(8),
        "l_hip": get_pt(11), "r_hip": get_pt(12)
    }

    # Calculate head Y (use nose as proxy for chin/head position)
    # Calculate shoulder Y (average of both shoulders)
    head_y = None
    if keypoints["nose"] is not None:
        head_y = float(keypoints["nose"][1])
    
    shoulder_y = None
    shoulder_coords = []
    if keypoints["l_sh"] is not None:
        shoulder_coords.append(keypoints["l_sh"][1])
    if keypoints["r_sh"] is not None:
        shoulder_coords.append(keypoints["r_sh"][1])
    
    if shoulder_coords:
        shoulder_y = float(np.mean(shoulder_coords))

    return head_y, shoulder_y, keypoints


def draw_debug_overlay_pullup(frame, kpts, head_y, shoulder_y):
    """
    Draws visualization overlay showing head position relative to shoulders.
    """
    # Draw shoulder line
    if kpts['l_sh'] is not None and kpts['r_sh'] is not None:
        cv2.line(frame, tuple(kpts['l_sh'].astype(int)), tuple(kpts['r_sh'].astype(int)), (255, 255, 0), 2)
    
    # Draw head position indicator
    if kpts['nose'] is not None and shoulder_y is not None:
        # Draw line from nose to shoulder level
        nose_pos = tuple(kpts['nose'].astype(int))
        shoulder_x = int((kpts['l_sh'][0] + kpts['r_sh'][0]) / 2) if kpts['l_sh'] is not None and kpts['r_sh'] is not None else nose_pos[0]
        shoulder_pos = (shoulder_x, int(shoulder_y))
        
        # Color based on position
        head_above_shoulder = head_y < shoulder_y  # Lower Y = higher in image
        line_color = (0, 255, 0) if head_above_shoulder else (0, 0, 255)  # Green if above, red if below
        
        cv2.line(frame, nose_pos, shoulder_pos, line_color, 2)
        cv2.circle(frame, nose_pos, 5, (0, 255, 255), -1)
    
    # Text overlay
    if head_y is not None and shoulder_y is not None:
        head_above_shoulder = head_y < shoulder_y  # Lower Y = higher in image
        status = "CHIN ABOVE BAR" if head_above_shoulder else "CHIN BELOW BAR"
        color = (0, 255, 0) if head_above_shoulder else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame


# ---------- Push-up Functions ----------

def get_pose_details_pushup(results):
    """
    Extracts keypoints and torso Y.
    """
    if results is None or len(results) == 0:
        return None, None

    r = results[0]
    if r.keypoints is None or r.keypoints.xy is None:
        return None, None

    kpts = r.keypoints.xy
    if kpts.shape[0] == 0:
        return None, None

    # Get main person
    dets = r.boxes
    if dets is None or len(dets) == 0:
        return None, None

    scores = dets.conf.cpu().numpy()
    main_idx = int(scores.argmax())
    person_kpts = kpts[main_idx].cpu().numpy()

    def get_pt(idx):
        if idx < len(person_kpts) and person_kpts[idx][0] != 0 and person_kpts[idx][1] != 0:
            return person_kpts[idx]
        return None

    keypoints = {
        "l_sh": get_pt(5), "r_sh": get_pt(6),
        "l_el": get_pt(7), "r_el": get_pt(8),
        "l_hip": get_pt(11), "r_hip": get_pt(12)
    }

    # Calculate Torso Y (average of shoulders and hips)
    y_coords = []
    for k in keypoints.values():
        if k is not None:
            y_coords.append(k[1])

    if not y_coords:
        return None, None

    torso_y = float(np.mean(y_coords))
    return torso_y, keypoints


def draw_debug_overlay_pushup(frame, kpts, angle, threshold):
    """
    Draws the torso-to-arm lines and the calculated angle on the frame.
    """
    # Color based on threshold
    color = (0, 255, 0) if angle < threshold else (0, 0, 255)  # Green if good, Red if bad

    # Draw Left Side (if visible)
    if kpts['l_sh'] is not None and kpts['l_hip'] is not None and kpts['l_el'] is not None:
        cv2.line(frame, tuple(kpts['l_sh'].astype(int)), tuple(kpts['l_hip'].astype(int)), (255, 255, 0), 2)  # Torso
        cv2.line(frame, tuple(kpts['l_sh'].astype(int)), tuple(kpts['l_el'].astype(int)), color, 3)  # Arm

    # Draw Right Side (if visible)
    if kpts['r_sh'] is not None and kpts['r_hip'] is not None and kpts['r_el'] is not None:
        cv2.line(frame, tuple(kpts['r_sh'].astype(int)), tuple(kpts['r_hip'].astype(int)), (255, 255, 0), 2)
        cv2.line(frame, tuple(kpts['r_sh'].astype(int)), tuple(kpts['r_el'].astype(int)), color, 3)

    # Text overlay
    cv2.putText(frame, f"Angle: {int(angle)} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame


# ---------- Page Functions ----------

def show_landing_page():
    """Display the landing page with mission, features, and contact info."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="pill">AI Rehab Studio</div>
            <h1 class="hero-title">Move better with confident, beautiful feedback.</h1>
            <p class="hero-subtitle">
                TrainR blends live computer vision with PT-approved coaching. Calibrate, choose your flow,
                and get cinematic cues that keep every rep intentional.
            </p>
            <div class="stat-row">
                <div class="stat"><strong>--</strong><span>sessions guided</span></div>
                <div class="stat"><strong>--</strong><span>tracking precision</span></div>
                <div class="stat"><strong>--</strong><span>movement templates</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cta1, col_cta2 = st.columns([1, 1])
    with col_cta1:
        if st.button("🚀 Launch today’s plan", key="cta_schedule", use_container_width=True, type="primary"):
            st.session_state.page = "workout_schedule"
            st.rerun()
    with col_cta2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="eyebrow">What’s inside</div>
                <p style="margin:4px 0 10px;">Live form overlays, therapist cues, and adaptive checklists for each movement.</p>
                <div class="metric-chip">🎥 Pose overlays</div>
                <div class="metric-chip" style="margin-top:8px;">🧠 Smart coaching</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Core capabilities")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card-v2">
                <div class="pill">01</div>
                <h3>Real-time form correction</h3>
                <p>Track key joints and get instant, actionable cues to prevent injuries and build consistency.</p>
            </div>
            <div class="feature-card-v2">
                <div class="pill">02</div>
                <h3>Nutrition pulse</h3>
                <p>Quick macros with AI-assisted estimates so logging feels effortless and sustainable.</p>
            </div>
            <div class="feature-card-v2">
                <div class="pill">03</div>
                <h3>Adaptive education</h3>
                <p>Short insights tailored to your habits and progress to keep you learning without overload.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="contact-card" style="margin-top:18px;">
            <div class="eyebrow">Stay in touch</div>
            <h3 style="margin:6px 0 6px;">Have feedback or want early features?</h3>
            <p style="margin:0 0 12px;">We’d love to hear from you. Send a note and let’s shape TrainR together.</p>
            <div class="metric-chip">📧 contact@trainr.app</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_workout_schedule():
    """Display today's workout schedule."""
    # Header
    today = datetime.now().strftime("%B %d, %Y")
    st.title("📅 Today’s workout")
    st.markdown(f"**{today}**")

    # Back button
    if st.button("← Back to Home", key="back_home"):
        st.session_state.page = "landing"
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Pull-up Exercise Card
    st.markdown(
        """
        <div class="flow-card">
            <div class="eyebrow">Strength</div>
            <h2 style="margin:6px 0;">🏋️‍♂️ Pull-ups</h2>
            <p style="margin: 0 0 12px;"><strong>3 sets of 2-3 reps</strong></p>
            <p style="margin:0;">Chin clears the bar with control. Focus on lat drive and stacked shoulders.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("▶️ Start Pull-ups", key="start_pullup", use_container_width=True, type="primary"):
        st.session_state.page = "exercise"
        st.session_state.exercise = "pullup"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Push-up Exercise Card
    st.markdown(
        """
        <div class="flow-card secondary">
            <div class="eyebrow">Control</div>
            <h2 style="margin:6px 0;">🏃 Push-ups</h2>
            <p style="margin: 0 0 12px;"><strong>3 sets of 5-8 reps</strong></p>
            <p style="margin:0;">Keep elbows tucked, ribs down, and move through a smooth 4s cadence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("▶️ Start Push-ups", key="start_pushup", use_container_width=True, type="primary"):
        st.session_state.page = "exercise"
        st.session_state.exercise = "pushup"
        st.rerun()


def show_exercise_analysis():
    """Display exercise analysis page with video upload."""
    # Back button
    if st.button("← Back to Workout Schedule", key="back_schedule"):
        st.session_state.page = "workout_schedule"
        st.session_state.exercise = None
        st.rerun()
    
    if st.session_state.exercise == "pullup":
        st.subheader("🏋️‍♂️ Pull-up Analysis")
        st.sidebar.header("Configuration")
        st.sidebar.info("**Standard:** Chin must go above the bar (head above shoulders) for a valid rep.")
        st.sidebar.info("Side or rear view recommended for best results.")
        
        @st.cache_resource
        def load_model():
            return YOLO("yolov8n-pose.pt")
        
        uploaded_file = st.file_uploader("Upload video (Side view recommended)", type=["mp4", "mov", "avi"], key="pullup_video")
        
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()
            video_path = tmp.name

            if st.button("Analyze Form", key="pullup_analyze"):
                model = load_model()
                cap = cv2.VideoCapture(video_path)

                # Setup for output video visualization
                video_col, metrics_col = st.columns([2, 1])
                with video_col:
                    stframe = st.empty()
                with metrics_col:
                    good_metric = st.empty()
                    total_metric = st.empty()

                good_count = 0
                total_reps = 0
                state = "down"  # Start in down position (dead hang)
                
                # Track range for calibration
                min_head_y = None  # Highest position (lowest Y value)
                max_head_y = None  # Lowest position (highest Y value)
                
                # Track if chin went above bar during current rep
                chin_above_bar = False

                frame_idx = 0
                FRAME_STRIDE = 3

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % FRAME_STRIDE != 0:
                        continue

                    # Inference
                    results = model(frame, verbose=False)
                    head_y, shoulder_y, kpts = get_pose_details_pullup(results)

                    if head_y is None or shoulder_y is None or kpts is None:
                        continue

                    # Auto-calibration of range
                    if min_head_y is None:
                        min_head_y = head_y
                        max_head_y = head_y
                    
                    min_head_y = min(min_head_y, head_y)
                    max_head_y = max(max_head_y, head_y)

                    # Check if chin is above bar (head above shoulders)
                    head_above_shoulder = head_y < shoulder_y  # Lower Y = higher in image

                    # Define UP/DOWN thresholds based on head position
                    head_range = max_head_y - min_head_y
                    if head_range > 0:
                        # Top position: head significantly above shoulders
                        top_thresh = min_head_y + 0.3 * head_range
                        # Bottom position: head at or below shoulders
                        bottom_thresh = min_head_y + 0.7 * head_range
                    else:
                        top_thresh = min_head_y
                        bottom_thresh = max_head_y

                    # State Machine
                    if state == "down" and head_y < top_thresh:  # Head goes up (lower Y)
                        state = "up"
                        chin_above_bar = False  # Reset for new rep

                    if state == "up":
                        # Check if chin goes above bar during this rep
                        if head_above_shoulder:
                            chin_above_bar = True

                        # Check for completion (back down)
                        if head_y > bottom_thresh:
                            state = "down"
                            total_reps += 1
                            
                            # Count as good if chin went above bar
                            if chin_above_bar:
                                good_count += 1

                    # Draw Debug Lines on Frame
                    debug_frame = draw_debug_overlay_pullup(frame.copy(), kpts, head_y, shoulder_y)

                    # Display
                    stframe.image(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    good_metric.markdown(f"### ✅ Valid Reps: {good_count}")
                    total_metric.markdown(f"### 📊 Total Reps: {total_reps}")

                cap.release()

                st.divider()
                st.write(f"**Analysis Complete.**")
                st.write(f"**Standard:** Chin above bar (head above shoulders)")
                st.write(f"- ✅ Valid reps (chin above bar): {good_count}")
                st.write(f"- 📊 Total reps detected: {total_reps}")
                
                if total_reps > 0:
                    invalid_count = total_reps - good_count
                    if invalid_count == 0:
                        st.balloons()
                        st.success(f"Perfect! All {good_count} reps had chin above bar.")
                    else:
                        st.warning(f"{invalid_count} rep(s) did not reach chin above bar.")
                else:
                    st.info("No reps detected. Make sure the video shows clear pull-up movements.")

    elif st.session_state.exercise == "pushup":
        st.subheader("🏃 Push-up Analysis")
        st.sidebar.header("Configuration")
        flare_threshold = st.sidebar.slider(
            "Max Elbow Angle (Degrees)",
            min_value=45,
            max_value=90,
            value=75,
            help="Higher = more lenient. Lower = stricter form. >75 usually implies flaring."
        )
        st.sidebar.info("Tip: Adjust this slider until 'Good' reps are green and 'Bad' reps are red.")

        @st.cache_resource
        def load_model():
            return YOLO("yolov8n-pose.pt")

        uploaded_file = st.file_uploader("Upload video (Front view best)", type=["mp4", "mov", "avi"], key="pushup_video")

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()
            video_path = tmp.name

            if st.button("Analyze Form", key="pushup_analyze"):
                model = load_model()
                cap = cv2.VideoCapture(video_path)

                # Setup for output video visualization
                video_col, metrics_col = st.columns([2, 1])
                with video_col:
                    stframe = st.empty()
                with metrics_col:
                    good_metric = st.empty()
                    bad_metric = st.empty()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

                good_count = 0
                bad_count = 0
                state = "up"
                min_y, max_y = None, None
                current_rep_angles = []

                frame_idx = 0
                FRAME_STRIDE = 3

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % FRAME_STRIDE != 0:
                        continue

                    # Inference
                    results = model(frame, verbose=False)
                    current_y, kpts = get_pose_details_pushup(results)

                    if current_y is None:
                        continue

                    # Auto-calibration of range
                    if min_y is None: min_y = current_y
                    if max_y is None: max_y = current_y
                    min_y = min(min_y, current_y)
                    max_y = max(max_y, current_y)

                    # Define UP/DOWN thresholds
                    range_span = max_y - min_y
                    down_thresh = min_y + 0.6 * range_span
                    up_thresh = min_y + 0.3 * range_span

                    # Calculate Angle
                    left_angle = 0
                    right_angle = 0
                    if kpts['l_sh'] is not None and kpts['l_hip'] is not None and kpts['l_el'] is not None:
                        left_angle = calculate_angle(kpts['l_hip'], kpts['l_sh'], kpts['l_el'])
                    if kpts['r_sh'] is not None and kpts['r_hip'] is not None and kpts['r_el'] is not None:
                        right_angle = calculate_angle(kpts['r_hip'], kpts['r_sh'], kpts['r_el'])

                    # Use max angle found to be safe, or average
                    valid_angles = [a for a in [left_angle, right_angle] if a > 10]  # filter noise
                    current_flare = np.mean(valid_angles) if valid_angles else 0

                    # State Machine
                    if state == "up" and current_y > down_thresh:
                        state = "down"
                        current_rep_angles = []

                    if state == "down":
                        if current_flare > 0:
                            current_rep_angles.append(current_flare)

                        # Check for completion
                        if current_y < up_thresh:
                            state = "up"
                            if current_rep_angles:
                                rep_max_flare = np.median(current_rep_angles)

                                # COMPARE AGAINST SLIDER THRESHOLD
                                if rep_max_flare > flare_threshold:
                                    bad_count += 1
                                else:
                                    good_count += 1

                    # Draw Debug Lines on Frame
                    debug_frame = draw_debug_overlay_pushup(frame.copy(), kpts, current_flare, flare_threshold)

                    # Display
                    stframe.image(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    good_metric.markdown(f"### Good: {good_count}")
                    bad_metric.markdown(f"### Bad: {bad_count}")

                cap.release()

                st.divider()
                st.write(f"**Analysis Complete.** Threshold used: {flare_threshold}°")
                if bad_count == 0 and good_count > 0:
                    st.balloons()
                    st.success("Perfect form! No flaring detected.")
                elif bad_count > 0:
                    st.error(f"Detected {bad_count} reps with flared elbows (> {flare_threshold}°).")


# ---------- Streamlit App ----------

st.set_page_config(page_title="TrainR - AI Fitness Companion", page_icon="💪", layout="wide")
inject_base_styles()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "exercise" not in st.session_state:
    st.session_state.exercise = None

# Route to appropriate page
if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "workout_schedule":
    show_workout_schedule()
elif st.session_state.page == "exercise":
    show_exercise_analysis()
else:
    show_landing_page()
