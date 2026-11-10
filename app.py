from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from detection.vehicle_detector import VehicleDetector
from detection.emergency_detector import EmergencyDetector
from traffic.adaptive_controller import AdaptiveSignalController
from traffic.models import LANE_NAMES
from simulation.junction_draw import draw_junction
from utils.event_log import add_event, events_to_frame

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "mpeg", "mpg"]
MAX_WIDTH = 640                 # CPU-friendly inference width for an 8 GB laptop
INFERENCE_INTERVAL = 1.0       # each camera analysed about once per second
FRAGMENT_INTERVAL = "250ms"    # UI refresh interval

st.set_page_config(page_title="Intelligent Traffic Signal Control", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stMetric"] {background:rgba(80,100,120,.08); padding:8px; border-radius:9px;}
.camera-title {font-weight:700; letter-spacing:.04em; padding:.25rem 0;}
</style>
""", unsafe_allow_html=True)

# Persistent application state. VideoCapture handles and detector objects stay in session state
# across Streamlit fragment refreshes.
def init_state():
    defaults = {
        "controller": AdaptiveSignalController(),
        "events": [],
        "detector": None,
        "camera_detectors": [None] * 4,
        "captures": [None] * 4,
        "temp_paths": [None] * 4,
        "camera_frames": [None] * 4,
        "camera_counts": [0] * 4,
        "camera_classes": [{"car": 0, "motorcycle": 0, "bus": 0, "truck": 0} for _ in range(4)],
        "camera_status": ["No video loaded"] * 4,
        "camera_last_infer": [0.0] * 4,
        "camera_turn": 0,
        "running": False,
        "paused": False,
        "last_tick": None,
        "last_controller_tick": None,
        "model_path": "yolov8n.pt",
        "uploaded_signatures": [None] * 4,
        "camera_fps": [25.0] * 4,
        "camera_frame_totals": [0] * 4,
        "playback_started_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()
controller: AdaptiveSignalController = st.session_state.controller


def close_video_resources():
    for cap in st.session_state.captures:
        if cap is not None:
            cap.release()
    for path in st.session_state.temp_paths:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
    st.session_state.captures = [None] * 4
    st.session_state.temp_paths = [None] * 4


def resize_for_inference(frame):
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        new_h = max(1, int(h * MAX_WIDTH / w))
        frame = cv2.resize(frame, (MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return frame


def density_label(count: int) -> str:
    if count >= 20:
        return "HIGH"
    if count >= 8:
        return "MEDIUM"
    return "LOW"


def prepare_videos(uploads):
    close_video_resources()
    for idx, upload in enumerate(uploads):
        if upload is None:
            st.session_state.camera_status[idx] = "No video loaded"
            st.session_state.camera_frames[idx] = None
            continue
        suffix = Path(upload.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getvalue())
            path = tmp.name
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            Path(path).unlink(missing_ok=True)
            raise RuntimeError(f"Could not open {upload.name}. Try MP4 (H.264) format.")
        st.session_state.captures[idx] = cap
        st.session_state.temp_paths[idx] = path
        st.session_state.camera_status[idx] = f"Ready · {upload.name}"
        st.session_state.camera_frames[idx] = None
        st.session_state.camera_counts[idx] = 0
        st.session_state.camera_classes[idx] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        st.session_state.camera_last_infer[idx] = 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.session_state.camera_fps[idx] = fps if fps and 1 <= fps <= 120 else 25.0
        st.session_state.camera_frame_totals[idx] = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


def stop_simulation(reset_controller=False):
    st.session_state.running = False
    st.session_state.paused = False
    close_video_resources()
    if reset_controller:
        controller.reset()
    st.session_state.last_tick = None
    st.session_state.last_controller_tick = None


st.title("🚦 Intelligent Traffic Signal Control")
st.caption("STAGE 1 · Four-camera video analytics · Adaptive signal timing · Emergency priority")
st.info("Upload one clip per approach. Press Start Simulation to play all feeds together. For an 8 GB laptop, inference is resized and processed round-robin at a modest rate to reduce CPU and memory load.")

with st.sidebar:
    st.header("System setup")
    model_path = st.text_input("Vehicle YOLO model", value=st.session_state.model_path)
    if st.button("Load vehicle model", use_container_width=True):
        try:
            with st.spinner("Loading YOLO model (first run may download weights)..."):
                # One independent model/tracker per camera avoids ByteTrack state leaking between feeds.
                st.session_state.camera_detectors = [VehicleDetector(model_path) for _ in range(4)]
                st.session_state.detector = st.session_state.camera_detectors[0]
                st.session_state.model_path = model_path
            add_event(st.session_state.events, "SYSTEM", f"Loaded four independent camera trackers: {model_path}")
            st.success("Vehicle model ready for all four cameras.")
        except Exception as exc:
            st.session_state.camera_detectors = [None] * 4
            st.session_state.detector = None
            st.error(f"Model load failed: {exc}")
    st.write("**CV model:**", "READY" if st.session_state.detector else "NOT LOADED")
    st.divider()
    st.subheader("Signal timing")
    min_green = st.slider("Minimum green (seconds)", 5, 20, 10)
    max_green = st.slider("Maximum green (seconds)", 20, 60, 50)
    amber_time = st.slider("Amber (seconds)", 2, 6, 3)
    all_red_time = st.slider("All-red clearance (seconds)", 1, 5, 2)
    fairness_weight = st.slider("Waiting-time fairness", 0.0, 2.0, 0.7, 0.1)
    controller.configure(min_green, max(max_green, min_green), amber_time, all_red_time, fairness_weight)
    st.caption("Educational simulation timings only; not certified for real roads.")

st.subheader("1. Four-approach CCTV surveillance wall")
cols = st.columns(2, gap="medium")
uploads = []
for idx in range(4):
    with cols[idx % 2]:
        st.markdown(f"<div class='camera-title'>● CAM 0{idx+1} · {LANE_NAMES[idx].upper()}</div>", unsafe_allow_html=True)
        upload = st.file_uploader(f"Upload {LANE_NAMES[idx]} video", type=VIDEO_TYPES, key=f"lane_video_{idx}")
        uploads.append(upload)

# Detect changed uploads and invalidate an old running session safely.
signatures = [((u.name, u.size) if u is not None else None) for u in uploads]
if signatures != st.session_state.uploaded_signatures:
    if st.session_state.running:
        stop_simulation()
    st.session_state.uploaded_signatures = signatures

control_cols = st.columns([1.2, 1, 1, 1])
with control_cols[0]:
    start_clicked = st.button("▶ START SIMULATION", type="primary", use_container_width=True, disabled=st.session_state.running)
with control_cols[1]:
    pause_clicked = st.button("⏸ PAUSE", use_container_width=True, disabled=not st.session_state.running or st.session_state.paused)
with control_cols[2]:
    resume_clicked = st.button("▶ RESUME", use_container_width=True, disabled=not st.session_state.running or not st.session_state.paused)
with control_cols[3]:
    reset_clicked = st.button("↻ RESET", use_container_width=True)

if start_clicked:
    if st.session_state.detector is None:
        st.error("Load the vehicle YOLO model in the sidebar first.")
    elif not all(uploads):
        st.error("Please upload all four camera videos before starting.")
    else:
        try:
            prepare_videos(uploads)
            st.session_state.running = True
            st.session_state.paused = False
            st.session_state.playback_started_at = time.monotonic()
            st.session_state.last_tick = time.monotonic()
            st.session_state.last_controller_tick = time.monotonic()
            add_event(st.session_state.events, "SYSTEM", "Four-camera simulation started")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not prepare videos: {exc}")

if pause_clicked:
    st.session_state.paused = True
    add_event(st.session_state.events, "SYSTEM", "Simulation paused")
if resume_clicked:
    st.session_state.paused = False
    st.session_state.playback_started_at = time.monotonic()
    st.session_state.last_tick = time.monotonic()
    st.session_state.last_controller_tick = time.monotonic()
    add_event(st.session_state.events, "SYSTEM", "Simulation resumed")
if reset_clicked:
    stop_simulation(reset_controller=True)
    st.session_state.camera_frames = [None] * 4
    st.session_state.camera_counts = [0] * 4
    st.session_state.camera_status = ["Reset · upload/start again"] * 4
    add_event(st.session_state.events, "SYSTEM", "Simulation reset")
    st.rerun()

# Fragment refreshes the video wall without rerunning upload widgets or the whole page.
@st.fragment(run_every=FRAGMENT_INTERVAL if st.session_state.running and not st.session_state.paused else None)
def live_simulation_panel():
    if st.session_state.running and not st.session_state.paused:
        now = time.monotonic()
        last = st.session_state.last_controller_tick or now
        elapsed = int(now - last)
        if elapsed > 0:
            controller.advance(elapsed)
            st.session_state.last_controller_tick = last + elapsed

        # Seek all feeds to the same elapsed playback timestamp. This keeps
        # the four clips synchronized and advances at their recorded frame rate
        # rather than playing only one decoded frame per UI refresh.
        started = st.session_state.playback_started_at or now
        playback_elapsed = max(0.0, now - started)
        for idx, cap in enumerate(st.session_state.captures):
            if cap is None:
                continue
            fps = st.session_state.camera_fps[idx]
            total = st.session_state.camera_frame_totals[idx]
            target_frame = int(playback_elapsed * fps)
            if total > 0:
                target_frame %= total  # loop each clip at its own duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                st.session_state.camera_status[idx] = "Playback ended / unreadable"
                continue
            frame = resize_for_inference(frame)
            st.session_state.camera_frames[idx] = frame
            st.session_state.camera_status[idx] = "PLAYING · synchronized playback"

        # Round-robin inference: one camera per refresh (about one inference per
        # camera per second at this refresh rate), keeping CPU use manageable.
        camera_idx = st.session_state.camera_turn % 4
        st.session_state.camera_turn += 1
        frame_for_cv = st.session_state.camera_frames[camera_idx]
        detector = st.session_state.camera_detectors[camera_idx]
        if frame_for_cv is not None and detector is not None:
            try:
                count, annotated, classes = detector.detect_frame_with_stats(frame_for_cv)
                st.session_state.camera_counts[camera_idx] = count
                st.session_state.camera_classes[camera_idx] = classes
                controller.set_lane_count(camera_idx, count)
                st.session_state.camera_frames[camera_idx] = annotated
            except Exception as exc:
                st.session_state.camera_status[camera_idx] = f"Detection error: {exc}"

    phase_col, lane_col, time_col, status_col = st.columns(4)
    phase_col.metric("Signal phase", controller.phase.replace("_", " ").title())
    lane_col.metric("Active approach", LANE_NAMES[controller.current_lane])
    time_col.metric("Phase remaining", f"{controller.phase_remaining}s")
    status_col.metric("Simulation", "PAUSED" if st.session_state.paused else ("RUNNING" if st.session_state.running else "STOPPED"))

    video_cols = st.columns(2, gap="medium")
    for idx, col in enumerate(video_cols):
        pass
    for idx in range(4):
        with video_cols[idx % 2]:
            st.markdown(f"**CAM 0{idx+1} · {LANE_NAMES[idx]}**")
            frame = st.session_state.camera_frames[idx]
            if frame is not None:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            else:
                st.info(st.session_state.camera_status[idx])
            m1, m2 = st.columns(2)
            m1.metric("Vehicles in frame", st.session_state.camera_counts[idx])
            m2.metric("Density", density_label(st.session_state.camera_counts[idx]))
            st.caption(st.session_state.camera_status[idx])

live_simulation_panel()

st.divider()
st.subheader("2. Virtual intersection & adaptive signal control")
map_col, demand_col = st.columns([1.1, 1])
with map_col:
    junction = draw_junction(controller.current_lane, controller.phase, controller.emergency_lane, controller.phase_remaining)
    st.image(junction, use_container_width=True)
    st.caption("Simplified conflict-group model: only one approach receives green at a time.")
with demand_col:
    st.markdown("**Live detected lane demand**")
    for idx, lane in enumerate(controller.lanes):
        c1, c2 = st.columns([1, 2])
        c1.metric(lane.name, lane.vehicle_count)
        c2.progress(min(lane.vehicle_count / 40.0, 1.0), text=f"{lane.density_label} · wait {lane.wait_seconds}s")
    st.caption(f"Selected green duration for active lane: {controller.next_green_duration()} seconds")
    st.caption("Signals advance automatically while the simulation is running.")

st.divider()
st.subheader("3. Emergency vehicle priority")
em1, em2 = st.columns([1, 1])
with em1:
    ambulance_file = st.file_uploader("Optional ambulance / emergency clip", type=VIDEO_TYPES, key="ambulance_upload")
    emergency_lane_name = st.selectbox("Emergency vehicle approach", LANE_NAMES, index=2)
    if ambulance_file is not None:
        st.video(ambulance_file)
with em2:
    custom_model_path = st.text_input("Optional custom ambulance YOLO weights", value=str(MODEL_DIR / "emergency_vehicle.pt"))
    emergency_class_name = st.text_input("Emergency class label", value="ambulance")
    if st.button("Analyze emergency clip with custom AI", use_container_width=True):
        if ambulance_file is None:
            st.warning("Upload an emergency clip first.")
        else:
            temp_path = None
            try:
                emergency_detector = EmergencyDetector(custom_model_path, emergency_class_name)
                suffix = Path(ambulance_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(ambulance_file.getvalue())
                    temp_path = tmp.name
                with st.spinner("Checking emergency clip..."):
                    found, frames_checked, confidence = emergency_detector.analyze_video(temp_path)
                if found:
                    controller.request_emergency(LANE_NAMES.index(emergency_lane_name))
                    add_event(st.session_state.events, "EMERGENCY_AI", f"Custom model detected {emergency_class_name} for {emergency_lane_name}")
                    st.success(f"Custom model confirmed {emergency_class_name} in {frames_checked} frames (confidence {confidence:.2f}).")
                else:
                    st.warning(f"No confirmed {emergency_class_name} found in {frames_checked} frames.")
            except Exception as exc:
                st.error(f"Custom emergency AI unavailable: {exc}")
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
    if st.button("🚨 Trigger emergency priority (demo input)", type="primary", use_container_width=True):
        controller.request_emergency(LANE_NAMES.index(emergency_lane_name))
        add_event(st.session_state.events, "EMERGENCY_DEMO", f"Manual demo request for {emergency_lane_name}; not automatic ambulance recognition")
        st.warning("Manual demo request activated; this is not automatic ambulance recognition.")
    if st.button("✓ Mark emergency vehicle cleared", use_container_width=True):
        controller.clear_emergency()
        add_event(st.session_state.events, "EMERGENCY", "Emergency vehicle cleared")
        st.success("Emergency priority released; the controller will complete its clearance phase automatically.")

if controller.emergency_lane is not None:
    st.error(f"Emergency priority requested / active — {LANE_NAMES[controller.emergency_lane]}")
else:
    st.success("Normal adaptive traffic control mode")

st.divider()
st.subheader("Analytics summary")
summary = []
for idx, lane in enumerate(controller.lanes):
    classes = st.session_state.camera_classes[idx]
    summary.append({"Approach": lane.name, "Vehicles in latest frame": lane.vehicle_count,
                    "Density": lane.density_label, "Vehicle classes": ", ".join(f"{k}: {v}" for k, v in classes.items())})
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

st.subheader("Event history")
event_frame = events_to_frame(st.session_state.events)
if not event_frame.empty:
    st.dataframe(event_frame.tail(30), use_container_width=True, hide_index=True)
    st.download_button("Download event log (CSV)", event_frame.to_csv(index=False).encode("utf-8"), "traffic_events.csv", "text/csv")
else:
    st.caption("No events recorded yet.")

st.caption("Academic software simulation only. This application does not control real traffic signals. Four uploaded clips are simulated camera feeds; use synchronized footage if presenting them as views of one real intersection.")
