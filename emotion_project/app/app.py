"""Streamlit demo — live multimodal emotion recognition powered by
pretrained foundation models.

Face  → dima806/facial_emotions_image_detection (ViT, 7 emotions → 5)
Audio → ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (8 → 5)

Run:
    streamlit run app/app.py
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ai.foundation_models import (
    audio_predict,
    face_predict,
    load_audio_bundle,
    load_face_bundle,
)
from src.utils.helpers import select_device
from src.utils.labels import EMOTIONS

EMOJI = {"Happy": "😊", "Sad": "😢", "Angry": "😠", "Fearful": "😨", "Neutral": "😐"}
COLOR = {
    "Happy": "#22c55e",
    "Sad": "#3b82f6",
    "Angry": "#ef4444",
    "Fearful": "#a855f7",
    "Neutral": "#64748b",
}

AUDIO_SAMPLE_RATE = 16000
AUDIO_CLIP_SECONDS = 3.0


# --------------------------------------------------------------------------- #
# Caching                                                                     #
# --------------------------------------------------------------------------- #
@st.cache_resource
def _get_bundles():
    device = select_device("auto")
    face_bundle = load_face_bundle(device)
    audio_bundle = load_audio_bundle(device)
    return device, face_bundle, audio_bundle


@st.cache_resource
def _get_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _detect_face_box(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_face_detector().detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def _crop_with_margin(frame_bgr, box, margin=0.25) -> np.ndarray | None:
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    mx, my = int(margin * w), int(margin * h)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _ema(prev: np.ndarray | None, new: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) + 1e-9)
    return 20.0 * float(np.log10(rms))


def _audio_recorder_loop(sr, seconds, out_q, stop_event):
    try:
        import sounddevice as sd
    except ImportError:
        return
    n_samples = int(sr * seconds)
    while not stop_event.is_set():
        try:
            rec = sd.rec(n_samples, samplerate=sr, channels=1, dtype="float32")
            sd.wait()
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
            continue
        if stop_event.is_set():
            break
        while not out_q.empty():
            try:
                out_q.get_nowait()
            except queue.Empty:
                break
        out_q.put(rec.flatten())


def _emotion_card(slot, label: str, confidence: float):
    color = COLOR.get(label, "#475569")
    emoji = EMOJI.get(label, "❔")
    slot.markdown(
        f"""
        <div style="background:{color}22;border:2px solid {color};
                    border-radius:16px;padding:24px;text-align:center;">
          <div style="font-size:72px;line-height:1;">{emoji}</div>
          <div style="font-size:32px;font-weight:700;color:{color};margin-top:8px;">
            {label.upper()}
          </div>
          <div style="font-size:16px;color:#64748b;margin-top:4px;">
            confidence {confidence * 100:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _probs_df(probs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"emotion": EMOTIONS, "probability": probs}).set_index("emotion")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Multimodal Emotion Recognition (AI)", layout="wide")
    st.markdown(
        "<h1 style='margin-bottom:0;'>🤖 Multimodal Emotion Recognition</h1>"
        "<p style='color:#64748b;margin-top:4px;'>"
        "Foundation models: ViT (face) + Wav2Vec2 (audio) → 5 emotions"
        "</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading pretrained models (first run downloads ~1.5GB)..."):
        device, face_bundle, audio_bundle = _get_bundles()

    st.sidebar.header("⚙️ Settings")
    smoothing = st.sidebar.slider("Prediction smoothing", 0.05, 1.0, 0.30, 0.05)
    silence_dbfs = st.sidebar.slider("Silence threshold (dBFS)", -60.0, -10.0, -40.0, 1.0)
    conf_threshold = st.sidebar.slider("Min confidence to show prediction", 0.20, 0.90, 0.40, 0.05)
    face_weight = st.sidebar.slider("Face weight in fusion", 0.0, 1.0, 0.40, 0.05)
    mirror = st.sidebar.checkbox("Mirror video (selfie)", value=True)
    fps_target = st.sidebar.slider("Target video FPS", 5, 30, 12)
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Face**: `{face_bundle.id2label}`\n\n**Audio**: `{audio_bundle.id2label}`",
    )
    st.sidebar.caption(f"Device: `{device}`")

    if "running" not in st.session_state:
        st.session_state.running = False

    c1, c2 = st.columns(2)
    if c1.button("▶ Start live", use_container_width=True, disabled=st.session_state.running):
        st.session_state.running = True
    if c2.button("⏹ Stop", use_container_width=True, disabled=not st.session_state.running):
        st.session_state.running = False

    top_left, top_right = st.columns([1.3, 1])
    video_slot = top_left.empty()
    audio_meter_slot = top_left.empty()
    top_right.subheader("Current emotion")
    main_card = top_right.empty()
    sub_metrics = top_right.empty()
    status = top_right.empty()

    st.divider()
    detail_l, detail_r = st.columns(2)
    detail_l.subheader("Face probabilities (ViT)")
    face_chart = detail_l.empty()
    detail_l.subheader("Audio probabilities (Wav2Vec2)")
    audio_chart = detail_l.empty()
    detail_r.subheader("Fusion (weighted average)")
    fusion_chart = detail_r.empty()

    if not st.session_state.running:
        main_card.info("Press **▶ Start live** to begin streaming.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.running = False
        st.error("Could not open webcam.")
        return

    audio_q: queue.Queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    recorder = threading.Thread(
        target=_audio_recorder_loop,
        args=(AUDIO_SAMPLE_RATE, AUDIO_CLIP_SECONDS, audio_q, stop_event),
        daemon=True,
    )
    recorder.start()

    smoothed_face = None
    smoothed_audio = None
    last_audio_dbfs = -60.0
    audio_silent = True
    frame_period = 1.0 / max(fps_target, 1)

    try:
        while st.session_state.running:
            tick_start = time.time()

            ok, frame = cap.read()
            if not ok:
                status.warning("Webcam read failed.")
                time.sleep(0.1)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            # --------- Face inference ---------
            box = _detect_face_box(frame)
            display = frame.copy()
            if box is not None:
                crop_bgr = _crop_with_margin(frame, box, margin=0.25)
                if crop_bgr is not None:
                    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    raw = face_predict(face_bundle, rgb)
                    smoothed_face = _ema(smoothed_face, raw, smoothing)
                    top_i = int(np.argmax(smoothed_face))
                    top_lbl = EMOTIONS[top_i]
                    cx = COLOR[top_lbl]
                    bgr = (int(cx[5:7], 16), int(cx[3:5], 16), int(cx[1:3], 16))
                    x, y, w, h = box
                    cv2.rectangle(display, (x, y), (x + w, y + h), bgr, 2)
                    cv2.putText(
                        display,
                        f"{top_lbl} {smoothed_face[top_i]:.2f}",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2,
                    )
            video_slot.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")

            # --------- Audio inference ---------
            try:
                samples = audio_q.get_nowait()
            except queue.Empty:
                samples = None

            if samples is not None:
                last_audio_dbfs = _rms_dbfs(samples)
                audio_silent = last_audio_dbfs < silence_dbfs

            if samples is not None and not audio_silent:
                raw_a = audio_predict(audio_bundle, samples, source_sr=AUDIO_SAMPLE_RATE)
                smoothed_audio = _ema(smoothed_audio, raw_a, smoothing)

            # --------- Audio meter ---------
            meter_norm = max(0.0, min(1.0, (last_audio_dbfs + 60.0) / 60.0))
            meter_color = "#94a3b8" if audio_silent else "#22c55e"
            audio_meter_slot.markdown(
                f"""
                <div style="margin-top:8px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:13px;color:#64748b;">
                    <span>🎤 Mic level</span>
                    <span>{last_audio_dbfs:.1f} dBFS {'· silent (ignored)' if audio_silent else ''}</span>
                  </div>
                  <div style="background:#e2e8f0;border-radius:6px;height:10px;
                              overflow:hidden;margin-top:4px;">
                    <div style="background:{meter_color};width:{meter_norm * 100:.1f}%;height:100%;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------- Fusion (weighted avg) ---------
            fusion_probs = None
            if smoothed_face is not None and smoothed_audio is not None:
                fusion_probs = face_weight * smoothed_face + (1.0 - face_weight) * smoothed_audio
            elif smoothed_face is not None:
                fusion_probs = smoothed_face
            elif smoothed_audio is not None:
                fusion_probs = smoothed_audio

            # --------- Main card ---------
            if fusion_probs is not None:
                top_i = int(np.argmax(fusion_probs))
                conf = float(fusion_probs[top_i])
                if conf >= conf_threshold:
                    _emotion_card(main_card, EMOTIONS[top_i], conf)
                else:
                    main_card.warning("Not confident — speak / face the camera.")
                bits = []
                if smoothed_face is not None:
                    fi = int(np.argmax(smoothed_face))
                    bits.append(
                        f"<div style='flex:1;text-align:center;'>"
                        f"<div style='font-size:12px;color:#64748b;'>FACE</div>"
                        f"<div style='font-size:18px;'>{EMOJI[EMOTIONS[fi]]} {EMOTIONS[fi]}</div>"
                        f"<div style='font-size:13px;color:#64748b;'>{smoothed_face[fi]*100:.0f}%</div>"
                        f"</div>"
                    )
                if smoothed_audio is not None:
                    ai = int(np.argmax(smoothed_audio))
                    bits.append(
                        f"<div style='flex:1;text-align:center;'>"
                        f"<div style='font-size:12px;color:#64748b;'>AUDIO</div>"
                        f"<div style='font-size:18px;'>{EMOJI[EMOTIONS[ai]]} {EMOTIONS[ai]}</div>"
                        f"<div style='font-size:13px;color:#64748b;'>{smoothed_audio[ai]*100:.0f}%</div>"
                        f"</div>"
                    )
                sub_metrics.markdown(
                    f"<div style='display:flex;gap:12px;margin-top:12px;'>{''.join(bits)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                main_card.info("Detecting face / waiting for audio…")

            # --------- Charts ---------
            if smoothed_face is not None:
                face_chart.bar_chart(_probs_df(smoothed_face), height=200)
            if smoothed_audio is not None:
                audio_chart.bar_chart(_probs_df(smoothed_audio), height=200)
            elif audio_silent:
                audio_chart.info("Audio silent — speak louder or lower the silence threshold.")
            if fusion_probs is not None:
                fusion_chart.bar_chart(_probs_df(fusion_probs), height=200)

            elapsed = time.time() - tick_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    finally:
        stop_event.set()
        cap.release()
        recorder.join(timeout=1.0)


if __name__ == "__main__":
    main()
