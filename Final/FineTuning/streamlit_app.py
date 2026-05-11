"""Professional Streamlit dashboard for pyannote diarization inference.

Reads configuration from environment variables set by the launcher cell:
  HF_TOKEN, STREAMLIT_PROC_DIR, STREAMLIT_DATA_DIR, STREAMLIT_OUT_DIR,
  STREAMLIT_FT_WEIGHTS, STREAMLIT_FILES (comma-separated VoxConverse IDs).

This version improves the UI and converts raw VoxConverse / pyannote speaker IDs
(spk13, spk07, SPEAKER_00, etc.) into readable labels such as Speaker 1,
Speaker 2, ... . Hypothesis speakers are mapped to reference speaker numbering
by maximum time-overlap when a reference RTTM is available.
"""
import os
import re
import string
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
from pyannote.audio import Model, Pipeline
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
PROC_DIR = Path(os.environ.get("STREAMLIT_PROC_DIR", "."))
DATA_DIR = Path(os.environ.get("STREAMLIT_DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("STREAMLIT_OUT_DIR", "."))
FT_WEIGHTS = Path(os.environ.get("STREAMLIT_FT_WEIGHTS", str(OUT_DIR / "segmentation_finetuned.pt")))
FILES = [f.strip() for f in os.environ.get("STREAMLIT_FILES", "").split(",") if f.strip()]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UI_UPLOAD_DIR = OUT_DIR / "streamlit_uploads"
UI_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Page setup and CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Diarization Inference Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --card-bg: rgba(255,255,255,0.055);
        --card-border: rgba(255,255,255,0.10);
        --muted: rgba(255,255,255,0.62);
        --accent: #ff4b4b;
    }
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
    section[data-testid="stSidebar"] {background: #151821; border-right: 1px solid rgba(255,255,255,0.08);}
    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--card-border);
        background: linear-gradient(135deg, rgba(255,75,75,0.16), rgba(58,134,255,0.10));
        border-radius: 24px;
        margin-bottom: 1.15rem;
    }
    .hero-title {font-size: 2.35rem; font-weight: 850; letter-spacing: -0.04em; margin: 0;}
    .hero-subtitle {font-size: 1rem; color: var(--muted); margin-top: 0.35rem;}
    .status-row {display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1rem;}
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.38rem 0.65rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(255,255,255,0.86);
        font-size: 0.82rem;
        font-weight: 600;
    }
    .section-title {font-size: 1.22rem; font-weight: 780; margin: 1.25rem 0 0.55rem 0; letter-spacing: -0.02em;}
    .metric-card {
        padding: 1rem 1rem;
        border-radius: 18px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        min-height: 112px;
    }
    .metric-label {font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.32rem;}
    .metric-value {font-size: 1.72rem; font-weight: 780; line-height: 1.1;}
    .metric-note {font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem;}
    .soft-card {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        margin-bottom: 1rem;
    }
    div[data-testid="stDataFrame"] {border-radius: 16px; overflow: hidden;}
    .stButton > button {border-radius: 14px; height: 3rem; font-weight: 720;}
    .stDownloadButton > button {border-radius: 14px; font-weight: 650;}
    hr {border-color: rgba(255,255,255,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

if not HF_TOKEN:
    st.error("HF_TOKEN is not set. Run the notebook launcher cell or export HF_TOKEN before starting Streamlit.")
    st.stop()


# -------------------------------------------------------------------------
# Cached pipeline loaders
# -------------------------------------------------------------------------
def _pipeline_from_pretrained(model_id):
    try:
        return Pipeline.from_pretrained(model_id, token=HF_TOKEN)
    except TypeError as exc:
        if "unexpected keyword argument 'token'" in str(exc):
            return Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)
        raise


@st.cache_resource(show_spinner="Loading baseline pyannote pipeline…")
def load_baseline():
    pipe = _pipeline_from_pretrained("pyannote/speaker-diarization-3.1")
    return pipe.to(torch.device(DEVICE))


@st.cache_resource(show_spinner="Loading fine-tuned pyannote pipeline…")
def load_finetuned():
    if not FT_WEIGHTS.exists():
        return None
    try:
        model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
    except TypeError:
        model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)

    state = torch.load(str(FT_WEIGHTS), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        raw = state["state_dict"]
        prefixed = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
        state = prefixed or raw
    model.load_state_dict(state, strict=False)
    model = model.to(torch.device(DEVICE)).eval()

    pipe = _pipeline_from_pretrained("pyannote/speaker-diarization-3.1")
    if hasattr(pipe, "_segmentation") and hasattr(pipe._segmentation, "model"):
        pipe._segmentation.model = model
    elif hasattr(pipe, "segmentation_model"):
        pipe.segmentation_model = model
    return pipe.to(torch.device(DEVICE))


# -------------------------------------------------------------------------
# Diarization helpers
# -------------------------------------------------------------------------
def ann_to_segments(annotation, duration):
    rows = []
    for seg, _, spk in annotation.itertracks(yield_label=True):
        start = max(0.0, float(seg.start))
        end = min(float(duration), float(seg.end))
        if end > start:
            rows.append({"start": start, "end": end, "speaker": str(spk)})
    return sorted(rows, key=lambda r: (r["start"], r["end"], r["speaker"]))


def load_ref(rttm_path):
    ref = Annotation(uri=Path(rttm_path).stem)
    with open(rttm_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if parts and parts[0] == "SPEAKER":
                start = float(parts[3])
                end = start + float(parts[4])
                ref[Segment(start, end), f"track_{i}"] = parts[7]
    return ref


def build_reference_speaker_map(reference_segments):
    """Map raw reference speakers to Speaker 1, Speaker 2, ... by first appearance."""
    mapping = {}
    ordered = sorted(reference_segments, key=lambda r: (r["start"], r["end"], r["speaker"]))
    for row in ordered:
        raw = row["speaker"]
        if raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
    return mapping


def build_first_appearance_map(segments, start_index=1):
    """Map any speaker set to Speaker N labels by first appearance."""
    mapping = {}
    ordered = sorted(segments, key=lambda r: (r["start"], r["end"], r["speaker"]))
    for row in ordered:
        raw = row["speaker"]
        if raw not in mapping:
            mapping[raw] = f"Speaker {start_index + len(mapping)}"
    return mapping


def compute_overlap_seconds(segment_a, segment_b):
    """Calculate overlap duration in seconds between two timestamp segments."""
    return max(0.0, min(segment_a["end"], segment_b["end"]) - max(segment_a["start"], segment_b["start"]))


def map_hypothesis_speakers_to_reference(reference_segments, hypothesis_segments, reference_label_map):
    """Map hypothesis speakers to readable reference labels using maximum overlap."""
    hyp_speakers = sorted({r["speaker"] for r in hypothesis_segments})
    ref_raw_labels = sorted({r["speaker"] for r in reference_segments})
    mapping = {}
    next_id = len(reference_label_map) + 1

    for hyp_spk in hyp_speakers:
        scores = {ref_spk: 0.0 for ref_spk in ref_raw_labels}
        hyp_rows = [r for r in hypothesis_segments if r["speaker"] == hyp_spk]
        for hyp_row in hyp_rows:
            for ref_row in reference_segments:
                scores[ref_row["speaker"]] += compute_overlap_seconds(hyp_row, ref_row)

        if scores:
            best_ref, best_overlap = max(scores.items(), key=lambda item: item[1])
        else:
            best_ref, best_overlap = None, 0.0

        if best_ref is not None and best_overlap > 0:
            mapping[hyp_spk] = reference_label_map.get(best_ref, best_ref)
        else:
            mapping[hyp_spk] = f"Speaker {next_id}"
            next_id += 1
    return mapping


def apply_display_speaker_labels(segments, label_map):
    """Return a copy of segments with readable display_speaker labels."""
    out = []
    for row in segments:
        new_row = dict(row)
        new_row["raw_speaker"] = row["speaker"]
        new_row["display_speaker"] = label_map.get(row["speaker"], row["speaker"])
        out.append(new_row)
    return out


def make_display_segments(ref_rows, result_rows):
    """Create display-ready reference and hypothesis segments with aligned labels."""
    display = {}
    reference_map = build_reference_speaker_map(ref_rows) if ref_rows else {}
    ref_display = apply_display_speaker_labels(ref_rows, reference_map) if ref_rows else None

    for system, rows in result_rows.items():
        if ref_rows:
            hyp_map = map_hypothesis_speakers_to_reference(ref_rows, rows, reference_map)
        else:
            hyp_map = build_first_appearance_map(rows)
        display[system] = apply_display_speaker_labels(rows, hyp_map)
    return ref_display, display, reference_map


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------
def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (x != y))
            prev = old
    return dp[-1]


def label_at(rows, t, key="display_speaker"):
    labels = [r[key] for r in rows if r["start"] <= t < r["end"]]
    return "+".join(sorted(set(labels))) if labels else "<none>"


def speaker_label_wer_cer(ref_rows, hyp_rows, duration, step=0.5):
    if not ref_rows or not hyp_rows:
        return None, None
    times = np.arange(0.0, duration, step)
    ref_tokens = [label_at(ref_rows, t) for t in times]
    hyp_tokens = [label_at(hyp_rows, t) for t in times]
    wer = edit_distance(ref_tokens, hyp_tokens) / max(len(ref_tokens), 1)
    ref_chars = " ".join(ref_tokens)
    hyp_chars = " ".join(hyp_tokens)
    cer = edit_distance(list(ref_chars), list(hyp_chars)) / max(len(ref_chars), 1)
    return 100 * wer, 100 * cer


def per_speaker_stats(rows):
    stats = {}
    total_speech = sum(max(0.0, r["end"] - r["start"]) for r in rows)
    for row in rows:
        spk = row["display_speaker"]
        duration = max(0.0, row["end"] - row["start"])
        stats.setdefault(spk, {"Speaker": spk, "Segments": 0, "Speaking time (s)": 0.0})
        stats[spk]["Segments"] += 1
        stats[spk]["Speaking time (s)"] += duration

    out = []
    for item in stats.values():
        item["Avg segment (s)"] = item["Speaking time (s)"] / max(item["Segments"], 1)
        item["Speech share %"] = 100 * item["Speaking time (s)"] / max(total_speech, 1e-9)
        out.append(item)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    for col in ["Speaking time (s)", "Avg segment (s)", "Speech share %"]:
        df[col] = df[col].round(2)
    return df.sort_values("Speaking time (s)", ascending=False).reset_index(drop=True)


def normalize_text(value):
    text = str(value).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def better_text(metric, baseline_value, finetuned_value):
    if baseline_value is None or finetuned_value is None:
        return "N/A"
    if abs(baseline_value - finetuned_value) < 1e-9:
        return "Same"
    return "Fine-tuned better" if finetuned_value < baseline_value else "Baseline better"


# -------------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------------
def speaker_palette(labels):
    labels = list(dict.fromkeys(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(labels), 1))
    return {label: cmap(i) for i, label in enumerate(labels)}


def build_timeline_plot(wav_path, duration, panels, reference_rows=None):
    y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    t = np.linspace(0, duration, len(y))

    all_rows = []
    if reference_rows:
        all_rows.extend(reference_rows)
    for _, rows in panels:
        all_rows.extend(rows)
    labels = sorted({r["display_speaker"] for r in all_rows}, key=lambda s: int(s.split()[-1]) if s.startswith("Speaker ") else 999)
    colors = speaker_palette(labels)

    n_tracks = len(panels) + (1 if reference_rows else 0)
    fig, axes = plt.subplots(
        n_tracks + 1,
        1,
        figsize=(16, 2.0 + 1.0 * n_tracks),
        gridspec_kw={"height_ratios": [1.45] + [0.85] * n_tracks},
    )
    if n_tracks == 0:
        axes = [axes]

    axes[0].plot(t, y, color="#9ca3af", linewidth=0.35)
    axes[0].fill_between(t, y, 0, color="#9ca3af", alpha=0.22)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(Path(wav_path).stem, fontweight="bold", fontsize=12)
    axes[0].grid(axis="x", alpha=0.18)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)

    idx = 1
    rows_to_plot = []
    if reference_rows:
        rows_to_plot.append(("Reference", reference_rows))
    rows_to_plot.extend(panels)

    for label, rows in rows_to_plot:
        ax = axes[idx]
        for row in rows:
            ax.barh(
                0,
                row["end"] - row["start"],
                left=row["start"],
                color=colors[row["display_speaker"]],
                edgecolor="white",
                linewidth=0.35,
                height=0.62,
            )
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontweight="bold")
        ax.set_xlim(0, duration)
        ax.grid(axis="x", alpha=0.18)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False)
        idx += 1

    axes[-1].set_xlabel("Time (seconds)")

    legend_handles = [plt.Line2D([0], [0], color=colors[label], lw=7) for label in labels[:20]]
    if legend_handles:
        fig.legend(
            legend_handles,
            labels[:20],
            loc="lower center",
            ncol=min(6, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def build_speaker_bar(stats_df, title):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if not stats_df.empty:
        ax.bar(stats_df["Speaker"], stats_df["Speaking time (s)"])
        ax.set_ylabel("Speaking time (s)")
        ax.set_xlabel("Speaker")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    return fig


def segments_df(rows, label):
    return pd.DataFrame([
        {
            "system": label,
            "start_s": round(r["start"], 2),
            "end_s": round(r["end"], 2),
            "duration_s": round(r["end"] - r["start"], 2),
            "speaker": r["display_speaker"],
            "raw_speaker": r.get("raw_speaker", r.get("speaker", "")),
        }
        for r in rows
    ])


def card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------
# Header
# -------------------------------------------------------------------------
ft_status = "Loaded" if FT_WEIGHTS.exists() else "Missing"
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🎙️ Diarization Inference Dashboard</div>
        <div class="hero-subtitle">Speaker diarization, transcription alignment, and metric evaluation for multi-speaker audio.</div>
        <div class="status-row">
            <span class="badge">Device: {DEVICE.upper()}</span>
            <span class="badge">Fine-tuned weights: {ft_status}</span>
            <span class="badge">Files loaded: {len(FILES)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# Sidebar input
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["VoxConverse file", "Upload audio"], index=0)

    wav_path = None
    uri = None
    has_ref = False

    if source == "VoxConverse file":
        if not FILES:
            st.error("STREAMLIT_FILES is empty. Set it in the launcher cell.")
        else:
            selected = st.selectbox("File", FILES, index=0)
            candidate = PROC_DIR / f"{selected}.wav"
            if candidate.exists():
                wav_path = candidate
                uri = selected
                has_ref = (DATA_DIR / f"{selected}.rttm").exists()
            else:
                st.error(f"Missing audio file: {candidate}")
    else:
        upload = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        if upload is not None:
            raw_path = UI_UPLOAD_DIR / upload.name
            raw_path.write_bytes(upload.getvalue())
            y, _sr = librosa.load(str(raw_path), sr=16000, mono=True)
            if y.size > 0:
                peak = float(np.max(np.abs(y))) or 1.0
                y = (y / peak) * 0.95
                norm = UI_UPLOAD_DIR / (Path(upload.name).stem + "_16k.wav")
                sf.write(str(norm), y, 16000, subtype="PCM_16")
                wav_path = norm
                uri = norm.stem

    st.markdown("---")
    st.header("Pipeline")
    choice = st.radio("Run", ["Both (compare)", "Baseline only", "Fine-tuned only"], index=0)
    run = st.button("▶ Run diarization", type="primary", use_container_width=True, disabled=wav_path is None)

    st.markdown("---")
    st.caption(f"Device: **{DEVICE}**")
    st.caption(f"Reference RTTM: **{'available' if has_ref else 'not available'}**")
    st.caption(f"Fine-tuned weights: **{ft_status}**")


# -------------------------------------------------------------------------
# Main content before run
# -------------------------------------------------------------------------
if wav_path is None:
    st.info("Select a VoxConverse file or upload an audio file, then run diarization.")
    st.stop()

duration = float(sf.info(str(wav_path)).duration)

st.markdown('<div class="section-title">File Overview</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    card("File", uri, "Selected audio")
with col2:
    card("Duration", f"{duration:.1f}s", "Audio length")
with col3:
    card("Reference", "Yes" if has_ref else "No", "RTTM availability")
with col4:
    card("Device", DEVICE.upper(), "Inference backend")
with col5:
    card("Mode", choice.replace(" only", ""), "Selected run")

st.markdown('<div class="section-title">Audio Preview</div>', unsafe_allow_html=True)
st.audio(str(wav_path))

# Cache diarization outputs so changing sidebar radios (pipeline mode) or tab widgets
# (e.g. segment table view) does not wipe results — the Run button is only True on
# the rerun that immediately follows a click.
cache_key = (uri, str(Path(wav_path).resolve()))
diar_cache = st.session_state.setdefault("diar_cache", {})

run_base = choice in ("Both (compare)", "Baseline only")
run_ft = choice in ("Both (compare)", "Fine-tuned only")
results = {}

if run:
    entry = diar_cache.setdefault(cache_key, {"results": {}})
    entry["duration"] = duration
    steps = max(1, sum([run_base, run_ft]))
    done = 0
    progress = st.progress(0.0, "Starting inference…")

    if run_base:
        progress.progress(done / steps, "Loading baseline pipeline…")
        baseline = load_baseline()
        progress.progress((done + 0.35) / steps, "Running baseline diarization…")
        t0 = time.time()
        ann = baseline(str(wav_path))
        elapsed = time.time() - t0
        results["Baseline"] = (ann, ann_to_segments(ann, duration), elapsed)
        entry["results"]["Baseline"] = results["Baseline"]
        done += 1

    if run_ft:
        progress.progress(done / steps, "Loading fine-tuned pipeline…")
        fine_tuned = load_finetuned()
        if fine_tuned is None:
            progress.empty()
            st.error(f"Fine-tuned weights not found at `{FT_WEIGHTS}`. Run the training/save section first.")
            st.stop()
        progress.progress((done + 0.35) / steps, "Running fine-tuned diarization…")
        t0 = time.time()
        ann = fine_tuned(str(wav_path))
        elapsed = time.time() - t0
        results["Fine-tuned"] = (ann, ann_to_segments(ann, duration), elapsed)
        entry["results"]["Fine-tuned"] = results["Fine-tuned"]
        done += 1

    progress.progress(1.0, "Done")
    progress.empty()

elif cache_key in diar_cache:
    full = diar_cache[cache_key]["results"]
    if run_base and "Baseline" in full:
        results["Baseline"] = full["Baseline"]
    if run_ft and "Fine-tuned" in full:
        results["Fine-tuned"] = full["Fine-tuned"]
    if not results:
        st.warning(
            "No cached results for this mode on the current file. "
            "Choose **Both (compare)** or the missing pipeline and click **Run diarization** once."
        )
        st.stop()
else:
    st.info("Select a VoxConverse file or upload an audio file, then click **Run diarization**.")
    st.stop()


# -------------------------------------------------------------------------
# Reference and display-label mapping
# -------------------------------------------------------------------------
ref_ann = None
ref_rows = None
if has_ref:
    ref_ann = load_ref(DATA_DIR / f"{uri}.rttm")
    ref_rows = ann_to_segments(ref_ann, duration)

raw_result_rows = {label: rows for label, (_ann, rows, _elapsed) in results.items()}
ref_display_rows, display_result_rows, reference_label_map = make_display_segments(ref_rows, raw_result_rows)


# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------
tab_overview, tab_metrics, tab_segments, tab_speakers, tab_downloads = st.tabs(
    ["Overview", "Metrics", "Segments", "Speaker Stats", "Downloads"]
)

with tab_overview:
    st.markdown('<div class="section-title">Timeline Comparison</div>', unsafe_allow_html=True)
    panels = [(label, display_result_rows[label]) for label in results]
    fig = build_timeline_plot(wav_path, duration, panels, reference_rows=ref_display_rows)
    st.pyplot(fig, use_container_width=True)

    if reference_label_map:
        with st.expander("Speaker label mapping", expanded=False):
            mapping_df = pd.DataFrame(
                [{"Raw VoxConverse ID": raw, "Display label": display} for raw, display in reference_label_map.items()]
            )
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)

with tab_metrics:
    st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)
    metric_rows = []
    uem = Timeline([Segment(0.0, duration)], uri=uri)

    for label, (ann, rows, elapsed) in results.items():
        display_rows = display_result_rows[label]
        row = {
            "system": label,
            "speakers": len({r["display_speaker"] for r in display_rows}),
            "segments": len(display_rows),
            "rtf": round(elapsed / duration, 4),
            "wall_time_s": round(elapsed, 2),
        }
        if ref_ann is not None and ref_display_rows is not None:
            der = 100 * DiarizationErrorRate(collar=0.25, skip_overlap=False)(ref_ann, ann, uem=uem)
            wer, cer = speaker_label_wer_cer(ref_display_rows, display_rows, duration)
            row["der_percent"] = round(der, 3)
            row["speaker_label_wer_percent"] = round(wer, 3) if wer is not None else None
            row["speaker_label_cer_percent"] = round(cer, 3) if cer is not None else None
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    if not metrics_df.empty:
        systems_present = set(metrics_df["system"].tolist())
        has_pair = {"Baseline", "Fine-tuned"}.issubset(systems_present)
        if has_pair:
            st.markdown('<div class="section-title">Summary cards</div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            for col, sys_name in ((left, "Baseline"), (right, "Fine-tuned")):
                with col:
                    st.markdown(f"**{sys_name}**")
                    row = metrics_df[metrics_df["system"] == sys_name].iloc[0].to_dict()
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        card("DER", f"{row.get('der_percent', 'N/A')}%", "Lower is better")
                    with c2:
                        card("WER", f"{row.get('speaker_label_wer_percent', 'N/A')}%", "Lower is better")
                    with c3:
                        card("CER", f"{row.get('speaker_label_cer_percent', 'N/A')}%", "Lower is better")
                    with c4:
                        card("RTF", f"{row.get('rtf', 'N/A')}", "Lower is faster")
        else:
            selected_system = st.selectbox(
                "Show metric cards for",
                metrics_df["system"].tolist(),
                index=len(metrics_df) - 1,
            )
            selected_row = metrics_df[metrics_df["system"] == selected_system].iloc[0].to_dict()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("DER", f"{selected_row.get('der_percent', 'N/A')}%", "Lower is better")
            with c2:
                card("WER", f"{selected_row.get('speaker_label_wer_percent', 'N/A')}%", "Lower is better")
            with c3:
                card("CER", f"{selected_row.get('speaker_label_cer_percent', 'N/A')}%", "Lower is better")
            with c4:
                card("RTF", f"{selected_row.get('rtf', 'N/A')}", "Lower is faster")

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    if ref_ann is not None and {"Baseline", "Fine-tuned"}.issubset(set(metrics_df["system"])):
        base = metrics_df[metrics_df["system"] == "Baseline"].iloc[0]
        ft = metrics_df[metrics_df["system"] == "Fine-tuned"].iloc[0]
        comparison = pd.DataFrame([
            {"metric": "DER", "baseline": base.get("der_percent"), "fine_tuned": ft.get("der_percent"), "result": better_text("DER", base.get("der_percent"), ft.get("der_percent"))},
            {"metric": "WER", "baseline": base.get("speaker_label_wer_percent"), "fine_tuned": ft.get("speaker_label_wer_percent"), "result": better_text("WER", base.get("speaker_label_wer_percent"), ft.get("speaker_label_wer_percent"))},
            {"metric": "CER", "baseline": base.get("speaker_label_cer_percent"), "fine_tuned": ft.get("speaker_label_cer_percent"), "result": better_text("CER", base.get("speaker_label_cer_percent"), ft.get("speaker_label_cer_percent"))},
            {"metric": "RTF", "baseline": base.get("rtf"), "fine_tuned": ft.get("rtf"), "result": better_text("RTF", base.get("rtf"), ft.get("rtf"))},
        ])
        st.markdown('<div class="section-title">Baseline vs Fine-tuned</div>', unsafe_allow_html=True)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

with tab_segments:
    st.markdown('<div class="section-title">Segments</div>', unsafe_allow_html=True)
    frames = []
    if ref_display_rows is not None:
        frames.append(segments_df(ref_display_rows, "Reference"))
    for label in results:
        frames.append(segments_df(display_result_rows[label], label))
    seg_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    table_mode = st.radio("Table view", ["Readable labels only", "Include raw labels"], horizontal=True)
    show_df = seg_df.drop(columns=["raw_speaker"], errors="ignore") if table_mode == "Readable labels only" else seg_df
    st.dataframe(show_df, use_container_width=True, height=520, hide_index=True)

with tab_speakers:
    st.markdown('<div class="section-title">Per-speaker Statistics</div>', unsafe_allow_html=True)
    available = list(results.keys())
    if {"Baseline", "Fine-tuned"}.issubset(set(available)):
        col_b, col_f = st.columns(2)
        for col, sys_name in ((col_b, "Baseline"), (col_f, "Fine-tuned")):
            with col:
                st.markdown(f"**{sys_name}**")
                stats_df = per_speaker_stats(display_result_rows[sys_name])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                if not stats_df.empty:
                    st.pyplot(
                        build_speaker_bar(stats_df, f"{sys_name}: speaking time by speaker"),
                        use_container_width=True,
                    )
    else:
        selected_stats_system = st.selectbox("System", available, index=len(available) - 1)
        stats_df = per_speaker_stats(display_result_rows[selected_stats_system])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        if not stats_df.empty:
            st.pyplot(
                build_speaker_bar(stats_df, f"{selected_stats_system}: speaking time by speaker"),
                use_container_width=True,
            )

with tab_downloads:
    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)
    frames = []
    if ref_display_rows is not None:
        frames.append(segments_df(ref_display_rows, "Reference"))
    for label in results:
        frames.append(segments_df(display_result_rows[label], label))
    seg_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    metric_rows = []
    uem = Timeline([Segment(0.0, duration)], uri=uri)
    for label, (ann, _rows, elapsed) in results.items():
        display_rows = display_result_rows[label]
        row = {
            "system": label,
            "speakers": len({r["display_speaker"] for r in display_rows}),
            "segments": len(display_rows),
            "rtf": round(elapsed / duration, 4),
            "wall_time_s": round(elapsed, 2),
        }
        if ref_ann is not None and ref_display_rows is not None:
            der = 100 * DiarizationErrorRate(collar=0.25, skip_overlap=False)(ref_ann, ann, uem=uem)
            wer, cer = speaker_label_wer_cer(ref_display_rows, display_rows, duration)
            row["der_percent"] = round(der, 3)
            row["speaker_label_wer_percent"] = round(wer, 3) if wer is not None else None
            row["speaker_label_cer_percent"] = round(cer, 3) if cer is not None else None
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇ Download segments CSV",
            seg_df.to_csv(index=False).encode(),
            f"segments_{uri}.csv",
            "text/csv",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "⬇ Download metrics CSV",
            metrics_df.to_csv(index=False).encode(),
            f"metrics_{uri}.csv",
            "text/csv",
            use_container_width=True,
        )

st.caption("Speaker labels shown in the UI are display labels. Raw VoxConverse and pyannote IDs are kept internally for metric calculation.")

