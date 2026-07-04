#!/usr/bin/env python3
"""
Reel / Short-Form Video Fact Checker — prototype pipeline.

Usage:
    python fact_check.py <url> [--frames 6] [--whisper-model small] [--model claude-sonnet-5]

Pipeline:
    1. Download video via yt-dlp (works for TikTok, Instagram Reels, YouTube Shorts, etc.)
    2. Extract audio -> transcribe locally with faster-whisper (no per-minute API cost)
    3. Extract a handful of evenly spaced, downscaled frames for visual context
    4. Send transcript + metadata + frames to Claude (with web search enabled) acting
       as a rigorous, impartial fact-checker
    5. Save a Markdown + JSON report

Env:
    ANTHROPIC_API_KEY must be set (see .env.example)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency 'anthropic'. Run: pip install -r requirements.txt")


# --------------------------------------------------------------------------
# Step 1: Download
# --------------------------------------------------------------------------

def download_media(url: str, workdir: Path, cookies_file: str | None = None,
                    cookies_from_browser: str | None = None) -> dict:
    """Download video + metadata via yt-dlp. Returns dict with paths and info.

    cookies_file: path to a Netscape-format cookies.txt (needed for Instagram, and
        sometimes for age-gated/private content on other platforms).
    cookies_from_browser: browser name (e.g. 'chrome', 'firefox') to pull cookies from
        directly, instead of using an exported file. The browser should be closed while
        this runs, or yt-dlp may fail to read its cookie database.
    """
    out_template = str(workdir / "source.%(ext)s")
    info_path = workdir / "info.json"

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "-o", out_template,
        "--format", "mp4/best",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # yt-dlp writes info json as source.info.json
    info_candidates = list(workdir.glob("source*.info.json"))
    if not info_candidates:
        raise RuntimeError("yt-dlp did not produce an info.json file")
    info = json.loads(info_candidates[0].read_text())

    video_candidates = [p for p in workdir.glob("source.*") if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if not video_candidates:
        raise RuntimeError("yt-dlp did not produce a video file")

    return {
        "video_path": video_candidates[0],
        "title": info.get("title", ""),
        "description": info.get("description", ""),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "upload_date": info.get("upload_date", ""),
        "duration": info.get("duration") or 0,
        "webpage_url": info.get("webpage_url", url),
        "platform": info.get("extractor_key", ""),
    }


# --------------------------------------------------------------------------
# Step 2: Audio extraction + transcription
# --------------------------------------------------------------------------

def extract_audio(video_path: Path, workdir: Path) -> Path:
    audio_path = workdir / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
    return audio_path


def transcribe(audio_path: Path, model_size: str = "small") -> str:
    """Local, free transcription via faster-whisper (CPU, int8)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


# --------------------------------------------------------------------------
# Step 3: Frame sampling
# --------------------------------------------------------------------------

def extract_frames(video_path: Path, duration: float, workdir: Path, n_frames: int = 6) -> list[Path]:
    """Grab n_frames evenly spaced, downscaled JPEG stills to keep vision costs low."""
    if duration <= 0:
        duration = 15  # fallback guess for shorts if metadata missing

    frames = []
    n_frames = max(1, n_frames)
    for i in range(n_frames):
        # avoid exact 0 and exact end to skip black frames/logos
        timestamp = (duration * (i + 1)) / (n_frames + 1)
        out_path = workdir / f"frame_{i:02d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-frames:v", "1", "-vf", "scale=512:-1", "-q:v", "4",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists():
            frames.append(out_path)
    return frames


def image_to_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


# --------------------------------------------------------------------------
# Step 4: Fact-check via Claude
# --------------------------------------------------------------------------

FACT_CHECK_SYSTEM_PROMPT = """You are a rigorous, impartial fact-checking analyst reviewing short-form \
video content (reels, TikToks, YouTube Shorts). You will receive the video's transcript, its \
platform metadata (title, caption/description, uploader, upload date), and a small number of \
sample frames.

Your job:
1. Identify the distinct, checkable factual claims made in the video (ignore pure opinion, jokes, \
or subjective takes — flag them as such rather than fact-checking them).
2. For each claim, use web search to verify it against credible, current sources. Prefer primary \
sources, official data, and established outlets over blogs, forums, or SEO content.
3. Rate each claim: True / Mostly True / Misleading / Mostly False / False / Unverifiable.
4. Give an overall verdict for the video plus a short confidence note (e.g. "high confidence" vs \
"limited evidence available").
5. Be skeptical of unsupported claims, but do not assume malicious intent — misinformation is often \
unintentional or a matter of missing context.
6. If the transcript is too short, garbled, or ambiguous to extract real claims, say so plainly \
instead of inventing a verdict.

Output format (Markdown):
## Overall Verdict
## Claim-by-Claim Breakdown
(for each: the claim, verdict, brief explanation, source(s))
## Missing Context / Caveats

Always cite the sources you actually used. Do not fabricate URLs or citations, create a list of the sources you use at the end of each response"""


def build_user_content(metadata: dict, transcript: str, frame_paths: list[Path]) -> list[dict]:
    text_block = f"""Video metadata:
- Platform: {metadata.get('platform', 'unknown')}
- Title: {metadata.get('title', '(none)')}
- Uploader: {metadata.get('uploader', '(unknown)')}
- Upload date: {metadata.get('upload_date', '(unknown)')}
- URL: {metadata.get('webpage_url', '')}
- Description/caption: {metadata.get('description', '(none)')}

Transcript:
\"\"\"
{transcript if transcript else '(no speech detected / transcription empty)'}
\"\"\"

{len(frame_paths)} sample frames from the video are attached below for visual context \
(on-screen text, charts, context that audio alone won't capture)."""

    content = [{"type": "text", "text": text_block}]
    for fp in frame_paths:
        content.append(image_to_block(fp))
    return content


def call_fact_checker(metadata: dict, transcript: str, frame_paths: list[Path], model: str) -> anthropic.types.Message:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=FACT_CHECK_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": build_user_content(metadata, transcript, frame_paths)}],
    )
    return response


def extract_report_text(response: anthropic.types.Message) -> str:
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n\n".join(parts).strip()


# --------------------------------------------------------------------------
# Step 5: Save report
# --------------------------------------------------------------------------

def save_report(metadata: dict, transcript: str, report_text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "_" for c in metadata.get("title", "video"))[:40] or "video"
    base = f"{timestamp}_{slug}"

    md_path = output_dir / f"{base}.md"
    md_path.write_text(
        f"# Fact-Check Report\n\n"
        f"**Source:** {metadata.get('webpage_url')}\n"
        f"**Uploader:** {metadata.get('uploader')}\n"
        f"**Checked at:** {timestamp}\n\n"
        f"---\n\n{report_text}\n"
    )

    json_path = output_dir / f"{base}.json"
    json_path.write_text(json.dumps({
        "metadata": {k: v for k, v in metadata.items() if k != "video_path"},
        "transcript": transcript,
        "report": report_text,
        "checked_at": timestamp,
    }, indent=2, default=str))

    return md_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fact-check a short-form video from a URL.")
    parser.add_argument("url", help="Link to the reel/TikTok/short/video")
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to sample (default 6)")
    parser.add_argument("--no-frames", action="store_true", help="Skip frame extraction (cheaper, transcript-only)")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size: tiny/base/small/medium (default small)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model id for the fact-checker (default claude-sonnet-5)")
    parser.add_argument("--output-dir", default="./reports", help="Where to save reports")
    parser.add_argument("--keep-temp", action="store_true", help="Keep downloaded media/frames instead of deleting")
    parser.add_argument("--cookies", default=None, help="Path to a Netscape-format cookies.txt (needed for Instagram)")
    parser.add_argument("--cookies-from-browser", default=None, help="Pull cookies directly from a browser, e.g. 'chrome' or 'firefox' (browser should be closed)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
    if not api_key.startswith("sk-ant-"):
        sys.exit(
            "ANTHROPIC_API_KEY doesn't look like a valid Anthropic key (should start with "
            "'sk-ant-'). Check for stray quotes/spaces in your .env file, or that you copied "
            "the full key from https://console.anthropic.com"
        )

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        print(f"[1/5] Downloading: {args.url}")
        metadata = download_media(args.url, workdir, cookies_file=args.cookies,
                                   cookies_from_browser=args.cookies_from_browser)
        print(f"      -> '{metadata['title']}' by {metadata['uploader']} ({metadata['duration']}s)")

        print("[2/5] Extracting audio...")
        audio_path = extract_audio(metadata["video_path"], workdir)

        print(f"[3/5] Transcribing (faster-whisper, model={args.whisper_model}, local/free)...")
        transcript = transcribe(audio_path, args.whisper_model)
        print(f"      -> {len(transcript.split())} words transcribed")

        frame_paths = []
        if not args.no_frames:
            print(f"[4/5] Sampling {args.frames} frames for visual context...")
            frame_paths = extract_frames(metadata["video_path"], metadata["duration"], workdir, args.frames)
            print(f"      -> {len(frame_paths)} frames extracted")
        else:
            print("[4/5] Skipping frame extraction (--no-frames)")

        print(f"[5/5] Sending to Claude ({args.model}) with web search enabled...")
        response = call_fact_checker(metadata, transcript, frame_paths, args.model)
        report_text = extract_report_text(response)

        report_path = save_report(metadata, transcript, report_text, Path(args.output_dir))

        print("\n" + "=" * 70)
        print(report_text)
        print("=" * 70)
        print(f"\nSaved report to: {report_path.resolve()}")

        if args.keep_temp:
            print(f"(--keep-temp set but tempdir {workdir} will still be removed on exit by TemporaryDirectory)")


if __name__ == "__main__":
    main()
