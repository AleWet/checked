#!/usr/bin/env python3
"""
Reel / Short-Form Video Fact Checker — prototype pipeline.

Usage:
    python fact_check.py <url> [--provider claude|openai|perplexity] [--frames 6] [--whisper-model small] [--model <id>]

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

DEFAULT_MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.5",
    "perplexity": "sonar",
}


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

BASE_CHECK_SYSTEM_PROMPT = """You are a fact-checking engine for short-form video content.

INPUT: transcript, platform metadata (title, caption, uploader, date), sample frames.

TASK:
- Extract the 3 most checkable factual claims. Skip opinions, jokes, subjective takes.
- Verify each claim via web search. Prefer primary sources and established outlets.
- If transcript is too short or ambiguous to extract real claims, return reliability_score: null and explain in one sentence.

OUTPUT: valid JSON only, no markdown, no commentary.

{
  "reliability_score": <integer 0-100>,
  "claims": [
    {
      "claim": "<exact claim as stated>",
      "verdict": "true" | "mostly_true" | "uncertain" | "mostly_false" | "false",
      "sources": ["<exact URL>", "<exact URL>"]
    }
  ]
}

Rules:
- reliability_score is the weighted average of claim verdicts (true=100, mostly_true=75, uncertain=50, mostly_false=25, false=0).
- Never fabricate URLs. If no source found, set verdict to "uncertain" and sources to [].
- Output nothing except the JSON object."""

PREMIUM_CHECK_SYSTEM_PROMPT = """You are a senior fact-checking analyst specialising in short-form video content
(Reels, TikToks, YouTube Shorts). You receive: full transcript, platform metadata
(title, caption, uploader, upload date, view count if available), and sample frames.

ANALYSIS PIPELINE:

STEP 1 — CLAIM EXTRACTION
Identify up to 5 distinct, checkable factual claims. For each, note:
- The exact quote or close paraphrase as stated in the video
- Timestamp or position (early / mid / late)
- Claim type: statistic | causal | historical | scientific | political | attribution
- Checkability score: 1-10 (10 = fully checkable, 1 = too vague to verify)
Ignore pure opinions, predictions, jokes. Flag them in a separate "non-claims" list.

STEP 2 — VERIFICATION
For each claim with checkability ≥ 4:
- Search for primary sources: government data, peer-reviewed studies, official reports
- Search for established outlets: Reuters, AP, BBC, NYT, peer-reviewed journals
- Note source publication date — flag if source is older than 2 years
- Note any expert consensus or dissent on the topic
- Identify what context the video omits that materially changes the claim's meaning

STEP 3 — VERDICT ASSIGNMENT
Rate each claim on this exact scale:
- TRUE (90-100): supported by multiple primary sources, no significant caveats
- MOSTLY_TRUE (70-89): supported but missing nuance or minor context
- UNCERTAIN (40-69): conflicting evidence or insufficient primary sources found
- MOSTLY_FALSE (15-39): contradicted by primary sources with limited supporting evidence
- FALSE (0-14): directly contradicted by multiple primary sources

STEP 4 — CONFIDENCE RATING
Rate your own confidence in the verdict:
- HIGH: multiple primary sources found, clear consensus
- MEDIUM: some sources found, limited consensus or dated data
- LOW: minimal sources found, topic contested or niche

OUTPUT: valid JSON only. No markdown, no commentary outside the JSON.

{
  "reliability_score": <integer 0-100>,
  "confidence": "high" | "medium" | "low",
  "content_summary": "<one sentence describing what the video claims overall>",
  "verdict_summary": "<two sentences max: overall assessment and single most important finding>",
  "claims": [
    {
      "id": 1,
      "claim": "<exact quote or close paraphrase>",
      "claim_type": "statistic|causal|historical|scientific|political|attribution",
      "position": "early|mid|late",
      "checkability": <1-10>,
      "verdict": "true|mostly_true|uncertain|mostly_false|false",
      "verdict_score": <0-100>,
      "explanation": "<two sentences max: what sources say and what context is missing>",
      "source_date_flag": true | false,
      "sources": [
        {
          "title": "<article or page title>",
          "outlet": "<publication name>",
          "url": "<exact URL>",
          "date": "<publication date if available>"
        }
      ]
    }
  ],
  "non_claims": ["<opinion or joke flagged, one line each>"],
  "missing_context": "<three sentences max: what the video omits that materially changes its meaning>",
  "red_flags": ["<specific manipulative framing, cherry-picked stat, or missing attribution if detected>"]
}

RULES:
- reliability_score = weighted average of verdict_scores, weighted by checkability.
- Never fabricate URLs. If no source found, set verdict to "uncertain", sources to [].
- source_date_flag = true if best available source is older than 2 years.
- red_flags is an empty array [] if none detected — never omit the field.
- Output nothing except the JSON object. No preamble, no closing note."""

TIER_PROMPTS = {
    "base": BASE_CHECK_SYSTEM_PROMPT,
    "premium": PREMIUM_CHECK_SYSTEM_PROMPT,
}

def build_transcript_block(metadata: dict, transcript: str, frame_paths: list[Path]) -> str:
    return f"""Video metadata:
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


# ---- Claude ----------------------------------------------------------------

def image_to_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }


def call_claude(metadata: dict, transcript: str, frame_paths: list[Path], model: str, system_prompt: str) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    content = [{"type": "text", "text": build_transcript_block(metadata, transcript, frame_paths)}]
    for fp in frame_paths:
        content.append(image_to_block(fp))

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=system_prompt,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": content}],
    )

    report_text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()

    usage = response.usage
    web_searches = usage.server_tool_use.web_search_requests if usage.server_tool_use else 0
    usage_dict = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "web_search_requests": web_searches or 0,
    }
    return report_text, usage_dict


# ---- OpenAI -----------------------------------------------------------------

def call_openai(metadata: dict, transcript: str, frame_paths: list[Path], model: str, system_prompt: str) -> tuple[str, dict]:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env

    content = [{"type": "input_text", "text": build_transcript_block(metadata, transcript, frame_paths)}]
    for fp in frame_paths:
        b64 = base64.standard_b64encode(fp.read_bytes()).decode("utf-8")
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        tools=[{"type": "web_search"}],
        input=[{"role": "user", "content": content}],
    )

    report_text = response.output_text.strip()
    usage_dict = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        # OpenAI's Responses API doesn't return a discrete web-search-call count in `usage`
        # the way Claude does; search cost is folded into token usage here.
        "web_search_requests": None,
    }
    return report_text, usage_dict


# ---- Perplexity --------------------------------------------------------------

def call_perplexity(metadata: dict, transcript: str, frame_paths: list[Path], model: str, system_prompt: str) -> tuple[str, dict]:
    from openai import OpenAI  # Perplexity's Sonar API is OpenAI-chat-completions-compatible

    client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

    completion = client.chat.completions.create(
        model=model,  # e.g. "sonar", "sonar-pro", "sonar-reasoning-pro" — web search is built-in, no tool config needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_transcript_block(metadata, transcript, [])},
        ],
    )

    report_text = completion.choices[0].message.content.strip()
    citations = getattr(completion, "citations", None) or (completion.model_extra or {}).get("citations")
    if citations:
        report_text += "\n\n## Sources\n" + "\n".join(f"- {url}" for url in citations)

    usage_dict = {
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "web_search_requests": None,  # bundled into Sonar's per-token pricing, not itemized separately
    }
    return report_text, usage_dict


def call_fact_checker(provider: str, metadata: dict, transcript: str, frame_paths: list[Path], model: str, system_prompt: str) -> tuple[str, dict]:
    if provider == "claude":
        return call_claude(metadata, transcript, frame_paths, model, system_prompt)
    elif provider == "openai":
        return call_openai(metadata, transcript, frame_paths, model, system_prompt)
    elif provider == "perplexity":
        return call_perplexity(metadata, transcript, frame_paths, model, system_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def log_usage(usage: dict, provider: str, model: str, tier: str, log_path: Path) -> None:
    """Append one line per run to a local usage log, so cost is trackable over time."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "tier": tier,
        **usage,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to sample and send to the AI (default 6). Lower = cheaper.")
    parser.add_argument("--no-frames", "--no-images", dest="no_frames", action="store_true",
                         help="Skip frame extraction and don't send any images to the AI — transcript-only, cheapest option (saves vision tokens).")
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size: tiny/base/small/medium (default small)")
    parser.add_argument("--provider", choices=["claude", "openai", "perplexity"], default="claude",
                         help="Which AI provider to use for the fact-check step (default claude)")
    parser.add_argument("--model", default=None,
                         help=f"Model id to use. Defaults per provider: {DEFAULT_MODELS}")
    parser.add_argument("--tier", choices=["base", "premium"], default="base",
                         help="Which system prompt tier to use (default base). 'premium' is currently a placeholder, identical to base.")
    parser.add_argument("--output-dir", default="./reports", help="Where to save reports")
    parser.add_argument("--keep-temp", action="store_true", help="Keep downloaded media/frames instead of deleting")
    parser.add_argument("--cookies", default=None, help="Path to a Netscape-format cookies.txt (needed for Instagram)")
    parser.add_argument("--cookies-from-browser", default=None, help="Pull cookies directly from a browser, e.g. 'chrome' or 'firefox' (browser should be closed)")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]

    key_env_vars = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "perplexity": "PERPLEXITY_API_KEY"}
    key_name = key_env_vars[args.provider]
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        sys.exit(f"{key_name} is not set. Add it to your .env file (see .env.example).")
    if args.provider == "claude" and not api_key.startswith("sk-ant-"):
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
        if args.no_frames:
            print("[4/5] Skipping frame extraction (--no-frames)")
        elif args.provider == "perplexity":
            print("[4/5] Skipping frame extraction (Perplexity Sonar models are used here in text-only mode)")
        else:
            print(f"[4/5] Sampling {args.frames} frames for visual context...")
            frame_paths = extract_frames(metadata["video_path"], metadata["duration"], workdir, args.frames)
            print(f"      -> {len(frame_paths)} frames extracted")

        print(f"[5/5] Sending to {args.provider} ({model}, tier={args.tier}) with web search enabled...")
        system_prompt = TIER_PROMPTS[args.tier]
        report_text, usage = call_fact_checker(args.provider, metadata, transcript, frame_paths, model, system_prompt)
        search_note = f"{usage['web_search_requests']} web searches" if usage["web_search_requests"] is not None else "search count n/a"
        print(f"      -> tokens: {usage['input_tokens']} in / {usage['output_tokens']} out ({search_note})")
        log_usage(usage, args.provider, model, args.tier, Path(args.output_dir) / "usage_log.jsonl")

        report_path = save_report(metadata, transcript, report_text, Path(args.output_dir))

        print("\n" + "=" * 70)
        print(report_text)
        print("=" * 70)
        print(f"\nSaved report to: {report_path.resolve()}")

        if args.keep_temp:
            print(f"(--keep-temp set but tempdir {workdir} will still be removed on exit by TemporaryDirectory)")


if __name__ == "__main__":
    main()