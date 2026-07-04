# Reel / Short-Form Fact Checker — Prototype

A single-command pipeline: give it a link (TikTok, Instagram Reel, YouTube Short, etc.),
it downloads the video, transcribes it, grabs a few frames, and sends it all to Claude
(with live web search enabled) to produce a fact-check report.

## 1. System requirements (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
```

`ffmpeg` is used for audio extraction and frame sampling. `yt-dlp` (installed via pip below)
handles the actual video downloading and supports most short-form platforms out of the box.

## 2. Set up the project

```bash
git clone <your-repo-or-just-copy-this-folder> reel_fact_checker
cd reel_fact_checker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

The first time you run a transcription, `faster-whisper` will download the chosen model
(e.g. ~250MB for "small") to a local cache — no ongoing cost, runs on CPU.

## 3. Add your API key

```bash
cp .env.example .env
# edit .env and paste your Anthropic API key
```

Get a key at https://console.anthropic.com if you don't have one yet.

## 4. Run it

```bash
python fact_check.py "https://www.tiktok.com/@someuser/video/1234567890"
```

Optional flags:

```bash
python fact_check.py "<url>" \
  --frames 6 \                  # how many still frames to sample (default 6, 0 disables visuals via --no-frames)
  --whisper-model small \       # tiny/base/small/medium — bigger = more accurate, slower
  --model claude-sonnet-5 \     # the fact-checking model
  --output-dir ./reports
```

A Markdown + JSON report is saved to `./reports/` and printed to the terminal.

## Design notes / why these choices

- **Transcription runs locally (faster-whisper)**, not through a paid API — this is the
  single biggest cost lever, since transcript length is usually your largest input to the
  fact-checking model.
- **Frames are downscaled and capped** (default 6, ~512px wide) rather than sending the
  full video, since image tokens are the next biggest cost driver in the AI call.
- **The fact-checking model is given a live web search tool**, so it isn't just judging
  claims from frozen training knowledge — it verifies against current sources and returns
  citations. This is what makes the "impartial and informed" requirement actually work in
  practice, not just in the system prompt.
- Currently uses `claude-sonnet-5`, which as of writing has introductory pricing that makes
  it noticeably cheaper than standard Sonnet-tier rates — worth checking
  https://docs.claude.com/en/docs/about-claude/models/overview periodically, since pricing
  and model names do change over time and this alpha targets "cheap and fast" above all.

## Known limitations (alpha)

- Instagram often requires a logged-in session/cookies for reliable downloads; yt-dlp
  supports passing a cookies file (`--cookies cookies.txt`) if you hit that wall — not wired
  into the script yet.
- No retry/backoff logic yet for flaky downloads or rate limits.
- No caching — the same URL run twice re-downloads and re-transcribes. Fine for alpha,
  wasteful at scale.
- No content moderation/guardrails around what gets downloaded — worth adding before any
  public-facing deployment.

## Path to scale (not built yet, but the code is structured for this)

The script's five stages (`download_media`, `extract_audio` + `transcribe`,
`extract_frames`, `call_fact_checker`, `save_report`) are already separate functions, so the
natural next steps are:

1. **Wrap it as an API** — put these functions behind a FastAPI app with a `POST /fact-check`
   endpoint that accepts a URL and returns a job ID.
2. **Move heavy work onto a queue** — Celery or RQ + Redis, so the API responds instantly and
   a worker pool handles download/transcribe/verify in the background. This is what lets you
   handle many submitted links concurrently instead of one at a time.
3. **Cache by URL/content hash** — before reprocessing, check if you've already fact-checked
   this exact video (viral reels get re-shared constantly); serve the cached report.
4. **Tiered models** — run a cheap/fast pass (e.g. `claude-haiku-4-5`) to triage "does this even
   contain checkable claims worth a deep pass," and only escalate to the fuller Sonnet-tier
   verification for videos that need it. Cuts average cost per video submitted.
5. **Object storage** — move downloaded media/transcripts/frames from local temp dirs to
   S3-compatible storage once this is a multi-user service.
6. **Batch API** for any non-realtime bulk backfills (e.g. re-checking a backlog) — roughly
   half the per-token cost when you don't need an immediate response.
