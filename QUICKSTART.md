# Podcast Studio — Quick Start Guide

## Prerequisites

- Python 3.13+ (installed via Homebrew)
- ffmpeg (installed via Homebrew: `brew install ffmpeg`)
- API keys in `config.env` (see below)

## First-Time Setup

1. **Clone the repo** (skip if already done):
   ```bash
   git clone https://github.com/KG191/podcast-studio.git ~/Documents/Podcast\ AI_AGI_ASI
   ```

2. **Create the virtual environment and install dependencies**:
   ```bash
   cd ~/Documents/Podcast\ AI_AGI_ASI
   python3 -m venv .venv
   source .venv/bin/activate
   pip install google-genai streamlit openai pillow requests
   ```

3. **Set up your API keys**:
   ```bash
   cp config.env.example config.env
   ```
   Then open `config.env` and replace the placeholders with your actual keys:

   | Key | Where to get it | Cost |
   |-----|----------------|------|
   | `GEMINI_API_KEY` | https://aistudio.google.com/ → "Get API key" | Free |
   | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | ~$0.17/episode |
   | `GNEWS_API_KEY` | https://gnews.io/ → sign up → dashboard | Free |

## Launch the App

```bash
cd ~/Documents/Podcast\ AI_AGI_ASI
source .venv/bin/activate
streamlit run podcast_app.py
```

The app opens automatically in your browser at http://localhost:8501.

## Workflow

1. **Search** — pick a category or enter a custom query to find news stories
2. **Select** — tick one or more stories
3. **Configure** — choose podcast brand, TTS voice, and episode title
4. **Preview voice** — click the preview button to hear the selected voice
5. **Generate script** — ChatGPT writes the podcast script (editable)
6. **Generate covers** — DALL-E creates 3 cover image options (3000x3000)
7. **Generate podcast** — Gemini TTS produces the MP3 audio
8. **Copy preview summary** — paste into RSS.com as the episode description
9. **Upload** — drag the MP3 and cover.png into your RSS.com dashboard

## Output

Each episode saves to its own folder:

```
Episode_Title/
    script.txt            ← podcast script
    cover.png             ← 3000x3000 cover image
    Episode_Title.mp3     ← upload to RSS.com
    preview_summary.txt   ← episode description for RSS.com
    metadata.json         ← episode metadata
    chunks/               ← intermediate audio (can be deleted)
```

## CLI Alternative

You can also generate audio directly from a script file without the UI:

```bash
python podcast_audio.py --script "path/to/script.txt" --voice Kore --title "Episode Title"
```

Useful flags:
- `--dry-run` — preview chunking without calling the API
- `--resume` — retry from where it left off after a failure
- `--voice Charon` — change voice (30 options available)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `source .venv/bin/activate` first |
| `API key not valid` | Check `config.env` has your actual keys, not placeholders |
| News search returns nothing | Try shorter keywords (e.g. "Trump AI" not full sentences) |
| Voice sounds different between chunks | Already handled — voice consistency prompt is built in |
