#!/usr/bin/env python3
"""
podcast_audio.py - Generate podcast audio from a text script using Gemini TTS.

Usage:
    python podcast_audio.py --script script.txt --voice Kore --title "Episode Title"
    python podcast_audio.py --script script.txt --voice Kore --title "Episode Title" --resume
    python podcast_audio.py --script script.txt --title "Episode Title" --dry-run

Requires:
    pip3 install google-genai
    ffmpeg (via Homebrew: brew install ffmpeg)
    API key from https://aistudio.google.com/
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# --- Gemini TTS ---
GEMINI_MODEL_ID = "gemini-2.5-flash-preview-tts"
SAMPLE_RATE = 24000       # Gemini outputs 24kHz PCM
SAMPLE_WIDTH = 2          # 16-bit (2 bytes per sample)
CHANNELS = 1              # Mono
GEMINI_MAX_WORDS = 800    # ~5.3 min at 150 wpm (safety margin below ~5:27 API cutoff)
GEMINI_DELAY = 8          # Seconds between API calls (rate limit safety)

GEMINI_VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede",
    "Autonoe", "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome",
    "Fenrir", "Gacrux", "Iapetus", "Kore", "Laomedeia", "Leda", "Orus",
    "Puck", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager",
    "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr",
    "Zubenelgenubi",
]

# --- ElevenLabs TTS (primary engine — full episodes, no chunking) ---
ELEVENLABS_MODEL = "eleven_turbo_v2_5"  # 40,000 char limit = full episode in one call
ELEVENLABS_MAX_CHARS = 39000            # Safety margin below 40,000 API limit
# Premade voices: name -> voice_id
ELEVENLABS_VOICES = {
    "Archer": "L0Dsvb3SLTyegXwtm47J",     # Friendly young British male, podcasts
    "Rachel": "21m00Tcm4TlvDq8ikWAM",     # Calm American female, narration
    "George": "JBFqnCBsd6RMkjVDRZzb",     # Deep British male, narration
    "Adam": "pNInz6obpgDQGcFmaJgB",       # Deep American male, narration
    "Antoni": "ErXwobaYiN019PkySvjV",     # Well-rounded American male
    "Bella": "EXAVITQu4vr4xnSDxMaL",      # Expressive American female
    "Josh": "TxGEqnHWrfWFTfGW9XjX",       # Deep American male
    "Domi": "AZnzlk1XvdvUeBnXmlld",       # Strong American female
    "Elli": "MF3mGyEYCl7XYWbV9V6O",       # Young American female
    "Liberty X": "iBo5PWT1qLiEyqhM7TrG",  # The Calm Edge voice
    "Beth": "8N2ng9i2uiUWqstgmWlH",       # The Calm Edge voice
    "Emma": "56bWURjYFHyYyVf490Dp",       # The Calm Edge voice
    "Serena": "RGb96Dcl0k5eVje8EBch",     # The Calm Edge voice
}

# --- OpenAI TTS (short clips <5 min) ---
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_MAX_CHARS = 3800   # Safety margin below 4096 API hard limit
OPENAI_VOICES = [
    "alloy", "ash", "ballad", "cedar", "coral", "echo",
    "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse",
]

# --- Shared ---
# Legacy aliases for backward compatibility
MODEL_ID = GEMINI_MODEL_ID
VOICES = GEMINI_VOICES
MAX_WORDS_PER_CHUNK = GEMINI_MAX_WORDS
DELAY_BETWEEN_CHUNKS = GEMINI_DELAY

OUTPUT_BITRATE = "128k"
OUTPUT_SAMPLE_RATE = 44100
PODCAST_NAME = "AI to AGI to ASI"

# ---------------------------------------------------------------------------
# API Key Loading
# ---------------------------------------------------------------------------


def load_api_key():
    """Load Gemini API key from environment variable or config.env file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    script_dir = Path(__file__).parent
    config_file = script_dir / "config.env"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: No API key found.")
    print("Either set the GEMINI_API_KEY environment variable, or create config.env:")
    print('  GEMINI_API_KEY=your_key_here')
    print("\nGet a free key at: https://aistudio.google.com/")
    sys.exit(1)


def init_client(api_key):
    """Initialize the Gemini client."""
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        print("ERROR: google-genai package not installed.")
        print("Install with: pip3 install google-genai")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------


def chunk_script(text, max_words=MAX_WORDS_PER_CHUNK):
    """
    Split script text into chunks suitable for Gemini TTS.

    Strategy:
    1. Split into paragraphs (on blank lines)
    2. Accumulate paragraphs into chunks up to max_words
    3. If a single paragraph exceeds max_words, split at sentence boundaries

    Returns list of (chunk_index, chunk_text) tuples (1-indexed).
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > max_words:
            # Flush current accumulated chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split oversized paragraph at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_word_count = 0
            for sentence in sentences:
                s_words = len(sentence.split())
                if sent_word_count + s_words > max_words and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_word_count = 0
                sent_chunk.append(sentence)
                sent_word_count += s_words
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))

        elif current_word_count + para_words > max_words:
            # Current chunk is full, start a new one
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words

        else:
            current_chunk.append(para)
            current_word_count += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [(i + 1, chunk) for i, chunk in enumerate(chunks)]


# ---------------------------------------------------------------------------
# Gemini TTS Audio Generation
# ---------------------------------------------------------------------------


def generate_chunk_audio(client, text, voice_name, chunk_index, total_chunks,
                         max_retries=3):
    """
    Call Gemini TTS API for a single text chunk.
    Returns raw PCM bytes (16-bit, 24kHz, mono).
    Auto-retries on rate limit (429) errors.
    """
    from google.genai import types

    word_count = len(text.split())
    print(f"  Generating chunk {chunk_index}/{total_chunks} "
          f"({word_count} words)...", end=" ", flush=True)

    # Strong voice direction prompt to lock down consistency across chunks.
    # Every chunk gets the identical prompt so the TTS model enters the same
    # vocal state each time — same register, pitch, pace, and energy.
    voice_prompt = (
        "You are narrating a podcast. Follow these strict rules with no exceptions:\n"
        "- Speak at exactly 150 words per minute. Never speed up or slow down.\n"
        "- Use a calm, measured, professional tone throughout.\n"
        "- Maintain the exact same vocal register, pitch, and energy level from "
        "the first word to the last.\n"
        "- Do not vary your intonation pattern, emotional expression, or speaking style.\n"
        "- Do not add dramatic pauses, excitement, emphasis changes, or vocal fry.\n"
        "- Do not whisper, raise your voice, or change your delivery in any way.\n"
        "- Keep your voice steady and monotonously consistent. Flat and even.\n"
        "- This is one continuous narration. Read it like a calm newsreader.\n\n"
    )
    prompted_text = voice_prompt + text

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompted_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                ),
            )

            # Extract PCM audio data from response
            part = response.candidates[0].content.parts[0]
            pcm_data = part.inline_data.data

            duration_sec = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
            print(f"OK ({duration_sec:.1f}s)")

            return pcm_data

        except Exception as e:
            error_str = str(e)
            # Auto-retry on rate limit errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay from error if available
                import re as _re
                delay_match = _re.search(r'retry\s*(?:in|Delay["\s:]*)\s*(\d+)', error_str, _re.IGNORECASE)
                wait_secs = int(delay_match.group(1)) + 5 if delay_match else 45

                if attempt < max_retries:
                    print(f"RATE LIMITED (waiting {wait_secs}s, attempt {attempt}/{max_retries})")
                    time.sleep(wait_secs)
                    print(f"  Retrying chunk {chunk_index}/{total_chunks}...", end=" ", flush=True)
                    continue

            print("FAILED")
            raise RuntimeError(f"Chunk {chunk_index} generation failed: {e}")


# ---------------------------------------------------------------------------
# OpenAI TTS Audio Generation (primary engine)
# ---------------------------------------------------------------------------


def chunk_script_by_chars(text, max_chars=OPENAI_MAX_CHARS):
    """
    Split script text into chunks by character limit at sentence boundaries.
    Used for OpenAI TTS which has a 4096-character limit per call.
    Returns list of (chunk_index, chunk_text) tuples (1-indexed).
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_chars = 0

    for para in paragraphs:
        para_chars = len(para)

        if para_chars > max_chars:
            # Flush current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chars = 0

            # Split oversized paragraph at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_chars = 0
            for sentence in sentences:
                s_chars = len(sentence)
                if sent_chars + s_chars + 1 > max_chars and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_chars = 0
                sent_chunk.append(sentence)
                sent_chars += s_chars + 1
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))

        elif current_chars + para_chars + 2 > max_chars:
            # Current chunk is full
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_chars = para_chars

        else:
            current_chunk.append(para)
            current_chars += para_chars + 2  # +2 for \n\n separator

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [(i + 1, chunk) for i, chunk in enumerate(chunks)]


def generate_chunk_audio_openai(text, voice_name, chunk_index, total_chunks,
                                api_key, output_wav, max_retries=3):
    """
    Generate audio for a single text chunk using OpenAI TTS.
    Requests WAV output so it feeds directly into the existing
    crossfade + mastering pipeline.
    Returns the output WAV path.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    char_count = len(text)
    word_count = len(text.split())
    print(f"  Generating chunk {chunk_index}/{total_chunks} "
          f"({word_count} words, {char_count} chars)...", end=" ", flush=True)

    # Voice direction via the instructions parameter — keeps delivery
    # identical across every chunk so there are no audible seams.
    voice_instructions = (
        "You are narrating a podcast. Follow these rules strictly: "
        "Speak at a steady, calm, measured pace — approximately 150 words per minute. "
        "Maintain the exact same vocal register, pitch, energy, and tone throughout. "
        "Do not vary your intonation, emotion, or delivery style. "
        "No dramatic pauses, no excitement, no whispering, no raised voice. "
        "Read evenly and steadily like a calm, professional newsreader."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=voice_name,
                input=text,
                instructions=voice_instructions,
                response_format="wav",
                speed=1.0,
            )

            # Save WAV bytes directly
            Path(output_wav).write_bytes(response.content)

            duration = get_audio_duration(output_wav)
            print(f"OK ({duration:.1f}s)")
            return output_wav

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_secs = 30
                if attempt < max_retries:
                    print(f"RATE LIMITED (waiting {wait_secs}s, attempt {attempt}/{max_retries})")
                    time.sleep(wait_secs)
                    print(f"  Retrying chunk {chunk_index}/{total_chunks}...", end=" ", flush=True)
                    continue

            print("FAILED")
            raise RuntimeError(f"Chunk {chunk_index} generation failed: {e}")


# ---------------------------------------------------------------------------
# ElevenLabs TTS Audio Generation (primary — full episodes, no chunking)
# ---------------------------------------------------------------------------


def generate_audio_elevenlabs(text, voice_name, api_key, output_mp3,
                              max_retries=3, podcast_brand=None):
    """
    Generate audio using ElevenLabs TTS.

    Uses eleven_turbo_v2_5 which handles up to 40,000 characters per call —
    a full 15-20 min podcast episode in ONE call with zero chunking.
    For scripts exceeding 40K chars, automatically chunks with voice
    continuity via previous_request_ids.

    Returns path to the output MP3 file.
    """
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    client = ElevenLabs(api_key=api_key)

    # Resolve voice name to ID
    voice_id = ELEVENLABS_VOICES.get(voice_name, voice_name)

    # The Calm Edge: Wendy Script Replication Framework ElevenLabs settings
    # Stability: 70-80, Clarity: 75-85, Style Exaggeration: 0-10 (LOW)
    if podcast_brand == "The Calm Edge":
        voice_settings = VoiceSettings(
            stability=0.75,           # Wendy framework: 70-80 range
            similarity_boost=0.80,    # Wendy framework: 75-85 (clarity)
            style=0.05,              # Wendy framework: 0-10 (LOW) — controlled authority
            use_speaker_boost=True,
            speed=1.0,
        )
    else:
        voice_settings = VoiceSettings(
            stability=0.72,           # High consistency — steady podcast narration
            similarity_boost=0.85,    # Close to original voice
            style=0.0,                # Neutral — no exaggerated expression
            use_speaker_boost=True,   # Enhanced clarity
            speed=1.0,                # Normal pace
        )

    char_count = len(text)
    word_count = len(text.split())

    if char_count <= ELEVENLABS_MAX_CHARS:
        # --- Single-call generation (the main advantage) ---
        print(f"  Generating full audio in one call "
              f"({word_count} words, {char_count} chars)...", end=" ", flush=True)

        for attempt in range(1, max_retries + 1):
            try:
                audio_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=ELEVENLABS_MODEL,
                    output_format="mp3_44100_128",
                    voice_settings=voice_settings,
                )
                audio_bytes = b"".join(audio_gen)
                Path(output_mp3).write_bytes(audio_bytes)

                duration = get_audio_duration(output_mp3)
                size_mb = len(audio_bytes) / (1024 * 1024)
                print(f"OK ({duration:.1f}s, {size_mb:.1f} MB)")
                return output_mp3

            except Exception as e:
                error_str = str(e)
                # Quota exceeded — give a clear, actionable message
                if "quota_exceeded" in error_str or "credits remaining" in error_str.lower():
                    print("QUOTA EXCEEDED")
                    # Extract remaining/required credits from error if possible
                    import re as _re
                    remaining = _re.search(r'(\d+)\s*credits\s*remaining', error_str)
                    required = _re.search(r'(\d+)\s*credits\s*are\s*required', error_str)
                    msg = "ElevenLabs quota exceeded."
                    if remaining and required:
                        msg += (f" You have {remaining.group(1)} credits remaining "
                                f"but need {required.group(1)}.")
                    msg += (" Upgrade your plan at https://elevenlabs.io/pricing "
                            "or switch to OpenAI TTS / Gemini TTS.")
                    raise RuntimeError(msg)
                if "429" in error_str or "rate" in error_str.lower():
                    wait_secs = 30
                    if attempt < max_retries:
                        print(f"RATE LIMITED (waiting {wait_secs}s, attempt {attempt}/{max_retries})")
                        time.sleep(wait_secs)
                        continue
                print("FAILED")
                raise RuntimeError(f"ElevenLabs generation failed: {e}")
    else:
        # --- Multi-call with voice continuity (rare — scripts over 40K chars) ---
        print(f"  Script is {char_count} chars — splitting into chunks...")
        chunks = _split_text_elevenlabs(text, ELEVENLABS_MAX_CHARS)
        all_audio = b""

        for i, chunk in enumerate(chunks):
            print(f"  Generating chunk {i + 1}/{len(chunks)} "
                  f"({len(chunk)} chars)...", end=" ", flush=True)

            for attempt in range(1, max_retries + 1):
                try:
                    kwargs = {}
                    # Pass next chunk preview for prosody continuity
                    if i + 1 < len(chunks):
                        kwargs["next_text"] = chunks[i + 1][:200]

                    audio_gen = client.text_to_speech.convert(
                        text=chunk,
                        voice_id=voice_id,
                        model_id=ELEVENLABS_MODEL,
                        output_format="mp3_44100_128",
                        voice_settings=voice_settings,
                        **kwargs,
                    )
                    chunk_audio = b"".join(audio_gen)
                    all_audio += chunk_audio
                    print(f"OK ({len(chunk_audio) / 1024:.0f} KB)")
                    break

                except Exception as e:
                    error_str = str(e)
                    if "quota_exceeded" in error_str or "credits remaining" in error_str.lower():
                        print("QUOTA EXCEEDED")
                        raise RuntimeError(
                            "ElevenLabs quota exceeded. Upgrade at "
                            "https://elevenlabs.io/pricing or switch to "
                            "OpenAI TTS / Gemini TTS."
                        )
                    if ("429" in error_str or "rate" in error_str.lower()) and attempt < max_retries:
                        print(f"RATE LIMITED (waiting 30s, attempt {attempt}/{max_retries})")
                        time.sleep(30)
                        continue
                    print("FAILED")
                    raise RuntimeError(f"ElevenLabs chunk {i + 1} failed: {e}")

            if i < len(chunks) - 1:
                time.sleep(1)

        Path(output_mp3).write_bytes(all_audio)
        duration = get_audio_duration(output_mp3)
        size_mb = len(all_audio) / (1024 * 1024)
        print(f"  Assembled {len(chunks)} chunks -> {Path(output_mp3).name} "
              f"({duration:.1f}s, {size_mb:.1f} MB)")
        return output_mp3


def _split_text_elevenlabs(text, max_chars):
    """Split text at sentence boundaries for ElevenLabs chunking."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# WAV File Handling
# ---------------------------------------------------------------------------


def save_wav(pcm_data, filepath):
    """Save raw PCM data as a WAV file."""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)


def get_completed_chunks(chunks_dir):
    """Return set of chunk indices that already have valid WAV files."""
    completed = set()
    if chunks_dir.exists():
        for f in chunks_dir.glob("chunk_*.wav"):
            match = re.match(r"chunk_(\d+)\.wav", f.name)
            if match and f.stat().st_size > 44:  # WAV header is 44 bytes
                completed.add(int(match.group(1)))
    return completed


# ---------------------------------------------------------------------------
# ffmpeg Operations
# ---------------------------------------------------------------------------


def check_ffmpeg():
    """Verify ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("ERROR: ffmpeg is installed but returned an error.")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found.")
        print("Install with: brew install ffmpeg")
        sys.exit(1)


def get_chunk_wpm(wav_path, word_count):
    """Calculate words-per-minute for a chunk to detect speed anomalies."""
    duration = get_audio_duration(wav_path)
    if duration > 0 and word_count > 0:
        return (word_count / duration) * 60
    return 150.0  # default assumption


def normalize_chunk_tempo(wav_path, target_wpm, word_count):
    """
    Normalize a chunk's tempo to match a target WPM.
    Corrects ANY deviation above 2% (previously 15%).
    Does NOT apply loudnorm — that's done once on the final audio.
    """
    actual_wpm = get_chunk_wpm(wav_path, word_count)
    if actual_wpm <= 0 or word_count <= 0:
        return

    ratio = actual_wpm / target_wpm

    # Skip only if extremely close (within 2%)
    if abs(ratio - 1.0) < 0.02:
        return

    tempo = max(0.5, min(2.0, ratio))
    normalized_path = wav_path.parent / f"{wav_path.stem}_norm.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", f"atempo={tempo:.4f}",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-sample_fmt", "s16",
        str(normalized_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        normalized_path.replace(wav_path)
        print(f"    {wav_path.stem}: {actual_wpm:.0f} -> {target_wpm:.0f} WPM (x{tempo:.3f})")
    else:
        if normalized_path.exists():
            normalized_path.unlink()


def concatenate_wavs(chunks_dir, output_wav, chunk_word_counts=None):
    """
    Assemble chunk WAV files into a single WAV with voice consistency fixes:
      1. Normalize every chunk's tempo to the MEDIAN WPM across all chunks
      2. Crossfade between chunks (200ms) to smooth transitions
      3. Final mastering pass: compression + loudnorm on the full audio
    """
    wav_files = sorted(
        chunks_dir.glob("chunk_*.wav"),
        key=lambda f: int(re.search(r'(\d+)', f.stem).group()),
    )

    if not wav_files:
        raise RuntimeError("No WAV chunks found to concatenate")

    # --- Step 1: Compute median WPM and normalize all chunks to it ---
    if chunk_word_counts:
        wpms = []
        for wav_path in wav_files:
            idx = int(re.search(r'(\d+)', wav_path.stem).group())
            wc = chunk_word_counts.get(idx, 0)
            if wc > 0:
                wpm = get_chunk_wpm(wav_path, wc)
                if wpm > 0:
                    wpms.append(wpm)

        if wpms:
            wpms_sorted = sorted(wpms)
            median_wpm = wpms_sorted[len(wpms_sorted) // 2]
            print(f"  Normalizing all chunks to median pace: {median_wpm:.0f} WPM")

            for wav_path in wav_files:
                idx = int(re.search(r'(\d+)', wav_path.stem).group())
                wc = chunk_word_counts.get(idx, 0)
                if wc > 0:
                    normalize_chunk_tempo(wav_path, median_wpm, wc)

    # --- Step 2: Crossfade concatenation (200ms blend at each boundary) ---
    crossfade_sec = 0.2

    if len(wav_files) == 1:
        # Single chunk — just copy
        shutil.copy2(wav_files[0], output_wav)
    else:
        # Build ffmpeg crossfade filter chain
        inputs = []
        for wav in wav_files:
            inputs.extend(["-i", str(wav.resolve())])

        filter_parts = []
        for i in range(len(wav_files) - 1):
            src = "[0]" if i == 0 else f"[a{i - 1:02d}]"
            dst = "[out]" if i == len(wav_files) - 2 else f"[a{i:02d}]"
            filter_parts.append(
                f"{src}[{i + 1}]acrossfade=d={crossfade_sec}:c1=tri:c2=tri{dst}"
            )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-sample_fmt", "s16",
            str(output_wav),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to hard concat if crossfade fails (e.g., very short chunks)
            print(f"  Crossfade failed, falling back to hard concat: {result.stderr[:200]}")
            list_file = chunks_dir / "concat_list.txt"
            with open(list_file, "w") as f:
                for wav in wav_files:
                    f.write(f"file '{wav.resolve()}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(output_wav),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            list_file.unlink()
            if result.returncode != 0:
                raise RuntimeError(f"WAV concatenation failed: {result.stderr}")

    print(f"  Assembled {len(wav_files)} chunks with {int(crossfade_sec * 1000)}ms crossfade -> {output_wav.name}")

    # --- Step 3: Final mastering pass (compression + loudnorm on full audio) ---
    # Light compression evens out volume/energy differences between chunks.
    # Single-pass loudnorm on the FULL file is more consistent than per-chunk.
    mastered_path = output_wav.parent / f"{output_wav.stem}_mastered.wav"
    master_filters = (
        "acompressor=threshold=-25dB:ratio=3:attack=200:release=1000:makeup=2,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(output_wav),
        "-af", master_filters,
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-sample_fmt", "s16",
        str(mastered_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        mastered_path.replace(output_wav)
        print("  Final mastering pass applied (compression + loudnorm)")
    else:
        if mastered_path.exists():
            mastered_path.unlink()
        print(f"  Warning: mastering pass failed, using unmastered audio")


def encode_mp3(input_wav, output_mp3, title, podcast_name=None):
    """Encode WAV to MP3 with podcast-standard settings and ID3 metadata."""
    brand = podcast_name or PODCAST_NAME
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-ar", str(OUTPUT_SAMPLE_RATE),
        "-ac", "1",
        "-b:a", OUTPUT_BITRATE,
        "-codec:a", "libmp3lame",
        "-metadata", f"title={title}",
        "-metadata", f"artist={brand}",
        "-metadata", f"album={brand}",
        "-metadata", "genre=Podcast",
        "-metadata", f"date={datetime.now().strftime('%Y')}",
        str(output_mp3),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MP3 encoding failed: {result.stderr}")

    size_mb = output_mp3.stat().st_size / (1024 * 1024)
    print(f"  Encoded MP3: {output_mp3.name} ({size_mb:.1f} MB)")


def get_audio_duration(filepath):
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Episode Directory Management
# ---------------------------------------------------------------------------


def sanitize_title(title):
    """Convert title to a filesystem-safe name."""
    safe = re.sub(r'[^\w\s-]', '', title).strip()
    return re.sub(r'\s+', '_', safe)


def create_episode_dir(base_dir, title):
    """Create structured episode directory. Returns (episode_dir, chunks_dir)."""
    episode_dir = Path(base_dir) / sanitize_title(title)
    chunks_dir = episode_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir, chunks_dir


def save_metadata(episode_dir, title, voice, script_path, num_chunks, mp3_name):
    """Save episode metadata as JSON."""
    metadata = {
        "title": title,
        "podcast": PODCAST_NAME,
        "voice": voice,
        "model": MODEL_ID,
        "source_script": str(script_path),
        "num_chunks": num_chunks,
        "mp3_file": mp3_name,
        "created": datetime.now().isoformat(),
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate podcast audio from a text script using Gemini TTS.",
        epilog='Example: python podcast_audio.py --script script.txt --voice Kore '
               '--title "The Ethics of AI Autonomy"',
    )
    parser.add_argument("--script", required=True,
                        help="Path to the script .txt file")
    parser.add_argument("--voice", default="Kore", choices=VOICES,
                        help="TTS voice name (default: Kore)")
    parser.add_argument("--title", required=True,
                        help="Episode title")
    parser.add_argument("--output-dir", default=None,
                        help="Base output directory (default: script's directory)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previously generated chunks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview chunking without calling the API")
    parser.add_argument("--max-words", type=int, default=MAX_WORDS_PER_CHUNK,
                        help=f"Max words per chunk (default: {MAX_WORDS_PER_CHUNK})")

    args = parser.parse_args()

    # --- Read script ---
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: Script file not found: {script_path}")
        sys.exit(1)

    script_text = script_path.read_text(encoding="utf-8")
    total_words = len(script_text.split())
    est_total_min = total_words / 150
    print(f"Script: {script_path.name} ({total_words} words, ~{est_total_min:.0f} min)")

    # --- Chunk the script ---
    chunks = chunk_script(script_text, max_words=args.max_words)
    print(f"Split into {len(chunks)} chunks:")
    for idx, chunk_text in chunks:
        wc = len(chunk_text.split())
        est = wc / 150
        print(f"  Chunk {idx}: {wc} words (~{est:.1f} min)")

    if args.dry_run:
        total_est = sum(len(c.split()) for _, c in chunks) / 150
        print(f"\nTotal estimated duration: ~{total_est:.0f} min")
        print(f"API calls needed: {len(chunks)}")
        print(f"Estimated processing time: ~{len(chunks) * 45 + (len(chunks)-1) * DELAY_BETWEEN_CHUNKS:.0f}s")
        print("\n[Dry run complete — no API calls made]")
        return

    # --- Verify ffmpeg ---
    check_ffmpeg()

    # --- Setup directories ---
    base_dir = args.output_dir or str(script_path.parent)
    episode_dir, chunks_dir = create_episode_dir(base_dir, args.title)

    # Copy script to episode directory
    dest_script = episode_dir / "script.txt"
    if not dest_script.exists():
        shutil.copy2(script_path, dest_script)

    print(f"Output: {episode_dir}")

    # --- Check for resume ---
    completed = get_completed_chunks(chunks_dir) if args.resume else set()
    if completed:
        remaining = len(chunks) - len(completed)
        print(f"Resuming: {len(completed)} chunks done, {remaining} remaining")

    # --- Initialize API ---
    api_key = load_api_key()
    client = init_client(api_key)

    # --- Generate audio chunks ---
    print(f"\nGenerating audio (voice: {args.voice})...")
    failed_chunks = []
    chunk_word_counts = {}

    for idx, chunk_text in chunks:
        chunk_word_counts[idx] = len(chunk_text.split())
        if idx in completed:
            print(f"  Chunk {idx}/{len(chunks)}: SKIPPED (already exists)")
            continue

        try:
            pcm_data = generate_chunk_audio(
                client, chunk_text, args.voice, idx, len(chunks)
            )
            save_wav(pcm_data, chunks_dir / f"chunk_{idx:03d}.wav")
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            failed_chunks.append(idx)

        # Rate limit delay between chunks
        if idx < len(chunks):
            time.sleep(DELAY_BETWEEN_CHUNKS)

    if failed_chunks:
        print(f"\nERROR: {len(failed_chunks)} chunk(s) failed: {failed_chunks}")
        print("Fix the issue and re-run with --resume to retry failed chunks.")
        sys.exit(1)

    # --- Concatenate and encode ---
    print("\nAssembling final audio...")

    safe_title = sanitize_title(args.title)
    full_wav = episode_dir / f"{safe_title}_full.wav"
    final_mp3 = episode_dir / f"{safe_title}.mp3"

    concatenate_wavs(chunks_dir, full_wav, chunk_word_counts=chunk_word_counts)
    encode_mp3(full_wav, final_mp3, title=args.title)

    # --- Save metadata ---
    save_metadata(episode_dir, args.title, args.voice,
                  script_path, len(chunks), final_mp3.name)

    # --- Final report ---
    duration = get_audio_duration(final_mp3)
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    size_mb = final_mp3.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 55}")
    print(f"  DONE!")
    print(f"  Episode:  {args.title}")
    print(f"  Voice:    {args.voice}")
    print(f"  Duration: {minutes}:{seconds:02d}")
    print(f"  File:     {final_mp3}")
    print(f"  Size:     {size_mb:.1f} MB")
    print(f"{'=' * 55}")
    print(f"\nReady to upload to RSS.com!")


if __name__ == "__main__":
    main()
