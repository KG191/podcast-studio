#!/usr/bin/env python3
"""
podcast_app.py - Streamlit UI for end-to-end podcast production.

Launch with:
    streamlit run podcast_app.py
"""

import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image

# Import TTS pipeline from existing script
from podcast_audio import (
    ELEVENLABS_VOICES,
    GEMINI_VOICES,
    OPENAI_VOICES,
    OPENAI_MAX_CHARS,
    check_ffmpeg,
    chunk_script,
    chunk_script_by_chars,
    concatenate_wavs,
    encode_mp3,
    generate_audio_elevenlabs,
    generate_chunk_audio,
    generate_chunk_audio_openai,
    get_audio_duration,
    init_client,
    sanitize_title,
    save_wav,
)

TTS_ENGINES = [
    "ElevenLabs (Recommended)",
    "OpenAI TTS (<5 min clips)",
    "Gemini TTS (Free, <5 min clips)",
]

# ElevenLabs voice display names with descriptions
ELEVENLABS_VOICE_LABELS = [
    "Archer — Friendly young British male, podcasts",
    "Rachel — Calm American female, narration",
    "George — Deep British male, narration",
    "Adam — Deep American male, narration",
    "Antoni — Well-rounded American male",
    "Bella — Expressive American female",
    "Josh — Deep American male",
    "Domi — Strong American female",
    "Elli — Young American female",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

PODCAST_BRANDS = [
    "AI to AGI to ASI",
    "The Calm Edge",
    "Think, Expand, Grow, Thrive",
]

CATEGORIES = ["AI", "Gender bias"]

OPENAI_MODEL = "gpt-5.2"
DALLE_MODEL = "dall-e-3"
COVER_SIZE = 3000  # Final cover image size (pixels)

# ---------------------------------------------------------------------------
# API Key Loading
# ---------------------------------------------------------------------------


def load_keys():
    """Load all API keys from config.env."""
    keys = {}
    config_file = BASE_DIR / "config.env"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    # Environment variables override file
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GNEWS_API_KEY", "ELEVENLABS_API_KEY"):
        env_val = os.environ.get(var)
        if env_val:
            keys[var] = env_val
    return keys


# ---------------------------------------------------------------------------
# News Search (GNews API)
# ---------------------------------------------------------------------------


def search_news(query, api_key, max_results=3):
    """Search GNews API for top stories. Returns list of article dicts."""
    # GNews works best with keyword AND queries, not exact phrases.
    # Split user input into keywords and join with AND.
    words = query.split()
    # Remove filler words that hurt search
    stopwords = {"and", "or", "the", "of", "in", "a", "an", "for", "to", "with", "on", "at", "by"}
    keywords = [w for w in words if w.lower() not in stopwords]
    if not keywords:
        keywords = words  # fallback if everything was filtered

    search_query = " AND ".join(keywords) if len(keywords) > 1 else keywords[0]

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": search_query,
        "lang": "en",
        "sortby": "relevance",
        "max": max_results,
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])

    # If AND query is too strict and returns nothing, retry with just keywords
    if not articles and len(keywords) > 2:
        params["q"] = " ".join(keywords[:3])
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

    return articles


# ---------------------------------------------------------------------------
# Script Generation (OpenAI)
# ---------------------------------------------------------------------------


def fetch_article_text(url):
    """Fetch and extract readable text from an article URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        # Simple HTML to text: strip tags, decode entities
        import re as _re
        from html import unescape
        text = resp.text
        # Remove script/style blocks
        text = _re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        # Remove tags
        text = _re.sub(r'<[^>]+>', ' ', text)
        # Decode HTML entities
        text = unescape(text)
        # Collapse whitespace
        text = _re.sub(r'\s+', ' ', text).strip()
        # Return first ~5000 chars (enough for context, within token limits)
        return text[:5000]
    except Exception:
        return ""


INTRO_OUTRO = {
    "AI to AGI to ASI": {
        "intro": (
            "Welcome to AI to AGI to ASI — the podcast that tracks the arc of "
            "artificial intelligence from where it stands today, through the pursuit "
            "of artificial general intelligence, to the horizon of artificial "
            "superintelligence. I'm your host, and each episode we cut through the "
            "noise to bring you the stories, the science, and the stakes that matter "
            "most as humanity navigates this transformative era."
        ),
        "outro": (
            "That's all for this episode of AI to AGI to ASI. If this conversation "
            "resonated with you, share it with someone who needs to hear it — and "
            "make sure you're subscribed so you never miss an episode. The journey "
            "from AI to AGI to ASI is unfolding in real time, and we'll be right here "
            "making sense of every step. Until next time — stay informed, stay critical, "
            "and stay human."
        ),
    },
    "The Calm Edge": {
        "intro": (
            "I'm Wendy — a performance coach for ambitious professionals navigating "
            "pressure, politics, and high-stakes decisions.\n\n"
            "This is The Calm Edge — where we think clearly, act deliberately, and "
            "remain composed when it matters most.\n\n"
            "And if you're looking for direct, strategic guidance tailored to your "
            "situation, you'll find it inside the Wendy Performance Coach app on the App Store."
        ),
        "outro": (
            "This has been The Calm Edge.\n\n"
            "When pressure rises, composure is a decision.\n\n"
            "For deeper, situation-specific guidance, the Wendy Performance Coach app "
            "is available on the App Store.\n\n"
            "Until next time — stay steady."
        ),
        "rules": (
            "THE CALM EDGE — MANDATORY EPISODE RULES:\n"
            "- Identity: Every episode must reinforce composure under pressure, strategic "
            "thinking over emotional reaction, psychological depth, executive restraint, "
            "and deliberate action.\n"
            "- AVOID: motivation tone, therapy tone, productivity culture, hustle language, "
            "inspirational energy. Never use: crush it, level up, dominate, hack, explosive growth.\n"
            "- ENCOURAGE language: composure, deliberate, strategic, positioning, signal, "
            "leverage, controlled response.\n"
            "- Episode structure: The Situation → The Psychological Pattern → "
            "The Strategic Reframe → The Calm Move (one action only).\n"
            "- Tone: Measured, analytical, slightly detached, emotionally regulated. "
            "Never excited, aggressive, casual, or hype-driven.\n"
            "- The Wendy Performance Coach app is mentioned ONLY in the intro and outro. "
            "Never mid-episode. Never urgently.\n"
            "- Episode length: 8-14 minutes. No filler.\n"
            "- Each episode must answer: What is truly happening psychologically?\n"
            "- Final test: Does the episode leave the listener more composed? If not, revise."
        ),
    },
    "Think, Expand, Grow, Thrive": {
        "intro": (
            "Welcome to Think, Expand, Grow, Thrive — the podcast dedicated to "
            "broadening your perspective and empowering your next move. I'm your host, "
            "and each episode we explore the ideas, trends, and breakthroughs that "
            "help you think bigger and live bolder."
        ),
        "outro": (
            "That wraps up this episode of Think, Expand, Grow, Thrive. If this "
            "episode sparked something in you, pass it along and subscribe for more. "
            "Keep thinking, keep expanding, keep growing — and above all, keep thriving. "
            "See you next time."
        ),
    },
}


def generate_script(article_urls, podcast_brand, api_key):
    """Generate a podcast script from selected article URLs."""
    client = OpenAI(api_key=api_key)

    # Fetch actual article content so ChatGPT doesn't have to guess
    article_contents = []
    for url in article_urls:
        text = fetch_article_text(url)
        if text:
            article_contents.append(f"SOURCE ({url}):\n{text}")
        else:
            article_contents.append(f"SOURCE ({url}):\n[Could not fetch content]")

    all_articles = "\n\n---\n\n".join(article_contents)

    brand_intros = INTRO_OUTRO.get(podcast_brand, {})
    intro_text = brand_intros.get("intro", "")
    outro_text = brand_intros.get("outro", "")
    brand_rules = brand_intros.get("rules", "")

    # Adjust length target per brand
    if brand_rules:
        # The Calm Edge: 8-14 min → ~1200-2100 words
        length_instruction = "Aim for approximately 1200-2100 words (8-14 minutes at speaking pace)."
    else:
        # Default brands: 20 min → ~3000-3500 words
        length_instruction = "Aim for approximately 3000-3500 words (20 minutes at speaking pace)."

    # Build brand-specific rules section
    brand_rules_block = ""
    if brand_rules:
        brand_rules_block = (
            f"\n\nBRAND-SPECIFIC RULES — FOLLOW THESE EXACTLY:\n{brand_rules}\n"
        )

    prompt = (
        f"You are a professional podcast scriptwriter. Generate a single-host "
        f"podcast script based on the article content provided below.\n\n"
        f"STRICT RULES:\n"
        f"- Output ONLY the script. No preamble, no commentary, no notes, no disclaimers.\n"
        f"- Do NOT say you cannot access links — the article text is provided below.\n"
        f"- Do NOT add any text before or after the script.\n"
        f"- Write seamless paragraphs. No timestamps, no segment markers, no headings.\n"
        f"- The script must begin with this exact intro:\n\"{intro_text}\"\n"
        f"- The script must end with this exact outro:\n\"{outro_text}\"\n"
        f"- Between the intro and outro, deliver insightful, well-structured commentary "
        f"on the article content. Analyse the implications, provide context, and engage "
        f"the listener as a knowledgeable, articulate host.\n"
        f"- {length_instruction}\n"
        f"{brand_rules_block}\n"
        f"ARTICLE CONTENT:\n\n{all_articles}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Cover Image Generation (OpenAI DALL-E 3)
# ---------------------------------------------------------------------------


def generate_cover_images(title, podcast_brand, api_key, script_text="", count=3):
    """Generate cover image options via DALL-E 3. Returns list of PIL Images."""
    client = OpenAI(api_key=api_key)

    # First, ask GPT to distil the script into a visual concept
    visual_brief = ""
    if script_text:
        brief_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": (
                f"You are a visual art director for a podcast. Based on the script below, "
                f"describe in 2-3 sentences the key visual concept for the episode cover image. "
                f"Focus on the core subject matter, mood, and any symbolic imagery that "
                f"represents the episode's theme. Be specific and concrete — describe what "
                f"should literally be depicted.\n\n"
                f"Episode title: {title}\n"
                f"Podcast: {podcast_brand}\n\n"
                f"Script:\n{script_text[:3000]}"
            )}],
        )
        visual_brief = brief_resp.choices[0].message.content

    prompt = (
        f"Create a professional, cinematic podcast cover image. "
        f"Episode: \"{title}\" from the podcast \"{podcast_brand}\". "
        f"\n\nVisual direction: {visual_brief}\n\n"
        f"Style requirements: Bold, high-contrast, visually striking. "
        f"Use dramatic lighting and rich colour palette. "
        f"Suitable for a square podcast cover at small and large sizes. "
        f"The imagery must directly represent the episode's specific subject matter. "
        f"Do NOT include any text, letters, words, or typography in the image."
    )

    images = []
    for _ in range(count):
        response = client.images.generate(
            model=DALLE_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url",
        )
        img_url = response.data[0].url
        img_data = requests.get(img_url, timeout=30).content
        img = Image.open(io.BytesIO(img_data))
        # Upscale to 3000x3000
        img = img.resize((COVER_SIZE, COVER_SIZE), Image.LANCZOS)
        images.append(img)

    return images


# ---------------------------------------------------------------------------
# TTS Pipeline (wraps podcast_audio.py functions)
# ---------------------------------------------------------------------------


def generate_podcast_audio(script_text, voice, title, podcast_brand,
                           episode_dir, progress_callback=None,
                           tts_engine="elevenlabs", openai_key=None,
                           gemini_key=None, elevenlabs_key=None):
    """
    Full TTS pipeline. Supports three engines:
      - ElevenLabs (default): single-call for full episodes, no chunking needed
      - OpenAI TTS: chunked at 4K chars, for short clips
      - Gemini TTS: chunked by word count, free tier
    Returns path to final MP3.
    """
    check_ffmpeg()
    safe_title = sanitize_title(title)

    if tts_engine == "elevenlabs":
        # --- ElevenLabs: generates MP3 directly (single call, no chunking) ---
        if progress_callback:
            progress_callback(1, 1)

        final_mp3 = episode_dir / f"{safe_title}.mp3"
        generate_audio_elevenlabs(
            script_text, voice, elevenlabs_key, final_mp3,
        )

        if progress_callback:
            progress_callback(1, 1)
        return final_mp3

    # --- OpenAI / Gemini: chunk-based pipeline ---
    chunks_dir = episode_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    use_openai = tts_engine == "openai"

    if use_openai:
        chunks = chunk_script_by_chars(script_text)
    else:
        chunks = chunk_script(script_text)

    chunk_word_counts = {}
    for idx, chunk_text in chunks:
        chunk_word_counts[idx] = len(chunk_text.split())
        if progress_callback:
            progress_callback(idx, len(chunks))

        if use_openai:
            wav_path = chunks_dir / f"chunk_{idx:03d}.wav"
            generate_chunk_audio_openai(
                chunk_text, voice, idx, len(chunks),
                openai_key, wav_path,
            )
        else:
            client = init_client(gemini_key)
            pcm_data = generate_chunk_audio(client, chunk_text, voice, idx, len(chunks))
            save_wav(pcm_data, chunks_dir / f"chunk_{idx:03d}.wav")

        if idx < len(chunks):
            delay = 2 if use_openai else 8
            time.sleep(delay)

    full_wav = episode_dir / f"{safe_title}_full.wav"
    final_mp3 = episode_dir / f"{safe_title}.mp3"

    concatenate_wavs(chunks_dir, full_wav, chunk_word_counts=chunk_word_counts)
    encode_mp3(full_wav, final_mp3, title=title, podcast_name=podcast_brand)

    return final_mp3


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Podcast Studio", page_icon="🎙", layout="wide")
    st.title("🎙 Podcast Studio")

    # Load API keys
    keys = load_keys()
    missing = []
    if not keys.get("GNEWS_API_KEY"):
        missing.append("GNEWS_API_KEY")
    if not keys.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        st.error(f"Missing API key(s) in config.env: {', '.join(missing)}")
        st.info("Add them to config.env in the project folder, then refresh.")
        st.stop()

    if not keys.get("ELEVENLABS_API_KEY"):
        st.warning("ELEVENLABS_API_KEY not found in config.env — ElevenLabs TTS will be unavailable. "
                    "Get a key at https://elevenlabs.io/")
    if not keys.get("GEMINI_API_KEY"):
        st.info("GEMINI_API_KEY not set — Gemini TTS will be unavailable.")

    # --- Step 1: Search News ---
    st.header("1. Find Stories")
    with st.form("search_form"):
        col1, col2 = st.columns([1, 3])
        with col1:
            category = st.selectbox("Category", CATEGORIES)
        with col2:
            custom_query = st.text_input("Or enter a custom search query")
        search_clicked = st.form_submit_button("🔍 Search News")

    if search_clicked:
        query = custom_query.strip() if custom_query.strip() else category
        with st.spinner(f"Searching for: **{query}**"):
            try:
                articles = search_news(query, keys["GNEWS_API_KEY"])
                st.session_state["articles"] = articles
                st.session_state["search_query"] = query
                if not articles:
                    st.warning(f"No results found for \"{query}\". Try different keywords.")
            except Exception as e:
                st.error(f"News search failed: {e}")

    # --- Step 2: Display Stories (optional) ---
    selected = []
    if st.session_state.get("articles"):
        st.header("2. Select Stories")
        articles = st.session_state["articles"]

        for i, article in enumerate(articles):
            checked = st.checkbox(
                f"**{article['title']}**",
                key=f"article_{i}",
            )
            st.caption(f"{article.get('description', 'No description')}  \n"
                       f"[Read article]({article['url']}) — {article.get('source', {}).get('name', 'Unknown')}")
            if checked:
                selected.append(article["url"])

        st.session_state["selected_urls"] = selected

    # --- Step 3: Configuration (always visible) ---
    st.header("3. Configure Episode")
    col1, col2 = st.columns(2)
    with col1:
        podcast_brand = st.selectbox("Podcast Brand", PODCAST_BRANDS)
        tts_engine = st.selectbox("TTS Engine", TTS_ENGINES)
    with col2:
        is_elevenlabs = tts_engine.startswith("ElevenLabs")
        is_openai_tts = tts_engine.startswith("OpenAI")

        if is_elevenlabs:
            voice_label = st.selectbox("TTS Voice", ELEVENLABS_VOICE_LABELS)
            voice = voice_label.split(" — ")[0]
        elif is_openai_tts:
            voice = st.selectbox("TTS Voice", OPENAI_VOICES,
                                 index=OPENAI_VOICES.index("nova"))
        else:
            voice = st.selectbox("TTS Voice", GEMINI_VOICES,
                                 index=GEMINI_VOICES.index("Kore"))
        preview_clicked = st.button("🔊 Preview Voice (5 sec)")

    # Voice preview
    if preview_clicked:
        with st.spinner(f"Generating preview for {voice}..."):
            preview_text = (
                "Welcome to the podcast. Today we explore the latest developments "
                "in artificial intelligence and what they mean for humanity."
            )
            try:
                if is_elevenlabs:
                    from elevenlabs.client import ElevenLabs as ELClient
                    from elevenlabs import VoiceSettings
                    el_client = ELClient(api_key=keys["ELEVENLABS_API_KEY"])
                    voice_id = ELEVENLABS_VOICES.get(voice, voice)
                    audio_gen = el_client.text_to_speech.convert(
                        text=preview_text,
                        voice_id=voice_id,
                        model_id="eleven_turbo_v2_5",
                        output_format="mp3_44100_128",
                        voice_settings=VoiceSettings(
                            stability=0.72, similarity_boost=0.85,
                            style=0.0, speed=1.0,
                        ),
                    )
                    audio_bytes = b"".join(audio_gen)
                    st.audio(audio_bytes, format="audio/mp3")
                elif is_openai_tts:
                    oai_client = OpenAI(api_key=keys["OPENAI_API_KEY"])
                    response = oai_client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=voice,
                        input=preview_text,
                        instructions="Speak in a steady, calm podcast host voice.",
                        response_format="wav",
                        speed=1.0,
                    )
                    st.audio(response.content, format="audio/wav")
                else:
                    from google.genai import types as genai_types
                    preview_client = init_client(keys["GEMINI_API_KEY"])
                    response = preview_client.models.generate_content(
                        model="gemini-2.5-flash-preview-tts",
                        contents=f"Read this in a steady podcast host voice: {preview_text}",
                        config=genai_types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=genai_types.SpeechConfig(
                                voice_config=genai_types.VoiceConfig(
                                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                        voice_name=voice,
                                    )
                                )
                            ),
                        ),
                    )
                    pcm_data = response.candidates[0].content.parts[0].inline_data.data
                    import io as _io, wave as _wave
                    wav_buf = _io.BytesIO()
                    with _wave.open(wav_buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(pcm_data)
                    st.audio(wav_buf.getvalue(), format="audio/wav")
            except Exception as e:
                st.error(f"Voice preview failed: {e}")

    # Default episode title from selected article, or empty
    default_title = ""
    if selected and st.session_state.get("articles"):
        default_title = st.session_state["articles"][0]["title"]
    episode_title = st.text_input("Episode Title", value=default_title)

    st.session_state["podcast_brand"] = podcast_brand
    st.session_state["voice"] = voice
    st.session_state["tts_engine"] = tts_engine
    st.session_state["episode_title"] = episode_title

    # --- Step 4: Script (Generate from stories, Upload, or Paste) ---
    st.header("4. Podcast Script")

    script_col1, script_col2 = st.columns(2)
    with script_col1:
        if st.button("📝 Generate Script from Stories", disabled=not selected):
            with st.spinner("Generating podcast script with ChatGPT..."):
                try:
                    script = generate_script(selected, podcast_brand, keys["OPENAI_API_KEY"])
                    st.session_state["script"] = script
                except Exception as e:
                    st.error(f"Script generation failed: {e}")
        if not selected:
            st.caption("Search and select stories above, or upload/paste a script.")
    with script_col2:
        uploaded_script = st.file_uploader(
            "Or upload a script (.txt)", type=["txt"], key="script_uploader"
        )
        if uploaded_script is not None:
            st.session_state["script"] = uploaded_script.read().decode("utf-8")

    if "script" in st.session_state:
        script = st.text_area(
            "Podcast Script (edit if needed)",
            value=st.session_state["script"],
            height=400,
            key="script_editor",
        )
        st.session_state["script"] = script
        word_count = len(script.split())
        st.caption(f"{word_count} words — ~{word_count / 150:.0f} min at normal pace")

        # Download button
        st.download_button(
            "⬇ Download Script",
            data=script,
            file_name=f"{sanitize_title(episode_title)}_script.txt",
            mime="text/plain",
        )

    # --- Step 5: Cover Image (independent from podcast generation) ---
    if "script" in st.session_state:
        st.header("5. Cover Image")

        if st.button("🎨 Generate 3 Cover Options"):
            with st.spinner("Generating cover images with DALL-E 3..."):
                try:
                    images = generate_cover_images(
                        st.session_state.get("episode_title", "Untitled"),
                        st.session_state.get("podcast_brand", PODCAST_BRANDS[0]),
                        keys["OPENAI_API_KEY"],
                        script_text=st.session_state.get("script", ""),
                    )
                    st.session_state["cover_images"] = images
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

        if "cover_images" in st.session_state:
            cols = st.columns(3)
            for i, img in enumerate(st.session_state["cover_images"]):
                with cols[i]:
                    st.image(img, caption=f"Option {i + 1}", use_container_width=True)

            cover_choice = st.radio(
                "Select cover image",
                options=[1, 2, 3],
                horizontal=True,
                key="cover_choice",
            )
            st.session_state["selected_cover"] = cover_choice - 1

    # --- Step 6: Generate Podcast (independent from cover image) ---
    if "script" in st.session_state:
        st.header("6. Generate Podcast")

        # Determine engine
        engine_selection = st.session_state.get("tts_engine", TTS_ENGINES[0])
        is_elevenlabs = engine_selection.startswith("ElevenLabs")
        is_openai = engine_selection.startswith("OpenAI")
        if is_elevenlabs:
            engine_label = "ElevenLabs"
            engine_key = "elevenlabs"
        elif is_openai:
            engine_label = "OpenAI TTS"
            engine_key = "openai"
        else:
            engine_label = "Gemini TTS"
            engine_key = "gemini"
        st.caption(f"Engine: **{engine_label}** | Voice: **{st.session_state.get('voice', 'Archer')}**")

        if st.button("🎙 Generate Podcast Audio"):
            title = st.session_state.get("episode_title", "Untitled")
            brand = st.session_state.get("podcast_brand", PODCAST_BRANDS[0])
            sel_voice = st.session_state.get("voice", "Archer")
            script_text = st.session_state["script"]

            # Create episode directory
            safe = sanitize_title(title)
            episode_dir = BASE_DIR / safe
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Save script
            script_path = episode_dir / "script.txt"
            script_path.write_text(script_text, encoding="utf-8")

            # Save cover image if one was selected
            if "cover_images" in st.session_state:
                cover_idx = st.session_state.get("selected_cover", 0)
                cover_img = st.session_state["cover_images"][cover_idx]
                cover_path = episode_dir / "cover.png"
                cover_img.save(str(cover_path), "PNG")

            # Generate audio
            progress_text = ("Generating full episode audio..." if is_elevenlabs
                             else f"Generating audio with {engine_label}...")
            progress_bar = st.progress(0, text=progress_text)

            def update_progress(current, total):
                progress_bar.progress(
                    current / total,
                    text=(f"Generating audio..." if total == 1
                          else f"Generating chunk {current}/{total}..."),
                )

            try:
                mp3_path = generate_podcast_audio(
                    script_text, sel_voice, title, brand,
                    episode_dir,
                    progress_callback=update_progress,
                    tts_engine=engine_key,
                    openai_key=keys.get("OPENAI_API_KEY") if is_openai else None,
                    gemini_key=keys.get("GEMINI_API_KEY") if engine_key == "gemini" else None,
                    elevenlabs_key=keys.get("ELEVENLABS_API_KEY") if is_elevenlabs else None,
                )

                progress_bar.progress(1.0, text="Done!")

                # Save metadata
                duration = get_audio_duration(mp3_path)
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                metadata = {
                    "title": title,
                    "podcast": brand,
                    "voice": sel_voice,
                    "duration": f"{minutes}:{seconds:02d}",
                    "mp3_file": mp3_path.name,
                    "cover_file": "cover.png" if "cover_images" in st.session_state else None,
                    "created": datetime.now().isoformat(),
                }
                with open(episode_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Success
                st.success(f"Podcast generated successfully!")
                st.markdown(f"""
                **Episode**: {title}
                **Duration**: {minutes}:{seconds:02d}
                **Files saved to**: `{episode_dir}`
                - `{mp3_path.name}` — upload to RSS.com
                - `cover.png` — upload as episode artwork
                - `script.txt` — episode script
                """)

                # Audio player
                audio_bytes = mp3_path.read_bytes()
                st.audio(audio_bytes, format="audio/mp3")

                # Generate podcast preview summary (max 4000 chars)
                st.header("7. Podcast Preview Summary")
                with st.spinner("Generating preview summary..."):
                    try:
                        oai_client = OpenAI(api_key=keys["OPENAI_API_KEY"])
                        summary_resp = oai_client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[{"role": "user", "content": (
                                f"Write a compelling podcast episode preview/description "
                                f"for the following podcast script. The preview should "
                                f"hook listeners and summarise the key topics covered. "
                                f"Keep it under 4000 characters.\n\n"
                                f"Podcast: {brand}\n"
                                f"Episode: {title}\n\n"
                                f"{script_text}"
                            )}],
                        )
                        summary = summary_resp.choices[0].message.content
                        # Enforce 4000 char limit
                        if len(summary) > 4000:
                            summary = summary[:3997] + "..."

                        st.text_area(
                            "Preview summary (copy for RSS.com episode description)",
                            value=summary,
                            height=250,
                        )
                        st.caption(f"{len(summary)} / 4000 characters")

                        # Save to episode folder
                        summary_path = episode_dir / "preview_summary.txt"
                        summary_path.write_text(summary, encoding="utf-8")
                        st.caption(f"Saved to `{summary_path}`")
                    except Exception as e:
                        st.error(f"Summary generation failed: {e}")

            except Exception as e:
                st.error(f"Audio generation failed: {e}")


if __name__ == "__main__":
    main()
