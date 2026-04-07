from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from faster_whisper import WhisperModel

from config import EPISODES_PATH

INPUT_PATH = EPISODES_PATH
OUTPUT_PATH = INPUT_PATH
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "podcast-pipeline/0.1"
FASTER_WHISPER_MODEL_NAME = os.getenv("FASTER_WHISPER_MODEL", "base")
FASTER_WHISPER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu")
FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")

TRANSCRIPT_HEADING_PATTERNS = [
    re.compile(r"^episode transcript$", re.IGNORECASE),
    re.compile(r"^transcript$", re.IGNORECASE),
    re.compile(r"^read transcript$", re.IGNORECASE),
]

_WHISPER_MODEL: WhisperModel | None = None


def load_episodes(input_path: Path = INPUT_PATH) -> list[dict[str, Any]]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_episodes(
    episodes: list[dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    output_path.write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated episodes in {output_path}")


def episode_has_transcript(episode: dict[str, Any]) -> bool:
    return bool(str(episode.get("transcript") or "").strip())


def fetch_episode_page(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    return response.text


def infer_audio_suffix(audio_url: str) -> str:
    parsed_url = urlparse(audio_url)
    decoded_path = unquote(parsed_url.path)
    suffix = Path(decoded_path).suffix
    return suffix if suffix else ".mp3"


def download_audio_file(audio_url: str) -> str | None:
    temp_path: str | None = None

    try:
        response = requests.get(
            audio_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=infer_audio_suffix(audio_url),
        ) as temp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
            temp_path = temp_file.name
    except requests.RequestException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

    return temp_path


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL

    if _WHISPER_MODEL is None:
        model_kwargs: dict[str, Any] = {}
        if FASTER_WHISPER_DEVICE != "auto":
            model_kwargs["device"] = FASTER_WHISPER_DEVICE
        if FASTER_WHISPER_COMPUTE_TYPE != "default":
            model_kwargs["compute_type"] = FASTER_WHISPER_COMPUTE_TYPE

        _WHISPER_MODEL = WhisperModel(
            FASTER_WHISPER_MODEL_NAME,
            **model_kwargs,
        )

    return _WHISPER_MODEL


def transcribe_audio_with_whisper(audio_url: str) -> str | None:
    audio_path = download_audio_file(audio_url)
    if not audio_path:
        return None

    try:
        segments, _info = get_whisper_model().transcribe(audio_path,
                                                         language="en",
                                                         vad_filter=True,
                                                         beam_size=1,
                                                         best_of=1)
        transcript = clean_transcript_text(" ".join(segment.text for segment in segments))
    except Exception as e:
        print(f"Whisper transcription failed for {audio_url}: {e}")
        return None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return transcript if transcript and len(transcript) >= 20 else None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def text_matches_transcript_heading(text: str) -> bool:
    normalized = " ".join(text.split())
    return any(pattern.match(normalized) for pattern in TRANSCRIPT_HEADING_PATTERNS)


def clean_transcript_text(raw_text: str, heading_text: str | None = None) -> str | None:
    text = normalize_whitespace(raw_text)
    if heading_text and text.lower().startswith(heading_text.lower()):
        text = text[len(heading_text) :].lstrip(" :\n")
        text = normalize_whitespace(text)

    return text or None


def find_transcript_heading(soup: BeautifulSoup) -> Tag | None:
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b", "p", "div", "span"]):
        text = tag.get_text(" ", strip=True)
        if text_matches_transcript_heading(text):
            return tag

    return None


def extract_text_from_same_container(heading_tag: Tag) -> str | None:
    container = heading_tag.parent if isinstance(heading_tag.parent, Tag) else None
    if container is None:
        return None

    full_text = container.get_text("\n", strip=True)
    heading_text = heading_tag.get_text(" ", strip=True)
    transcript_text = clean_transcript_text(full_text, heading_text=heading_text)
    if transcript_text and len(transcript_text) >= 200:
        return transcript_text

    return None


def extract_text_from_following_blocks(heading_tag: Tag) -> str | None:
    blocks: list[str] = []

    next_node = heading_tag.parent if isinstance(heading_tag.parent, Tag) else heading_tag
    for sibling in next_node.find_next_siblings():
        if sibling.name in {"h1", "h2", "h3", "h4"}:
            break

        sibling_text = sibling.get_text("\n", strip=True)
        if sibling_text:
            blocks.append(sibling_text)

    transcript_text = clean_transcript_text("\n\n".join(blocks))
    if transcript_text and len(transcript_text) >= 200:
        return transcript_text

    return None


def extract_text_from_transcript_labeled_containers(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all(True):
        attr_text = " ".join(
            str(value)
            for key, value in tag.attrs.items()
            if key in {"id", "class", "data-testid", "aria-label"}
        ).lower()
        if "transcript" not in attr_text:
            continue

        transcript_text = clean_transcript_text(tag.get_text("\n", strip=True))
        if transcript_text and len(transcript_text) >= 200:
            return transcript_text

    return None


def extract_transcript_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    heading_tag = find_transcript_heading(soup)
    if heading_tag is not None:
        transcript_text = extract_text_from_same_container(heading_tag)
        if transcript_text:
            return transcript_text

        transcript_text = extract_text_from_following_blocks(heading_tag)
        if transcript_text:
            return transcript_text

    return extract_text_from_transcript_labeled_containers(soup)


def enrich_episode_with_transcript(episode: dict[str, Any]) -> dict[str, Any]:
    enriched_episode = dict(episode)
    if episode_has_transcript(episode):
        return enriched_episode

    episode_link = episode.get("episode_link")
    if not episode_link:
        return enriched_episode

    html = fetch_episode_page(episode_link)
    if not html:
        audio_url = episode.get("audio_url")
        if not audio_url:
            return enriched_episode

        transcript = transcribe_audio_with_whisper(audio_url)
        if transcript:
            enriched_episode["transcript"] = transcript
            enriched_episode["transcript_source"] = "whisper_audio"
        return enriched_episode

    transcript = extract_transcript_from_html(html)
    if transcript:
        enriched_episode["transcript"] = transcript
        enriched_episode["transcript_source"] = "episode_page"
        return enriched_episode

    audio_url = episode.get("audio_url")
    if not audio_url:
        return enriched_episode

    transcript = transcribe_audio_with_whisper(audio_url)
    if transcript:
        enriched_episode["transcript"] = transcript
        enriched_episode["transcript_source"] = "whisper_audio"

    return enriched_episode


def enrich_episodes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_episode_with_transcript(episode) for episode in episodes]


def run_transcript_enrichment(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    episodes = load_episodes(input_path)
    updated_episodes = enrich_episodes(episodes)
    save_episodes(updated_episodes, output_path)

    already_had_transcript = sum(1 for episode in episodes if episode_has_transcript(episode))
    newly_enriched = sum(
        1
        for before, after in zip(episodes, updated_episodes, strict=False)
        if not episode_has_transcript(before) and episode_has_transcript(after)
    )
    attempted = sum(1 for episode in episodes if not episode_has_transcript(episode))

    return {
        "status": "success",
        "episodes_total": len(episodes),
        "attempted": attempted,
        "newly_enriched": newly_enriched,
        "skipped_existing": already_had_transcript,
        "still_missing": attempted - newly_enriched,
    }


def main() -> None:
    result = run_transcript_enrichment()
    print(
        "Transcript enrichment complete. "
        f"Attempted: {result['attempted']}. "
        f"Newly enriched: {result['newly_enriched']}. "
        f"Skipped existing: {result['skipped_existing']}. "
        f"Still missing: {result['still_missing']}."
    )


if __name__ == "__main__":
    main()
