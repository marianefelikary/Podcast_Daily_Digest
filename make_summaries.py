from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from config import EPISODES_PATH, SUMMARY_STATUS_PATH

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
MAX_WORKERS = int(os.getenv("SUMMARY_MAX_WORKERS", "4"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "1.0"))


class WorthListening(str, Enum):
    YES = "yes"
    MAYBE = "maybe"
    NO = "no"


class EpisodeSummary(BaseModel):
    full_summary: str = Field(
        description="A summary of the most important points in the episode."
    )
    key_points: list[str] = Field(description="List of key points from the episode.")
    notable_items: list[str] = Field(
        default_factory=list,
        description="Up to 10 important names, tools, products, or companies mentioned.",
    )
    worth_listening: WorthListening = Field(
        description="Whether the episode is worth listening to."
    )


class SummaryJobResult(BaseModel):
    episode_id_or_guid: str | None
    status: str
    error: str | None = None
    summary: EpisodeSummary | None = None


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return api_key.strip()


def get_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def load_episodes(path: Path = EPISODES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as temp_file:
        json.dump(payload, temp_file, indent=2, ensure_ascii=False)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)

    temp_path.replace(path)


def episode_has_summary(episode: dict[str, Any]) -> bool:
    return bool(episode.get("summary"))


def episode_is_eligible(episode: dict[str, Any]) -> bool:
    return bool(episode.get("transcript")) and not episode_has_summary(episode)


def build_prompt(transcript: str) -> str:
    return f"""
You are summarizing one podcast episode transcript for a daily email digest.

Rules:

- Be specific
- Focus only on the most important information
- Do not invent details that are not in the transcript
- Prefer concrete takeaways over vague wording
- Do not include anything related to ads and sponsors that the podcast mentions

Transcript:
{transcript}
""".strip()


def parse_summary_response(response: Any) -> EpisodeSummary:
    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, EpisodeSummary):
            return parsed
        return EpisodeSummary.model_validate(parsed)

    response_text = getattr(response, "text", None)
    if not response_text:
        raise ValueError("Gemini response did not include parsed data or text.")

    return EpisodeSummary.model_validate_json(response_text)


def is_retryable_generation_error(error: Exception) -> bool:
    message = str(error).upper()
    retryable_markers = ["429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
    return any(marker in message for marker in retryable_markers)


def generate_summary_with_retries(
    client: genai.Client,
    episode: dict[str, Any],
) -> EpisodeSummary:
    last_error: Exception | None = None

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_prompt(episode["transcript"]),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EpisodeSummary,
                ),
            )
            return parse_summary_response(response)
        except ValidationError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES or not is_retryable_generation_error(exc):
                raise
            time.sleep(LLM_BACKOFF_SECONDS * (2**attempt))

    raise RuntimeError(f"Summary generation failed after retries: {last_error}")


def generate_summary_for_episode(
    client: genai.Client,
    episode: dict[str, Any],
) -> SummaryJobResult:
    episode_id = episode.get("episode_id_or_guid")
    transcript = episode.get("transcript")

    if not transcript:
        return SummaryJobResult(
            episode_id_or_guid=episode_id,
            status="skipped_no_transcript",
        )

    if episode_has_summary(episode):
        return SummaryJobResult(
            episode_id_or_guid=episode_id,
            status="skipped_existing_summary",
        )

    try:
        summary = generate_summary_with_retries(client, episode)
    except ValidationError as exc:
        return SummaryJobResult(
            episode_id_or_guid=episode_id,
            status="failed_validation",
            error=str(exc),
        )
    except Exception as exc:
        return SummaryJobResult(
            episode_id_or_guid=episode_id,
            status="failed_generation",
            error=str(exc),
        )

    return SummaryJobResult(
        episode_id_or_guid=episode_id,
        status="summarized",
        summary=summary,
    )


def summarize_episodes_in_parallel(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated_episodes = [dict(episode) for episode in episodes]
    status_records: list[dict[str, Any]] = []

    indexed_episodes = list(enumerate(updated_episodes))
    eligible_jobs = [
        (index, episode) for index, episode in indexed_episodes if episode_is_eligible(episode)
    ]

    for _index, episode in indexed_episodes:
        episode_id = episode.get("episode_id_or_guid")
        if not episode.get("transcript"):
            status_records.append(
                SummaryJobResult(
                    episode_id_or_guid=episode_id,
                    status="skipped_no_transcript",
                ).model_dump()
            )
        elif episode_has_summary(episode):
            status_records.append(
                SummaryJobResult(
                    episode_id_or_guid=episode_id,
                    status="skipped_existing_summary",
                ).model_dump()
            )

    if not eligible_jobs:
        return updated_episodes, status_records

    worker_count = max(1, min(MAX_WORKERS, len(eligible_jobs)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(generate_summary_for_episode, get_client(), episode): index
            for index, episode in eligible_jobs
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            episode = updated_episodes[index]
            episode_id = episode.get("episode_id_or_guid")

            try:
                result = future.result()
            except Exception as exc:
                status_records.append(
                    SummaryJobResult(
                        episode_id_or_guid=episode_id,
                        status="failed_worker",
                        error=str(exc),
                    ).model_dump()
                )
                continue

            status_records.append(result.model_dump(exclude={"summary"}))
            if result.summary is not None:
                episode["summary"] = result.summary.model_dump()

    return updated_episodes, status_records


def run_summary_generation() -> dict[str, Any]:
    episodes = load_episodes()
    updated_episodes, status_records = summarize_episodes_in_parallel(episodes)
    safe_write_json(EPISODES_PATH, updated_episodes)
    safe_write_json(SUMMARY_STATUS_PATH, status_records)

    status_counts: dict[str, int] = {}
    for record in status_records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "status": "success",
        "episodes_total": len(updated_episodes),
        "status_counts": status_counts,
        "summarized": status_counts.get("summarized", 0),
    }


def main() -> None:
    result = run_summary_generation()
    print(
        f"Wrote updated episodes to {EPISODES_PATH} and summary statuses to {SUMMARY_STATUS_PATH}. "
        f"New summaries: {result['summarized']}."
    )


if __name__ == "__main__":
    main()
