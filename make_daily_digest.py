from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from config import DAILY_DIGEST_PATH, DAILY_DIGEST_STATUS_PATH, EPISODES_PATH, SUMMARY_STATUS_PATH

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "1.0"))


class EpisodeSummary(BaseModel):
    full_summary: str = Field(description="The episode-level summary.")
    key_points: list[str] = Field(description="Key takeaways for the episode.")
    notable_items: list[str] = Field(
        default_factory=list,
        description="Important names, tools, products, or companies mentioned.",
    )
    worth_listening: str = Field(description="Whether the episode is worth listening to.")


class DailyDigestSummary(BaseModel):
    key_points: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3 to 5 distinct, non-overlapping key points across the day.",
    )
    notable_items: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 important names, tools, products, or companies that matter most overall.",
    )


class DigestArtifact(BaseModel):
    digest_date: str
    created_at: str
    episode_count: int
    source_episode_ids: list[str]
    source_podcast_titles: list[str]
    digest: DailyDigestSummary


class DigestStatusRecord(BaseModel):
    digest_date: str
    status: str
    error: str | None = None
    episode_count: int = 0
    source_episode_ids: list[str] = Field(default_factory=list)


class SummaryStatusRecord(BaseModel):
    episode_id_or_guid: str | None
    status: str
    error: str | None = None


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return api_key.strip()


def get_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
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


def load_episodes(path: Path = EPISODES_PATH) -> list[dict[str, Any]]:
    return load_json_file(path, default=[])


def load_existing_digests(path: Path = DAILY_DIGEST_PATH) -> list[dict[str, Any]]:
    return load_json_file(path, default=[])


def load_summary_status_records(path: Path = SUMMARY_STATUS_PATH) -> list[dict[str, Any]]:
    return load_json_file(path, default=[])


def get_digest_date(episodes: list[dict[str, Any]]) -> str:
    published_dates = sorted(
        {
            published_date
            for episode in episodes
            if isinstance((published_date := episode.get("published_date")), str) and published_date
        }
    )
    if len(published_dates) == 1:
        return published_dates[0]

    return date.today().isoformat()


def is_valid_summary_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False

    try:
        EpisodeSummary.model_validate(value)
    except ValidationError:
        return False

    return True


def get_current_digest_episode_ids(summary_status_records: list[dict[str, Any]]) -> list[str]:
    episode_ids: list[str] = []

    for record in summary_status_records:
        try:
            parsed_record = SummaryStatusRecord.model_validate(record)
        except ValidationError:
            continue

        if parsed_record.status != "summarized":
            continue

        if parsed_record.episode_id_or_guid:
            episode_ids.append(parsed_record.episode_id_or_guid)

    return episode_ids


def select_digest_batch(
    episodes: list[dict[str, Any]],
    digest_episode_ids: list[str],
) -> list[dict[str, Any]]:
    digest_episode_id_set = set(digest_episode_ids)
    return [
        episode
        for episode in episodes
        if episode.get("episode_id_or_guid") in digest_episode_id_set
        and is_valid_summary_payload(episode.get("summary"))
    ]


def build_episode_summary_input(episodes: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for episode in episodes:
        summary = EpisodeSummary.model_validate(episode["summary"])
        lines.append(
            "\n".join(
                [
                    f"Full Summary: {summary.full_summary}",
                    "Key Points:",
                    *[f"- {key_point}" for key_point in summary.key_points],
                ]
            )
        )

    return "\n\n".join(lines)


def build_prompt(episode_summaries: str) -> str:
    return f"""
You are creating one daily digest from multiple podcast episode summaries.

Your job is to combine the episode summaries into one concise overview of the day.

Instructions:
- Focus only on the most important takeaways across all podcasts
- Merge overlapping themes instead of repeating them
- Do not restate the same idea multiple times
- Keep only the most valuable and distinct insights
- Prefer concrete conclusions over vague wording
- Be concise and selective
- Do not invent details that are not present in the episode summaries

Return:
- key_points: 3 to 5 distinct, non-overlapping key points across the day
- notable_items: up to 5 important names, tools, products, or companies that matter most overall

Episode summaries:
{episode_summaries}
""".strip()


def parse_digest_response(response: Any) -> DailyDigestSummary:
    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, DailyDigestSummary):
            return parsed
        return DailyDigestSummary.model_validate(parsed)

    response_text = getattr(response, "text", None)
    if not response_text:
        raise ValueError("Gemini response did not include parsed data or text.")

    return DailyDigestSummary.model_validate_json(response_text)


def is_retryable_generation_error(error: Exception) -> bool:
    message = str(error).upper()
    retryable_markers = ["429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
    return any(marker in message for marker in retryable_markers)


def generate_daily_digest(
    client: genai.Client,
    episodes: list[dict[str, Any]],
) -> DailyDigestSummary:
    last_error: Exception | None = None

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_prompt(build_episode_summary_input(episodes)),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DailyDigestSummary,
                ),
            )
            return parse_digest_response(response)
        except ValidationError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES or not is_retryable_generation_error(exc):
                raise
            time.sleep(LLM_BACKOFF_SECONDS * (2**attempt))

    raise RuntimeError(f"Daily digest generation failed after retries: {last_error}")


def build_digest_artifact(
    digest_date: str,
    episodes: list[dict[str, Any]],
    digest: DailyDigestSummary,
) -> DigestArtifact:
    source_episode_ids = [
        episode_id
        for episode in episodes
        if isinstance((episode_id := episode.get("episode_id_or_guid")), str) and episode_id
    ]
    source_podcast_titles = sorted(
        {
            podcast_title
            for episode in episodes
            if isinstance((podcast_title := episode.get("podcast_title")), str) and podcast_title
        }
    )

    return DigestArtifact(
        digest_date=digest_date,
        created_at=datetime.now().isoformat(timespec="seconds"),
        episode_count=len(episodes),
        source_episode_ids=source_episode_ids,
        source_podcast_titles=source_podcast_titles,
        digest=digest,
    )


def save_digest_artifact(artifact: DigestArtifact, path: Path = DAILY_DIGEST_PATH) -> None:
    existing_digests = load_existing_digests(path)
    existing_digests.append(artifact.model_dump())
    safe_write_json(path, existing_digests)


def save_status_record(record: DigestStatusRecord, path: Path = DAILY_DIGEST_STATUS_PATH) -> None:
    existing_records = load_json_file(path, default=[])
    existing_records.append(record.model_dump())
    safe_write_json(path, existing_records)


def find_existing_digest(
    existing_digests: list[dict[str, Any]],
    digest_date: str,
    source_episode_ids: list[str],
) -> dict[str, Any] | None:
    target_ids = sorted(source_episode_ids)
    for digest in existing_digests:
        if digest.get("digest_date") != digest_date:
            continue
        existing_ids = digest.get("source_episode_ids")
        if isinstance(existing_ids, list) and sorted(existing_ids) == target_ids:
            return digest
    return None


def run_daily_digest() -> dict[str, Any]:
    episodes = load_episodes()
    summary_status_records = load_summary_status_records()
    digest_episode_ids = get_current_digest_episode_ids(summary_status_records)
    digest_batch = select_digest_batch(episodes, digest_episode_ids)
    digest_date = get_digest_date(digest_batch) if digest_batch else date.today().isoformat()
    source_episode_ids = [
        episode_id
        for episode in digest_batch
        if isinstance((episode_id := episode.get("episode_id_or_guid")), str) and episode_id
    ]

    if len(digest_batch) <= 1:
        record = DigestStatusRecord(
            digest_date=digest_date,
            status="skipped_not_enough_summaries",
            episode_count=len(digest_batch),
            source_episode_ids=source_episode_ids,
        )
        save_status_record(record)
        return {
            "status": "skipped",
            "reason": "not_enough_summaries",
            "digest_date": digest_date,
            "episode_count": len(digest_batch),
            "source_episode_ids": source_episode_ids,
        }

    existing_digests = load_existing_digests()
    if find_existing_digest(existing_digests, digest_date, source_episode_ids):
        save_status_record(
            DigestStatusRecord(
                digest_date=digest_date,
                status="skipped_existing_digest",
                episode_count=len(digest_batch),
                source_episode_ids=source_episode_ids,
            )
        )
        return {
            "status": "skipped",
            "reason": "existing_digest",
            "digest_date": digest_date,
            "episode_count": len(digest_batch),
            "source_episode_ids": source_episode_ids,
        }

    try:
        digest = generate_daily_digest(get_client(), digest_batch)
        artifact = build_digest_artifact(digest_date, digest_batch, digest)
    except ValidationError as exc:
        record = DigestStatusRecord(
            digest_date=digest_date,
            status="failed_validation",
            error=str(exc),
            episode_count=len(digest_batch),
            source_episode_ids=source_episode_ids,
        )
        save_status_record(record)
        raise
    except Exception as exc:
        record = DigestStatusRecord(
            digest_date=digest_date,
            status="failed_generation",
            error=str(exc),
            episode_count=len(digest_batch),
            source_episode_ids=source_episode_ids,
        )
        save_status_record(record)
        raise

    save_digest_artifact(artifact)
    save_status_record(
        DigestStatusRecord(
            digest_date=digest_date,
            status="generated",
            episode_count=artifact.episode_count,
            source_episode_ids=artifact.source_episode_ids,
        )
    )
    return {
        "status": "generated",
        "digest_date": digest_date,
        "episode_count": artifact.episode_count,
        "source_episode_ids": artifact.source_episode_ids,
    }


def main() -> None:
    result = run_daily_digest()
    if result["status"] == "generated":
        print(f"Wrote daily digest to {DAILY_DIGEST_PATH}")
        return
    if result["reason"] == "not_enough_summaries":
        print("Skipped digest generation because fewer than 2 summarized episodes were available.")
        return
    print("Skipped digest generation because the current digest already exists.")


if __name__ == "__main__":
    main()
