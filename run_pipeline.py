from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config import PIPELINE_RUNS_PATH
from enrich_episode_transcripts import run_transcript_enrichment
from ingest_podcast_feeds import run_ingest
from make_daily_digest import run_daily_digest
from make_summaries import run_summary_generation
from send_email import send_digest_email, send_episode_batch_email


load_dotenv()


class PipelineError(RuntimeError):
    pass


def log_stage_start(stage_number: int, total_stages: int, label: str) -> None:
    print(f"\n[{stage_number}/{total_stages}] {label}")


def log_stage_result(result: dict[str, Any]) -> None:
    status = result.get("status", "unknown").upper()
    details = ", ".join(
        f"{key}={value}"
        for key, value in result.items()
        if key not in {"status", "episodes"} and value is not None
    )
    print(f"{status}: {details}" if details else status)


def ensure_stage_success(stage_name: str, result: dict[str, Any]) -> None:
    if result.get("status") == "failed":
        raise PipelineError(f"{stage_name} failed: {result.get('error') or result.get('reason')}")


def summary_stage_has_failures(summary_result: dict[str, Any]) -> bool:
    status_counts = summary_result.get("status_counts", {})
    if not isinstance(status_counts, dict):
        return False
    return any(str(status).startswith("failed") and count for status, count in status_counts.items())


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


def append_pipeline_run(record: dict[str, Any], path: Path = PIPELINE_RUNS_PATH) -> None:
    existing_records = load_json_file(path, default=[])
    existing_records.append(record)
    safe_write_json(path, existing_records)


def run_pipeline() -> dict[str, Any]:
    total_stages = 5

    log_stage_start(1, total_stages, "Fetch new episodes")
    ingest_result = run_ingest()
    log_stage_result(ingest_result)

    log_stage_start(2, total_stages, "Fetch or generate transcripts")
    transcript_result = run_transcript_enrichment()
    log_stage_result(transcript_result)

    log_stage_start(3, total_stages, "Generate episode summaries")
    summary_result = run_summary_generation()
    log_stage_result(summary_result)
    if summary_stage_has_failures(summary_result):
        print("\n[4/5] Generate daily digest")
        print("SKIPPED: summary stage had failures in this run")
        print("\n[5/5] Send digest email")
        print("SKIPPED: summary stage had failures in this run")
        return {
            "status": "success",
            "ingest": ingest_result,
            "transcripts": transcript_result,
            "summaries": summary_result,
            "digest": {
                "status": "skipped",
                "reason": "summary_failures_present",
            },
            "email": {
                "status": "skipped",
                "reason": "summary_failures_present",
            },
        }

    log_stage_start(4, total_stages, "Generate daily digest")
    digest_result = run_daily_digest()
    log_stage_result(digest_result)

    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()
    log_stage_start(5, total_stages, "Send digest email")
    if digest_result.get("status") == "generated":
        email_result = send_digest_email(
            digest_date=digest_result["digest_date"],
            recipient=recipient,
        )
    elif (
        digest_result.get("status") == "skipped"
        and digest_result.get("reason") == "not_enough_summaries"
        and digest_result.get("episode_count") == 1
    ):
        email_result = send_episode_batch_email(
            digest_date=digest_result["digest_date"],
            episode_ids=digest_result.get("source_episode_ids", []),
            recipient=recipient,
        )
    else:
        print("SKIPPED: no new digest generated in this run")
        return {
            "status": "success",
            "ingest": ingest_result,
            "transcripts": transcript_result,
            "summaries": summary_result,
            "digest": digest_result,
            "email": {
                "status": "skipped",
                "reason": "no_new_digest_generated",
            },
        }

    log_stage_result(email_result)
    if email_result.get("status") == "skipped" and email_result.get("reason") == "missing_recipient":
        raise PipelineError("email failed: RECIPIENT_EMAIL is not configured")
    ensure_stage_success("email", email_result)

    return {
        "status": "success",
        "ingest": ingest_result,
        "transcripts": transcript_result,
        "summaries": summary_result,
        "digest": digest_result,
        "email": email_result,
    }


def main() -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = run_pipeline()
        append_pipeline_run(
            {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "status": "success",
                "result": result,
            }
        )
        print("\nPipeline complete.")
    except Exception as error:
        append_pipeline_run(
            {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"\nPIPELINE FAILED: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
