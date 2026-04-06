from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    CREDENTIALS_PATH,
    DAILY_DIGEST_PATH,
    DIGEST_EMAIL_STATUS_PATH,
    EPISODES_PATH,
    TOKEN_PATH,
)

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
DAILY_DIGESTS_PATH = DAILY_DIGEST_PATH
EMAIL_STATUS_PATH = DIGEST_EMAIL_STATUS_PATH

load_dotenv()

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


def get_authenticated_credentials() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError("credentials.json not found.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


def load_digests() -> list[dict[str, Any]]:
    payload = load_json_file(DAILY_DIGESTS_PATH, default=[])
    if not isinstance(payload, list):
        raise ValueError("daily_digests.json must contain a JSON array.")
    return payload


def load_episodes() -> list[dict[str, Any]]:
    payload = load_json_file(EPISODES_PATH, default=[])
    if not isinstance(payload, list):
        raise ValueError("episodes.json must contain a JSON array.")
    return payload


def load_email_status_records() -> list[dict[str, Any]]:
    payload = load_json_file(EMAIL_STATUS_PATH, default=[])
    if not isinstance(payload, list):
        raise ValueError("digest_email_status.json must contain a JSON array.")
    return payload


def was_digest_sent(
    digest_date: str,
    recipient: str,
    status_records: list[dict[str, Any]],
) -> bool:
    for record in status_records:
        if (
            record.get("digest_date") == digest_date
            and record.get("recipient") == recipient
            and record.get("status") == "sent"
        ):
            return True
    return False


def record_email_status(
    digest_date: str,
    recipient: str,
    status: str,
    error: str | None = None,
    message_id: str | None = None,
) -> None:
    records = load_email_status_records()
    records.append(
        {
            "digest_date": digest_date,
            "recipient": recipient,
            "status": status,
            "message_id": message_id,
            "error": error,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    safe_write_json(EMAIL_STATUS_PATH, records)


def pick_digest(digests: list[dict[str, Any]], digest_date: str | None) -> dict[str, Any] | None:
    if digest_date:
        for digest in digests:
            if digest.get("digest_date") == digest_date:
                return digest
        return None

    if not digests:
        return None

    return max(digests, key=lambda item: item.get("digest_date", ""))


def build_episode_index(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        episode["episode_id_or_guid"]: episode
        for episode in episodes
        if isinstance(episode, dict) and isinstance(episode.get("episode_id_or_guid"), str)
    }


def normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def build_subject(digest: dict[str, Any]) -> str:
    digest_date = digest.get("digest_date", datetime.now().date().isoformat())
    return f"Daily Podcast Digest - {digest_date}"


def build_episode_only_subject(digest_date: str) -> str:
    return f"Daily Podcast Update - {digest_date}"


def build_episode_section(episode: dict[str, Any]) -> list[str]:
    summary = episode.get("summary", {})
    full_summary = str(summary.get("full_summary", "")).strip()
    key_points = normalize_list(summary.get("key_points"))
    notable_items = normalize_list(summary.get("notable_items"))
    worth_listening = str(summary.get("worth_listening", "")).strip()

    lines = [
        f"Title: {episode.get('episode_title', 'Untitled Episode')}",
        f"Podcast: {episode.get('podcast_title', 'Unknown Podcast')}",
        f"Published Date: {episode.get('published_date', 'Unknown')}",
        f"Link: {episode.get('episode_link', 'N/A')}",
        "",
        "Full Summary:",
        full_summary or "No full summary available.",
        "",
        "Key Points:",
    ]

    if key_points:
        for point in key_points:
            lines.append(f"- {point}")
    else:
        lines.append("- No key points available.")

    lines.extend(["", "Notable Items:"])
    if notable_items:
        for item in notable_items:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.extend(["", f"Worth Listening: {worth_listening or 'unknown'}"])
    return lines


def build_body(digest: dict[str, Any], episode_index: dict[str, dict[str, Any]]) -> str:
    digest_date = digest.get("digest_date", "Unknown date")
    digest_payload = digest.get("digest", {})
    digest_key_points = normalize_list(digest_payload.get("key_points"))
    digest_notable_items = normalize_list(digest_payload.get("notable_items"))
    source_episode_ids = normalize_list(digest.get("source_episode_ids"))

    lines: list[str] = [
        f"Daily Podcast Digest for {digest_date}",
        "",
        "Main Digest Summary",
    ]

    if digest_key_points:
        for point in digest_key_points:
            lines.append(f"- {point}")
    else:
        lines.append("- No digest key points available.")

    if digest_notable_items:
        lines.extend(
            [
                "",
                "Top Notable Items",
                ", ".join(digest_notable_items),
            ]
        )

    lines.extend(["", "Episode Details"])

    included_episodes = 0
    for episode_id in source_episode_ids:
        episode = episode_index.get(episode_id)
        if not episode:
            continue

        lines.extend(["", *build_episode_section(episode)])
        included_episodes += 1

    if included_episodes == 0:
        return ""

    return "\n".join(lines).strip()


def build_episode_only_body(
    digest_date: str,
    episode_ids: list[str],
    episode_index: dict[str, dict[str, Any]],
) -> str:
    included_episodes = [
        episode_index[episode_id]
        for episode_id in episode_ids
        if episode_id in episode_index
    ]
    if not included_episodes:
        return ""

    lines: list[str] = [
        f"Daily Podcast Update for {digest_date}",
        "",
        "Episode Details",
    ]

    for episode in included_episodes:
        lines.extend(["", *build_episode_section(episode)])

    return "\n".join(lines).strip()


def create_message(recipient: str, subject: str, body: str) -> dict[str, str]:
    message = MIMEText(body, "plain", "utf-8")
    message["To"] = recipient
    message["Subject"] = subject

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return {"raw": encoded_message}


def send_message(recipient: str, subject: str, body: str) -> str:
    creds = get_authenticated_credentials()
    service = build("gmail", "v1", credentials=creds)
    message = create_message(recipient=recipient, subject=subject, body=body)
    sent_message = (
        service.users()
        .messages()
        .send(userId="me", body=message)
        .execute()
    )
    return str(sent_message.get("id", ""))


def send_digest_email(
    recipient: str,
    digest_date: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not recipient:
        return {
            "status": "skipped",
            "reason": "missing_recipient",
        }

    digests = load_digests()
    episodes = load_episodes()
    status_records = load_email_status_records()

    digest = pick_digest(digests, digest_date)
    if not digest:
        return {
            "status": "skipped",
            "reason": "missing_digest",
            "digest_date": digest_date,
        }

    resolved_digest_date = str(digest.get("digest_date", "")).strip()
    if not resolved_digest_date:
        return {
            "status": "skipped",
            "reason": "missing_digest_date",
        }

    if not force and was_digest_sent(resolved_digest_date, recipient, status_records):
        return {
            "status": "skipped",
            "reason": "already_sent",
            "digest_date": resolved_digest_date,
            "recipient": recipient,
        }

    body = build_body(digest, build_episode_index(episodes))
    if not body:
        return {
            "status": "skipped",
            "reason": "empty_body",
            "digest_date": resolved_digest_date,
            "recipient": recipient,
        }

    subject = build_subject(digest)

    try:
        message_id = send_message(recipient=recipient, subject=subject, body=body)
    except HttpError as error:
        status_code = getattr(error.resp, "status", "unknown")
        reason = getattr(error.resp, "reason", "") or "Unknown error"
        details = ""
        if getattr(error, "content", None):
            try:
                details = error.content.decode("utf-8")
            except UnicodeDecodeError:
                details = str(error.content)
        error_text = f"{status_code}: {reason} {details}".strip()
        record_email_status(
            digest_date=resolved_digest_date,
            recipient=recipient,
            status="failed",
            error=error_text,
        )
        return {
            "status": "failed",
            "reason": "gmail_http_error",
            "digest_date": resolved_digest_date,
            "recipient": recipient,
            "error": error_text,
        }
    except Exception as error:
        record_email_status(
            digest_date=resolved_digest_date,
            recipient=recipient,
            status="failed",
            error=str(error),
        )
        return {
            "status": "failed",
            "reason": "send_error",
            "digest_date": resolved_digest_date,
            "recipient": recipient,
            "error": str(error),
        }

    record_email_status(
        digest_date=resolved_digest_date,
        recipient=recipient,
        status="sent",
        message_id=message_id,
    )
    return {
        "status": "sent",
        "digest_date": resolved_digest_date,
        "recipient": recipient,
        "message_id": message_id,
    }


def send_episode_batch_email(
    recipient: str,
    digest_date: str,
    episode_ids: list[str],
) -> dict[str, Any]:
    if not recipient:
        return {
            "status": "skipped",
            "reason": "missing_recipient",
        }

    status_records = load_email_status_records()
    if was_digest_sent(digest_date, recipient, status_records):
        return {
            "status": "skipped",
            "reason": "already_sent",
            "digest_date": digest_date,
            "recipient": recipient,
        }

    episodes = load_episodes()
    body = build_episode_only_body(digest_date, episode_ids, build_episode_index(episodes))
    if not body:
        return {
            "status": "skipped",
            "reason": "empty_body",
            "digest_date": digest_date,
            "recipient": recipient,
        }

    try:
        message_id = send_message(
            recipient=recipient,
            subject=build_episode_only_subject(digest_date),
            body=body,
        )
    except HttpError as error:
        status_code = getattr(error.resp, "status", "unknown")
        reason = getattr(error.resp, "reason", "") or "Unknown error"
        details = ""
        if getattr(error, "content", None):
            try:
                details = error.content.decode("utf-8")
            except UnicodeDecodeError:
                details = str(error.content)
        error_text = f"{status_code}: {reason} {details}".strip()
        record_email_status(
            digest_date=digest_date,
            recipient=recipient,
            status="failed",
            error=error_text,
        )
        return {
            "status": "failed",
            "reason": "gmail_http_error",
            "digest_date": digest_date,
            "recipient": recipient,
            "error": error_text,
        }
    except Exception as error:
        record_email_status(
            digest_date=digest_date,
            recipient=recipient,
            status="failed",
            error=str(error),
        )
        return {
            "status": "failed",
            "reason": "send_error",
            "digest_date": digest_date,
            "recipient": recipient,
            "error": str(error),
        }

    record_email_status(
        digest_date=digest_date,
        recipient=recipient,
        status="sent",
        message_id=message_id,
    )
    return {
        "status": "sent",
        "digest_date": digest_date,
        "recipient": recipient,
        "message_id": message_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the latest podcast digest via Gmail.")
    parser.add_argument(
        "--recipient",
        default=os.getenv("RECIPIENT_EMAIL", "").strip(),
        help="Recipient email address. Defaults to RECIPIENT_EMAIL from .env.",
    )
    parser.add_argument(
        "--digest-date",
        default=None,
        help="Optional digest date in YYYY-MM-DD format. Defaults to the latest digest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send even if the digest was already marked as sent to this recipient.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipient = args.recipient

    try:
        result = send_digest_email(
            recipient=recipient,
            digest_date=args.digest_date,
            force=args.force,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Failed to load input data: {error}")
        return

    if result["status"] == "sent":
        print(
            f"Sent digest {result['digest_date']} to {result['recipient']}. "
            f"Gmail message id: {result['message_id']}"
        )
        return

    if result["reason"] == "missing_recipient":
        print("Recipient email is required. Use --recipient or set RECIPIENT_EMAIL.")
        return
    if result["reason"] == "missing_digest":
        target = args.digest_date or "latest"
        print(f"No digest found for selection: {target}.")
        return
    if result["reason"] == "missing_digest_date":
        print("Selected digest is missing digest_date. Not sending.")
        return
    if result["reason"] == "already_sent":
        print(f"Digest {result['digest_date']} was already sent to {result['recipient']}. Skipping.")
        return
    if result["reason"] == "empty_body":
        print(f"No digest content available to send for {result['digest_date']}.")
        return

    print(f"Failed to send digest email: {result.get('error', result['reason'])}")


if __name__ == "__main__":
    main()
