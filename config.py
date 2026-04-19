from __future__ import annotations

from pathlib import Path

PODCAST_FEED_URLS = [
    "https://rss.buzzsprout.com/2175779.rss",
    "https://anchor.fm/s/f7cac464/podcast/rss",
]

BASE_DIR = Path(__file__).parent

EPISODES_PATH = BASE_DIR / "episodes.json"
SUMMARY_STATUS_PATH = BASE_DIR / "summary_status.json"
DAILY_DIGEST_PATH = BASE_DIR / "daily_digests.json"
DAILY_DIGEST_STATUS_PATH = BASE_DIR / "daily_digest_status.json"
DIGEST_EMAIL_STATUS_PATH = BASE_DIR / "digest_email_status.json"
PIPELINE_RUNS_PATH = BASE_DIR / "pipeline_runs.json"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"
