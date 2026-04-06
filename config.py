from __future__ import annotations

from pathlib import Path

PODCAST_FEED_URLS = [
    "https://rss.buzzsprout.com/2175779.rss",
    "https://anchor.fm/s/f7cac464/podcast/rss",
]

EPISODES_PATH = Path("episodes.json")
SUMMARY_STATUS_PATH = Path("summary_status.json")
DAILY_DIGEST_PATH = Path("daily_digests.json")
DAILY_DIGEST_STATUS_PATH = Path("daily_digest_status.json")
DIGEST_EMAIL_STATUS_PATH = Path("digest_email_status.json")
PIPELINE_RUNS_PATH = Path("pipeline_runs.json")
CREDENTIALS_PATH = Path("credentials.json")
TOKEN_PATH = Path("token.json")
