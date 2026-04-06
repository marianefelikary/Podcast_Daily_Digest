from __future__ import annotations

import json
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable

import feedparser

from config import EPISODES_PATH, PODCAST_FEED_URLS

OUTPUT_PATH = EPISODES_PATH


def parse_feed(feed_url: str) -> feedparser.FeedParserDict:
    return feedparser.parse(feed_url)


def get_podcast_title(feed: feedparser.FeedParserDict) -> str | None:
    return feed.get("feed", {}).get("title")


def parse_published_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    for field_name in ("published", "updated", "created"):
        raw_value = entry.get(field_name)
        if not raw_value:
            continue

        try:
            parsed = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, IndexError):
            continue

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)

        return parsed

    return None


def should_process_after_checkpoint(
    published_at: datetime | None,
    latest_processed_date: date | None,
) -> bool:
    if published_at is None:
        return False

    if latest_processed_date is None:
        return True

    return published_at.date() >= latest_processed_date


class EpisodePageLinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_anchor = False
        self.current_href: str | None = None
        self.current_text_parts: list[str] = []
        self.found_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or self.found_href is not None:
            return

        self.in_anchor = True
        self.current_text_parts = []
        self.current_href = dict(attrs).get("href")

    def handle_data(self, data: str) -> None:
        if self.in_anchor:
            self.current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.in_anchor:
            return

        anchor_text = "".join(self.current_text_parts).strip().lower()
        if self.current_href and anchor_text == "episode page":
            self.found_href = self.current_href

        self.in_anchor = False
        self.current_href = None
        self.current_text_parts = []


def extract_episode_page_from_content(entry: feedparser.FeedParserDict) -> str | None:
    for content_item in entry.get("content", []):
        html = content_item.get("value")
        if not html:
            continue

        parser = EpisodePageLinkExtractor()
        parser.feed(html)
        if parser.found_href:
            return parser.found_href

    return None


def get_episode_link(entry: feedparser.FeedParserDict) -> str | None:
    return entry.get("link") or extract_episode_page_from_content(entry)


def get_episode_key(entry: feedparser.FeedParserDict) -> str | None:
    return entry.get("id") or entry.get("guid") or entry.get("link")


def iter_candidate_audio_links(entry: feedparser.FeedParserDict) -> Iterable[str]:
    for enclosure in entry.get("enclosures", []):
        href = enclosure.get("href") or enclosure.get("url")
        if href:
            yield href

    for link in entry.get("links", []):
        rel = (link.get("rel") or "").lower()
        link_type = (link.get("type") or "").lower()
        href = link.get("href")

        if not href:
            continue

        if rel == "enclosure" or link_type.startswith("audio/"):
            yield href


def get_audio_url(entry: feedparser.FeedParserDict) -> str | None:
    for audio_url in iter_candidate_audio_links(entry):
        return audio_url

    return None


def normalize_entry(
    podcast_title: str | None,
    entry: feedparser.FeedParserDict,
) -> dict[str, Any]:
    published_at = parse_published_datetime(entry)

    return {
        "podcast_title": podcast_title,
        "episode_title": entry.get("title"),
        "published_date": published_at.date().isoformat() if published_at else None,
        "episode_link": get_episode_link(entry),
        "episode_id_or_guid": get_episode_key(entry),
        "audio_url": get_audio_url(entry),
    }


def load_existing_episodes(output_path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    return json.loads(output_path.read_text(encoding="utf-8"))


def parse_existing_published_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def get_latest_processed_date(episodes: list[dict[str, Any]]) -> date | None:
    dates = [
        parsed_date
        for episode in episodes
        if (parsed_date := parse_existing_published_date(episode.get("published_date"))) is not None
    ]
    return max(dates) if dates else None


def get_existing_episode_keys(episodes: list[dict[str, Any]]) -> set[str]:
    return {
        episode_key
        for episode in episodes
        if isinstance((episode_key := episode.get("episode_id_or_guid")), str) and episode_key
    }


def sort_episodes(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        episodes,
        key=lambda episode: (
            episode.get("published_date") or "",
            episode.get("podcast_title") or "",
            episode.get("episode_title") or "",
        ),
        reverse=True,
    )


def collect_recent_episodes(
    feed_urls: list[str],
    latest_processed_date: date | None,
    existing_episode_keys: set[str],
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []

    for feed_url in feed_urls:
        parsed_feed = parse_feed(feed_url)
        podcast_title = get_podcast_title(parsed_feed)

        for entry in parsed_feed.get("entries", []):
            published_at = parse_published_datetime(entry)
            episode_key = get_episode_key(entry)

            if not should_process_after_checkpoint(published_at, latest_processed_date):
                continue

            if episode_key and episode_key in existing_episode_keys:
                continue

            normalized_episode = normalize_entry(podcast_title, entry)
            episodes.append(normalized_episode)
            if episode_key:
                existing_episode_keys.add(episode_key)

    return episodes


def print_episodes(episodes: list[dict[str, Any]]) -> None:
    print(f"Matched {len(episodes)} episode(s)")
    for index, episode in enumerate(episodes, start=1):
        print(f"\nEpisode {index}")
        pprint(episode, sort_dicts=False)


def save_episodes_json(
    episodes: list[dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    output_path.write_text(
        json.dumps(episodes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")


def run_ingest(
    feed_urls: list[str] = PODCAST_FEED_URLS,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    existing_episodes = load_existing_episodes(output_path)
    latest_processed_date = get_latest_processed_date(existing_episodes)
    existing_episode_keys = get_existing_episode_keys(existing_episodes)

    new_episodes = collect_recent_episodes(
        feed_urls,
        latest_processed_date=latest_processed_date,
        existing_episode_keys=existing_episode_keys,
    )

    all_episodes = sort_episodes(existing_episodes + new_episodes)
    save_episodes_json(all_episodes, output_path=output_path)
    return {
        "status": "success",
        "new_episodes": len(new_episodes),
        "total_episodes": len(all_episodes),
        "episodes": new_episodes,
    }


def main() -> None:
    result = run_ingest()
    new_episodes = result["episodes"]
    print_episodes(new_episodes)
    print(
        f"Ingest complete. New episodes: {result['new_episodes']}. "
        f"Total episodes: {result['total_episodes']}."
    )


if __name__ == "__main__":
    main()
