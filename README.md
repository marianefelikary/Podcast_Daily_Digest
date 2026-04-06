# Podcast Digest Pipeline

## Overview
This is a personal Python project built to help me keep up with the podcasts I used to listen to regularly, but no longer have time to follow in full.

The pipeline monitors selected podcast feeds, fetches or generates transcripts, summarizes each episode with Gemini, combines the results into a daily digest, and sends the digest by email.

## Problem It Solves
Listening to multiple podcast episodes every day takes time. Even when the content is valuable, it is not always practical to listen to everything in full.

This project turns new podcast episodes into short, useful summaries so I can stay updated without manually reviewing every episode.

## Current Scope
The current implementation supports:
- reading episodes from a fixed list of podcast RSS feeds
- storing episode metadata in a local JSON file
- extracting transcripts from episode pages when available
- falling back to local transcription with `faster-whisper` when needed
- generating structured episode summaries with Gemini
- generating a combined daily digest from multiple episode summaries
- sending the result by email through Gmail API
- tracking local processing and email delivery state with JSON files

This is currently a local, file-based pipeline. It is not deployed as a hosted service and does not use a database.

## Workflow

### 1. Fetch New Episodes
The pipeline reads the configured RSS feeds and stores any newly discovered episodes.

### 2. Fetch or Generate Transcripts
For episodes that do not already have transcripts, the pipeline first tries to extract transcript text from the episode page. If that fails, it falls back to local audio transcription with `faster-whisper`.

### 3. Generate Episode Summaries
For each episode with a transcript, the pipeline generates a structured summary with Gemini, including:
- `full_summary`
- `key_points`
- `notable_items`
- `worth_listening`

### 4. Build the Daily Digest
The pipeline combines the individual episode summaries into a single daily digest that keeps only the most important takeaways across podcasts.

### 5. Send the Email
The digest is sent through the Gmail API using the authenticated Gmail account as the sender and the configured recipient email as the destination.

## Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed locally
- a Google Cloud project with Gmail API enabled
- a valid `credentials.json` for Gmail OAuth
- a Gemini API key

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
python3 run_pipeline.py
```