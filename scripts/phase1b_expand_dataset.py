"""
Phase 1b -- Expand Dataset via YouTube Scraping (Option C)
==========================================================
Scrapes comments from 12 Gujarati YouTube channels.
Runs on CPU, no GPU needed. Uses YouTube Data API v3.

Requirements:
    pip install google-api-python-client langdetect pandas tqdm

Usage:
    Set YOUTUBE_API_KEY below (or set env var YOUTUBE_API_KEY)
    python scripts/phase1b_expand_dataset.py

Output:
    data/raw/expanded_comments_raw.csv       — all scraped comments
    data/processed/expanded_gujlish.csv      — filtered Gujlish sentences
    data/processed/combined_full_dataset.csv — merged with original 21k
"""

import sys, io, os, time, csv, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
for d in [DATA_RAW, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)

# Set your YouTube API key here OR set env var YOUTUBE_API_KEY
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

# Max comments to fetch per channel (to stay within API quota)
# YouTube API free quota: 10,000 units/day. Each commentThreads call = 1 unit per page (100 comments)
# 500 comments per channel × 12 channels = 6,000 units — well within limit
MAX_COMMENTS_PER_CHANNEL = 500
MAX_VIDEOS_PER_CHANNEL   = 10   # scan top 10 videos per channel

# ── Target Channels ───────────────────────────────────────────────────────────
# These are popular Gujarati YouTube channels with high comment volume.
# Comments are typically Gujlish (Gujarati-English code-mixed).
# Channel IDs obtained from YouTube channel URLs.
GUJARATI_CHANNELS = [
    # Music & Entertainment (very high comment engagement)
    {"name": "Jignesh Kaviraj",       "id": "UCzwbPDCgFBBCVRkXXHAqTIQ"},
    {"name": "Kirtidan Gadhvi",        "id": "UCZrXFjuB9UbOIGAHMiLFVWw"},
    {"name": "Vikram Thakor",          "id": "UCjBqHJnCfYqUBhIfOuAbdpg"},

    # Comedy & Drama
    {"name": "VB Robawala",            "id": "UC_RlBf_0Wl7Rf_x9_NfzFCg"},
    {"name": "Gujjubhai",              "id": "UCcO3AVVX9iQD8Ny4I7eDk9Q"},

    # Food & Lifestyle (very popular, lots of Gujlish comments)
    {"name": "Gujarati Zaika",         "id": "UCaxuBXo_V1pFRRGMGfEXpvg"},
    {"name": "Kabita Kitchen Gujarati","id": "UCi1_JaZaDn4rvRoQXqhPAGw"},

    # Spiritual / Devotional (massive audience in Gujarat)
    {"name": "Morari Bapu",            "id": "UCrT4sUjqKjsS5WQBM_nfFcA"},

    # News & Current Affairs
    {"name": "Sandesh News",           "id": "UCBGkHBvO_cLnmMZZ3W6WGBQ"},
    {"name": "TV9 Gujarati",           "id": "UCbEMtP4sE3rEKxiGUGbV59A"},

    # Cricket / Sports (lots of code-mixed in cricket commentary)
    {"name": "Gujarati Sports",        "id": "UCf-Z0u97P_a4pEJFpUKKniw"},

    # Education (Gujarati medium competitive exam prep)
    {"name": "Ojas Bharti",            "id": "UC5aVWeiSHvpjomibx6MNXOQ"},
]

# ── Gujarati Script Detection ─────────────────────────────────────────────────
GUJARATI_UNICODE_RANGE = re.compile(r'[\u0A80-\u0AFF]')
ENGLISH_PATTERN        = re.compile(r'[a-zA-Z]')

def is_gujlish(text: str, min_gujarati: int = 1, min_english: int = 1) -> bool:
    """Returns True if text has at least some Gujarati script AND English characters."""
    if not text or len(text.strip()) < 5:
        return False
    has_gujarati = len(GUJARATI_UNICODE_RANGE.findall(text)) >= min_gujarati
    has_english  = len(ENGLISH_PATTERN.findall(text)) >= min_english
    return has_gujarati and has_english

def clean_comment(text: str) -> str:
    """Basic cleaning: remove URLs, newlines, excess whitespace."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── YouTube API ───────────────────────────────────────────────────────────────
def get_youtube_client():
    try:
        from googleapiclient.discovery import build
        return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    except ImportError:
        print("ERROR: google-api-python-client not installed.")
        print("Run: pip install google-api-python-client")
        sys.exit(1)

def get_channel_videos(youtube, channel_id: str, max_videos: int = 10) -> list:
    """Get top video IDs from a channel."""
    video_ids = []
    try:
        response = youtube.search().list(
            channelId=channel_id,
            part='id',
            maxResults=max_videos,
            order='viewCount',
            type='video'
        ).execute()
        for item in response.get('items', []):
            if item['id'].get('videoId'):
                video_ids.append(item['id']['videoId'])
    except Exception as e:
        print(f"  Warning: Could not fetch videos — {e}")
    return video_ids

def get_video_comments(youtube, video_id: str, max_comments: int = 100) -> list:
    """Get comments for a video. Returns list of comment strings."""
    comments = []
    page_token = None
    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                videoId=video_id,
                part='snippet',
                maxResults=min(100, max_comments - len(comments)),
                pageToken=page_token,
                textFormat='plainText',
                order='relevance'
            ).execute()
            for item in response.get('items', []):
                text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                text = clean_comment(text)
                if text:
                    comments.append(text)
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        except Exception as e:
            err = str(e)
            if 'disabled comments' in err.lower() or 'commentsDisabled' in err:
                break  # skip videos with disabled comments
            print(f"  Warning: Comment fetch error — {err[:80]}")
            break
    return comments

# ── Main Scraping Loop ─────────────────────────────────────────────────────────
def main():
    if YOUTUBE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: YouTube API key not set!")
        print("Either:")
        print("  1. Edit YOUTUBE_API_KEY in this script")
        print("  2. Run: set YOUTUBE_API_KEY=your_actual_key  (Windows)")
        print("     then: python scripts/phase1b_expand_dataset.py")
        print()
        print("To get a free YouTube API key:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a project → Enable 'YouTube Data API v3'")
        print("  3. Create an API key → Paste it here")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 1b — Expand Dataset (YouTube Scraping)")
    print("=" * 60)
    print(f"Channels to scrape : {len(GUJARATI_CHANNELS)}")
    print(f"Max per channel    : {MAX_COMMENTS_PER_CHANNEL} comments")
    print(f"Expected raw total : ~{len(GUJARATI_CHANNELS) * MAX_COMMENTS_PER_CHANNEL:,} comments")
    print()

    youtube = get_youtube_client()

    all_raw  = []     # all comments
    gujlish  = []     # filtered Gujlish only

    for ch in tqdm(GUJARATI_CHANNELS, desc="Channels"):
        ch_name = ch['name']
        ch_id   = ch['id']
        print(f"\n  [{ch_name}] Fetching top videos...")

        video_ids = get_channel_videos(youtube, ch_id, MAX_VIDEOS_PER_CHANNEL)
        if not video_ids:
            print(f"  [{ch_name}] No videos found — channel ID may be wrong, skipping")
            continue

        ch_comments = 0
        for vid_id in video_ids:
            if ch_comments >= MAX_COMMENTS_PER_CHANNEL:
                break
            comments = get_video_comments(
                youtube, vid_id,
                max_comments=MAX_COMMENTS_PER_CHANNEL - ch_comments
            )
            for c in comments:
                all_raw.append({
                    "text": c,
                    "channel": ch_name,
                    "video_id": vid_id,
                    "is_gujlish": is_gujlish(c)
                })
                if is_gujlish(c):
                    gujlish.append({
                        "text": c,
                        "channel": ch_name,
                        "label": None   # unlabeled — for MLM pre-training
                    })
            ch_comments += len(comments)
            time.sleep(0.1)  # gentle rate limiting

        print(f"  [{ch_name}] {ch_comments} comments scraped, {sum(1 for r in all_raw[-ch_comments:] if r['is_gujlish'])} Gujlish")

    # ── Save raw ──────────────────────────────────────────────────────────────
    raw_path = DATA_RAW / "expanded_comments_raw.csv"
    pd.DataFrame(all_raw).to_csv(raw_path, index=False, encoding='utf-8-sig')

    # ── Save filtered Gujlish ─────────────────────────────────────────────────
    guj_df    = pd.DataFrame(gujlish).drop_duplicates(subset=['text'])
    guj_path  = DATA_PROCESSED / "expanded_gujlish.csv"
    guj_df.to_csv(guj_path, index=False, encoding='utf-8-sig')

    # ── Merge with existing dataset ───────────────────────────────────────────
    orig_path = DATA_PROCESSED / "gujlish_clean.csv"
    if orig_path.exists():
        orig_df = pd.read_csv(orig_path, encoding='utf-8-sig')
        # Original has 'text' and 'label' columns
        # New scraped has 'text' and 'label' (None for unlabeled)
        combined = pd.concat([
            orig_df[['text', 'label']],
            guj_df[['text', 'label']]
        ], ignore_index=True).drop_duplicates(subset=['text'])

        # Save combined — labeled rows for training, all rows for MLM
        combined_path = DATA_PROCESSED / "combined_full_dataset.csv"
        combined.to_csv(combined_path, index=False, encoding='utf-8-sig')

        labeled   = combined[combined['label'].notna()]
        unlabeled = combined[combined['label'].isna()]

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Original dataset           : {len(orig_df):,} sentences")
        print(f"New Gujlish scraped        : {len(guj_df):,} sentences")
        print(f"Combined total             : {len(combined):,} sentences")
        print(f"  — Labeled (for training) : {len(labeled):,}")
        print(f"  — Unlabeled (for MLM)    : {len(unlabeled):,}")
        print()
        print(f"Saved to: {combined_path}")
    else:
        print(f"\nWARNING: Original dataset not found at {orig_path}")
        print(f"Saving new data only to {guj_path}")
        print(f"New Gujlish sentences: {len(guj_df):,}")

    print("\n✅ Phase 1b complete!")
    print(f"Raw comments saved  : {raw_path}")
    print(f"Gujlish filtered    : {guj_path}")
    print("\nNext step: Send combined_full_dataset.csv to friend's GPU machine")
    print("           and run run_option_a.py with the expanded dataset.")

if __name__ == "__main__":
    main()
