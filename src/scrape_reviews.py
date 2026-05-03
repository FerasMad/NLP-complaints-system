"""
Scrape Arabic 1-3 star reviews from Saudi food delivery + restaurant apps on Google Play.
Used to source data for التوصيل and دقة الطلب categories.

Pagination via continuation_token so we get thousands per app, not just one page.
"""
import csv
import time

from google_play_scraper import Sort, reviews

# Saudi food delivery + restaurant apps
APPS = {
    "hungerstation": "com.hungerstation.android.web",
    "jahez": "net.jahez",
    "mrsool": "com.mrsool",
    "talabat": "com.talabat",
    "thechefz": "com.thechefz.app",
    "yum": "com.yum.android.startup",
    "carriage": "qa.com.carriage",
    "toyou": "com.toyou.app",
    "albaik": "com.albaik.android",
    "alromansiah": "com.alromansiah.client",
    "kfc_sa": "com.kfc.sa",
    "shawerma_house": "com.shawermahouse",
}

# How many reviews to pull per (app, score, sort) combo. Pagination loops in steps of 200.
TARGET_PER_BUCKET = 1000
PAGE_SIZE = 200
SCORES = [1, 2, 3]  # negative + mid (3-star often has detailed complaints)
SORTS = [Sort.NEWEST, Sort.MOST_RELEVANT]

OUT = "data/raw/play_store_reviews.csv"


def fetch_paginated(app_id, score, sort, target):
    """Pull up to `target` reviews using continuation_token pagination."""
    out = []
    token = None
    while len(out) < target:
        try:
            result, token = reviews(
                app_id,
                lang="ar",
                country="sa",
                sort=sort,
                count=min(PAGE_SIZE, target - len(out)),
                filter_score_with=score,
                continuation_token=token,
            )
        except Exception as e:
            print(f"      error: {e}")
            break
        if not result:
            break
        for r in result:
            out.append({
                "app": app_id,
                "rating": r["score"],
                "text": r["content"] or "",
            })
        if token is None:
            break
        time.sleep(0.5)
    return out


def fetch_app(app_id):
    all_rows = []
    for score in SCORES:
        for sort in SORTS:
            sort_name = "newest" if sort == Sort.NEWEST else "relevant"
            print(f"    score={score} sort={sort_name}: ", end="", flush=True)
            rows = fetch_paginated(app_id, score, sort, TARGET_PER_BUCKET)
            print(f"{len(rows)} reviews")
            all_rows.extend(rows)
            time.sleep(1)
    return all_rows


def main():
    rows = []
    for name, app_id in APPS.items():
        print(f"Scraping {name} ({app_id})...")
        try:
            rows.extend(fetch_app(app_id))
        except Exception as e:
            print(f"  failed entirely: {e}")
            continue

    # drop empty texts and dedupe by text
    seen = set()
    deduped = []
    for r in rows:
        t = r["text"].strip()
        if not t or t in seen:
            continue
        seen.add(t)
        deduped.append(r)
    print(f"\nScraped {len(rows)} raw rows, {len(deduped)} unique non-empty.")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["app", "rating", "text"])
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
