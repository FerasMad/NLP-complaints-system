"""
Scrape Arabic 1-2 star reviews from Saudi food delivery apps on Google Play.
Used to source data for التوصيل and دقة الطلب categories.
"""
import csv
import time
from google_play_scraper import reviews, Sort

# Saudi food delivery apps
APPS = {
    "hungerstation": "com.hungerstation.android.web",
    "jahez": "net.jahez",
    "mrsool": "com.mrsool",
    "talabat": "com.talabat",
}

OUT = "data/raw/play_store_reviews.csv"


def fetch_app(app_id, count=500):
    """Get up to `count` Arabic 1-2 star reviews."""
    all_rows = []
    for score in [1, 2]:
        try:
            result, _ = reviews(
                app_id,
                lang="ar",
                country="sa",
                sort=Sort.NEWEST,
                count=count,
                filter_score_with=score,
            )
            for r in result:
                all_rows.append({
                    "app": app_id,
                    "rating": r["score"],
                    "text": r["content"] or "",
                })
            print(f"  {app_id} score={score}: {len(result)} reviews")
        except Exception as e:
            print(f"  {app_id} score={score}: error {e}")
        time.sleep(1)
    return all_rows


def main():
    rows = []
    for name, app_id in APPS.items():
        print(f"Scraping {name}...")
        rows.extend(fetch_app(app_id))

    # drop empty texts
    rows = [r for r in rows if r["text"].strip()]
    print(f"\nTotal scraped: {len(rows)} reviews")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["app", "rating", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
