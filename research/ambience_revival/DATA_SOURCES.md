# Data Sources for Ambience Revival

The single biggest reason v3 ambience failed was **not enough genuine ambience data**. The class was inferred from ratings + scattered hints rather than collected deliberately. If we retry, we collect deliberately.

## Bar to clear before retraining

We need approximately:

- **300+ clean ambience-positive rows**, balanced across the 12 subtypes
- **600+ hard-negative rows** (food / staff / delivery / cleanliness / wait / order / price complaints that mention place-words like المكان, الفرع, الجلسة)
- **At least 80 hand-written test cases** (provided in `test_examples.csv`)

If we can't reach the bar, drop the experiment. Don't retrain on insufficient data — that's how v3 died.

## Sources, ranked by signal-to-noise

### Tier 1 — Permitted, high signal

**Google Places API (Place Details → Reviews)**
- Official Google API, free quota up to ~100k requests/month
- Reviews include star rating, text, and language tag — filter for `ar` and Saudi cities
- Documentation: <https://developers.google.com/maps/documentation/places/web-service/details>
- License: per Google's TOS, store results only as long as needed for your use; cite Google Maps in any visualization
- **Signal:** medium-high. Reviews on Google Places skew toward in-restaurant experiences (vs. delivery apps), which means more ambience mentions
- **Cost:** ~free for our volume

**AraMA / AraMAMS (Arabic multi-aspect sentiment datasets)**
- Academic Arabic ABSA datasets that include an "ambience" or "environment" aspect
- Search for: AraMA on HuggingFace Hub, AraMAMS on the original authors' GitHub
- License: typically academic/research, check per dataset
- **Signal:** high (already aspect-tagged). **Risk:** small (a few hundred rows of ambience at best)
- Use as gold seed, not as the bulk

**Hand-written by the team**
- Each labeler writes 10–15 examples per ambience subtype
- Use the prompts in `BOUNDARIES.md` as templates
- **Signal:** highest (no labeling noise). **Cost:** ~1 day of team work

### Tier 2 — Use carefully

**Twitter / X via the X API v2 (academic or paid tier)**
- Search query: `place_country:SA lang:ar (مطعم OR محل OR كافيه) -is:retweet`
- Then filter by ambience keywords from `keywords.py`
- License: per X TOS, requires API access (not scraping)
- **Risk:** very noisy, lots of off-topic results. Expect 10–20% yield after filtering
- **Avoid:** scraping the public-facing site. It violates TOS and gets your IP banned

**Yelp / TripAdvisor exports for restaurants in Saudi Arabia**
- These platforms have less Saudi/Gulf coverage than Google Places, but worth a pass
- License: their TOS prohibits scraping. **Only use the official API** if available, or partnership data
- **Signal:** medium

### Tier 3 — Do not use

- **HungerStation / Jahez / Mrsool / Talabat scraping** — these were used for v4 production data but are unlikely to have ambience signal. Delivery reviews are about food and the courier, almost never about the restaurant's seating. Don't waste effort here.
- **Public scrapers without permission** — TOS-violating, ethical risk, and the labeling burden of cleaning the noise outweighs the volume.
- **AI-generated synthetic ambience data** — the v3 ambience problem was distribution mismatch with reality, not insufficient volume. Synthesizing more of the wrong distribution makes it worse.

## Suggested collection plan

| Source | Target rows | Effort | Yield estimate |
|---|---|---|---|
| Google Places API (Riyadh + Jeddah + Dammam, top 200 restaurants each) | 600 reviews → ~120 ambience after filter | 1 day setup + 1 day filtering | high signal |
| AraMA / AraMAMS academic seed | 100–200 ambience rows | 0.5 days | very high signal |
| Hand-written by team (4 labelers × 12 subtypes × 5 examples) | 240 rows | 1 day | very high signal |
| Hard-negative pool (existing v4 training data, sampled) | 600 confusable rows (food/staff/clean mentioning place-words) | 0.5 days | n/a — these go to filter as `hard_negative_candidates.csv` |

Total target: **~500 ambience candidates + 600 hard-negatives**. After labeling pass: hopefully 300+ clean ambience positives. If not, abort.

## Storage and licensing rules

- Keep raw API responses in `data/raw/ambience_v5/<source>/` — gitignored, never push to GitHub
- Store cleaned CSVs in `data/processed/ambience_v5/` — also gitignored
- Maintain a `data/processed/ambience_v5/SOURCES.md` tracking exactly which row came from which source, with timestamp and license note
- For Google Places: include the `place_id` and `review_id` in the row metadata so we can honor delete-requests later
- Never publish the raw collected data anywhere outside the team's private storage

## What "ethical" means here

- Use official APIs with rate-limit-respecting clients
- Honor robots.txt and TOS
- Don't republish raw third-party reviews — only train weights and report aggregate metrics
- Cite data sources in any paper, model card, or portfolio writeup
