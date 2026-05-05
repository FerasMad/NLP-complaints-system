# Design

Visual system for the live HuggingFace Space at `hf_space/app.py`. Identity is warm, hospitable, Saudi-rooted, judged as a portfolio artifact by recruiters.

## Theme

Light, warm, daylight. Reads like a magazine spread about Saudi food culture, not a SaaS dashboard. Cream paper surface, terracotta as committed accent, deep ink type. Earth tones throughout, no neon, no glow.

## Color tokens

Defined as CSS custom properties in `hf_space/app.py`:

| Token | Hex | OKLCH | Role |
|---|---|---|---|
| `--cream` | `#F5EFE6` | `0.95 0.014 75` | page background, hero text on terracotta |
| `--paper` | `#EEE6D7` | `0.92 0.020 75` | card / textarea surface |
| `--ink` | `#2D211A` | `0.22 0.018 50` | primary text, primary CTA |
| `--ink-muted` | `#5C4F45` | `0.45 0.015 50` | secondary text, labels |
| `--terracotta` | `#C75D3D` | `0.58 0.13 38` | hero band, rank-1 prediction, accent |
| `--terracotta-deep` | `#A14828` | `0.50 0.13 38` | hero gradient base, hover states, link border |
| `--terracotta-tint` | `rgba(199,93,61,0.10)` | — | focus ring, link underline |
| `--olive` | `#5F6845` | `0.45 0.060 110` | secondary accent, rank-2 numerals |
| `--olive-tint` | `rgba(95,104,69,0.10)` | — | rank-2 background |
| `--border` | `#D9CFC0` | `0.83 0.012 75` | hairline dividers |
| `--border-strong` | `#C9BDA8` | `0.77 0.015 75` | secondary button border |

**Color strategy: Committed.** Terracotta carries 25-35% of visual surface (the entire hero band plus the rank-1 prediction tile). Not an accent dot.

## Typography

Single family. **Almarai** (Google Fonts), weights 300, 400, 700, 800. Arabic and Latin glyphs in one file; no font-swap on language change.

Why Almarai: warm humanist character without the Tajawal default reflex; Saudi heritage in name and origin; weights span 300-800 for hierarchy.

### Scale (modular, ratio ≥ 1.25)

| Use | Size | Weight |
|---|---|---|
| Hero h1 | `clamp(2.2rem, 5.4vw, 3.6rem)` | 800 |
| Lede | `clamp(1rem, 1.5vw, 1.15rem)` | 400 |
| Result top-1 category | `clamp(1.15rem, 2.2vw, 1.45rem)` | 800 |
| Result top-1 percentage | `clamp(1.4rem, 3vw, 2rem)` | 800 |
| Body | `1rem` | 400 |
| Section label | `0.74rem` (uppercase, tracked 0.16em) | 700 |
| Small / meta-key | `0.68rem` (uppercase, tracked 0.14em) | 700 |

Line-height: 1.7 for body, 1.15 for hero, 1.75 for prose. RTL throughout for Arabic; LTR isolated for English meta values inside `.meta-item` and `.result-pct`.

## Layout

980px max width, single column. Three vertical sections:

1. **Hero band** (terracotta drench, RTL, asymmetric — title hugs right edge)
2. **Workspace** (cream surface, generous padding, input → result → examples → about, separated by hairline dividers)
3. **Footer** (paper surface, links right, attribution left)

Spacing: fluid `clamp()` for section padding, fixed `gap` inside components. No 8px-grid orthodoxy; rhythm via varied generous gaps (24px/32px/48px/56px).

## Components

### Hero
Full-width terracotta band with subtle radial highlights (top-right warm, bottom-left shadow). Eyebrow → headline → lede → meta strip with 4 stat columns. Meta strip border-top hairline in cream-alpha.

### Textarea (input)
Paper surface, generous padding (22px / 24px), 14px radius, 1px hairline border. Focus state: terracotta border + 4px terracotta-tint focus ring. Min-height 140px, vertical resize.

### Buttons
- **Primary** (`button.primary`): ink background, cream text. Hover: terracotta-deep. Subtle 1px translate-Y on hover. The primary CTA is ink (not terracotta) so the rank-1 result owns the terracotta.
- **Secondary** (`button.secondary`): transparent, ink-muted text, border-strong border. Hover: paper background, ink text.

### Result rows
Three-row grid (`auto 1fr auto`): rank meta column (Arabic + English) | category | percentage. Three rank treatments:

- **Rank 1**: terracotta drench, cream type. Largest type, strongest visual weight.
- **Rank 2**: olive-tint background with subtle olive-alpha border. Ink type.
- **Rank 3**: paper background with hairline border. Ink-muted type.

No side-stripe borders. No bars. Color is the rank signal; type weight is the priority signal.

### Empty / message states
Dashed hairline border, paper background, terracotta middot mark, bilingual headline + sub. Used when input is empty, too short, or non-Arabic.

### Example chips
Pill-shaped (999px radius), transparent ground, 1px hairline border. Hover: ink ground, cream type. RTL text alignment.

### Accordion
Collapses the about section. Hairline border-top instead of full card. Body uses prose styles with terracotta-tinted link underlines.

### Footer
Paper surface, hairline border-top. Made-by left, links right (Source on GitHub, Model on HF Hub). Links are ink with terracotta-deep hover and a hairline underline that fades in.

## Motion

Restraint. Three transitions only:

- Buttons: `background 140ms ease, transform 140ms ease`
- Textarea focus: `border-color 160ms, box-shadow 160ms`
- Result row hover: `transform 140ms` (translate-X by -2px in RTL)

No entrance animations, no scroll-triggered reveals. The brand is hospitable, not theatrical.

`@media (prefers-reduced-motion: reduce)` clamps all durations to 0.01ms.

## Imagery

None. The Space is a tool with strong typographic and color identity; brand bans on "zero imagery for food briefs" are waived because this is a dev-tool register exception. Identity carries via palette weight, type, and layout.

## Bans observed

- No side-stripe borders (>1px colored accent on cards / rows)
- No gradient text (`background-clip: text`)
- No glassmorphism / backdrop-filter blur as decoration
- No identical card grids
- No em dashes in copy (commas, colons, periods, parens only)
- No reflex-default fonts (Inter / DM Sans / Plex / Cormorant / Fraunces)
- No generic dark + neon SaaS palette (the prior version, now replaced)

## Responsive

Single breakpoint at 640px. Mobile collapses the result grid to two rows (rank + category, then percentage on its own line, left-aligned). Buttons stretch to fill row.

## Files

- `hf_space/app.py` — application code, embedded CSS, identity definitions
- `hf_space/README.md` — HF Space metadata frontmatter (title, emoji, sdk_version)
- `hf_space/requirements.txt` — pinned deps for the Space container
