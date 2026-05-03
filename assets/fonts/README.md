# Thmanyah Typeface

This directory contains the Thmanyah Typeface (3 families × 5 weights, WOFF2 format) used for the project's visual identity.

**Source:** [Thmanyah](https://thmanyah.com/) — a Saudi media company. Their typeface is freely distributed under their custom license (see `THMANYAH_FONT_LICENSE.pdf`).

## Families

| Family | Use case | Weights |
|---|---|---|
| `thmanyahsans` | UI body text, buttons, labels | Light, Regular, Medium, Bold, Black |
| `thmanyahseriftext` | Long-form body text (paragraph) | Light, Regular, Medium, Bold, Black |
| `thmanyahserifdisplay` | Headlines, hero copy | Light, Regular, Medium, Bold, Black |

## CSS @font-face declarations

The Gradio app (`app/space_app.py`) loads these via inline @font-face CSS using `gr.Files()` to serve them. For HF Spaces, the same files live under `hf_space/fonts/`.

Example CSS for external use:

```css
@font-face {
    font-family: "Thmanyah Sans";
    src: url("/fonts/thmanyahsans-Regular.woff2") format("woff2");
    font-weight: 400;
}
@font-face {
    font-family: "Thmanyah Display";
    src: url("/fonts/thmanyahserifdisplay-Bold.woff2") format("woff2");
    font-weight: 700;
}
body { font-family: "Thmanyah Sans", "Segoe UI", sans-serif; }
h1 { font-family: "Thmanyah Display", serif; font-weight: 700; }
```

## License

See `THMANYAH_FONT_LICENSE.pdf` (extracted from the original distribution). Free for the use cases that license describes (typically open-source / non-commercial / portfolio).
