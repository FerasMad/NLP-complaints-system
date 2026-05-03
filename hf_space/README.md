---
title: Arabic Restaurant Complaints Classifier
emoji: 🍽️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: 8-class Arabic restaurant complaint classifier — 95% test accuracy
tags:
  - arabic
  - nlp
  - classification
  - bert
  - saudi
  - dialectal-arabic
models:
  - CAMeL-Lab/bert-base-arabic-camelbert-mix
  - UBC-NLP/MARBERT
  - aubmindlab/bert-base-arabertv02
language:
  - ar
---

# Arabic Restaurant Complaints Classifier

Classify Arabic restaurant complaints into 8 actionable categories. Saudi-Gulf dialect specialization.

## Performance

| Metric | Value |
|---|---:|
| Test accuracy | 95.05% |
| Test weighted F1 | 95.08% |
| Test macro F1 | 92.03% |
| Min class F1 (عامة) | 84.84% |

## Categories

| Arabic | English |
|---|---|
| التوصيل | Delivery |
| السعر والقيمة | Price / value |
| النظافة | Cleanliness |
| جودة الطعام | Food quality |
| خدمة الموظفين | Staff service |
| دقة الطلب | Order accuracy |
| عامة | General |
| وقت الانتظار | Wait time |

## Source

https://github.com/FerasMad/NLP-complaints-system
