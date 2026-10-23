---
title: GeoAI Outdoor Planner
emoji: 🌤️
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Where and when to go outdoors, from live geo data + an SLM
---

# GeoAI Outdoor Planner

Ask where and when to go outdoors; answers are grounded in live OpenStreetMap, Open-Meteo weather and CAMS
air-quality data and written by a small open language model
([Qwen3.5-2B + LoRA](https://huggingface.co/NamasteWorld/geoai-outdoor-qwen3.5-2b-lora)).

This page is the permanent front door. The assistant runs on the author's GPU workstation; the page reads the
current API address from [`backend.json`](backend.json) and calls it directly.

* API: `POST <url in backend.json>/ask` with `{"query": "..."}`, or `GET <url>/ask?q=...`
* Code: https://github.com/harshit-singh-hs/GeoAI-Hackathon-2026

Map data © OpenStreetMap contributors. Weather © Open-Meteo.com (CC BY 4.0). Air quality: CAMS via Open-Meteo.
