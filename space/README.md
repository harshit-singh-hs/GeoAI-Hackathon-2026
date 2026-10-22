---
title: GeoAI Outdoor Planner
emoji: 🌤️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# GeoAI Outdoor Planner — public endpoint

Ask where and when to go outdoors; answers are grounded in live OpenStreetMap, Open-Meteo weather and CAMS
air-quality data and written by a small open language model (Qwen3.5-2B + LoRA).

* Web UI: `/`
* API: `POST /ask` with `{"query": "..."}` or `GET /ask?q=...`
* Health: `/health`

This Space is a thin gateway: the assistant runs on the author's GPU workstation and is reached through a tunnel.
Map data © OpenStreetMap contributors. Weather © Open-Meteo.com (CC BY 4.0). Air quality: CAMS via Open-Meteo.
