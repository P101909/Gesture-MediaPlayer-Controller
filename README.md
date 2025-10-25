<!-- Header Banner -->

<h1 align="center">🎯 Gesture Recognition & Control</h1>
<p align="center">
  <b>CMPT 310 · Fall 2025 · SFU</b><br>
  <i>Hybrid AI system that blends motion detection and gesture recognition for video players.</i>
</p>

---

## 🌟 Overview

This project is a **hybrid hand gesture recognition system** that merges:
- 🤚 **Static gestures** (Play, Pause, Stop) recognized via **MediaPipe Model Maker**, and  
- 👋 **Dynamic gestures** (Wave Left/Right, Double Swipe) detected using **motion tracking** of hand landmarks.  

It’s designed for **real-time media control**, **ASL integration**, and **custom gesture interactions**.

---

## ✨ Features

| 🎨 Type | 🧠 Example | ⚙️ Description |
|----------|-------------|----------------|
| 🖐 **Static Gestures** | 👍 Play · ✋ Pause · 👊 Stop | Trained using **MediaPipe Model Maker** (`.task` model). |
| 👋 **Dynamic Gestures** | Wave Left → ⏮️ Previous · Wave Right → ⏭️ Next | Real-time **motion tracking** of wrist x-direction. |
| 🔊 **Custom Gestures** | Swipe Up/Down for Volume · Double Swipe Left = Skip 30s | Hand motion logic or a small temporal **LSTM** model. |
| 🤟 **ASL Mode (Optional)** | A, B, C... | Recognize ASL letters via **ASL Alphabet Dataset**. |
| 🎥 **Live Camera Control** | Control videos / apps using gestures | Implemented with **OpenCV + MediaPipe Tasks**. |

---

## 🧱 Project Structure

```bash
Gesture_Project/
├── config/
│   └── .......          # Map HaGRID → custom labels
├── set_data/
│   ├── .../                    # Raw dataset or captured images
│   ├── .../              # After label remap
│   └── .../                 # train/val/test
├── models/
│   └── ...
├── scripts/
│   ├── ....py
│   ├── ....py
│   ├── ....py
│   ├── ....py     
│   └── ....py
├── .gitignore
├── Makefile
└── README.md
