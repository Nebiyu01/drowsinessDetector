# Drowsiness Detector

A real time webcam based drowsiness detector that flags prolonged eye closure and yawning using facial landmarks. The system runs fully on device, requires no cloud services, and is designed to behave calmly rather than firing on every blink.

## Features

* Real time face and landmark tracking using MediaPipe
* Eye Aspect Ratio and Mouth Aspect Ratio signals
* Time window logic to avoid noisy alerts
* Per user calibration to adapt thresholds
* Event based logging with timestamps and reasons
* Runs at real time speed on CPU

## How it works

The detector processes webcam frames and extracts facial landmarks. From these landmarks, it computes two signals.

* Eye Aspect Ratio, a geometric measure of eye openness
* Mouth Aspect Ratio, a measure of mouth opening

Drowsiness is detected when either signal crosses a calibrated threshold for a sustained duration. Alerts use cooldown logic to prevent repeated firing.

## Calibration

Calibration personalizes the eye threshold for each user.

1. Run the calibration script
2. Look straight at the camera with eyes open for 10 seconds
3. The script computes your average EAR
4. The alert threshold is set to average EAR multiplied by 0.75

The values are saved to calibration.json and loaded automatically by the main app.

## Event logging

Alerts are logged to logs.csv only when an alert fires. Each row contains:

* ISO timestamp
* Unix timestamp
* EAR value at trigger time
* MAR value at trigger time
* Trigger reason, eyes_closed or yawn
* Current FPS

Optional session markers can be added with keyboard shortcuts for cleaner analysis.

## Metrics

Example results from local testing on macOS laptop webcam:

* Runtime speed around 28 to 30 FPS on CPU
* Eye closure alerts triggered below calibrated EAR threshold near 0.21
* Yawn alerts triggered above MAR threshold of 0.6
* Normal blinking did not trigger alerts after calibration

## Project structure

* main.py, webcam pipeline and detection logic
* calibrate.py, per user calibration
* logger.py, CSV event logging
* analyze_logs.py, session based analysis
* logs.csv, generated event logs
* calibration.json, generated calibration data

## Setup

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python mediapipe numpy
```

## Run

Calibration:

```bash
python calibrate.py
```

Detector:

```bash
python main.py
```

Keyboard controls:

* q to quit
* s to mark session start
* e to mark session end

## Notes

This project focuses on reliable behavior and clear evaluation rather than model size or visuals. All processing runs locally and no video frames are stored.

## Resume summary

Built a real time webcam drowsiness detector using facial landmarks, per user calibration, and event based logging, running fully on device at around 30 FPS on CPU.
