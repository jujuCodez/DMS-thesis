# Driver Attention Monitoring System
### AI-Assisted Driver Monitoring for Fleet Operations

A Computer Engineering undergraduate thesis developed at **De La Salle University**, integrating computer vision, mobile application development, and fleet telemetry to monitor driver attention and deliver contextual alerts.

The project combines camera-based driver-state classification with facial landmarks, head-pose calibration, and temporal decision rules. A Flutter application provides monitoring and feedback, while a Streamlit dashboard supports review of aggregated driver events.

[Architecture](docs/ARCHITECTURE.md) · [Prototype setup](prototype/README.md) · [Validation notes](prototype/VALIDATION.md)

## Core Capabilities

| Component | Implementation |
|---|---|
| **Driver-state classification** | MobileViT v2 inference through ONNX Runtime, with preprocessing and prediction handling in Dart |
| **Attention monitoring** | Native MediaPipe face-landmarker interface, calibrated head-pose baselines, sustained-event detection, and alert cooldowns |
| **Mobile application** | Flutter interface for camera streaming, calibration, monitoring settings, profiles, and log review |
| **Driver feedback** | Audio and text-to-speech alerts, with optional Gemini-generated phrasing |
| **Telemetry and storage** | SQLite and Firestore logging, GPS integration, event aggregation, and timing instrumentation |
| **Fleet analytics** | Streamlit dashboard for reviewing driver-event summaries and geographic information |
| **Model tooling** | PyTorch/timm utilities for checkpoint inference timing and experimental TorchScript conversion |

The Flutter classifier uses four output labels: **drowsy**, **focused**, **using phone**, and **yawning**. Separate monitoring rules handle conditions such as looking away and loss of face detection.

## System Architecture

```text
Camera Stream (MJPEG)
        |
        v
Frame Decoding and Preprocessing
        |
        +--> Native Face Landmarks and Head Pose
        |
        +--> MobileViT v2 Classification (ONNX Runtime)
        |
        v
Calibration, Dwell-Time Rules, and Alert Management
        |
        +--> Monitoring Interface and Audio/TTS Feedback
        |           |
        |           +--> Optional Gemini-Assisted Alert Wording
        |
        v
Event Logging: SQLite / Firestore
        |
        v
Fleet Dashboard: Streamlit
```

The vision pipeline and temporal rules determine monitoring states. The optional language-model integration provides alert wording.

## Repository Structure

| Directory or File | Description |
|---|---|
| [`prototype/lib/`](prototype/lib/) | Flutter application source |
| [`prototype/driver_log_dashboard.py`](prototype/driver_log_dashboard.py) | Fleet-event analytics dashboard |
| [`prototype/fps_model_test.py`](prototype/fps_model_test.py) | Synthetic-input model inference benchmark |
| [`prototype/pth_pt.py`](prototype/pth_pt.py) | Experimental checkpoint-to-TorchScript utility |
| `model_trainers/` | Earlier model-training material |
| `app_codes_v8/` and `android test code/` | Earlier application and platform experiments |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component architecture and implementation notes |
| [`prototype/README.md`](prototype/README.md) | Environment requirements and setup guidance |
| [`prototype/VALIDATION.md`](prototype/VALIDATION.md) | Completed checks and verification scope |

Earlier Python/Kivy experiments are retained as part of the project's development history. Their model configurations and deployment paths differ from the Flutter prototype.

## Setup and Reproducibility

This repository contains the mobile application sources, dashboard, and supporting model utilities. Running the complete system requires the Flutter manifest and dependency lockfile, native platform projects, face-landmarker implementation, model and audio assets, and development-backend configuration.

See the [setup guide](prototype/README.md) for component-specific prerequisites. Credentials and original Firebase project configuration are excluded.

Trained model weights, datasets, camera firmware, and some platform/build files are not included. Refer to the [setup guide](prototype/README.md) for requirements and [validation notes](prototype/VALIDATION.md) for the scope of the available checks.

## Research Scope

The system is an academic prototype intended for technical study and experimentation. Deployment would require further validation of detection performance, reliability, authentication, and the handling of driver information.

The [architecture documentation](docs/ARCHITECTURE.md) describes the monitoring pipeline, logging behavior, and current implementation considerations. Use synthetic data and an isolated development backend for testing.

## Academic Context and Contributors

**Thesis:** *Driver Attention Monitoring System with AI-Based Assistant for Fleet Operations*  
**Institution:** De La Salle University  
**Program:** Bachelor of Science in Computer Engineering

Developed collaboratively by:

- Michael Judrick M. Aringo
- Henson Adrian T. Lee
- Gabriel Alexis C. Ocampo
- Jared Craig C. Ong

## Usage and Licensing

A project-wide license has not been established for this source release. Please contact the authors regarding reuse or redistribution. Third-party libraries, models, and tools remain subject to their respective licenses.
