# Driver Attention Monitoring System

Computer Engineering undergraduate team thesis at De La Salle University: camera-based driver-state monitoring, alerts, and fleet-log review.

**Interviewers: start with [the review guide](docs/REVIEW_GUIDE.md).** The [Flutter prototype](prototype/) is a source snapshot, not a complete independently buildable mobile project.

## Implemented components

- Flutter/Dart UI: MJPEG frames, calibration, monitoring controls, profiles, logs and test tools.
- MobileViT v2 ONNX inference and a native MediaPipe face-landmarker channel, with pose baselines, dwell timers and alert cooldowns.
- Audio/TTS alerts with optional Gemini wording and request instrumentation.
- Local SQLite and Firestore log sinks, GPS, timing and event aggregation.
- Python/Streamlit fleet dashboard with event summaries and map views.
- PyTorch/timm model timing and a TorchScript conversion experiment.

Flutter classifier label order: `drowsy`, `focused`, `using phone`, `yawning`. Looking-away and no-face conditions are handled separately. Earlier Python/Kivy experiments have different configurations.

## Repository map

| Path | Purpose |
|---|---|
| [prototype/lib/](prototype/lib/) | Flutter source extracted from the thesis archive |
| [prototype/driver_log_dashboard.py](prototype/driver_log_dashboard.py) | Fleet dashboard |
| [prototype/fps_model_test.py](prototype/fps_model_test.py) | Synthetic-input model timing |
| [prototype/pth_pt.py](prototype/pth_pt.py) | Historical TorchScript conversion experiment |
| [docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md) | Walkthrough and engineering findings |
| [prototype/README.md](prototype/README.md) | Setup prerequisites and missing artifacts |
| [prototype/PUBLICATION_NOTES.md](prototype/PUBLICATION_NOTES.md) | Publication changes |
| model_trainers/, app_codes_v8/, android test code/ | Existing earlier material, preserved |
| [docs/legacy-kivy-readme.md](docs/legacy-kivy-readme.md) | Historical previous README |

## System flow

```text
MJPEG camera -> preprocessing -> native landmarks + ONNX classification
 -> calibration / dwell rules -> UI + audio / TTS (+ optional Gemini)
 -> local SQLite / optional Firestore logs -> Streamlit fleet dashboard
```

## Reproduction status

The archive contains three Python scripts and Flutter `lib/` sources. It does NOT contain the original `pubspec.yaml`/lockfile, Android/iOS scaffolding, native `face_landmarker` implementation, model/audio assets, firmware, datasets, checkpoints, or a reproducible accuracy-evaluation bundle. No hardware run, mobile build, live-cloud integration or accuracy result was independently reproduced for publication. See the static validation report.

## Team and provenance

Thesis: **Driver Attention Monitoring System with AI-Based Assistant for Fleet Operations**. Team: Michael Judrick M. Aringo, Henson Adrian T. Lee, Gabriel Alexis C. Ocampo, Jared Craig C. Ong. Repository ownership does not imply sole authorship.

Snapshot source: `DSP-1-2425-C1-Thesis Prototype Codes.zip`. Original-file hashes and publication changes are documented. Earlier repository history remains intact.

## Research limitations

This is a research prototype, not a validated automotive safety product. Archived authentication stores plaintext passwords and needs replacement before real accounts are used. Location, identity and conversation logs need appropriate access controls, consent and retention. Use synthetic data and an isolated backend for review.

## Licensing

No new license grant is made by this upload. The old README's MIT/commercial-use wording was inconsistent, and no project license file accompanied the archive. Confirm permissions with the authors before reuse; dependencies retain their own licenses.
