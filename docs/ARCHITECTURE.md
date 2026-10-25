# Architecture

## Application lifecycle

[`main.dart`](../prototype/lib/main.dart) initializes Firebase, loads application settings, configures local storage sinks, and starts the settings-aware GPS bridge. The Flutter application exposes monitoring, calibration, settings, and log-management screens.

## Monitoring pipeline

[`camera_engine.dart`](../prototype/lib/features/driver_monitoring/ui/camera_engine.dart) coordinates the native face-landmarker interface, ONNX session, head-pose calibration, sustained-event rules, and alert cooldowns.

[`image_utils.dart`](../prototype/lib/features/driver_monitoring/utils/image_utils.dart) provides image transformations and the classifier output order: `drowsy`, `focused`, `using phone`, `yawning`. Preprocessing and label order must match the trained model.

The `face_landmarker` method channel provides the interface to native landmark processing. Its native implementation and model assets are required to run the complete application.

## Feedback

[`tts_service.dart`](../prototype/lib/features/driver_monitoring/utils/tts_service.dart) manages spoken feedback. [`gemini_service.dart`](../prototype/lib/features/driver_monitoring/utils/gemini_service.dart) optionally generates short alert wording. Monitoring decisions remain in the vision and temporal-rule pipeline.

## Storage and observability

`driver_logs.dart`, `metrics.dart`, and `metrics_exporter.dart` implement event aggregation and instrumentation. SQLite supports local records, while Firestore supports cloud log storage. The [Streamlit dashboard](../prototype/driver_log_dashboard.py) reads per-minute aggregates for fleet summaries and map views.

## Model utilities

[`fps_model_test.py`](../prototype/fps_model_test.py) benchmarks model execution using synthetic tensors. It excludes camera decoding, preprocessing, native landmarks, networking, and feedback.

[`pth_pt.py`](../prototype/pth_pt.py) is a TorchScript conversion experiment. Its architecture, output dimensions, and checkpoint parameters must match before using the exported model.

## Implementation considerations

- Prototype authentication stores plaintext passwords; replace it before using real accounts.
- Enabling monitoring starts GPS and enables driver logging in the engine. Settings and permission behavior need integration testing.
- The exposure heuristic samples compressed JPEG bytes and should be evaluated against decoded luminance.
- Checkpoint conversion uses `strict=False`; missing or unexpected weights require inspection.
- Client-side API keys are extractable; deployed AI integrations need a server-side credential boundary.
- Driver identity, location, and conversations require appropriate access control, retention, and consent.
