# Prototype setup and source review

This folder is an incomplete source snapshot, not a ready-to-run Flutter project.

## Flutter prerequisites

Restore the original `pubspec.yaml`, lockfile, Flutter/Dart versions, Android/iOS scaffolding and platform permissions. Restore the `face_landmarker` native method channel described in `lib/face_landmarker_channel.dart`.

Missing assets include `assets/models/DMS_mobilevitv2_100.onnx`, `face_landmarker.task`, and audio referenced in source. Restore asset declarations and verify model preprocessing/label metadata.

Generate `lib/firebase_options.dart` using `flutterfire configure` for YOUR isolated development project. The comment-only `.example` file cannot substitute for generated configuration. Configure Firestore rules appropriately; original cloud identifiers are excluded.

Packages inferred from imports (not a verified version lock): audioplayers, cloud_firestore, file_selector, firebase_core, flutter_file_dialog, flutter_tts, geolocator, http, image, onnxruntime, path, permission_handler, share_plus, shared_preferences, speech_to_text, sqflite, and Flutter SDK.

Configure MJPEG in Settings or via `CAMERA_STREAM_URL` at build time. Generic default: `http://192.168.4.1:81/stream`.

Optional Gemini uses `String.fromEnvironment('GEMINI_API_KEY')`; an empty key disables generation. `--dart-define=GEMINI_API_KEY=...` is for personal development only: compiled client keys are extractable. Deployment needs a server-side secret boundary. The archived preview model name may need updating in your own account.

## Dashboard

From the repository root in an isolated Python environment:

```sh
python -m pip install -r prototype/requirements-dashboard.txt
python -m streamlit run prototype/driver_log_dashboard.py
```

Configure Application Default Credentials for YOUR test project, for example `GOOGLE_APPLICATION_CREDENTIALS` pointing to a local service-account file. No records or credentials are supplied. The dashboard connects to Firestore at startup. `fetch_logs` documents `driver_logs_new`: identifiers, timestamps, GPS aggregates and warning/alert counts. Do not connect reviewers to production.

## Model utilities

```sh
python -m pip install -r prototype/requirements-model-tools.txt
python prototype/fps_model_test.py
```

`DMS_MODEL_DIR` selects the directory of checkpoint filenames in `MODEL_PATHS`; missing files are reported. `WARMUP` and `ITERS` control timing. This measures synthetic-input model execution, not end-to-end throughput or accuracy. Use trusted checkpoints only; compatibility loading can fall back to ordinary `torch.load`.

`pth_pt.py` reads `DMS_CHECKPOINT` (default `models/mobilenetv4.pth`, relative to working directory), writing a `_scripted.pt` sibling. This historical TorchScript experiment is not an ONNX exporter. Verify fixed architecture/class count and `strict=False` missing/unexpected parameters before relying on the output.

Dependencies were inferred, not reconstructed as a tested locked environment. See `VALIDATION.md` for actual checks.
