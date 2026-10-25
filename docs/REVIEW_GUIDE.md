# Interviewer review guide

## Suggested walkthrough

1. `prototype/lib/main.dart`: initialization, settings and local sink wiring.
2. `prototype/lib/features/driver_monitoring/ui/camera_engine.dart`: native landmarks, ONNX inference, calibration, dwell rules, exposure guard, cooldowns.
3. Adjacent `utils/image_utils.dart`: four-label ordering and grayscale-to-three-channel tensor preprocessing; must match the trained model.
4. `utils/metrics.dart`, `metrics_exporter.dart`, `driver_logs.dart`: compare aggregation with `prototype/driver_log_dashboard.py`.
5. `utils/gemini_service.dart` and `tts_service.dart`: optional LLM wording is distinct from the ONNX vision classifier.
6. `prototype/fps_model_test.py`: synthetic model timing rather than an accuracy evaluation or total system latency.

## Static-review observations

| Area | Finding / follow-up |
|---|---|
| Authentication | `auth_controller.dart` and `local_app_db.dart` store/compare plaintext passwords. Replace before real accounts. |
| Reproduction | Native channel, assets and Flutter build metadata missing. Restore before claiming a working checkout. |
| Settings | `CameraEngine.setEnabled(true)` enables logs and starts GPS alongside separate settings. Verify effective consent and permissions. |
| Exposure heuristic | `fastApproxLuma` samples compressed JPEG bytes. Compare with decoded luminance; it is not calibrated brightness. |
| Conversion | Fixed architecture and `strict=False`. Verify classifier weights and reject unintended mismatches. |
| Timing | Excludes camera decode, preprocessing, native landmarks, network and alerts. Report end-to-end measures separately. |
| LLM | Key externalized; archived preview model retained. Verify model availability and deploy a server-side secret boundary. |
| Privacy | GPS, identity and conversations need access, consent and retention controls. Use synthetic review data. |

These are source observations, not reproduced runtime failures. Research algorithms have not been silently rewritten.

## Discussion prompts

- Why combine classification with pose rules?
- How do timers trade false alerts against detection delay?
- How are class ordering, preprocessing and model versions validated together?
- How should dropped frames, network failures and initialization errors be tested?
- Which implementation and experimental work did each team member contribute?

This is collaborative academic work. No per-file authorship, patentability, new accuracy result or production readiness is inferred from the ZIP.
