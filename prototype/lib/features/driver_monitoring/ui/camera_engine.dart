// lib/features/driver_monitoring/ui/camera_engine.dart
import 'dart:async'; // for unawaited & StreamSubscription
import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

// 👇 adjust this path if your channel file lives elsewhere
import '../../../face_landmarker_channel.dart';

import '../../driver_monitoring/utils/image_utils.dart'
    show
        kClassNames,
        fastApproxLuma,
        imageToCHWGray3Float32, // grayscale → replicate to 3ch
        argMax;

// Metrics (network, algo stages, etc.)
import '../../driver_monitoring/utils/metrics.dart' show Metrics;

// TTS (dynamic, single-interaction flow)
import '../../driver_monitoring/utils/tts_service.dart';

// App settings (to toggle Gemini/TTS + monitoring)
import '../../driver_monitoring/utils/app_settings.dart' show AppSettings;

// Driver logs (per-minute aggregates)
import '../../driver_monitoring/utils/driver_logs.dart' show DriverLogs;

class EngineUi {
  final String status; // OK / WARN / ALERT / NOFACE / OFF / INIT / ...
  final String info;
  final String? reason;
  final double? pitch, yaw, roll;

  // Classification result
  final String? label;
  final double? conf;

  const EngineUi({
    required this.status,
    required this.info,
    this.reason,
    this.pitch,
    this.yaw,
    this.roll,
    this.label,
    this.conf,
  });
}

/// Owns the MediaPipe face landmarker bridge, ONNX session, dwell logic,
/// exposure guard, metrics instrumentation, alert + TTS behavior, and GPS.
class CameraEngine {
  CameraEngine({required this.onUi});

  // -------- UI callback --------
  final void Function(EngineUi) onUi;

  // -------- ML state --------
  bool _busy = false;
  OrtSession? _session;
  bool get ready => _session != null; // Face detector removed (MediaPipe now)

  // MediaPipe bridge (native)
  late final FaceLandmarkerChannel _mp;

  // -------- Enable / disable --------
  bool _enabled = false;
  bool get enabled => _enabled;

  // -------- Start-up delay --------
  late DateTime _startupDelayEndsAt;

  // -------- Head pose + baseline --------
  double? _pitch, _yaw, _roll;
  double? _pitch0, _yaw0, _roll0; // calibration baseline
  double _pitchThresh = 20, _yawThresh = 20, _rollThresh = 20;

  // -------- Dwell timers --------
  double? _noFaceSince; // seconds
  double? _poseBadSince; // seconds
  double? _clsBadSince; // seconds

  // -------- Exposure jump guard --------
  int _emaLuma = -1; // 0..255 EMA of brightness
  int _aeCooldown = 0; // frames to wait before re-arming skip
  int _skipFrames = 0;

  // -------- Audio + TTS --------
  final _beepPlayer = AudioPlayer();
  DateTime? _lastBeepAt;
  static const double _BEEP_COOLDOWN_S = 3.0;
  static const bool _ALLOW_BEEP_DURING_INTERACTION = false;

  final _tts = TTSService();

  // -------- GPS bridge --------
  final _gps = GpsBridge.I;

  // -------- (Optional) network metadata from stream --------
  int? _pendingNetSeq;
  double? _pendingTransportMs;

  // -------- Settings listeners --------
  void _applyTtsSettings() {
    _tts.enabled = AppSettings.I.ttsEnabled;     // Voice alerts ON/OFF
    _tts.useGemini = AppSettings.I.ttsUseGemini; // Dynamic wording ON/OFF
  }

  void _applyMonitoringSetting() {
    // Reflect AppSettings flag into the engine + metrics recorder
    setEnabled(AppSettings.I.monitoringEnabled);
  }

  // ================= Lifecycle =================
  Future<void> init() async {
    _startupDelayEndsAt = DateTime.now().add(const Duration(seconds: 5));

    // Watch settings for Gemini/TTS + Monitoring toggles
    AppSettings.I.addListener(_applyTtsSettings);
    AppSettings.I.addListener(_applyMonitoringSetting);
    _applyTtsSettings();

    // MediaPipe FaceLandmarker (native)
    _mp = FaceLandmarkerChannel();
    await _mp.initialize(
      modelAssetPath: 'face_landmarker.task',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputBlendshapes: true,
      outputTransformationMatrices: true,
    );

    // ONNX session
    OrtEnv.instance.init();
    final raw = await rootBundle.load('assets/models/DMS_mobilevitv2_100.onnx');
    final bytes = raw.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, OrtSessionOptions());

    await _beepPlayer.setReleaseMode(ReleaseMode.stop);

    onUi(const EngineUi(status: 'INIT', info: 'Initializing…'));

    // Sync initial monitoring state last (after we’re ready)
    _applyMonitoringSetting();
  }

  void dispose() {
    AppSettings.I.removeListener(_applyTtsSettings);
    AppSettings.I.removeListener(_applyMonitoringSetting);

    _mp.close();
    _session?.release();
    OrtEnv.instance.release();
    _beepPlayer.dispose();

    // stop GPS if the page/engine is torn down
    _gps.stop();
  }

  // ================= Enable / Disable =================
  void setEnabled(bool v) {
    if (_enabled == v) return;
    _enabled = v;

    // Keep the metrics logger in lockstep with the engine
    Metrics.I.setEnabled(_enabled);

    if (!_enabled) {
      // Stop any ongoing interaction and clear transient state
      _stopSignals();
      _resetSoft();
      onUi(const EngineUi(
        status: 'OFF',
        info: 'Monitoring is OFF',
      ));

      // Stop logs + GPS
      DriverLogs.I.setEnabled(false);
      _gps.stop();
    } else {
      // Re-arm with a brief startup delay to avoid spurious alerts
      _startupDelayEndsAt = DateTime.now().add(const Duration(seconds: 3));
      _resetSoft();
      onUi(const EngineUi(
        status: 'INIT',
        info: 'Re-initializing…',
      ));

      // Identity is provided by AuthController via DriverLogs.setIdentity()
      // Start logs + GPS
      DriverLogs.I.setEnabled(true);
      _gps.start();
    }
  }

  Future<void> _stopSignals() async {
    try {
      await _beepPlayer.stop();
    } catch (_) {}
    try {
      await _tts.stop();
    } catch (_) {}
  }

  void _resetSoft() {
    // Clear dwell + exposure guard + pose
    _noFaceSince = null;
    _poseBadSince = null;
    _clsBadSince = null;

    _emaLuma = -1;
    _aeCooldown = 0;
    _skipFrames = 0;

    _pitch = _yaw = _roll = null;

    _pendingNetSeq = null;
    _pendingTransportMs = null;
  }

  // ================ Public helpers for page ================

  String? _poseYpr() {
    if (_yaw == null || _pitch == null || _roll == null) return null;
    return '${_yaw!.toStringAsFixed(1)},${_pitch!.toStringAsFixed(1)},${_roll!.toStringAsFixed(1)}';
  }

  void calibrateNow() {
    _pitch0 = _pitch;
    _yaw0 = _yaw;
    _roll0 = _roll;
  }

  /// Optional manual identity override (not required if you use AuthController).
  void setDriverIdentity({String? fullName, String? username}) {
    DriverLogs.I.setIdentity(fullName: fullName, username: username);
  }

  // ================ Frame entrypoints (from MjpegView) ================
  Future<void> onFrameMeta(Uint8List jpg, int frameId, double downloadMs) async {
    // ✅ Record a network-timing row immediately (source FPS metric)
    // This logs when a frame *arrives* from the stream, independent of inference.
    try {
      Metrics.I.noteNetworkFrame(
        downloadMs: downloadMs,
        bytes: jpg.length,
        seq: frameId,
        frameId: frameId,
      );
    } catch (err, st) {
      debugPrint('⚠️ noteNetworkFrame failed: $err\n$st');
    }

    // ✅ Stash meta for the next algo frame row
    // This keeps transport_ms visible side-by-side with inference times.
    _pendingNetSeq = frameId;
    _pendingTransportMs = downloadMs;

    // Continue to ML pipeline
    await onFrame(jpg);
  }


  /// Legacy entrypoint (bytes only).
  Future<void> onFrame(Uint8List jpg) async {
    // Hard gate: if disabled, do nothing
    if (!_enabled) return;

    // Exposure jump guard (keeps UI stable on AE jumps)
    final luma = fastApproxLuma(jpg);
    if (_emaLuma < 0) {
      _emaLuma = luma;
    } else {
      _emaLuma = (((_emaLuma * 7) + luma) / 8).round();
    }
    final delta = (luma - _emaLuma).abs();
    if (_skipFrames == 0 && _aeCooldown == 0 && delta > 70) {
      _skipFrames = 6; // allow AE to settle
      _aeCooldown = 10;
    }
    if (_skipFrames > 0) {
      _skipFrames--;
      if (_aeCooldown > 0) _aeCooldown--;
      return;
    }
    if (_aeCooldown > 0) _aeCooldown--;

    if (!ready) return;
    if (_busy) return;

    _busy = true;
    try {
      await _processJpegForML(jpg);
    } finally {
      _busy = false;
    }
  }

  // ================= Internal: ML pipeline + metrics =================
  Future<void> _processJpegForML(Uint8List jpg) async {
    // TOTAL algo stopwatch
    final swTotal = Stopwatch()..start();

    // --- Decode JPEG ---
    final swDecode = Stopwatch()..start();
    final decoded = img.decodeJpg(jpg);
    final decodeMs = swDecode.elapsedMicroseconds / 1000.0;
    if (decoded == null) {
      onUi(const EngineUi(status: 'OK', info: 'JPEG decode failed'));
      await Metrics.I.noteAlgoFrame(
        frameId: _pendingNetSeq,
        jpegSize: jpg.length,
        transportMs: _pendingTransportMs,
        decodeMs: decodeMs,
        faceMs: 0,
        preprocMs: 0,
        modelMs: 0,
        rxToModelMs: decodeMs,
        totalMs: swTotal.elapsedMicroseconds / 1000.0,
        status: 'DECODE_FAIL',
        label: null,
        conf: null,
        poseYpr: null,
        lookingAway: null,
      );
      _pendingNetSeq = null;
      _pendingTransportMs = null;
      return;
    }

    final w = decoded.width, h = decoded.height;

    // --- Face detect (MediaPipe Tasks, VIDEO mode) ---
    final swFace = Stopwatch()..start();
    final tsMs = DateTime.now().millisecondsSinceEpoch; // int, not double
    final mp = await _mp.detect(jpg, timestampMs: tsMs); // positional jpeg, named timestamp
    final faceMs = swFace.elapsedMicroseconds / 1000.0;
    final nowS = DateTime.now().millisecondsSinceEpoch / 1000.0;

    // NOFACE dwell (null-safe)
    if (mp == null || mp.faces.isEmpty) {
      _noFaceSince ??= nowS;
      _poseBadSince = null;
      _clsBadSince = null;

      final nofaceElapsed = nowS - (_noFaceSince ?? nowS);
      String statusStr;
      final inStartup = DateTime.now().isBefore(_startupDelayEndsAt);

      if (nofaceElapsed >= 3.0) {
        if (inStartup) {
          onUi(const EngineUi(status: 'NOFACE', reason: 'startup', info: 'No face detected'));
          statusStr = 'NOFACE';
        } else {
          onUi(const EngineUi(status: 'ALERT', reason: 'noface', info: 'No face detected'));
          // ALERT path: fire-and-forget beep + TTS to avoid blocking frames
          unawaited(_maybeBeep());
          if (_tts.enabled) {
            unawaited(_tts.speakAlert(causeKey: 'noface').then((ok) {
              if (ok) {
                DriverLogs.I.markAlertSpoken('no_face'); // ← canonical key
              }
            }));
          }
          statusStr = 'ALERT';
        }
      } else if (nofaceElapsed >= 1.0) {
        onUi(const EngineUi(status: 'NOFACE', reason: 'noface', info: 'No face detected'));
        if (!inStartup) {
          final played = await _maybeBeep();
          if (played) {
            DriverLogs.I.markWarnPlayed('no_face');
          }
        }
        statusStr = 'NOFACE';
      } else {
        onUi(const EngineUi(status: 'NOFACE', reason: 'noface', info: 'No face detected'));
        statusStr = 'NOFACE';
      }

      await Metrics.I.noteAlgoFrame(
        frameId: _pendingNetSeq,
        jpegSize: jpg.length,
        transportMs: _pendingTransportMs,
        decodeMs: decodeMs,
        faceMs: faceMs,
        preprocMs: 0,
        modelMs: 0,
        rxToModelMs: decodeMs + faceMs,
        totalMs: swTotal.elapsedMicroseconds / 1000.0,
        status: statusStr,
        label: null,
        conf: null,
        poseYpr: null,
        lookingAway: null,
      );

      _pendingNetSeq = null;
      _pendingTransportMs = null;
      return;
    } else {
      _noFaceSince = null;
    }

    // ----- Pick largest face by bbox area from landmarks -----
    Rect? bestRect;
    for (final face in mp.faces) {
      final r = _rectFromLandmarks(face, w, h, pad: 0.30);
      if (bestRect == null ||
          (r.width * r.height) > (bestRect.width * bestRect.height)) {
        bestRect = r;
      }
    }
    final rect = bestRect!;

    // Head pose from transformation matrix if available (null-safe)
    if (mp.matrices.isNotEmpty) {
      final ypr = _matrixToYpr(mp.matrices.first);
      _yaw = ypr.$1;
      _pitch = ypr.$2;
      _roll = ypr.$3;
    } else {
      _yaw = _pitch = _roll = null; // fall back to "no pose"
    }

    // --- Preprocess for ONNX (crop, resize, grayscale→3ch, normalize) ---
    final swPrep = Stopwatch()..start();
    final cropped = img.copyCrop(
      decoded,
      x: rect.left.toInt(),
      y: rect.top.toInt(),
      width: rect.width.toInt(),
      height: rect.height.toInt(),
    );
    final resized = img.copyResize(cropped, width: 224, height: 224);

    final tensor = imageToCHWGray3Float32(
      resized,
      mean: const [0.485, 0.456, 0.406],
      std:  const [0.229, 0.224, 0.225],
    );
    final preprocMs = swPrep.elapsedMicroseconds / 1000.0;

    // Suppress during startup delay
    if (DateTime.now().isBefore(_startupDelayEndsAt)) {
      onUi(EngineUi(
        status: 'INIT',
        info: 'Initializing…',
        pitch: _pitch,
        yaw: _yaw,
        roll: _roll,
        label: null,
        conf: null,
      ));
      await Metrics.I.noteAlgoFrame(
        frameId: _pendingNetSeq,
        jpegSize: jpg.length,
        transportMs: _pendingTransportMs,
        decodeMs: decodeMs,
        faceMs: faceMs,
        preprocMs: preprocMs,
        modelMs: 0,
        rxToModelMs: decodeMs + faceMs + preprocMs,
        totalMs: swTotal.elapsedMicroseconds / 1000.0,
        status: 'INIT',
        label: null,
        conf: null,
        poseYpr: _poseYpr(),
        lookingAway: _poseIsOutOfRange(),
      );

      _pendingNetSeq = null;
      _pendingTransportMs = null;
      return;
    }

    // --- ONNX inference ---
    final swModel = Stopwatch()..start();
    final inputTensor =
        OrtValueTensor.createTensorWithDataList(tensor, [1, 3, 224, 224]);
    final outputs =
        await _session?.runAsync(OrtRunOptions(), {'input': inputTensor});
    inputTensor.release();
    final modelMs = swModel.elapsedMicroseconds / 1000.0;

    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      onUi(EngineUi(
        status: 'OK',
        info: 'ONNX output is empty',
        pitch: _pitch,
        yaw: _yaw,
        roll: _roll,
        label: null,
        conf: null,
      ));
      await Metrics.I.noteAlgoFrame(
        frameId: _pendingNetSeq,
        jpegSize: jpg.length,
        transportMs: _pendingTransportMs,
        decodeMs: decodeMs,
        faceMs: faceMs,
        preprocMs: preprocMs,
        modelMs: modelMs,
        rxToModelMs: decodeMs + faceMs + preprocMs,
        totalMs: swTotal.elapsedMicroseconds / 1000.0,
        status: 'ONNX_EMPTY',
        label: null,
        conf: null,
        poseYpr: _poseYpr(),
        lookingAway: _poseIsOutOfRange(),
      );

      _pendingNetSeq = null;
      _pendingTransportMs = null;
      return;
    }

    final scores = ((outputs.first!.value as List).first as List)
        .map((e) => (e as num).toDouble())
        .toList();
    for (final o in outputs) {
      o?.release();
    }
    final predIdx = argMax(scores);
    final label = kClassNames[predIdx];
    final conf = scores[predIdx];

    // --- Dwell tracking / status ---
    final poseAlert = _poseIsOutOfRange();
    if (poseAlert) {
      _poseBadSince ??= nowS;
    } else {
      _poseBadSince = null;
    }

    final isBadClass = (label != 'focused');
    if (isBadClass) {
      _clsBadSince ??= nowS;
    } else {
      _clsBadSince = null;
    }

    final poseElapsed = _poseBadSince != null ? nowS - _poseBadSince! : 0.0;
    final clsElapsed = _clsBadSince != null ? nowS - _clsBadSince! : 0.0;

    String statusStr = 'OK';

    // ALERT >= 3s -> TTSService (dynamic flow)
    if (poseElapsed >= 3.0 || clsElapsed >= 3.0) {
      final causeKey = (poseElapsed >= 3.0) ? 'pose' : 'cls_$predIdx';
      final info =
          '$label (score: ${conf.toStringAsFixed(3)})  |  Pitch: ${_fmt(_pitch)}°, Yaw: ${_fmt(_yaw)}°, Roll: ${_fmt(_roll)}°';

      onUi(EngineUi(
        status: 'ALERT',
        reason: causeKey,
        info: info,
        pitch: _pitch,
        yaw: _yaw,
        roll: _roll,
        label: label,
        conf: conf,
      ));

      // Fire-and-forget beep + TTS so frames keep flowing
      unawaited(_maybeBeep());
      final catA = _mapCategory(label, causeKey);
      if (_tts.enabled) {
        unawaited(_tts.speakAlert(causeKey: causeKey, label: label).then((ok) {
          if (ok && catA != null) {
            DriverLogs.I.markAlertSpoken(catA);
          }
        }));
      }

      statusStr = 'ALERT';
    }
    // WARN 1..3s (beep only)
    else if (poseElapsed >= 1.0 || clsElapsed >= 1.0) {
      onUi(EngineUi(
        status: 'WARN',
        reason: poseElapsed >= 1.0 ? 'pose' : 'classifier',
        info:
            '$label (score: ${conf.toStringAsFixed(3)})  |  Pitch: ${_fmt(_pitch)}°, Yaw: ${_fmt(_yaw)}°, Roll: ${_fmt(_roll)}°',
        pitch: _pitch,
        yaw: _yaw,
        roll: _roll,
        label: label,
        conf: conf,
      ));

      // Beep; only if it actually played, mark warn
      final played = await _maybeBeep();
      if (played) {
        final catW =
            _mapCategory(label, poseElapsed >= 1.0 ? 'pose' : 'classifier');
        if (catW != null) DriverLogs.I.markWarnPlayed(catW);
      }

      statusStr = 'WARN';
    }
    // Otherwise OK
    else {
      onUi(EngineUi(
        status: 'OK',
        info:
            'focused (score: ${conf.toStringAsFixed(3)})  |  Pitch: ${_fmt(_pitch)}°, Yaw: ${_fmt(_yaw)}°, Roll: ${_fmt(_roll)}°',
        pitch: _pitch,
        yaw: _yaw,
        roll: _roll,
        label: label,
        conf: conf,
      ));
      statusStr = 'OK';
    }

    // --- Metrics summary for this frame ---
    await Metrics.I.noteAlgoFrame(
      frameId: _pendingNetSeq,
      jpegSize: jpg.length,
      transportMs: _pendingTransportMs,
      decodeMs: decodeMs,
      faceMs: faceMs,
      preprocMs: preprocMs,
      modelMs: modelMs,
      rxToModelMs: decodeMs + faceMs + preprocMs,
      totalMs: swTotal.elapsedMicroseconds / 1000.0,
      status: statusStr,
      label: label,
      conf: conf,
      poseYpr: _poseYpr(),
      lookingAway: _poseIsOutOfRange(),
    );

    // Clear pending meta after a successful frame
    _pendingNetSeq = null;
    _pendingTransportMs = null;
  }

  // ================= Alerts / beep =================
  /// Returns true if the beep actually started playing; false if suppressed
  /// by interaction guard or cooldown.
  Future<bool> _maybeBeep() async {
    if (!_ALLOW_BEEP_DURING_INTERACTION && _tts.isInteractionActive) {
      return false;
    }

    final now = DateTime.now();
    if (_lastBeepAt != null &&
        now.difference(_lastBeepAt!).inMilliseconds <
            (_BEEP_COOLDOWN_S * 1000)) {
      return false;
    }
    _lastBeepAt = now;
    try {
      await _beepPlayer.play(AssetSource('alarm.wav'));
      return true;
    } catch (_) {
      return false;
    }
  }

  // ================= Helpers =================

  /// Build a face crop rectangle from normalized [Landmark]s.
  Rect _rectFromLandmarks(List<Landmark> lms, int w, int h, {double pad = 0.3}) {
    double minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (final lm in lms) {
      if (lm.x < minX) minX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y > maxY) maxY = lm.y;
    }

    // Expand by [pad] and clamp to image bounds
    final cx = (minX + maxX) / 2.0;
    final cy = (minY + maxY) / 2.0;
    final ww = (maxX - minX) * (1 + pad);
    final hh = (maxY - minY) * (1 + pad);

    double left = ((cx - ww / 2) * w).clamp(0.0, w.toDouble());
    double top = ((cy - hh / 2) * h).clamp(0.0, h.toDouble());
    double right = ((cx + ww / 2) * w).clamp(0.0, w.toDouble());
    double bottom = ((cy + hh / 2) * w).clamp(0.0, h.toDouble()); // NOTE: was h; keep bounds safe

    // Ensure minimum size
    if (right - left < 10) right = (left + 10).clamp(0.0, w.toDouble());
    if (bottom - top < 10) bottom = (top + 10).clamp(0.0, h.toDouble());

    return Rect.fromLTRB(left, top, right, bottom);
  }

  /// Extract yaw/pitch/roll (degrees) from a 4x4 row-major transform matrix.
  /// Convention: ZYX (yaw around Z, pitch around Y, roll around X).
  /// Returns (yaw, pitch, roll).
  (double, double, double) _matrixToYpr(List<double> m) {
    // Rotation (upper-left 3x3)
    final r00 = m[0],  r01 = m[1],  r02 = m[2];
    final r10 = m[4],  r11 = m[5],  r12 = m[6];
    final r20 = m[8],  r21 = m[9],  r22 = m[10];

    // Yaw-Pitch-Roll (ZYX)
    final yawZ   = math.atan2(r10, r00);               // around Z
    final pitchY = math.asin((-r20).clamp(-1.0, 1.0)); // around Y
    final rollX  = math.atan2(r21, r22);               // around X

    double toDeg(double rad) => rad * 180.0 / math.pi;
    return (toDeg(yawZ), toDeg(pitchY), toDeg(rollX));
  }

  bool _poseIsOutOfRange() {
    if (_pitch == null || _yaw == null || _roll == null) return false;
    final p0 = _pitch0 ?? 0.0;
    final y0 = _yaw0 ?? 0.0;
    final r0 = _roll0 ?? 0.0;
    final dp = (_pitch! - p0).abs();
    final dy = (_yaw! - y0).abs();
    final dr = (_roll! - r0).abs();
    return (dp > _pitchThresh) || (dy > _yawThresh) || (dr > _rollThresh);
  }

  /// Map ONNX label / reason into **canonical categories**:
  /// yawning | using_phone | drowsy | looking_away | no_face
  String? _mapCategory(String? label, String? reason) {
    final l = (label ?? '').toLowerCase();
    final r = (reason ?? '').toLowerCase();

    if (r == 'pose') return 'looking_away';
    if (r == 'noface' || r == 'no_face') return 'no_face';

    if (r.startsWith('cls')) {
      if (l.contains('yawn')) return 'yawning';
      if (l.contains('phone')) return 'using_phone';
      if (l.contains('sleep') || l.contains('drowsy')) return 'drowsy';
      if (l.contains('look') || l.contains('away') || l.contains('distract')) {
        return 'looking_away';
      }
    }
    if (l.isNotEmpty && l != 'focused') return 'looking_away';
    return null;
  }

  String _fmt(double? v) => v == null ? 'NA' : v.toStringAsFixed(1);
}

/// =============================================================================
/// GpsBridge: 1 Hz GPS updates into DriverLogs while monitoring is enabled.
/// Call start()/stop() from CameraEngine.setEnabled.
/// =============================================================================
class GpsBridge {
  GpsBridge._();
  static final GpsBridge I = GpsBridge._();

  StreamSubscription<Position>? _sub;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);

  Future<void> start() async {
    if (_sub != null) return;

    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return;
    }

    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      perm = await Geolocator.requestPermission();
    }
    if (perm == LocationPermission.denied || perm == LocationPermission.deniedForever) {
      return; // no permission; silently skip GPS
    }

    // Prime with current position
    try {
      final cur = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.bestForNavigation,
      );
      DriverLogs.I.updateGps(cur.latitude, cur.longitude);
      _lastSent = DateTime.now();
    } catch (_) {}

    // Platform-specific settings for ~1 Hz
    final LocationSettings settings;
    if (Platform.isAndroid) {
      settings = AndroidSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0,
        intervalDuration: const Duration(seconds: 1),
        forceLocationManager: false,
      );
    } else if (Platform.isIOS) {
      settings = AppleSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0, // report even if not moving
        pauseLocationUpdatesAutomatically: true,
        allowBackgroundLocationUpdates: false,
        showBackgroundLocationIndicator: false,
      );
    } else {
      settings = const LocationSettings(
        accuracy: LocationAccuracy.best,
        distanceFilter: 0,
      );
    }

    _sub = Geolocator.getPositionStream(locationSettings: settings).listen((pos) {
      final now = DateTime.now();
      if (now.difference(_lastSent).inMilliseconds >= 900) {
        DriverLogs.I.updateGps(pos.latitude, pos.longitude);
        _lastSent = now;
      }
    });
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
  }
}
