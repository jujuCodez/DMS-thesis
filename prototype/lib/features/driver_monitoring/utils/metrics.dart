// Local SQLite metrics logger (BigQuery-friendly schema).
// - Writes one row per event into table "metrics" in LocalAppDb.
// - Stable schema matches MetricsExporter._cols so CSV/summary work offline.
// - Public API:
//     Metrics.I.setEnabled(bool)
//     Metrics.I.setUsername(String?)
//     Metrics.I.setAppVersion(String?)
//     Metrics.I.setFrameContext({String? cause, String? testingPhase})
//     Metrics.I.setFpsOnly(bool v)                    // <-- NEW (FPS-only toggle)
//     Metrics.I.noteNetworkFrame(...)
//     Metrics.I.noteAlgoFrame(...)
//     Metrics.I.noteTtsFlow(...)
//     Metrics.I.noteStt(...)

import 'dart:math' as math;
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

import '../../../data/local_app_db.dart';

class Metrics {
  Metrics._();
  static final Metrics I = Metrics._();

  // -------- Enable/disable (no-ops when disabled) --------
  bool enabled = true;
  void setEnabled(bool v) => enabled = v;

  // -------- FPS-only toggle (log only network frames) --------
  bool _fpsOnly = false;
  bool get fpsOnly => _fpsOnly;
  void setFpsOnly(bool v) => _fpsOnly = v;

  // -------- Session identity --------
  final String sessionId =
      DateTime.now().toUtc().toIso8601String().replaceAll(':', '-') +
          '_${math.Random().nextInt(900000) + 100000}';
  String? username;
  String device = kIsWeb
      ? 'web'
      : (Platform.isAndroid)
          ? 'android'
          : (Platform.isIOS)
              ? 'ios'
              : 'other';
  String? appVer;

  void setUsername(String? name) => username = name;
  void setAppVersion(String? v) => appVer = v;

  // -------- Optional per-frame context (env/testing phase) --------
  // Call setFrameContext(cause: 'day'|'night', testingPhase: 'focused'|...|'not_testing')
  String? _frameCause;                  // e.g., 'day' | 'night'
  String _testingPhase = 'not_testing'; // always has a value

  void setFrameContext({String? cause, String? testingPhase}) {
    if (cause != null) _frameCause = cause;
    if (testingPhase != null) _testingPhase = testingPhase; // never null
  }

  // Optional network sequence counter
  int _netSeq = 0;

  // Table bootstrap/migrations (run once)
  bool _tableReady = false;

  Future<void> _ensureTable() async {
    if (_tableReady) return;
    final db = LocalAppDb.I.db;

    // Base table (if not exists)
    await db.execute('''
      CREATE TABLE IF NOT EXISTS metrics (
        client_ts TEXT,
        session TEXT,
        username TEXT,
        device TEXT,
        app_ver TEXT,
        name TEXT,

        frame_id INTEGER,
        jpeg_size INTEGER,
        transport_ms REAL,
        jpeg_decode_ms REAL,
        face_ms REAL,
        preproc_ms REAL,
        model_ms REAL,
        rx_to_model_ms REAL,
        algo_total_ms REAL,
        status TEXT,
        label TEXT,
        testing_phase TEXT,   -- replaces selected_label
        cause TEXT,

        dur_ms REAL,
        phase TEXT,
        type TEXT,
        had_user_reply INTEGER,
        text_len INTEGER,
        net_seq INTEGER,
        conf REAL,

        pose_ypr TEXT,        -- "yaw,pitch,roll" (e.g. "-3.2,1.0,0.0")
        looking_away TEXT     -- 'looking_away' | 'not_looking_away'
      );
    ''');

    // Indexes for common queries/exports
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_metrics_client_ts ON metrics(client_ts);',
    );
    await db.execute(
      'CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, client_ts);',
    );

    // Defensive: add any missing columns (if schema predates this file)
    final cols = await db.rawQuery('PRAGMA table_info(metrics)');
    final have = {for (final r in cols) (r['name'] as String)};
    Future<void> add(String name, String type) async {
      if (!have.contains(name)) {
        await db.execute('ALTER TABLE metrics ADD COLUMN $name $type;');
      }
    }

    // Keep in sync with exporter _cols
    await add('session', 'TEXT');
    await add('username', 'TEXT');
    await add('device', 'TEXT');
    await add('app_ver', 'TEXT');
    await add('name', 'TEXT');
    await add('frame_id', 'INTEGER');
    await add('jpeg_size', 'INTEGER');
    await add('transport_ms', 'REAL');
    await add('jpeg_decode_ms', 'REAL');
    await add('face_ms', 'REAL');
    await add('preproc_ms', 'REAL');
    await add('model_ms', 'REAL');
    await add('rx_to_model_ms', 'REAL');
    await add('algo_total_ms', 'REAL');
    await add('status', 'TEXT');
    await add('label', 'TEXT');
    await add('testing_phase', 'TEXT');
    await add('cause', 'TEXT');
    await add('dur_ms', 'REAL');
    await add('phase', 'TEXT');
    await add('type', 'TEXT');
    await add('had_user_reply', 'INTEGER');
    await add('text_len', 'INTEGER');
    await add('net_seq', 'INTEGER');
    await add('conf', 'REAL');
    await add('pose_ypr', 'TEXT');
    await add('looking_away', 'TEXT');

    _tableReady = true;
  }

  // ---------------- Public API ----------------

  /// MJPEG/network timing per frame (download time + size).
  /// Adds row: name='frame_network'
  void noteNetworkFrame({
    required double downloadMs, // time for each image to get to phone (ms)
    required int bytes, // JPEG size (bytes)
    int? seq, // optional sequence number
    int? frameId, // optional frame id
  }) {
    if (!enabled) return;
    _write({
      'name': 'frame_network',
      'frame_id': frameId,
      'jpeg_size': bytes,
      'transport_ms': downloadMs,
      'net_seq': seq ?? ++_netSeq,

      // ---- Context on every network row ----
      'cause': _frameCause,               // env (day/night)
      'testing_phase': _testingPhase,     // always set (incl. 'not_testing')
      'type': 'frame_net',                // mark row type
      'dur_ms': downloadMs,               // mirror here for convenience
    });
  }

  /// Per-frame algorithm summary (vision).
  /// Adds row: name='frame_summary'
  Future<void> noteAlgoFrame({
    int? frameId,
    int? jpegSize,
    double? transportMs,        // optionally copy from noteNetworkFrame
    required double decodeMs,   // jpeg_decode_ms
    required double faceMs,     // head detection ms
    required double preprocMs,  // receive->tensor ms
    required double modelMs,    // model latency ms
    required double rxToModelMs,// extra: rx->model ready ms
    required double totalMs,    // total algorithm time ms
    required String status,     // OK | WARN | ALERT | NOFACE | INIT | ...
    String? label,              // classification string (model output)
    double? conf,               // optional confidence
    String? cause,              // pose | noface | cls_X ... (falls back to env)
    String? testingPhaseOverride, // optional override for this row only
    String? poseYpr,            // "yaw,pitch,roll" with 1 decimal (or null)
    bool? lookingAway,          // true => 'looking_away', false => 'not_looking_away', null => blank
    String? type,               // override row type (defaults to 'frame')
  }) {
    if (!enabled || _fpsOnly) return Future.value();

    final effectiveCause =
        (cause == null || cause.isEmpty) ? _frameCause : cause;
    final effectivePhase =
        (testingPhaseOverride == null || testingPhaseOverride.isEmpty)
            ? _testingPhase
            : testingPhaseOverride;

    final lookingAwayStr = (lookingAway == null)
        ? null
        : (lookingAway ? 'looking_away' : 'not_looking_away');

    return _write({
      'name': 'frame_summary',
      'frame_id': frameId,
      'jpeg_size': jpegSize,
      'transport_ms': transportMs,
      'jpeg_decode_ms': decodeMs,
      'face_ms': faceMs,
      'preproc_ms': preprocMs,
      'model_ms': modelMs,
      'rx_to_model_ms': rxToModelMs,
      'algo_total_ms': totalMs,
      'status': status,
      'label': label,
      'conf': conf,

      // ---- Context stamped on every frame row ----
      'cause': effectiveCause,            // env (e.g., day|night)
      'testing_phase': effectivePhase,    // always set
      'type': type ?? 'frame',            // normalize to 'frame'
      'dur_ms': totalMs,                  // mirror algo_total_ms for convenience

      // ---- Pose / behavior extras ----
      'pose_ypr': poseYpr,                // "yaw,pitch,roll"
      'looking_away': lookingAwayStr,     // 'looking_away' | 'not_looking_away'
    });
  }

  /// LLM/voice flow timings.
  /// Writes TWO rows to keep each measurement atomic:
  ///   1) name='llm'        { phase, dur_ms }
  ///   2) name='llm_to_tts' { phase, dur_ms }
  Future<void> noteTtsFlow({
    required String phase, // 'initial' | 'followup'
    required double llmMs, // LLM latency (ms)
    required double llmToTtsMs, // LLM -> start of TTS (ms)
    bool? hadUserReply,
    int? textLen,
  }) async {
    if (!enabled || _fpsOnly) return;
    await _write({
      'name': 'llm',
      'phase': phase,
      'dur_ms': llmMs,
      'had_user_reply': hadUserReply == null ? null : (hadUserReply ? 1 : 0),
      'text_len': textLen,
    });
    await _write({
      'name': 'llm_to_tts',
      'phase': phase,
      'dur_ms': llmToTtsMs,
      'had_user_reply': hadUserReply == null ? null : (hadUserReply ? 1 : 0),
      'text_len': textLen,
    });
  }

  /// STT latency logs.
  /// Adds row: name='stt_latency' { type: 'first'|'final', dur_ms }
  Future<void> noteStt({
    required bool hadReply,
    required double msToFirst, // speech-to-text latency (ms)
    required int textLen,
    String type = 'first_or_final',
  }) {
    if (!enabled || _fpsOnly) return Future.value();
    return _write({
      'name': 'stt_latency',
      'type': type,
      'dur_ms': msToFirst,
      'had_user_reply': hadReply ? 1 : 0,
      'text_len': textLen,
    });
  }

  // ---------------- Internals ----------------

  Future<void> _write(Map<String, dynamic> body) async {
    if (!enabled) return;

    await _ensureTable();
    final db = LocalAppDb.I.db;

    // Envelope present on every row (stable CSV columns)
    final envelope = <String, dynamic>{
      'client_ts': DateTime.now().toUtc().toIso8601String(), // TEXT ISO
      'session': sessionId, // TEXT
      'username': username, // TEXT (nullable)
      'device': device, // TEXT
      'app_ver': appVer, // TEXT (nullable)
    };

    // Normalize all known fields so schema stays stable (missing -> null).
    final normalized = <String, dynamic>{
      // Common
      'name': body['name'] ?? 'event',

      // Frame / network / vision
      'frame_id': null,
      'jpeg_size': null,
      'transport_ms': null,
      'jpeg_decode_ms': null,
      'face_ms': null,
      'preproc_ms': null,
      'model_ms': null,
      'rx_to_model_ms': null,
      'algo_total_ms': null,
      'status': null,
      'label': null,
      'testing_phase': null,  // replaces selected_label
      'cause': null,

      // LLM / STT / misc
      'dur_ms': null,
      'phase': null,          // keep for llm/stt rows
      'type': null,
      'had_user_reply': null,
      'text_len': null,

      // Optional extras
      'net_seq': null,
      'conf': null,
      'pose_ypr': null,
      'looking_away': null,
    };

    // Overlay event body on normalized nulls
    normalized.addAll(body);

    // Final payload (exact set of columns used by exporter)
    final payload = {
      ...envelope,
      ...normalized,
    };

    await db.insert('metrics', payload);
  }
}
