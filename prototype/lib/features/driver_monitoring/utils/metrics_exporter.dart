// lib/features/driver_monitoring/utils/metrics_exporter.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter_file_dialog/flutter_file_dialog.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';

import '../../../data/local_app_db.dart';

class MetricsExporter {
  // ---------- Stable CSV column order ----------
  // Keep this in sync with the Metrics writer and TestToolsPage viewer.
  // NOTE: For frame rows, these columns are expected to be present:
  // cause (env), testing_phase, type='frame', dur_ms (= algo_total_ms).
  static const _cols = [
    'client_ts','session','username','device','app_ver','name',
    'frame_id','jpeg_size','transport_ms','jpeg_decode_ms','face_ms',
    'preproc_ms','model_ms','rx_to_model_ms','algo_total_ms',
    'status','label','testing_phase',           // <= renamed (was selected_label)
    'cause','dur_ms','phase','type','had_user_reply',
    'text_len','net_seq','conf',
    'pose_ypr','looking_away',                  // <= new extras
  ];

  // ---------- Public API used by Settings/Test pages ----------
  static Future<String?> exportMetricsSaveAs(BuildContext context) async {
    final rows = await _loadRows();
    final bytes = _csvBytes(rows);
    final fileName = 'metrics_${_stamp()}.csv';
    return _saveSmart(
      context: context,
      bytes: bytes,
      fileName: fileName,
      // keep these labels for pickers; we strip mimeType from XFile constructors to avoid version issues
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
  }

  static Future<String?> exportLatencySummarySaveAs(BuildContext context) async {
    final summary = await buildSummary();
    final jsonStr = const JsonEncoder.withIndent('  ').convert(summary);
    final bytes = Uint8List.fromList(utf8.encode(jsonStr));
    final fileName = 'metrics_summary_${_stamp()}.json';
    return _saveSmart(
      context: context,
      bytes: bytes,
      fileName: fileName,
      mimeType: 'application/json',
      saveLabel: 'JSON',
      extension: 'json',
    );
  }

  // ---------- Core data builders ----------
  static Future<List<Map<String, Object?>>> _loadRows() async {
    final db = LocalAppDb.I.db;
    // Export *everything* (frame rows + ground truths + misc).
    // The CSV viewer / later analysis can filter by 'name' or 'type'.
    return db.query('metrics', orderBy: 'client_ts ASC');
  }

  static Uint8List _csvBytes(Iterable<Map<String, Object?>> rows) {
    final sb = StringBuffer()..writeln(_cols.join(','));
    String esc(v) => '"${(v ?? '').toString().replaceAll('"', '""')}"';
    for (final r in rows) {
      sb.writeln(_cols.map((k) => esc(r[k])).join(','));
    }
    return Uint8List.fromList(utf8.encode(sb.toString()));
  }

  static Map<String, num> _percentiles(List<double> xs) {
    if (xs.isEmpty) return {'p50': 0, 'p95': 0, 'p99': 0};
    xs.sort();
    double pick(double p) {
      final i = (p * (xs.length - 1)).round().clamp(0, xs.length - 1);
      return xs[i];
    }
    return {'p50': pick(0.50), 'p95': pick(0.95), 'p99': pick(0.99)};
  }

  // ---- Smoothness helpers (FPS stats from inter-arrival times) ----
  static List<DateTime> _timesFromRows(List<Map<String, Object?>> rows) =>
      rows
          .map((r) => DateTime.tryParse((r['client_ts'] ?? '').toString()))
          .whereType<DateTime>()
          .toList()
        ..sort();

  /// Convert time deltas (s) to instantaneous FPS (1/dt), ignoring <=0 or >2s stalls.
  static List<double> _instFpsFromTimes(List<DateTime> times) {
    final fps = <double>[];
    for (int i = 1; i < times.length; i++) {
      final dt = times[i].difference(times[i - 1]).inMilliseconds / 1000.0;
      if (dt > 0 && dt <= 2.0) {
        fps.add(1.0 / dt);
      }
    }
    return fps;
  }

  /// Percentile of a sorted list (0..1). Returns 0 if empty.
  static double _percentile(List<double> xs, double p) {
    if (xs.isEmpty) return 0.0;
    xs.sort();
    final idx = (p * (xs.length - 1)).round().clamp(0, xs.length - 1);
    return xs[idx];
  }

  /// Average of the worst K% samples (K in 0..1). Returns 0 if not enough samples.
  static double _avgWorstK(List<double> xs, double k) {
    if (xs.isEmpty || k <= 0) return 0.0;
    xs.sort(); // ascending -> worst (lowest fps) first
    final count = (xs.length * k).ceil().clamp(1, xs.length);
    final slice = xs.sublist(0, count);
    final sum = slice.fold<double>(0.0, (a, b) => a + b);
    return sum / slice.length;
  }

  static Future<Map<String, dynamic>> buildSummary() async {
    final rows = await _loadRows();

    // ---------------- Split frame types ----------------
    // Inference-side frames (algo timings)
    final frames = rows.where((r) {
      final name = (r['name'] ?? '').toString();
      final type = (r['type'] ?? '').toString();
      if (type.isNotEmpty) return type == 'frame';
      return name == 'frame_summary';
    }).toList();

    // Source/network-side frames (download arrival)
    final netFrames = rows.where((r) {
      final name = (r['name'] ?? '').toString();
      final type = (r['type'] ?? '').toString();
      if (type.isNotEmpty) return type == 'frame_net';
      return name == 'frame_network';
    }).toList();

    // ---------------- Helpers ----------------
    double _safeNum(Map<String, Object?> r, String k) =>
        (r[k] as num?)?.toDouble() ?? 0.0;

    int? _readInt(dynamic v) {
      if (v == null) return null;
      if (v is int) return v;
      if (v is num) return v.toInt();
      final s = v.toString();
      return int.tryParse(s);
    }

    double _sequenceSpanFpsGeneric(
      List<Map<String, Object?>> fr, {
      String primarySeqKey = 'frame_id',
      String fallbackSeqKey = 'net_seq',
    }) {
      final times = _timesFromRows(fr);
      if (times.length < 2) return 0.0;

      final durSec = (times.last.difference(times.first).inMilliseconds) / 1000.0;
      if (durSec <= 0) return 0.0;

      int? minSeq, maxSeq;
      for (final r in fr) {
        int? id = _readInt(r[primarySeqKey]);
        id ??= _readInt(r[fallbackSeqKey]);
        if (id != null) {
          minSeq = (minSeq == null || id < minSeq!) ? id : minSeq;
          maxSeq = (maxSeq == null || id > maxSeq!) ? id : maxSeq;
        }
      }
      if (minSeq != null && maxSeq != null && maxSeq! >= minSeq!) {
        // Uses span of produced frames; insensitive to skipped logging.
        return (maxSeq! - minSeq! + 1) / durSec;
      }

      // Fallback: trimmed median of inter-arrival gaps, ignoring stalls/jumps.
      final fps = _instFpsFromTimes(times);
      if (fps.isEmpty) return fr.length / durSec;
      fps.sort();
      // Robust: use median FPS (equiv. to median of gaps)
      final mid = (fps.length - 1) ~/ 2;
      return fps.length.isOdd ? fps[mid] : (fps[mid] + fps[mid + 1]) / 2.0;
    }

    Map<String, dynamic> frameSummary;

    // ---------------- Build frame summary ----------------
    if (frames.isEmpty && netFrames.isEmpty) {
      frameSummary = {
        'frames': 0,
        'net_frames': 0,
        'sustained_fps': 0.0,
        'sustained_fps_src': 0.0,
        'sustained_fps_infer': 0.0,
        'fps_source': 'none',
        'smoothness_fps': {
          'source': {
            'p1_low_fps': 0.0,
            'p0_1_low_fps': 0.0,
            'avg_low_1pct_fps': 0.0,
            'avg_low_0_1pct_fps': 0.0,
            'samples': 0,
          },
          'inference': {
            'p1_low_fps': 0.0,
            'p0_1_low_fps': 0.0,
            'avg_low_1pct_fps': 0.0,
            'avg_low_0_1pct_fps': 0.0,
            'samples': 0,
          },
        },
        'note': 'No frame rows found (frame_net/frame_network or frame/frame_summary)',
        'latency_ms': {
          'transport': _percentiles(const []),
          'jpeg_decode': _percentiles(const []),
          'face_detect': _percentiles(const []),
          'preprocess': _percentiles(const []),
          'model_infer': _percentiles(const []),
          'algo_total': _percentiles(const []),
          'end_to_end_no_capture': _percentiles(const []),
        },
      };
    } else {
      // FPS from the SOURCE (network) — primary
      final srcFps = _sequenceSpanFpsGeneric(
        netFrames,
        primarySeqKey: 'net_seq',   // prefer network sequence
        fallbackSeqKey: 'frame_id', // fallback if present
      );

      // FPS from INFERENCE rows — secondary
      final inferFps = _sequenceSpanFpsGeneric(
        frames,
        primarySeqKey: 'frame_id',
        fallbackSeqKey: 'net_seq',
      );

      // ---- Smoothness (1% / 0.1% low) ----
      // Prefer network times for smoothness; fall back to inference.
      final netTimes = _timesFromRows(netFrames);
      final infTimes = _timesFromRows(frames);
      final netFpsVec = _instFpsFromTimes(netTimes);
      final infFpsVec = _instFpsFromTimes(infTimes);

      Map<String, dynamic> _smoothnessBlock(List<double> fps) {
        if (fps.isEmpty) {
          return {
            'p1_low_fps': 0.0,
            'p0_1_low_fps': 0.0,
            'avg_low_1pct_fps': 0.0,
            'avg_low_0_1pct_fps': 0.0,
            'samples': 0,
          };
        }
        final p1 = _percentile([...fps], 0.01);    // 1st percentile FPS (≈ 99th pct frame-time)
        final p01 = _percentile([...fps], 0.001);  // 0.1st percentile FPS (≈ 99.9th pct frame-time)
        final a1 = _avgWorstK([...fps], 0.01);     // average FPS of worst 1% samples
        final a01 = _avgWorstK([...fps], 0.001);   // average FPS of worst 0.1% samples
        return {
          'p1_low_fps': p1,
          'p0_1_low_fps': p01,
          'avg_low_1pct_fps': a1,
          'avg_low_0_1pct_fps': a01,
          'samples': fps.length,
        };
      }

      final smoothSource = _smoothnessBlock(netFpsVec);
      final smoothInfer  = _smoothnessBlock(infFpsVec);

      // Latency percentiles from inference frames
      final transport = <double>[];
      final decode    = <double>[];
      final face      = <double>[];
      final pre       = <double>[];
      final model     = <double>[];
      final algoTotal = <double>[];
      final e2e       = <double>[]; // transport + decode + face + preproc + model

      for (final r in frames) {
        final t  = _safeNum(r, 'transport_ms');
        final jd = _safeNum(r, 'jpeg_decode_ms');
        final fd = _safeNum(r, 'face_ms');
        final pp = _safeNum(r, 'preproc_ms');
        final mi = _safeNum(r, 'model_ms');
        final at = _safeNum(r, 'algo_total_ms');

        if (t > 0) transport.add(t);
        if (jd > 0) decode.add(jd);
        if (fd > 0) face.add(fd);
        if (pp > 0) pre.add(pp);
        if (mi > 0) model.add(mi);
        if (at > 0) algoTotal.add(at);
        final e2eSum = t + jd + fd + pp + mi;
        if (e2eSum > 0) e2e.add(e2eSum);
      }

      frameSummary = {
        // counts
        'frames': frames.length,
        'net_frames': netFrames.length,

        // FPS: use SOURCE as the canonical metric; keep inference for reference
        'sustained_fps': netFrames.isNotEmpty ? srcFps : inferFps,
        'sustained_fps_src': srcFps,
        'sustained_fps_infer': inferFps,
        'fps_source': netFrames.isNotEmpty ? 'network' : 'inference',

        // Smoothness (1%/0.1% low; both percentile-style and avg-of-worst-k)
        'smoothness_fps': {
          'source': smoothSource,
          'inference': smoothInfer,
          'preferred': netFrames.isNotEmpty ? 'source' : 'inference',
        },

        // Latencies (from inference rows)
        'latency_ms': {
          'transport': _percentiles(transport),
          'jpeg_decode': _percentiles(decode),
          'face_detect': _percentiles(face),
          'preprocess': _percentiles(pre),
          'model_infer': _percentiles(model),
          'algo_total': _percentiles(algoTotal),
          'end_to_end_no_capture': _percentiles(e2e),
        },
      };
    }

    // ---------------- Voice chat (LLM/TTS) summary ----------------
    Map<String, dynamic> _buildVoiceBlock(List<Map<String, Object?>> rowsByName) {
      List<double> _durAll(Iterable<Map<String, Object?>> rs) =>
          rs
              .map((r) => (r['dur_ms'] as num?)?.toDouble() ?? 0.0)
              .where((v) => v > 0)
              .toList();

      List<double> _durPhase(String phase) =>
          _durAll(rowsByName.where((r) => (r['phase'] ?? '') == phase));

      final overall = _percentiles(_durAll(rowsByName));
      final initial = _percentiles(_durPhase('initial'));
      final follow  = _percentiles(_durPhase('followup'));

      return {
        'overall': overall,
        'initial': initial,
        'followup': follow,
        'count': rowsByName.length,
      };
    }

    final llmRows      = rows.where((r) => (r['name'] ?? '') == 'llm').toList();
    final llmToTtsRows = rows.where((r) => (r['name'] ?? '') == 'llm_to_tts').toList();

    final voiceChat = {
      'present': llmRows.isNotEmpty || llmToTtsRows.isNotEmpty,
      'llm_ms': _buildVoiceBlock(llmRows),
      'llm_to_tts_ms': _buildVoiceBlock(llmToTtsRows),
    };

    // ---------------- Compose final ----------------
    return {
      ...frameSummary,
      'voice_chat': voiceChat,
    };
  }

  // ---------- Cross-platform save helpers ----------
  static Future<String?> _saveSmart({
    required BuildContext context,
    required Uint8List bytes,
    required String fileName,
    required String mimeType,
    required String saveLabel,
    required String extension,
  }) async {
    // 1) Android: SAF “Save to…”
    if (!kIsWeb && Platform.isAndroid) {
      try {
        final params = SaveFileDialogParams(
          data: bytes,
          fileName: fileName,
          mimeTypesFilter: [mimeType],
        );
        final savedUri = await FlutterFileDialog.saveFile(params: params);
        return savedUri == null ? null : 'Saved via Android system picker';
      } catch (_) {
        // Fall through to Share sheet.
      }
    }

    // 2) Desktop (and Android/iOS fallback): native Save As…
    try {
      final location = await getSaveLocation(
        suggestedName: fileName,
        acceptedTypeGroups: [XTypeGroup(label: saveLabel, extensions: [extension])],
      );
      if (location != null) {
        final xf = XFile.fromData(bytes, name: fileName); // omit mimeType for compat
        await xf.saveTo(location.path);
        return location.path;
      }
    } catch (_) {
      // Ignore; try share.
    }

    // 3) Share sheet fallback
    final tmpPath = await _writeTemp(bytes, fileName);
    try {
      await Share.shareXFiles([XFile(tmpPath)]);
      return 'Shared from temp: $tmpPath';
    } catch (_) {
      return tmpPath; // at least we left a file behind
    }
  }

  static Future<String> _writeTemp(Uint8List bytes, String fileName) async {
    final dir = (await _tempDir()).path;
    final path = p.join(dir, fileName);
    final f = File(path);
    await f.writeAsBytes(bytes, flush: true);
    return path;
  }

  static Future<Directory> _tempDir() async {
    if (kIsWeb) {
      return Directory.systemTemp.createTemp('metrics_web_');
    }
    return Directory.systemTemp;
  }

  static String _stamp() {
    final n = DateTime.now();
    String two(int x) => x.toString().padLeft(2, '0');
    return '${n.year}${two(n.month)}${two(n.day)}_${n.hour.toString().padLeft(2,'0')}${n.minute.toString().padLeft(2,'0')}${n.second.toString().padLeft(2,'0')}';
  }
}
