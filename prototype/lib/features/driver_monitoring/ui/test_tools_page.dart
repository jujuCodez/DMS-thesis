// lib/features/driver_monitoring/ui/test_tools_page.dart
import 'dart:async';
import 'dart:convert';
import 'dart:ui' show FontFeature;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'chat_logs_page.dart';
import '../../../data/local_app_db.dart';
import '../utils/app_settings.dart';
import '../utils/metrics_exporter.dart' as mx;
import '../utils/metrics.dart' show Metrics;

/// Monitoring & Testing Lab (local only; no Firestore).
class TestToolsPage extends StatefulWidget {
  const TestToolsPage({super.key});

  @override
  State<TestToolsPage> createState() => _TestToolsPageState();
}

class _TestToolsPageState extends State<TestToolsPage> {
  // --- Monitoring toggle (mirrors global setting) ---
  bool get _monitoringOn => AppSettings.I.monitoringEnabled;

  // --- Environment marker (Day / Night) ---
  static const List<String> _envOptions = <String>['day', 'night'];
  String _env = 'day';

  // --- Phase labeling ---
  static const List<String> _phases = <String>[
    'not_testing',
    'using_normally',
    'focused',
    'drowsy',
    'yawning',
    'using_phone',
  ];
  String _activePhase = 'not_testing';
  DateTime? _phaseStartUtc;
  Timer? _phaseTick;

  // --- Auto end (max duration, seconds) ---
  int? _maxPhaseSeconds;
  Duration? get _maxPhaseDuration =>
      _maxPhaseSeconds == null ? null : Duration(seconds: _maxPhaseSeconds!);

  // --- Live diagnostics (rolling 30s + last frame) ---
  Timer? _poll;
  double _fpsMedian = 0;
  Map<String, num> _p50 = const {};
  Map<String, num> _p95 = const {};
  int _frameCount = 0;

  // last frame snapshot
  String _lastLabel = '—';
  double _lastConf = 0.0;
  String _lastStatus = '—';
  double _lastTransport = 0.0;
  double _lastDecode = 0.0;
  double _lastPreproc = 0.0;
  double _lastModel = 0.0;
  double _lastAlgoTotal = 0.0;

  // Optional: periodically log a sample of the current phase + model output
  bool _autoSample = false;
  Timer? _sampleTimer;

  // New: FPS-only mode (network-only metric rows)
  bool _fpsOnly = false;

  // Column set (must match exporter)
  static const _cols = [
    'client_ts','session','username','device','app_ver','name',
    'frame_id','jpeg_size','transport_ms','jpeg_decode_ms','face_ms',
    'preproc_ms','model_ms','rx_to_model_ms','algo_total_ms',
    'status','label','testing_phase',
    'cause','dur_ms','phase','type','had_user_reply',
    'text_len','net_seq','conf',
    'pose_ypr','looking_away',
  ];

  // Reusable WHERE predicates
  static const String _whereFrames =
      "(type = 'frame' OR name = 'frame_summary')";
  static const String _whereNet =
      "(type = 'frame_net' OR name = 'frame_network')";

  @override
  void initState() {
    super.initState();
    AppSettings.I.addListener(_onSettingsChanged);

    // Pick up current FPS-only mode from Metrics singleton.
    _fpsOnly = Metrics.I.fpsOnly;

    // Prime per-frame context so every frame_summary row carries env/testing_phase.
    Metrics.I.setFrameContext(cause: _env, testingPhase: 'not_testing');

    // live panels refresh
    _poll = Timer.periodic(const Duration(seconds: 1), (_) => _refreshLiveStats());
    unawaited(_refreshLiveStats());

    // phase elapsed ticker + auto-end check
    _phaseTick = Timer.periodic(const Duration(seconds: 1), (_) {
      _maybeAutoEndPhase();
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    AppSettings.I.removeListener(_onSettingsChanged);
    _poll?.cancel();
    _sampleTimer?.cancel();
    _phaseTick?.cancel();
    // end any running phase
    unawaited(_endPhaseIfRunning());
    super.dispose();
  }

  void _onSettingsChanged() {
    if (!mounted) return;
    setState(() {});
  }

  // ---------- Ground-truth writer (local SQLite only) ----------
  Future<void> _writeTruth(Map<String, Object?> body) async {
    final db = LocalAppDb.I.db;

    // Envelope (stable columns)
    final envelope = <String, Object?>{
      'client_ts': DateTime.now().toUtc().toIso8601String(),
      'session'  : Metrics.I.sessionId,
      'username' : Metrics.I.username,
      'device'   : Metrics.I.device,
      'app_ver'  : Metrics.I.appVer,
    };

    // Normalized payload
    final Map<String, Object?> norm = { for (final c in _cols) c: null };
    norm['name'] = 'ground_truth';

    // Merge provided body
    norm.addAll(body);

    await db.insert('metrics', {...envelope, ...norm});
  }

  // ---------- Environment control (Day/Night) ----------
  Future<void> _setEnv(String newEnv) async {
    if (_env == newEnv) return;
    setState(() => _env = newEnv);

    Metrics.I.setFrameContext(cause: newEnv);

    await _writeTruth({
      'type': 'env',
      'label': newEnv,
      'phase': 'instant',
      'dur_ms': 0,
    });
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Environment set to ${newEnv.toUpperCase()}')),
    );

    // Nudge live panel refresh
    unawaited(_refreshLiveStats());
  }

  // ---------- Phase control ----------
  Future<void> _setPhase(String newPhase) async {
    if (_activePhase == newPhase) return;

    await _endPhaseIfRunning();

    if (newPhase != 'not_testing' && !_monitoringOn) {
      AppSettings.I.setMonitoringEnabled(true);
    }

    if (newPhase != 'not_testing') {
      _phaseStartUtc = DateTime.now().toUtc();

      Metrics.I.setFrameContext(testingPhase: newPhase);

      await _writeTruth({
        'type': 'phase',
        'label': newPhase,
        'phase': 'start',
        'cause': _env,
      });

      _maybeAutoEndPhase();
    } else {
      Metrics.I.setFrameContext(testingPhase: 'not_testing');
      _phaseStartUtc = null;
    }

    setState(() => _activePhase = newPhase);
    unawaited(_refreshLiveStats());
  }

  Future<void> _endPhaseIfRunning() async {
    if (_activePhase != 'not_testing' && _phaseStartUtc != null) {
      final end = DateTime.now().toUtc();
      final durMs = end.difference(_phaseStartUtc!).inMilliseconds;
      await _writeTruth({
        'type': 'phase',
        'label': _activePhase,
        'phase': 'end',
        'dur_ms': durMs,
        'cause': _env,
      });
    }
    _phaseStartUtc = null;
  }

  void _maybeAutoEndPhase() {
    if (_activePhase == 'not_testing') return;
    if (_phaseStartUtc == null) return;
    final cap = _maxPhaseDuration;
    if (cap == null) return;

    final elapsed = DateTime.now().toUtc().difference(_phaseStartUtc!);
    if (elapsed >= cap) {
      unawaited(_endPhaseIfRunning());
      Metrics.I.setFrameContext(testingPhase: 'not_testing');
      setState(() {
        _activePhase = 'not_testing';
      });
      if (mounted) {
        final ss = _maxPhaseSeconds!;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Phase auto-ended after ${ss}s (max reached)')),
        );
      }
      unawaited(_refreshLiveStats());
    }
  }

  void _toggleAutoSample(bool on) {
    setState(() => _autoSample = on);
    _sampleTimer?.cancel();
    if (on) {
      _sampleTimer = Timer.periodic(const Duration(seconds: 5), (_) async {
        await _writeTruth({
          'type': 'phase_sample',
          'label': _activePhase,
          'status': _lastLabel,
          'conf': _lastConf,
          'algo_total_ms': _lastAlgoTotal,
          'cause': _env,
        });
      });
    }
  }

  Future<void> _manualSample() async {
    await _writeTruth({
      'type': 'phase_sample_manual',
      'label': _activePhase,
      'status': _lastLabel,
      'conf': _lastConf,
      'algo_total_ms': _lastAlgoTotal,
      'cause': _env,
    });
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Manual sample recorded')),
    );
    unawaited(_refreshLiveStats());
  }

  Future<void> _toggleEyesClosedBurst() async {
    final now = DateTime.now().toUtc();
    await _writeTruth({'type': 'eyes_closed', 'phase': 'start', 'cause': _env});
    await Future.delayed(const Duration(seconds: 2));
    await _writeTruth({
      'type': 'eyes_closed',
      'phase': 'end',
      'dur_ms': DateTime.now().toUtc().difference(now).inMilliseconds,
      'cause': _env,
    });
    unawaited(_refreshLiveStats());
  }

  Future<void> _markYawn() => _writeTruth({
    'type': 'yawn', 'phase': 'instant', 'dur_ms': 0, 'cause': _env,
  }).then((_) => _refreshLiveStats());

  Future<void> _markEvent(String label) => _writeTruth({
    'type': 'marker', 'label': label, 'phase': 'instant', 'dur_ms': 0, 'cause': _env,
  }).then((_) => _refreshLiveStats());

  // ---------- Purge local metrics ----------
  Future<void> _purgeMetrics() async {
    final db = LocalAppDb.I.db;
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete all local metrics?'),
        content: const Text(
          'This will remove all rows from the local "metrics" table. '
          'Exports you already saved won’t be affected.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton.tonal(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;

    try {
      final count = await LocalAppDb.I.db.delete('metrics');
      try { await LocalAppDb.I.db.execute('VACUUM'); } catch (_) {}
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Local metrics cleared ($count rows)')),
      );
      await _refreshLiveStats();

      // After purge, keep current env/testing phase context active for new frames.
      Metrics.I.setFrameContext(
        cause: _env,
        testingPhase: _activePhase,
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Purge failed: $e')),
      );
    }
  }

  // ---------- Live stats (last 30s) + last frame ----------
  Future<void> _refreshLiveStats() async {
    final db = LocalAppDb.I.db;
    final sinceIso = DateTime.now().toUtc()
        .subtract(const Duration(seconds: 30))
        .toIso8601String();

    // 30s window for algorithm frames (may be empty in FPS-only mode)
    final frameRows = await db.query(
      'metrics',
      where: '$_whereFrames AND client_ts >= ?',
      whereArgs: [sinceIso],
      orderBy: 'client_ts ASC',
    );

    // 30s window for network frames (for FPS + transport)
    final netRows = await db.query(
      'metrics',
      where: '$_whereNet AND client_ts >= ?',
      whereArgs: [sinceIso],
      orderBy: 'client_ts ASC',
    );

    // Median FPS from network rows (frame_network), ignoring outliers (<=0 or >2s)
    double fpsMedian = 0.0;
    if (netRows.length >= 3) {
      final times = netRows
          .map((r) => DateTime.tryParse(r['client_ts']?.toString() ?? ''))
          .whereType<DateTime>()
          .toList();
      final gaps = <double>[];
      for (int i = 1; i < times.length; i++) {
        final dt = times[i].difference(times[i - 1]).inMilliseconds / 1000.0;
        if (dt > 0 && dt <= 2.0) gaps.add(dt);
      }
      if (gaps.isNotEmpty) {
        gaps.sort();
        final mIdx = (gaps.length - 1) ~/ 2;
        final med = gaps.length.isOdd
            ? gaps[mIdx]
            : (gaps[mIdx] + gaps[mIdx + 1]) / 2.0;
        if (med > 0) fpsMedian = 1.0 / med;
      }
    }

    // Helper: collect positive (non-null) numbers only
    List<double> _vecPos(List<Map<String, Object?>> rows, String k) => rows
        .map((r) => (r[k] as num?)?.toDouble())
        .where((v) => v != null && v! > 0)
        .cast<double>()
        .toList();

    Map<String, num> pct(List<double> xs) {
      if (xs.isEmpty) return {'p50': 0, 'p95': 0};
      xs.sort();
      double q(double p) {
        final i = (p * (xs.length - 1)).round().clamp(0, xs.length - 1);
        return xs[i];
      }
      return {'p50': q(0.50), 'p95': q(0.95)};
    }

    // Prefer transport percentiles from *network* rows; fallback to frame rows.
    final transportNet = _vecPos(netRows, 'transport_ms');
    final transportFrm = _vecPos(frameRows, 'transport_ms');
    final transportSrc = transportNet.isNotEmpty ? transportNet : transportFrm;

    final decode    = _vecPos(frameRows, 'jpeg_decode_ms');
    final pre       = _vecPos(frameRows, 'preproc_ms');
    final model     = _vecPos(frameRows, 'model_ms');

    // e2e per row to avoid index drift; prefer algo_total_ms if present
    final e2e = <double>[];
    for (final r in frameRows) {
      final at = (r['algo_total_ms'] as num?)?.toDouble();
      if (at != null && at > 0) {
        e2e.add(at);
      } else {
        final t = (r['transport_ms'] as num?)?.toDouble() ?? 0;
        final d = (r['jpeg_decode_ms'] as num?)?.toDouble() ?? 0;
        final p = (r['preproc_ms'] as num?)?.toDouble() ?? 0;
        final m = (r['model_ms'] as num?)?.toDouble() ?? 0;
        final sum = t + d + p + m;
        if (sum > 0) e2e.add(sum);
      }
    }

    final p50 = <String, num>{
      'transport': pct(transportSrc)['p50'] ?? 0,
      'decode'   : pct(decode)['p50'] ?? 0,
      'preproc'  : pct(pre)['p50'] ?? 0,
      'model'    : pct(model)['p50'] ?? 0,
      'e2e'      : pct(e2e)['p50'] ?? 0,
    };

    final p95 = <String, num>{
      'transport': pct(transportSrc)['p95'] ?? 0,
      'decode'   : pct(decode)['p95'] ?? 0,
      'preproc'  : pct(pre)['p95'] ?? 0,
      'model'    : pct(model)['p95'] ?? 0,
      'e2e'      : pct(e2e)['p95'] ?? 0,
    };

    // Last frame snapshot (prefer algo frame; fallback to last network frame for transport)
    final lastFrame = await db.query(
      'metrics',
      where: _whereFrames,
      orderBy: 'client_ts DESC, rowid DESC',
      limit: 1,
    );
    final lastNet = await db.query(
      'metrics',
      where: _whereNet,
      orderBy: 'client_ts DESC, rowid DESC',
      limit: 1,
    );

    if (!mounted) return;
    setState(() {
      _fpsMedian = fpsMedian;
      _p50 = p50;
      _p95 = p95;

      // Count: show how many algo frames contributed to diagnostics window
      _frameCount = frameRows.length;

      if (lastFrame.isNotEmpty) {
        final r = lastFrame.first;
        _lastLabel     = (r['label'] ?? '—').toString();
        _lastConf      = (r['conf'] as num?)?.toDouble() ?? 0.0;
        _lastStatus    = (r['status'] ?? '—').toString();
        _lastTransport = (r['transport_ms'] as num?)?.toDouble() ?? 0.0;
        _lastDecode    = (r['jpeg_decode_ms'] as num?)?.toDouble() ?? 0.0;
        _lastPreproc   = (r['preproc_ms'] as num?)?.toDouble() ?? 0.0;
        _lastModel     = (r['model_ms'] as num?)?.toDouble() ?? 0.0;
        _lastAlgoTotal = (r['algo_total_ms'] as num?)?.toDouble() ?? 0.0;
      } else if (lastNet.isNotEmpty) {
        // FPS-only case: at least show last transport_ms
        final r = lastNet.first;
        _lastTransport = (r['transport_ms'] as num?)?.toDouble() ?? 0.0;
        _lastLabel = '—';
        _lastConf = 0.0;
        _lastStatus = '—';
        _lastDecode = 0.0;
        _lastPreproc = 0.0;
        _lastModel = 0.0;
        _lastAlgoTotal = 0.0;
      }
    });
  }

  Duration _phaseElapsed() {
    if (_phaseStartUtc == null) return Duration.zero;
    return DateTime.now().toUtc().difference(_phaseStartUtc!);
  }

  Color _phaseColor(String phase, BuildContext ctx) {
    final cs = Theme.of(ctx).colorScheme;
    switch (phase) {
      case 'focused':        return cs.primary;
      case 'drowsy':         return cs.error;
      case 'yawning':        return cs.tertiary;
      case 'using_phone':    return cs.secondary;
      case 'using_normally': return cs.primaryContainer;
      default:               return cs.outline;
    }
  }

  Color _envColor(BuildContext ctx) {
    final cs = Theme.of(ctx).colorScheme;
    return _env == 'day' ? cs.tertiary : cs.secondary;
  }

  IconData _envIcon() => _env == 'day' ? Icons.wb_sunny_outlined : Icons.nights_stay_outlined;

  // ---------- UI ----------
  @override
  Widget build(BuildContext context) {
    final elapsed = _phaseElapsed();
    final elapsedStr = elapsed == Duration.zero
        ? '—'
        : '${elapsed.inMinutes.remainder(60).toString().padLeft(2, '0')}:'
          '${elapsed.inSeconds.remainder(60).toString().padLeft(2, '0')}';
    final maxStr = (_activePhase != 'not_testing' && _maxPhaseSeconds != null)
        ? ' / ${_maxPhaseSeconds}s'
        : '';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitoring & Testing Lab'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(36),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            child: Row(
              children: [
                // Phase chip
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _phaseColor(_activePhase, context).withOpacity(0.12),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: _phaseColor(_activePhase, context).withOpacity(0.4)),
                  ),
                  child: Text(
                    _activePhase == 'not_testing'
                        ? 'NOT TESTING'
                        : 'TESTING: ${_activePhase.toUpperCase()}',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: _phaseColor(_activePhase, context),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                // Env chip (Day / Night)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _envColor(context).withOpacity(0.12),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: _envColor(context).withOpacity(0.4)),
                  ),
                  child: Row(
                    children: [
                      Icon(_envIcon(), size: 16, color: _envColor(context)),
                      const SizedBox(width: 6),
                      Text(
                        _env.toUpperCase(),
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          color: _envColor(context),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                // FPS-only chip
                if (_fpsOnly)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Theme.of(context).colorScheme.secondary.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(color: Theme.of(context).colorScheme.secondary.withOpacity(0.4)),
                    ),
                    child: Text(
                      'FPS ONLY',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Theme.of(context).colorScheme.secondary,
                      ),
                    ),
                  ),
                const SizedBox(width: 10),
                Text('Elapsed: $elapsedStr$maxStr',
                    style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 24),
        children: [
          // Monitoring enable
          SwitchListTile(
            title: const Text('Monitoring enabled'),
            subtitle: const Text('Toggle the live pipeline from here'),
            value: _monitoringOn,
            onChanged: (v) {
              AppSettings.I.setMonitoringEnabled(v);
              // Nudge UI to refresh last frame quickly.
              unawaited(_refreshLiveStats());
            },
          ),
          // FPS-only metrics toggle
          SwitchListTile(
            title: const Text('Log FPS only'),
            subtitle: const Text('Only store network (transport) rows for FPS; suppress algorithm/voice metrics'),
            value: _fpsOnly,
            onChanged: (v) {
              setState(() => _fpsOnly = v);
              Metrics.I.setFpsOnly(v);
              unawaited(_refreshLiveStats());
            },
          ),
          const SizedBox(height: 8),

          // Phase + Environment selectors
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Testing phase',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      for (final phase in _phases)
                        ChoiceChip(
                          label: Text(phase),
                          selected: _activePhase == phase,
                          onSelected: (sel) {
                            if (sel) unawaited(_setPhase(phase));
                          },
                        ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // ---- Auto end after N seconds ----
                  Row(
                    children: [
                      const Text('Auto end after'),
                      const SizedBox(width: 12),
                      DropdownButton<int?>(
                        value: _maxPhaseSeconds,
                        items: const <DropdownMenuItem<int?>>[
                          DropdownMenuItem<int?>(value: null, child: Text('No limit')),
                          DropdownMenuItem<int?>(value: 30, child: Text('30 s')),
                          DropdownMenuItem<int?>(value: 60, child: Text('60 s')),
                          DropdownMenuItem<int?>(value: 120, child: Text('120 s')),
                          DropdownMenuItem<int?>(value: 180, child: Text('180 s')),
                          DropdownMenuItem<int?>(value: 300, child: Text('300 s')),
                          DropdownMenuItem<int?>(value: 600, child: Text('600 s')),
                          DropdownMenuItem<int?>(value: 900, child: Text('900 s')),
                        ],
                        onChanged: (v) {
                          setState(() => _maxPhaseSeconds = v);
                          _maybeAutoEndPhase();
                        },
                      ),
                    ],
                  ),

                  const SizedBox(height: 16),
                  const Text('Environment (for your notes)',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    children: [
                      for (final e in _envOptions)
                        ChoiceChip(
                          avatar: Icon(
                            e == 'day' ? Icons.wb_sunny_outlined : Icons.nights_stay_outlined,
                            size: 16,
                          ),
                          label: Text(e),
                          selected: _env == e,
                          onSelected: (sel) {
                            if (sel) unawaited(_setEnv(e));
                          },
                        ),
                      const Tooltip(
                        message:
                            'Toggling logs a row (type="env", label="<day|night>"). Env is also included as "cause" in samples.',
                        child: Icon(Icons.info_outline, size: 18),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 12,
                    runSpacing: 8,
                    children: [
                      FilledButton.tonal(
                        onPressed: _activePhase == 'not_testing'
                            ? null
                            : () => unawaited(_endPhaseIfRunning().then((_) {
                                  if (!mounted) return;
                                  setState(() {
                                    _activePhase = 'not_testing';
                                  });
                                  Metrics.I.setFrameContext(testingPhase: 'not_testing');
                                  unawaited(_refreshLiveStats());
                                })),
                        child: const Text('End phase'),
                      ),
                      OutlinedButton.icon(
                        onPressed: _manualSample,
                        icon: const Icon(Icons.add_task_outlined),
                        label: const Text('Manual sample now'),
                      ),
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Checkbox(
                            value: _autoSample,
                            onChanged: (v) => _toggleAutoSample(v ?? false),
                          ),
                          const Text('Auto-sample every 5s'),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Live classification snapshot
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Live classification',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  _StatRow(label: 'Label', value: _lastLabel),
                  _StatRow(label: 'Confidence', value: '${(_lastConf * 100).toStringAsFixed(0)}%'),
                  _StatRow(label: 'Status', value: _lastStatus),
                  _StatRow(label: 'Env (cause)', value: _env.toUpperCase()),
                  const SizedBox(height: 8),
                  const Text('Last frame timings (ms)',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  _StatRow(label: 'transport', value: _lastTransport.toStringAsFixed(1)),
                  _StatRow(label: 'jpeg_decode', value: _lastDecode.toStringAsFixed(1)),
                  _StatRow(label: 'preproc', value: _lastPreproc.toStringAsFixed(1)),
                  _StatRow(label: 'model', value: _lastModel.toStringAsFixed(1)),
                  _StatRow(label: 'algo_total', value: _lastAlgoTotal.toStringAsFixed(1)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Live diagnostics
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Diagnostics (last 30s)',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  _StatRow(label: 'Frames (window)', value: '$_frameCount'),
                  _StatRow(label: 'Median FPS (src)', value: _fpsMedian.toStringAsFixed(1)),
                  const SizedBox(height: 6),
                  _LatencyTable(p50: _p50, p95: _p95),
                  const SizedBox(height: 6),
                  Text(
                    'Tip: aim for e2e p95 ≤ 120 ms (ignores stalls and clock jumps).',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Quick markers
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Quick markers',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 12,
                    runSpacing: 12,
                    children: [
                      FilledButton(
                        onPressed: _toggleEyesClosedBurst,
                        child: const Text('Eyes Closed (2s burst)'),
                      ),
                      OutlinedButton.icon(
                        onPressed: _markYawn,
                        icon: const Icon(Icons.face_6_outlined),
                        label: const Text('Mark Yawn'),
                      ),
                      OutlinedButton(
                        onPressed: () => _markEvent('A'),
                        child: const Text('Marker A'),
                      ),
                      OutlinedButton(
                        onPressed: () => _markEvent('B'),
                        child: const Text('Marker B'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Exports, purge, summary, **CSV Viewer**
          Card(
            
            child: Column(
              children: [
                ListTile(
                  leading: const Icon(Icons.table_view_outlined),
                  title: const Text('Preview local metrics (table)'),
                  subtitle: const Text('Quick CSV-like viewer (last 300 rows)'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => MetricsCsvViewerPage(
                          defaultColumns: _cols,
                          initialFilterName: 'all',
                          limit: 300,
                        ),
                      ),
                    );
                  },
                ),
                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.save_alt),
                  title: const Text('Export raw metrics (CSV)'),
                  subtitle: const Text('All rows in the local metrics table'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () async {
                    final res = await mx.MetricsExporter.exportMetricsSaveAs(context);
                    if (!mounted) return;
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(res == null ? 'Export cancelled' : 'Exported: $res')),
                    );
                  },
                ),
                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.timeline_outlined),
                  title: const Text('Export latency summary (JSON)'),
                  subtitle: const Text('Percentiles & FPS calculated on-device'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () async {
                    final res = await mx.MetricsExporter.exportLatencySummarySaveAs(context);
                    if (!mounted) return;
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text(res == null ? 'Export cancelled' : 'Exported: $res')),
                    );
                  },
                ),
                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.delete_forever_outlined, color: Colors.red),
                  title: const Text('Purge local metrics'),
                  subtitle: const Text('Delete all rows from the local "metrics" table'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: _purgeMetrics,
                ),
                // inside the Card with exports/purge/summary list tiles:
                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.forum_outlined),
                  title: const Text('Chat logs (LLM/TTS)'),
                  subtitle: const Text('View & export conversations, LLM calls, and TTS events'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => const ChatLogsPage()),
                    );
                  },
                ),

                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.analytics_outlined),
                  title: const Text('Show on-device summary'),
                  subtitle: const Text('Quick JSON preview in a dialog'),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () async {
                    final summary = await mx.MetricsExporter.buildSummary();
                    if (!mounted) return;
                    showDialog(
                      context: context,
                      builder: (_) => AlertDialog(
                        title: const Text('Latency Summary'),
                        content: SingleChildScrollView(
                          child: Text(const JsonEncoder.withIndent('  ').convert(summary)),
                        ),
                        actions: [
                          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
                        ],
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _StatRow extends StatelessWidget {
  final String label;
  final String value;
  const _StatRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(child: Text(label)),
          Text(value, style: const TextStyle(fontFeatures: [FontFeature.tabularFigures()])),
        ],
      ),
    );
  }
}

class _LatencyTable extends StatelessWidget {
  final Map<String, num> p50;
  final Map<String, num> p95;
  const _LatencyTable({required this.p50, required this.p95});

  @override
  Widget build(BuildContext context) {
    final rows = ['transport', 'decode', 'preproc', 'model', 'e2e'];
    String fmt(num n) => n.toStringAsFixed(0);
    return DataTable(
      columns: const [
        DataColumn(label: Text('Stage')),
        DataColumn(label: Text('p50 ms')),
        DataColumn(label: Text('p95 ms')),
      ],
      rows: [
        for (final r in rows)
          DataRow(cells: [
            DataCell(Text(r)),
            DataCell(Text(fmt(p50[r] ?? 0))),
            DataCell(Text(fmt(p95[r] ?? 0))),
          ])
      ],
      headingRowHeight: 32,
      dataRowMinHeight: 28,
      dataRowMaxHeight: 28,
      columnSpacing: 24,
    );
  }
}

/// ============================
/// Simple CSV-like table viewer
/// ============================
class MetricsCsvViewerPage extends StatefulWidget {
  final List<String> defaultColumns;
  final String initialFilterName; // 'all' | 'frame_summary' | 'ground_truth'
  final int limit;

  const MetricsCsvViewerPage({
    super.key,
    required this.defaultColumns,
    this.initialFilterName = 'all',
    this.limit = 300,
  });

  @override
  State<MetricsCsvViewerPage> createState() => _MetricsCsvViewerPageState();
}

class _MetricsCsvViewerPageState extends State<MetricsCsvViewerPage> {
  String _filter = 'all';
  bool _loading = false;
  List<Map<String, Object?>> _rows = [];
  late List<String> _columns;

  @override
  void initState() {
    super.initState();
    _filter = widget.initialFilterName;
    _columns = [...widget.defaultColumns];
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final db = LocalAppDb.I.db;

      String? where;
      List<Object?>? args;
      if (_filter != 'all') {
        // Accept both modern and legacy frame rows when requested
        if (_filter == 'frame_summary') {
          where = "(type = 'frame' OR name = 'frame_summary')";
        } else {
          where = 'name = ?';
          args = [_filter];
        }
      }

      final rows = await db.query(
        'metrics',
        where: where,
        whereArgs: args,
        orderBy: 'client_ts DESC, rowid DESC',
        limit: widget.limit,
      );

      final presentKeys = <String>{};
      for (final r in rows) {
        presentKeys.addAll(r.keys.map((k) => k.toString()));
      }
      final extras = presentKeys.where((k) => !_columns.contains(k)).toList()..sort();
      final cols = [...widget.defaultColumns, ...extras];

      setState(() {
        _rows = rows;
        _columns = cols;
      });
    } catch (_) {
      // swallow
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _fmtCell(Object? v) {
    if (v == null) return '';
    if (v is num) {
      final d = v.toDouble();
      if (d == d.roundToDouble()) return d.toStringAsFixed(0);
      return d.toStringAsFixed(3);
    }
    return v.toString();
  }

  Future<void> _copyAsCsv() async {
    if (_rows.isEmpty) return;
    final header = _columns.join(',');
    final lines = <String>[header];
    for (final r in _rows) {
      final fields = _columns.map((c) {
        final raw = _fmtCell(r[c]);
        final needsQuote = raw.contains(',') || raw.contains('\n') || raw.contains('"');
        final escaped = raw.replaceAll('"', '""');
        return needsQuote ? '"$escaped"' : escaped;
      }).join(',');
      lines.add(fields);
    }
    final csv = lines.join('\n');
    await Clipboard.setData(ClipboardData(text: csv));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Copied table to clipboard as CSV')),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Added 'frame_network' to quick filter options.
    final nameChoices = const [
      'all','frame_summary','frame_network','ground_truth','llm','llm_to_tts','stt_latency'
    ];
    return Scaffold(
      appBar: AppBar(
        title: const Text('Metrics Preview'),
        actions: [
          IconButton(
            tooltip: 'Copy as CSV',
            onPressed: _rows.isEmpty ? null : _copyAsCsv,
            icon: const Icon(Icons.copy_all_outlined),
          ),
          IconButton(
            tooltip: 'Refresh',
            onPressed: _loading ? null : _load,
            icon: const Icon(Icons.refresh),
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(48),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
            child: Row(
              children: [
                const Text('Filter:'),
                const SizedBox(width: 12),
                DropdownButton<String>(
                  value: _filter,
                  items: [
                    for (final n in nameChoices)
                      DropdownMenuItem(value: n, child: Text(n)),
                  ],
                  onChanged: (v) {
                    if (v == null) return;
                    setState(() => _filter = v);
                    _load();
                  },
                ),
                const SizedBox(width: 24),
                Text('Rows: ${_rows.length}/${widget.limit}'),
              ],
            ),
          ),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _rows.isEmpty
              ? const Center(child: Text('No rows'))
              : Scrollbar(
                  thumbVisibility: true,
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(minWidth: 900),
                      child: SingleChildScrollView(
                        child: DataTable(
                          headingRowHeight: 36,
                          dataRowMinHeight: 28,
                          dataRowMaxHeight: 28,
                          columnSpacing: 20,
                          columns: [
                            for (final c in _columns) DataColumn(label: Text(c)),
                          ],
                          rows: [
                            for (final r in _rows)
                              DataRow(
                                cells: [
                                  for (final c in _columns)
                                    DataCell(Text(_fmtCell(r[c]))),
                                ],
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
    );
  }
}
