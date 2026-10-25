import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// Metrics (optional but recommended)
import '../../utils/metrics.dart' show Metrics;

/// Lightweight MJPEG viewer with:
/// - per-frame callback (for ML)
/// - UI FPS throttle
/// - stall watchdog & auto-reconnect
/// - oversized-frame drop guard
/// - network timing metrics (download time + jpeg size per frame)
/// - optional rolling stats callback (RX FPS, latency percentiles)
/// - **enabled** flag: when false, disconnects stream and shows placeholder
class MjpegView extends StatefulWidget {
  final String streamUrl;

  /// UI throttle (frames/sec) — only affects preview painting.
  /// Does NOT affect onFrame or onStats rates.
  final int previewFps;

  /// Legacy per-frame callback (JPEG bytes only).
  /// Prefer [onFrameMeta] to also receive frameId and transport time.
  final void Function(Uint8List jpg)? onFrame;

  /// New: per-frame callback with metadata.
  /// - [jpg]: the JPEG bytes for the frame
  /// - [frameId]: monotonically increasing sequence id for correlation
  /// - [downloadMs]: network download time (SOI→EOI) in milliseconds
  final void Function(Uint8List jpg, int frameId, double downloadMs)? onFrameMeta;

  /// Restart delay after a disconnect.
  final Duration reconnectDelay;

  /// Restart if no bytes for this long.
  final Duration stallTimeout;

  /// Drop/restart if a single JPEG grows beyond this size.
  final int maxFrameBytes;

  /// When false, the widget cleanly disconnects and shows a placeholder.
  final bool enabled;

  /// Optional periodic stats callback (see [MjpegStats]).
  /// Fired roughly every [statsInterval] when frames are flowing.
  final void Function(MjpegStats stats)? onStats;

  /// Window for sustained FPS and latency distribution.
  /// We keep timestamps & latencies within this sliding window.
  final Duration statsWindow;

  /// How often to emit stats via [onStats].
  final Duration statsInterval;

  const MjpegView({
    super.key,
    required this.streamUrl,
    this.previewFps = 10,
    this.onFrame,
    this.onFrameMeta,
    this.reconnectDelay = const Duration(seconds: 1),
    this.stallTimeout = const Duration(milliseconds: 1500),
    this.maxFrameBytes = 700 * 1024, // ~700 KB safety (tune for your stream)
    this.enabled = true,
    this.onStats,
    // More responsive/smooth for RX FPS by default:
    this.statsWindow = const Duration(seconds: 3),
    this.statsInterval = const Duration(milliseconds: 500),
  });

  @override
  State<MjpegView> createState() => _MjpegViewState();
}

/// Immutable network stats emitted via [MjpegView.onStats].
class MjpegStats {
  /// Incoming stream FPS (SOI→EOI), not throttled by UI and independent of ML.
  final double sustainedFps;

  /// Frames in the current window used to compute [sustainedFps].
  final int framesInWindow;

  /// Download time (ms) distribution over the window (SOI→EOI).
  final double meanMs;
  final double p50Ms;
  final double p95Ms;
  final double p99Ms;
  final double maxMs;

  /// Counters since last emission.
  final int dropsOversize; // frames dropped for being too large
  final int restarts;      // automatic reconnects
  final int stalls;        // stall watchdog triggers

  const MjpegStats({
    required this.sustainedFps,
    required this.framesInWindow,
    required this.meanMs,
    required this.p50Ms,
    required this.p95Ms,
    required this.p99Ms,
    required this.maxMs,
    required this.dropsOversize,
    required this.restarts,
    required this.stalls,
  });
}

class _MjpegViewState extends State<MjpegView> {
  Uint8List? _lastPreviewJpeg;
  http.Client? _client;
  StreamSubscription<List<int>>? _sub;

  // UI throttle
  int _lastUiUs = 0;

  // Stall watchdog
  Timer? _stallTimer;
  int _lastByteUs = 0;

  // Network timing (approximate)
  // Timestamp (us) when the current JPEG's SOI was first seen in the buffer.
  int _currentFrameStartUs = 0;

  // Sequence number for frames (helps correlate with downstream metrics)
  int _seq = 0;

  // Rolling stats
  final Queue<int> _tsUs = Queue<int>(); // frame end timestamps (us) in window
  final Queue<_Sample> _lat = Queue<_Sample>(); // download ms samples in window
  int _dropsOversizeSinceLastStats = 0;
  int _restartsSinceLastStats = 0;
  int _stallsSinceLastStats = 0;
  int _lastStatsEmitUs = 0;

  bool get _isConnected => _client != null && _sub != null;

  @override
  void initState() {
    super.initState();
    if (widget.enabled) _start();
  }

  @override
  void didUpdateWidget(covariant MjpegView oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Toggle handling
    if (!widget.enabled && _isConnected) {
      _teardown();
      setState(() => _lastPreviewJpeg = null);
      return;
    }
    if (widget.enabled && !_isConnected) {
      _start();
      return;
    }

    // If URL changed while enabled, restart
    if (widget.enabled && oldWidget.streamUrl != widget.streamUrl) {
      _scheduleRestart(immediate: true);
    }
  }

  @override
  void dispose() {
    _teardown();
    super.dispose();
  }

  void _teardown() {
    _stallTimer?.cancel();
    _stallTimer = null;
    _sub?.cancel();
    _sub = null;
    _client?.close();
    _client = null;
    _currentFrameStartUs = 0;
  }

  void _armWatchdog() {
    _stallTimer?.cancel();
    _stallTimer = Timer.periodic(const Duration(milliseconds: 400), (_) {
      if (!mounted || !widget.enabled) return;
      final nowUs = DateTime.now().microsecondsSinceEpoch;
      if (_lastByteUs != 0 &&
          nowUs - _lastByteUs > widget.stallTimeout.inMicroseconds) {
        _stallsSinceLastStats++;
        _scheduleRestart();
      }
    });
  }

  Future<void> _start() async {
    if (!mounted || !widget.enabled) return;
    _client = http.Client();
    _armWatchdog();

    try {
      final req = http.Request('GET', Uri.parse(widget.streamUrl));
      // Hint MJPEG and keep-alive
      req.headers['Accept'] = 'multipart/x-mixed-replace';
      req.headers['Connection'] = 'keep-alive';
      req.headers['Cache-Control'] = 'no-cache';

      final resp = await _client!.send(req);

      if (resp.statusCode != 200) {
        throw Exception('HTTP ${resp.statusCode}');
      }

      final buf = <int>[];
      _sub = resp.stream.listen(
        (chunk) {
          if (!widget.enabled) return; // guard if toggled mid-stream
          _lastByteUs = DateTime.now().microsecondsSinceEpoch;
          buf.addAll(chunk);

          // Find JPEGs by SOI/EOI markers
          int soi = _idx(buf, const [0xFF, 0xD8]);
          int eoi = _idx(buf, const [0xFF, 0xD9]);

          // Mark the start time (first time we notice an SOI for this frame)
          if (soi != -1 && _currentFrameStartUs == 0) {
            _currentFrameStartUs = _lastByteUs;
          }

          // Resync if buffer grows absurdly without finding a frame
          if (buf.length > 2 * 1024 * 1024 && (soi == -1 || eoi == -1)) {
            // keep last 1 KB and drop the rest to find the next SOI quickly
            final tail = buf.sublist(buf.length - 1024);
            buf
              ..clear()
              ..addAll(tail);
            soi = _idx(buf, const [0xFF, 0xD8]);
            eoi = _idx(buf, const [0xFF, 0xD9]);

            // Reset timing since we're resyncing
            _currentFrameStartUs = 0;
            if (soi != -1) _currentFrameStartUs = _lastByteUs;
          }

          while (soi != -1 && eoi != -1 && eoi > soi) {
            final frameLen = eoi + 2 - soi;

            // Drop pathological frames (e.g., huge JPEG due to exposure jump)
            if (frameLen > widget.maxFrameBytes) {
              _dropsOversizeSinceLastStats++;
              buf.removeRange(0, eoi + 2);
              soi = _idx(buf, const [0xFF, 0xD8]);
              eoi = _idx(buf, const [0xFF, 0xD9]);

              // Next frame timing restarts at the next SOI
              _currentFrameStartUs = 0;
              if (soi != -1) _currentFrameStartUs = _lastByteUs;
              continue;
            }

            final jpg = Uint8List.fromList(buf.sublist(soi, eoi + 2));

            // ---- Network/download timing for this frame ----
            // Roughly: time from first-seen SOI to receiving the EOI (last chunk timestamp).
            final startUs =
                _currentFrameStartUs == 0 ? _lastByteUs : _currentFrameStartUs;
            final downloadMs = (_lastByteUs - startUs) / 1000.0;

            // Per-frame metric + sequence (also primes transport hint for algo)
            final seq = ++_seq;
            Metrics.I.noteNetworkFrame(
              downloadMs: downloadMs,
              bytes: jpg.length,
              seq: seq,
              frameId: seq,
            );

            // Reset for next frame
            _currentFrameStartUs = 0;

            // 1) ML callbacks (every frame)
            if (widget.onFrameMeta != null) {
              widget.onFrameMeta!.call(jpg, seq, downloadMs);
            }
            widget.onFrame?.call(jpg); // legacy bytes-only

            // 2) Update rolling stats window
            _pushStatsSample(downloadMs: downloadMs, frameEndUs: _lastByteUs);
            _maybeEmitStats();

            // 3) UI – throttle to previewFps
            final nowUs = _lastByteUs;
            final minGapUs = (1000000 / widget.previewFps).round();
            if (nowUs - _lastUiUs >= minGapUs) {
              _lastUiUs = nowUs;
              if (mounted) setState(() => _lastPreviewJpeg = jpg);
            }

            // Consume processed bytes
            buf.removeRange(0, eoi + 2);
            soi = _idx(buf, const [0xFF, 0xD8]);
            eoi = _idx(buf, const [0xFF, 0xD9]);

            if (soi != -1) {
              // New frame; start timing when we first see its SOI
              _currentFrameStartUs = _lastByteUs;
            }
          }
        },
        onError: (_) {
          _scheduleRestart();
        },
        onDone: _scheduleRestart,
        cancelOnError: true,
      );
    } catch (_) {
      _scheduleRestart();
    }
  }

  int _idx(List<int> data, List<int> pat) {
    final m = pat.length, n = data.length;
    if (m == 0 || n < m) return -1;
    // naive scan is fine; payload sizes are moderate
    for (int i = 0; i <= n - m; i++) {
      int j = 0;
      while (j < m && data[i + j] == pat[j]) j++;
      if (j == m) return i;
    }
    return -1;
  }

  void _scheduleRestart({bool immediate = false}) {
    if (!mounted || !widget.enabled) {
      _teardown();
      return;
    }
    _sub?.cancel();
    _client?.close();
    _stallTimer?.cancel();
    _sub = null;
    _client = null;
    _stallTimer = null;
    _currentFrameStartUs = 0;

    _restartsSinceLastStats++;

    final delay = immediate ? Duration.zero : widget.reconnectDelay;
    Future.delayed(delay, () {
      if (mounted && widget.enabled) _start();
    });
  }

  // -------------------- Rolling stats --------------------

  void _pushStatsSample({required double downloadMs, required int frameEndUs}) {
    final windowUs = widget.statsWindow.inMicroseconds;

    // Maintain end timestamps list for FPS (frames/second in window)
    _tsUs.addLast(frameEndUs);
    while (_tsUs.isNotEmpty && frameEndUs - _tsUs.first > windowUs) {
      _tsUs.removeFirst();
    }

    // Maintain latency samples queue
    _lat.addLast(_Sample(frameEndUs, downloadMs));
    while (_lat.isNotEmpty && frameEndUs - _lat.first.tUs > windowUs) {
      _lat.removeFirst();
    }
  }

  void _maybeEmitStats() {
    if (widget.onStats == null) return;
    final nowUs = _lastByteUs;
    if (_lastStatsEmitUs != 0 &&
        nowUs - _lastStatsEmitUs < widget.statsInterval.inMicroseconds) {
      return;
    }
    _lastStatsEmitUs = nowUs;

    if (_tsUs.length < 2 || _lat.isEmpty) return;

    final spanUs = (_tsUs.last - _tsUs.first).clamp(1, 1 << 62);
    final fps = _tsUs.length * 1000000.0 / spanUs;

    // Copy latencies and compute distribution
    final vals = _lat.map((s) => s.ms).toList()..sort();
    double pick(double q) {
      final i = ((vals.length - 1) * q).clamp(0, vals.length - 1).toInt();
      return vals[i];
    }

    final mean =
        vals.reduce((a, b) => a + b) / (vals.isEmpty ? 1.0 : vals.length);
    final stats = MjpegStats(
      sustainedFps: fps,
      framesInWindow: _tsUs.length,
      meanMs: mean,
      p50Ms: pick(0.50),
      p95Ms: pick(0.95),
      p99Ms: pick(0.99),
      maxMs: vals.last,
      dropsOversize: _dropsOversizeSinceLastStats,
      restarts: _restartsSinceLastStats,
      stalls: _stallsSinceLastStats,
    );

    // Reset counters that are "since last emission"
    _dropsOversizeSinceLastStats = 0;
    _restartsSinceLastStats = 0;
    _stallsSinceLastStats = 0;

    try {
      widget.onStats?.call(stats);
    } catch (_) {
      // ignore consumer errors
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.enabled) {
      return ColoredBox(
        color: Colors.black12,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: const [
              Icon(Icons.pause_circle_filled, size: 48, color: Colors.grey),
              SizedBox(height: 8),
              Text('Monitoring is OFF',
                  style: TextStyle(color: Colors.grey, fontSize: 16)),
            ],
          ),
        ),
      );
    }

    final imgBytes = _lastPreviewJpeg;
    if (imgBytes == null) {
      return const Center(child: Text('Connecting…'));
    }
    return Image.memory(
      imgBytes,
      fit: BoxFit.cover,
      gaplessPlayback: true,
    );
  }
}

class _Sample {
  final int tUs; // frame end time (us)
  final double ms; // download duration (ms)
  const _Sample(this.tUs, this.ms);
}
