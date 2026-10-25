// lib/features/driver_monitoring/ui/camera_page.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';

import '../../driver_monitoring/ui/widgets/mjpeg_view.dart';
import '../../driver_monitoring/ui/calibrate_page.dart' show CalibrationBridge;
import '../../driver_monitoring/utils/app_settings.dart';
import 'camera_engine.dart';

class CameraFaceOnnxPage extends StatefulWidget {
  const CameraFaceOnnxPage({super.key});

  @override
  State<CameraFaceOnnxPage> createState() => _CameraFaceOnnxPageState();
}

class _CameraFaceOnnxPageState extends State<CameraFaceOnnxPage> {
  // ---- Stream source ----
  late String _espStreamUrl;

  // ---- Monitoring master switch (from settings/prefs) ----
  late bool _monitoringEnabled;

  // ---- Reconnect/stall guard ----
  int _reconnectBump = 0; // bump the key to rebuild MjpegView (reconnect)
  DateTime _lastFrameAt = DateTime.fromMillisecondsSinceEpoch(0);
  Timer? _stallTimer;

  // ---- Engine ----
  late final CameraEngine _engine;

  // ---- UI state from engine ----
  String _status = 'INIT';
  String _info = 'Initializing…';
  String _alertReason = '';
  double? _pitch, _yaw, _roll;
  String? _label;
  double? _conf;

  @override
  void initState() {
    super.initState();

    // Initial settings
    _espStreamUrl = AppSettings.I.cameraUrl;
    _monitoringEnabled = AppSettings.I.monitoringEnabled; // default OFF in AppSettings

    // Engine + UI bridge
    _engine = CameraEngine(onUi: (update) {
      _status = update.status;
      _info = update.info;
      _alertReason = update.reason ?? '';
      _pitch = update.pitch;
      _yaw = update.yaw;
      _roll = update.roll;
      _label = update.label;
      _conf = update.conf;
      if (mounted) setState(() {});
    });

    _engine.init().then((_) {
      // Honor persisted ON/OFF on load
      _engine.setEnabled(_monitoringEnabled);
    });

    // Optional: quick calibrate bridge
    CalibrationBridge.register(_onCalibrateFromPage);

    // Watchdog to kick the stream if it stalls
    _stallTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (!_monitoringEnabled) return; // paused when OFF
      final idleMs = DateTime.now().difference(_lastFrameAt).inMilliseconds;
      if (idleMs > 1500) {
        if (mounted) setState(() => _reconnectBump++);
      }
    });

    // React to settings (url / enabled / metrics)
    AppSettings.I.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    _stallTimer?.cancel();
    CalibrationBridge.unregister(_onCalibrateFromPage);
    AppSettings.I.removeListener(_onSettingsChanged);
    _engine.dispose();
    super.dispose();
  }

  // ========= Settings reactions =========
  void _onSettingsChanged() {
    // Camera URL change → force reconnect when ON
    final newUrl = AppSettings.I.cameraUrl;
    if (newUrl != _espStreamUrl) {
      _espStreamUrl = newUrl;
      if (_monitoringEnabled) _reconnectBump++;
    }

    // Monitoring toggle → gate engine & stream
    final newEnabled = AppSettings.I.monitoringEnabled;
    if (newEnabled != _monitoringEnabled) {
      _monitoringEnabled = newEnabled;
      _engine.setEnabled(newEnabled);
      _lastFrameAt = DateTime.now(); // keep watchdog calm while flipping
      if (newEnabled) _reconnectBump++; // start pulling frames now
    }

    if (mounted) setState(() {});
  }

  void _onCalibrateFromPage() {
    _engine.calibrateNow();
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Head pose calibrated')),
    );
  }

  String _fmt(double? v) => v == null ? 'NA' : v.toStringAsFixed(1);

  // ========= UI helpers =========
  Color _statusColor(String s, BuildContext ctx) {
    switch (s) {
      case 'ALERT':
        return Colors.red;
      case 'WARN':
        return Colors.amber;
      case 'NOFACE':
        return Colors.orange;
      case 'OFF':
        return Colors.grey;
      case 'INIT':
        return Theme.of(ctx).colorScheme.primary;
      default:
        return Colors.green;
    }
  }

  Widget _monitoringSwitchChip(BuildContext context, bool enabled) {
    final cs = Theme.of(context).colorScheme;
    return Material(
      color: cs.surface.withOpacity(0.92),
      elevation: 2,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.shield_moon, size: 18),
            const SizedBox(width: 8),
            Text(
              'Monitoring',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: cs.onSurface,
              ),
            ),
            const SizedBox(width: 8),
            Switch(
              value: enabled,
              onChanged: (v) {
                // Persist + notify listeners (this page gets _onSettingsChanged)
                AppSettings.I.setMonitoringEnabled(v);
                if (v) setState(() => _reconnectBump++); // kick stream on
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _offOverlay(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Card(
          elevation: 3,
          margin: const EdgeInsets.symmetric(horizontal: 24),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.power_settings_new, size: 42, color: cs.primary),
                const SizedBox(height: 12),
                const Text(
                  'Monitoring is OFF',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  'Video stream and on-device ML are paused.\n'
                  'Tap below to start monitoring.',
                  style: TextStyle(color: cs.onSurfaceVariant),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                FilledButton.icon(
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Enable Monitoring'),
                  onPressed: () => AppSettings.I.setMonitoringEnabled(true),
                ),
                const SizedBox(height: 12),
                Text(
                  'Source: ${AppSettings.I.cameraUrl}',
                  style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _statusBanner(BuildContext context, String effectiveStatus) {
    final overlayColor = _statusColor(effectiveStatus, context);
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: overlayColor.withOpacity(0.88),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            switch (effectiveStatus) {
              'ALERT' => '⚠️ ALERT: ${_alertReason.toUpperCase()}',
              'NOFACE' => 'No Face Detected',
              'WARN' => 'Warning',
              'OFF' => 'Monitoring is OFF',
              'INIT' => 'Initializing…',
              _ => 'OK',
            },
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
          Text(
            _monitoringEnabled ? _info : 'Video stream & ML are paused.',
            style: const TextStyle(color: Colors.white, fontSize: 14),
            textAlign: TextAlign.center,
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
          if (_monitoringEnabled) ...[
            const SizedBox(height: 6),
            Text(
              'Pitch: ${_fmt(_pitch)}°  |  Yaw: ${_fmt(_yaw)}°  |  Roll: ${_fmt(_roll)}°',
              style: const TextStyle(color: Colors.white70, fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ],
          if (_monitoringEnabled && _label != null && _conf != null) ...[
            const SizedBox(height: 6),
            Text(
              'Class: $_label  •  ${_conf!.toStringAsFixed(3)}',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final effectiveStatus = _monitoringEnabled ? _status : 'OFF';

    return Scaffold(
      // Calibrate FAB only when monitoring is ON
      floatingActionButton: _monitoringEnabled
          ? FloatingActionButton.extended(
              onPressed: _onCalibrateFromPage,
              icon: const Icon(Icons.tune),
              label: const Text('Calibrate'),
            )
          : null,

      body: Stack(
        children: [
          // ---------------- Stream consumer ----------------
          // When monitoring is OFF, we do NOT render MjpegView at all.
          if (_monitoringEnabled)
            Positioned(
              left: 0,
              top: 0,
              child: Opacity(
                opacity: 0.0, // invisible preview; still ticks onFrame
                child: SizedBox(
                  width: 2,
                  height: 2,
                  child: MjpegView(
                    key: ValueKey(_reconnectBump),
                    streamUrl: _espStreamUrl,
                    previewFps: 10,
                    onFrame: (Uint8List jpg) async {
                      _lastFrameAt = DateTime.now();
                      await _engine.onFrame(jpg);
                    },
                  ),
                ),
              ),
            ),

          // OFF overlay (center card)
          if (!_monitoringEnabled) Positioned.fill(child: _offOverlay(context)),

          // Top row: status banner + monitoring toggle chip
          Positioned(
            top: 40,
            left: 16,
            right: 16,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_monitoringEnabled)
                  Expanded(child: _statusBanner(context, effectiveStatus))
                else
                  const Spacer(),
                const SizedBox(width: 12),
                _monitoringSwitchChip(context, _monitoringEnabled),
              ],
            ),
          ),

          // Bottom classification chip
          if (_monitoringEnabled && _label != null && _conf != null)
            Positioned(
              bottom: 28,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.70),
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Text(
                    '${_label!}  •  ${_conf!.toStringAsFixed(3)}',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.3,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
