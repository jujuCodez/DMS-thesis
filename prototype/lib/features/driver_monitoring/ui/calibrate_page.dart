// lib/features/driver_monitoring/ui/calibrate_page.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';

// Bring in your MJPEG widget
import '../../driver_monitoring/ui/widgets/mjpeg_view.dart';
import '../../driver_monitoring/utils/app_settings.dart';

/// ---------------------------------------------------------------------------
/// CalibrationBridge
/// ---------------------------------------------------------------------------
/// Minimal "hub" merged into this file. `camera_page.dart` registers a callback
/// so this page can trigger calibration (copy current head pose to baseline).
///
/// How to wire from camera_page.dart:
///   - In initState():  CalibrationBridge.register(_applyCalibrationFromPage);
///   - In dispose():    CalibrationBridge.unregister(_applyCalibrationFromPage);
///   - Implement _applyCalibrationFromPage() to set _pitch0/_yaw0/_roll0
///     from current _pitch/_yaw/_roll inside setState().
class CalibrationBridge {
  static void Function()? _onCalibrate;

  static void register(void Function() cb) {
    _onCalibrate = cb;
  }

  static void unregister(void Function() cb) {
    if (identical(_onCalibrate, cb)) {
      _onCalibrate = null;
    }
  }

  static void calibrate() {
    final fn = _onCalibrate;
    if (fn != null) {
      fn();
    }
  }
}

/// ---------------------------------------------------------------------------
/// CalibratePage
/// ---------------------------------------------------------------------------
/// Shows the live camera stream and provides a single "Confirm Calibration"
/// button that triggers the bridge. After calibrating, it pops back.
class CalibratePage extends StatefulWidget {
  const CalibratePage({super.key});

  @override
  State<CalibratePage> createState() => _CalibratePageState();
}

class _CalibratePageState extends State<CalibratePage> {
  late String _espStreamUrl;
  int _reconnectBump = 0;

  @override
  void initState() {
    super.initState();
    _espStreamUrl = AppSettings.I.cameraUrl; // dynamic stream from settings
    AppSettings.I.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    AppSettings.I.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    final newUrl = AppSettings.I.cameraUrl;
    if (newUrl != _espStreamUrl) {
      setState(() {
        _espStreamUrl = newUrl;
        _reconnectBump++; // force MJPEG preview to rebuild
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Calibrate Head Pose"),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 20),

            // Live camera preview (same widget used on the main tab)
            SizedBox(
              width: 240,
              height: 240,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: MjpegView(
                  key: ValueKey(_reconnectBump),
                  streamUrl: _espStreamUrl,
                  previewFps: 10,
                  onFrame: (Uint8List jpg) {
                    // Could overlay head pose lines later if needed
                  },
                ),
              ),
            ),

            const SizedBox(height: 16),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 20),
              child: Text(
                "Sit comfortably, look straight ahead, and hold still.\n"
                "When you're ready, press Confirm Calibration.",
                textAlign: TextAlign.center,
              ),
            ),

            const SizedBox(height: 24),
            ElevatedButton.icon(
              icon: const Icon(Icons.check),
              label: const Text("Confirm Calibration"),
              onPressed: () {
                // Call into the registered callback on the camera page
                CalibrationBridge.calibrate();

                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("Pose baseline calibrated")),
                );

                Navigator.of(context).maybePop();
              },
            ),

            const SizedBox(height: 16),
            TextButton.icon(
              icon: const Icon(Icons.refresh),
              label: const Text("Reconnect Preview"),
              onPressed: () => setState(() => _reconnectBump++),
            ),
          ],
        ),
      ),
    );
  }
}
