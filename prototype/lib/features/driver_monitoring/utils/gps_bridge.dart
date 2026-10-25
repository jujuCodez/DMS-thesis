import 'dart:async';
import 'package:geolocator/geolocator.dart';
import 'app_settings.dart';
import 'driver_logs.dart';

class GpsBridge {
  GpsBridge._();
  static final GpsBridge I = GpsBridge._();

  StreamSubscription<Position>? _sub;

  void init() {
    // react to monitoring toggle
    AppSettings.I.addListener(_sync);
    _sync(); // do an initial check
  }

  Future<void> dispose() async {
    AppSettings.I.removeListener(_sync);
    await _stop();
  }

  Future<void> _sync() async {
    final shouldRun = AppSettings.I.monitoringEnabled;
    if (shouldRun) {
      if (_sub == null) await _start();
    } else {
      await _stop();
    }
  }

  Future<void> _start() async {
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      perm = await Geolocator.requestPermission();
    }
    if (perm == LocationPermission.denied ||
        perm == LocationPermission.deniedForever) {
      return;
    }

    // Prime with current position (optional)
    try {
      final cur = await Geolocator.getCurrentPosition();
      DriverLogs.I.updateGps(cur.latitude, cur.longitude);
    } catch (_) {}

    _sub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 10, // meters
      ),
    ).listen((pos) {
      DriverLogs.I.updateGps(pos.latitude, pos.longitude);
    });
  }

  Future<void> _stop() async {
    await _sub?.cancel();
    _sub = null;
  }
}
