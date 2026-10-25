// lib/features/driver_monitoring/utils/app_settings.dart
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'metrics.dart' show Metrics;
import 'driver_logs.dart' show DriverLogs;

class AppSettings extends ChangeNotifier {
  AppSettings._();
  static final AppSettings I = AppSettings._();

  // ===== Preference keys =====
  static const _kMonitoring        = 'monitoring_enabled';
  static const _kCameraUrl         = 'camera_url';
  static const _kMetricsEnable     = 'metrics_enabled';
  static const _kThemeMode         = 'theme_mode'; // 'system' | 'light' | 'dark'
  static const _kDriverLogsEnable  = 'driver_logs_enabled';
  static const _kTtsEnable         = 'tts_enabled';      // master voice alerts
  static const _kTtsUseGemini      = 'tts_use_gemini';   // dynamic phrasing on/off

  // ===== In-memory settings =====

  // Monitoring master switch (default: OFF)
  bool _monitoringEnabled = false;
  bool get monitoringEnabled => _monitoringEnabled;

  // Camera stream URL
  String _cameraUrl = const String.fromEnvironment('CAMERA_STREAM_URL', defaultValue: 'http://192.168.4.1:81/stream'); // default stream
  String get cameraUrl => _cameraUrl;

  // Metrics (default: OFF)
  bool _metricsEnabled = false;
  bool get metricsEnabled => _metricsEnabled;

  // Driver logs upload (default: OFF)
  bool _driverLogsEnabled = false;
  bool get driverLogsEnabled => _driverLogsEnabled;

  // Voice/TTS alerts (default: ON)
  bool _ttsEnabled = true;
  bool get ttsEnabled => _ttsEnabled;

  // Gemini dynamic wording for TTS (default: OFF)
  bool _ttsUseGemini = false;
  bool get ttsUseGemini => _ttsUseGemini;

  // Theme
  ThemeMode _themeMode = ThemeMode.system;
  ThemeMode get themeMode => _themeMode;

  /// Initialize from SharedPreferences and apply side-effects.
  Future<void> init() async {
    final sp = await SharedPreferences.getInstance();

    _monitoringEnabled = sp.getBool(_kMonitoring) ?? false;
    _cameraUrl         = sp.getString(_kCameraUrl) ?? _cameraUrl;
    _metricsEnabled    = sp.getBool(_kMetricsEnable) ?? false;
    _driverLogsEnabled = sp.getBool(_kDriverLogsEnable) ?? false;
    _ttsEnabled        = sp.getBool(_kTtsEnable) ?? true;
    _ttsUseGemini      = sp.getBool(_kTtsUseGemini) ?? false;

    final tm = sp.getString(_kThemeMode);
    _themeMode = switch (tm) {
      'light' => ThemeMode.light,
      'dark'  => ThemeMode.dark,
      _       => ThemeMode.system,
    };

    // Apply to subsystems
    Metrics.I.setEnabled(_metricsEnabled);
    DriverLogs.I.setEnabled(_driverLogsEnabled);
    // Note: TTS flags are read by CameraEngine (listener on AppSettings).
  }

  // ===== Setters (persist + notify) =====

  Future<void> setMonitoringEnabled(bool v) async {
    if (_monitoringEnabled == v) return;
    _monitoringEnabled = v;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool(_kMonitoring, v);
    notifyListeners();
  }

  Future<void> setCameraUrl(String url) async {
    if (_cameraUrl == url) return;
    _cameraUrl = url;
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_kCameraUrl, url);
    notifyListeners();
  }

  Future<void> setMetricsEnabled(bool v) async {
    if (_metricsEnabled == v) return;
    _metricsEnabled = v;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool(_kMetricsEnable, v);
    Metrics.I.setEnabled(v);
    notifyListeners();
  }

  Future<void> setDriverLogsEnabled(bool v) async {
    if (_driverLogsEnabled == v) return;
    _driverLogsEnabled = v;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool(_kDriverLogsEnable, v);
    DriverLogs.I.setEnabled(v);
    notifyListeners();
  }

  Future<void> setTtsEnabled(bool v) async {
    if (_ttsEnabled == v) return;
    _ttsEnabled = v;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool(_kTtsEnable, v);
    notifyListeners(); // CameraEngine listens and applies to TTSService
  }

  Future<void> setTtsUseGemini(bool v) async {
    if (_ttsUseGemini == v) return;
    _ttsUseGemini = v;
    final sp = await SharedPreferences.getInstance();
    await sp.setBool(_kTtsUseGemini, v);
    notifyListeners(); // CameraEngine listens and applies to TTSService
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    if (_themeMode == mode) return;
    _themeMode = mode;
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_kThemeMode, switch (mode) {
      ThemeMode.light  => 'light',
      ThemeMode.dark   => 'dark',
      ThemeMode.system => 'system',
    });
    notifyListeners();
  }

  // ===== Utilities =====

  /// Permanently delete **local** driver logs stored in SQLite.
  /// Returns `true` if the operation completed without throwing.
  Future<bool> deleteLocalDriverLogs() async {
    try {
      await DriverLogs.I.clearLocal(); // implement in DriverLogs
      return true;
    } catch (_) {
      return false;
    }
  }
}
