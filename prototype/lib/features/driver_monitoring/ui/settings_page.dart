// lib/features/driver_monitoring/ui/settings_page.dart
import 'package:flutter/material.dart';

import '../utils/app_settings.dart';
import 'calibrate_page.dart';
import 'profile_page.dart';
import 'local_logs_page.dart'; // ⟵ opens the SQLite logs screen
import '../utils/metrics_exporter.dart' as mx; // ⟵ Save As… helpers
import 'test_tools_page.dart'; // ⟵ Test Lab
import '../../../data/local_app_db.dart'; // ⟵ for purging local driver logs

class SettingsPage extends StatefulWidget {
  final VoidCallback? onLogout;        // Provided by AppShell
  final String? currentUsername;       // For display

  const SettingsPage({super.key, this.onLogout, this.currentUsername});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  // App settings (local)
  final _cameraController = TextEditingController();

  bool _exporting = false; // disable taps while exporting

  @override
  void initState() {
    super.initState();
    AppSettings.I.addListener(_onSettingsChanged);
    _cameraController.text = AppSettings.I.cameraUrl;
  }

  @override
  void dispose() {
    AppSettings.I.removeListener(_onSettingsChanged);
    _cameraController.dispose();
    super.dispose();
  }

  void _onSettingsChanged() {
    if (!mounted) return;
    setState(() {
      _cameraController.text = AppSettings.I.cameraUrl;
    });
  }

  Future<void> _saveCameraUrl() async {
    AppSettings.I.setCameraUrl(_cameraController.text.trim());
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Camera URL saved')),
    );
  }

  Future<void> _exportMetrics() async {
    if (_exporting) return;
    setState(() => _exporting = true);
    try {
      final result = await mx.MetricsExporter.exportMetricsSaveAs(context);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(result == null ? 'Export cancelled' : 'Exported: $result')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  Future<void> _exportLatencySummary() async {
    if (_exporting) return;
    setState(() => _exporting = true);
    try {
      final result = await mx.MetricsExporter.exportLatencySummarySaveAs(context);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(result == null ? 'Export cancelled' : 'Exported: $result')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Export failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  Future<void> _purgeLocalDriverLogs() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete all local driver logs?'),
        content: const Text(
          'This removes all rows from local driver log tables in SQLite.\n'
          'Cloud logs (Firestore) are not affected.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton.tonal(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;

    try {
      final db = LocalAppDb.I.db;

      // Find tables that look like driver logs; adjust patterns if yours differ.
      final tables = await db.rawQuery(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND (name LIKE 'driver_logs%' OR name LIKE 'local_driver_logs%')",
      );

      int totalRows = 0;
      int affectedTables = 0;
      for (final row in tables) {
        final name = (row['name'] ?? '').toString();
        if (name.isEmpty) continue;
        final n = await db.delete(name);
        totalRows += n;
        affectedTables++;
      }

      // Reclaim space
      try { await db.execute('VACUUM'); } catch (_) {}

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            affectedTables == 0
                ? 'No local driver log tables found.'
                : 'Deleted $totalRows rows from $affectedTables table(s).',
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Purge failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final enabled = AppSettings.I.monitoringEnabled;
    final username = widget.currentUsername ?? '';

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 20, 16, 16),
          children: [
            // ===== Account =====
            ListTile(
              leading: const Icon(Icons.person),
              title: Text(
                username.isEmpty ? 'Logged in' : 'Logged in as $username',
              ),
              subtitle: const Text('Manage your profile and sign out'),
              trailing: widget.onLogout == null
                  ? null
                  : TextButton.icon(
                      onPressed: widget.onLogout,
                      icon: const Icon(Icons.logout),
                      label: const Text('Log out'),
                    ),
            ),
            const Divider(height: 32),

            // ===== Monitoring toggle =====
            SwitchListTile(
              title: const Text('Monitoring enabled'),
              subtitle: const Text('Turn all detection/alerts on or off'),
              value: enabled,
              onChanged: (v) => AppSettings.I.setMonitoringEnabled(v),
            ),
            const Divider(height: 32),

            // ===== Alerts (TTS & Gemini) =====
            Card(
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Voice alerts (TTS)'),
                    subtitle: Text(
                      AppSettings.I.ttsEnabled
                          ? 'Enabled — the app will speak alerts'
                          : 'Disabled — no spoken alerts',
                    ),
                    value: AppSettings.I.ttsEnabled,
                    onChanged: (v) => AppSettings.I.setTtsEnabled(v),
                  ),
                  const Divider(height: 1),
                  SwitchListTile(
                    title: const Text('Use Gemini for voice lines'),
                    subtitle: Text(
                      AppSettings.I.ttsUseGemini
                          ? 'Enabled — dynamic phrasing by Gemini'
                          : 'Disabled — use built-in fixed phrases',
                    ),
                    value: AppSettings.I.ttsUseGemini,
                    onChanged: (v) => AppSettings.I.setTtsUseGemini(v),
                  ),
                ],
              ),
            ),

            const Divider(height: 32),

            // ===== Metrics toggle (local only) =====
            SwitchListTile(
              title: const Text('Collect metrics locally (for CSV export)'),
              subtitle: Text(
                AppSettings.I.metricsEnabled
                    ? 'Enabled — timings will be stored locally for export'
                    : 'Disabled — no metrics will be recorded',
              ),
              value: AppSettings.I.metricsEnabled,
              onChanged: (v) => AppSettings.I.setMetricsEnabled(v),
            ),
            const SizedBox(height: 8),

            // ===== Exports (Save As…) =====
            Card(
              clipBehavior: Clip.antiAlias,
              child: Column(
                children: [
                  ListTile(
                    leading: const Icon(Icons.save_alt),
                    title: const Text('Export metrics (Save As…)'),
                    subtitle: const Text(
                      'CSV of per-frame timings: transport, jpeg_decode, face, preproc, model, totals',
                    ),
                    trailing: _exporting
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.chevron_right),
                    onTap: _exporting ? null : _exportMetrics,
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.timeline_outlined),
                    title: const Text('Export latency summary (Save As…)'),
                    subtitle: const Text(
                      'JSON with latency percentiles and FPS stats',
                    ),
                    trailing: _exporting
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.chevron_right),
                    onTap: _exporting ? null : _exportLatencySummary,
                  ),
                ],
              ),
            ),

            const Divider(height: 32),

            // ===== Appearance (Theme) =====
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Appearance',
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<ThemeMode>(
                      value: AppSettings.I.themeMode,
                      decoration: const InputDecoration(
                        labelText: 'Theme mode',
                        border: OutlineInputBorder(),
                      ),
                      items: const [
                        DropdownMenuItem(
                          value: ThemeMode.system,
                          child: Text('System'),
                        ),
                        DropdownMenuItem(
                          value: ThemeMode.light,
                          child: Text('Light'),
                        ),
                        DropdownMenuItem(
                          value: ThemeMode.dark,
                          child: Text('Dark'),
                        ),
                      ],
                      onChanged: (m) {
                        if (m != null) AppSettings.I.setThemeMode(m);
                      },
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // ===== Tools / Pages =====
            ListTile(
              leading: const Icon(Icons.tune),
              title: const Text('Calibrate Head Pose'),
              subtitle: const Text('Set your comfortable neutral pose'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CalibratePage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.science_outlined),
              title: const Text('Test Lab'),
              subtitle: const Text('Ground truth markers • Live latency • Exports'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const TestToolsPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.badge),
              title: const Text('Profile'),
              subtitle: const Text('Name, username, car brand/model'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => ProfilePage(
                      username: widget.currentUsername ?? '',
                    ),
                  ),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.storage),
              title: const Text('Local Driver Logs (SQLite)'),
              subtitle: const Text('View / export local per-minute logs'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const LocalLogsPage()),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.delete_forever, color: Colors.red),
              title: const Text('Purge local driver logs (SQLite)'),
              subtitle: const Text('Delete all rows from local driver log tables'),
              trailing: const Icon(Icons.chevron_right),
              onTap: _purgeLocalDriverLogs,
            ),

            const Divider(height: 32),

            // Driver logs toggle
            SwitchListTile(
              title: const Text('Driver Logs to Firestore'),
              subtitle: Text(
                AppSettings.I.driverLogsEnabled
                    ? 'Enabled — per-minute aggregates will upload to driver_logs'
                    : 'Disabled — nothing will be uploaded',
              ),
              value: AppSettings.I.driverLogsEnabled,
              onChanged: (v) => AppSettings.I.setDriverLogsEnabled(v),
            ),
            const Divider(height: 32),

            // ===== Camera URL (local) =====
            const Text(
              'ESP32 Camera',
              style: TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _cameraController,
              decoration: const InputDecoration(
                labelText: "Stream URL",
                hintText: "http://192.168.x.x:81/stream",
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _saveCameraUrl(),
            ),
            const SizedBox(height: 8),
            FilledButton.tonal(
              onPressed: _saveCameraUrl,
              child: const Text('Save camera URL'),
            ),
          ],
        ),
      ),
    );
  }
}
