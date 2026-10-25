import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../utils/driver_logs.dart' show DriverLogs;

/// ─────────────────────────────────────────────────────────────────────────
/// Local SQLite sink + simple viewer/export UI in ONE file.
/// ─────────────────────────────────────────────────────────────────────────

/// A sink contract the DriverLogs will use (very small).
abstract class _LogsSink {
  Future<void> writeMinute({
    required String docId,
    required Map<String, dynamic> payload,
  });

  Future<void> dispose() async {}
}

/// SQLite implementation (singleton).
class _LocalSqliteLogsSink implements _LogsSink {
  _LocalSqliteLogsSink._();
  static final _LocalSqliteLogsSink I = _LocalSqliteLogsSink._();

  Database? _db;

  Future<Database> _open() async {
    if (_db != null) return _db!;
    final dir = await getDatabasesPath();
    final path = p.join(dir, 'driver_logs.db');
    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, _) async {
        await db.execute('''
          CREATE TABLE IF NOT EXISTS driver_logs_new (
            id TEXT PRIMARY KEY,
            name TEXT,
            username TEXT,
            session TEXT,
            minute_start TEXT,
            minute_end TEXT,
            frames_seen INTEGER,
            avg_latitude REAL,
            avg_longitude REAL,
            gps_samples INTEGER,
            counts_json TEXT,
            client_ts TEXT
          );
        ''');
        await db.execute('CREATE INDEX IF NOT EXISTS idx_driver_logs_minute ON driver_logs_new(minute_start DESC);');
        await db.execute('CREATE INDEX IF NOT EXISTS idx_driver_logs_username ON driver_logs_new(username);');
      },
    );
    return _db!;
  }

  @override
  Future<void> writeMinute({
    required String docId,
    required Map<String, dynamic> payload,
  }) async {
    final db = await _open();
    await db.insert(
      'driver_logs_new',
      {
        'id': docId,
        'name': payload['name'],
        'username': payload['username'],
        'session': payload['session'],
        'minute_start': payload['minute_start'],
        'minute_end': payload['minute_end'],
        'frames_seen': payload['frames_seen'],
        'avg_latitude': payload['avg_latitude'],
        'avg_longitude': payload['avg_longitude'],
        'gps_samples': payload['gps_samples'],
        'counts_json': jsonEncode(payload['counts'] ?? const {}),
        'client_ts': payload['client_ts'],
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<int> countRows() async {
    final db = await _open();
    final res = await db.rawQuery('SELECT COUNT(*) AS c FROM driver_logs_new');
    return (res.first['c'] as int?) ?? 0;
  }

  Future<List<Map<String, dynamic>>> recent({int limit = 100}) async {
    final db = await _open();
    return db.query(
      'driver_logs_new',
      orderBy: 'minute_start DESC',
      limit: limit,
    );
  }

  Future<void> clearAll() async {
    final db = await _open();
    await db.delete('driver_logs_new');
  }

  Future<File> exportCsv() async {
    final rows = await recent(limit: 1000000);
    final buf = StringBuffer();
    buf.writeln([
      'id','name','username','session','minute_start','minute_end',
      'frames_seen','avg_latitude','avg_longitude','gps_samples','counts_json','client_ts'
    ].join(','));

    for (final r in rows) {
      String esc(v) => '"${(v ?? '').toString().replaceAll('"', '""')}"';
      buf.writeln([
        esc(r['id']), esc(r['name']), esc(r['username']), esc(r['session']),
        esc(r['minute_start']), esc(r['minute_end']),
        esc(r['frames_seen']), esc(r['avg_latitude']), esc(r['avg_longitude']),
        esc(r['gps_samples']), esc(r['counts_json']), esc(r['client_ts']),
      ].join(','));
    }

    final dir = await getDatabasesPath();
    final path = p.join(dir, 'driver_logs_export.csv');
    final file = File(path);
    await file.writeAsString(buf.toString(), flush: true);
    return file;
  }

  @override
  Future<void> dispose() async {
    await _db?.close();
    _db = null;
  }
}

/// The page UI.
class LocalLogsPage extends StatefulWidget {
  const LocalLogsPage({super.key});

  /// One-liner to switch DriverLogs to SQLite.
  static void activateForDriverLogs() {
    // Requires the tiny patch in driver_logs.dart (setSink + use sink in _flush).
    // ignore: invalid_use_of_visible_for_testing_member, invalid_use_of_protected_member
    (DriverLogs.I as dynamic).setSink(_LocalSqliteLogsSink.I);
  }

  @override
  State<LocalLogsPage> createState() => _LocalLogsPageState();
}

class _LocalLogsPageState extends State<LocalLogsPage> {
  int _count = 0;
  List<Map<String, dynamic>> _rows = const [];
  bool _busy = false;
  String? _notice;

  Future<void> _refresh() async {
    setState(() { _busy = true; _notice = null; });
    try {
      _count = await _LocalSqliteLogsSink.I.countRows();
      _rows = await _LocalSqliteLogsSink.I.recent(limit: 200);
    } catch (e) {
      _notice = 'Load failed: $e';
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _activate() async {
    try {
      LocalLogsPage.activateForDriverLogs();
      setState(() => _notice = 'SQLite logging activated for Driver Logs.');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Driver Logs → SQLite is now active')),
        );
      }
    } catch (e) {
      setState(() => _notice = 'Activation failed. Did you add the tiny patch in driver_logs.dart? ($e)');
    }
  }

  Future<void> _exportCsv() async {
    setState(() { _busy = true; _notice = null; });
    try {
      final file = await _LocalSqliteLogsSink.I.exportCsv();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Exported to: ${file.path}')),
      );
    } catch (e) {
      setState(() => _notice = 'Export failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _clear() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Clear all logs?'),
        content: const Text('This deletes all local SQLite driver logs.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;

    setState(() { _busy = true; _notice = null; });
    try {
      await _LocalSqliteLogsSink.I.clearAll();
      await _refresh();
    } catch (e) {
      setState(() => _notice = 'Clear failed: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Local Driver Logs (SQLite)')),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
              child: Row(
                children: [
                  Expanded(child: Text('Rows: $_count')),
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    onPressed: _busy ? null : _activate,
                    icon: const Icon(Icons.save_alt),
                    label: const Text('Activate SQLite logging'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _exportCsv,
                    icon: const Icon(Icons.download),
                    label: const Text('Export CSV'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _clear,
                    icon: const Icon(Icons.delete_outline),
                    label: const Text('Clear'),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    tooltip: 'Refresh',
                    onPressed: _busy ? null : _refresh,
                    icon: const Icon(Icons.refresh),
                  ),
                ],
              ),
            ),
            if (_notice != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(_notice!, style: const TextStyle(color: Colors.orange)),
              ),
            const Divider(height: 1),
            Expanded(
              child: _busy
                  ? const Center(child: CircularProgressIndicator())
                  : _rows.isEmpty
                      ? const Center(child: Text('No local logs yet.'))
                      : ListView.separated(
                          itemCount: _rows.length,
                          separatorBuilder: (_, __) => const Divider(height: 1),
                          itemBuilder: (_, i) {
                            final r = _rows[i];
                            final minute = (r['minute_start'] ?? '') as String;
                            final user = (r['username'] ?? '') as String;
                            final name = (r['name'] ?? '') as String;
                            final frames = r['frames_seen'];
                            return ListTile(
                              dense: true,
                              leading: const Icon(Icons.event_note),
                              title: Text('$minute — $name'),
                              subtitle: Text('user: $user  •  frames: $frames  •  id: ${r['id']}'),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () {
                                showDialog<void>(
                                  context: context,
                                  builder: (_) => AlertDialog(
                                    title: Text('Minute ${r['minute_start']}'),
                                    content: SingleChildScrollView(
                                      child: SelectableText(const JsonEncoder.withIndent('  ').convert({
                                        ...r,
                                        'counts': _safeDecode(r['counts_json']),
                                      })),
                                    ),
                                    actions: [
                                      TextButton(
                                        onPressed: () => Navigator.pop(context),
                                        child: const Text('Close'),
                                      ),
                                    ],
                                  ),
                                );
                              },
                            );
                          },
                        ),
            ),
          ],
        ),
      ),
    );
  }

  static dynamic _safeDecode(Object? v) {
    try {
      if (v == null) return null;
      return jsonDecode(v.toString());
    } catch (_) {
      return v;
    }
  }
}
