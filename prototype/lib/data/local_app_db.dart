// lib/data/local_app_db.dart
import 'dart:io';
import 'dart:math';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

/// Global switch: if true, Auth/Profile/Metrics/Logs use SQLite instead of Firebase.
class LocalBackendSwitch {
  LocalBackendSwitch._();
  static final LocalBackendSwitch I = LocalBackendSwitch._();
  bool useLocal = false;
}

/// One SQLite file that replaces Firebase collections:
/// - useracc          (auth)           id, username (unique), password, created_at
/// - drivers          (profile)        username (unique), names, car, timestamps
/// - metrics          (events)         flat schema (same columns every row)
/// - driver_logs_new  (minute logs)    your minute payloads
class LocalAppDb {
  LocalAppDb._();
  static final LocalAppDb I = LocalAppDb._();

  Database? _db;
  String? _activePath;

  String get activePath => _activePath ?? '<not opened>';

  // -------- Open / Import / Export --------

  Future<void> openDefault() async {
    final dir = await getDatabasesPath();
    final path = p.join(dir, 'dms_local.db');
    await _openAtPath(path);
  }

  /// Copy an external .db file into app data and open it.
  Future<void> openFromFile(String externalPath) async {
    final dir = await getDatabasesPath();
    final path = p.join(dir, 'dms_local.db');

    final src = File(externalPath);
    if (!await src.exists()) {
      throw ArgumentError('File not found: $externalPath');
    }
    await _db?.close();
    _db = null;

    final dst = File(path);
    if (await dst.exists()) await dst.delete();
    await src.copy(path);

    await _openAtPath(path);
  }

  /// Copy the current DB file out to a user path (Save As).
  Future<void> exportDbToPath(String savePath) async {
    final dbPath = _activePath;
    if (dbPath == null || !(await File(dbPath).exists())) {
      throw StateError('No active DB to export.');
    }
    final out = File(savePath);
    if (await out.exists()) await out.delete();
    await File(dbPath).copy(savePath);
  }

  /// Optional: export logs as CSV to a user path.
  Future<void> exportDriverLogsCsvToPath(String savePath) async {
    final rows = await db.query('driver_logs_new', orderBy: 'minute_start DESC');
    final sink = File(savePath).openWrite();
    sink.writeln([
      'id','name','username','session','minute_start','minute_end',
      'frames_seen','avg_latitude','avg_longitude','gps_samples','counts_json','client_ts'
    ].join(','));
    String esc(v) => '"${(v ?? '').toString().replaceAll('"', '""')}"';
    for (final r in rows) {
      sink.writeln([
        esc(r['id']), esc(r['name']), esc(r['username']), esc(r['session']),
        esc(r['minute_start']), esc(r['minute_end']),
        esc(r['frames_seen']), esc(r['avg_latitude']), esc(r['avg_longitude']),
        esc(r['gps_samples']), esc(r['counts_json']), esc(r['client_ts']),
      ].join(','));
    }
    await sink.flush();
    await sink.close();
  }

  Database get db {
    final d = _db;
    if (d == null) {
      throw StateError('Local DB not opened. Call openDefault() or openFromFile().');
    }
    return d;
  }

  Future<void> _openAtPath(String path) async {
    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, _) async => _createSchema(db),
      onOpen: (db) async => _createSchema(db), // ensure tables + columns on old files
    );
    _activePath = path;
  }

  Future<void> _createSchema(Database db) async {
    // ---- Core tables (latest schema) ----
    await db.execute('''
      CREATE TABLE IF NOT EXISTS useracc (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT
      );
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS drivers (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        first_name TEXT,
        last_name TEXT,
        full_name TEXT,
        car_brand TEXT,
        car_model TEXT,
        created_at TEXT,
        updated_at TEXT
      );
    ''');

    // Keep in sync with exporter + Metrics.I._ensureTable()
    await db.execute('''
      CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        testing_phase TEXT,      -- NEW
        cause TEXT,
        dur_ms REAL,
        phase TEXT,
        type TEXT,
        had_user_reply INTEGER,
        text_len INTEGER,
        net_seq INTEGER,
        conf REAL,
        pose_ypr TEXT,           -- NEW
        looking_away TEXT        -- NEW ('looking_away' | 'not_looking_away')
      );
    ''');

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

    // ---- Migrate older DBs: add any missing columns ----
    await _ensureColumns(db, 'metrics', const {
      'session'        : 'TEXT',
      'username'       : 'TEXT',
      'device'         : 'TEXT',
      'app_ver'        : 'TEXT',
      'name'           : 'TEXT',
      'frame_id'       : 'INTEGER',
      'jpeg_size'      : 'INTEGER',
      'transport_ms'   : 'REAL',
      'jpeg_decode_ms' : 'REAL',
      'face_ms'        : 'REAL',
      'preproc_ms'     : 'REAL',
      'model_ms'       : 'REAL',
      'rx_to_model_ms' : 'REAL',
      'algo_total_ms'  : 'REAL',
      'status'         : 'TEXT',
      'label'          : 'TEXT',
      'testing_phase'  : 'TEXT',   // ensure new label field exists
      'cause'          : 'TEXT',
      'dur_ms'         : 'REAL',
      'phase'          : 'TEXT',
      'type'           : 'TEXT',
      'had_user_reply' : 'INTEGER',
      'text_len'       : 'INTEGER',
      'net_seq'        : 'INTEGER',
      'conf'           : 'REAL',
      'pose_ypr'       : 'TEXT',   // ensure new pose field exists
      'looking_away'   : 'TEXT',   // ensure new behavior field exists
    });

    // ---- Helpful indexes ----
    await db.execute('CREATE INDEX IF NOT EXISTS idx_useracc_username ON useracc(username);');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_drivers_username ON drivers(username);');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_logs_minute ON driver_logs_new(minute_start DESC);');

    // Speeds up live panels/exports
    await db.execute('CREATE INDEX IF NOT EXISTS idx_metrics_client_ts ON metrics(client_ts);');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, client_ts);');
  }

  /// Add any columns missing from [table] using SQLite's ALTER TABLE ADD COLUMN.
  Future<void> _ensureColumns(Database db, String table, Map<String, String> wanted) async {
    final info = await db.rawQuery('PRAGMA table_info($table)');
    final have = <String>{for (final r in info) (r['name'] as String)};
    for (final entry in wanted.entries) {
      if (!have.contains(entry.key)) {
        await db.execute('ALTER TABLE $table ADD COLUMN ${entry.key} ${entry.value};');
      }
    }
  }

  // ---------------- AUTH ----------------
  Future<String?> signUp({required String username, required String password}) async {
    final exists = await db.query('useracc', where: 'username=?', whereArgs: [username], limit: 1);
    if (exists.isNotEmpty) return 'Username already taken';
    await db.insert('useracc', {
      'id': _randomId(),
      'username': username,
      'password': password,
      'created_at': _nowIso(),
    });
    return null;
  }

  Future<Map<String, dynamic>?> findUser(String username) async {
    final rows = await db.query('useracc', where: 'username=?', whereArgs: [username], limit: 1);
    return rows.isEmpty ? null : rows.first;
  }

  // ---------------- PROFILE ----------------
  Future<Map<String, dynamic>?> getProfile(String username) async {
    final rows = await db.query('drivers', where: 'username=?', whereArgs: [username], limit: 1);
    return rows.isEmpty ? null : rows.first;
  }

  Future<void> upsertProfile({
    required String username,
    String? firstName,
    String? lastName,
    String? fullName,
    String? carBrand,
    String? carModel,
  }) async {
    fullName ??= _composeName(firstName, lastName) ?? username;
    final now = _nowIso();

    final rows = await db.query('drivers', where: 'username=?', whereArgs: [username], limit: 1);
    if (rows.isEmpty) {
      await db.insert('drivers', {
        'id': _randomId(),
        'username': username,
        'first_name': firstName,
        'last_name': lastName,
        'full_name': fullName,
        'car_brand': carBrand,
        'car_model': carModel,
        'created_at': now,
        'updated_at': now,
      });
    } else {
      await db.update('drivers', {
        'first_name': firstName,
        'last_name': lastName,
        'full_name': fullName,
        'car_brand': carBrand,
        'car_model': carModel,
        'updated_at': now,
      }, where: 'username=?', whereArgs: [username]);
    }
  }

  // ---------------- METRICS ----------------
  Future<void> insertMetric(Map<String, dynamic> m) async {
    await db.insert('metrics', {
      'client_ts'      : m['client_ts'],
      'session'        : m['session'],
      'username'       : m['username'],
      'device'         : m['device'],
      'app_ver'        : m['app_ver'],
      'name'           : m['name'],
      'frame_id'       : m['frame_id'],
      'jpeg_size'      : m['jpeg_size'],
      'transport_ms'   : m['transport_ms'],
      'jpeg_decode_ms' : m['jpeg_decode_ms'],
      'face_ms'        : m['face_ms'],
      'preproc_ms'     : m['preproc_ms'],
      'model_ms'       : m['model_ms'],
      'rx_to_model_ms' : m['rx_to_model_ms'],
      'algo_total_ms'  : m['algo_total_ms'],
      'status'         : m['status'],
      'label'          : m['label'],
      'testing_phase'  : m['testing_phase'],   // NEW
      'cause'          : m['cause'],
      'dur_ms'         : m['dur_ms'],
      'phase'          : m['phase'],
      'type'           : m['type'],
      'had_user_reply' : m['had_user_reply'] == true ? 1 : (m['had_user_reply'] == null ? null : 0),
      'text_len'       : m['text_len'],
      'net_seq'        : m['net_seq'],
      'conf'           : m['conf'],
      'pose_ypr'       : m['pose_ypr'],        // NEW
      'looking_away'   : m['looking_away'],    // NEW
    });
  }

  // ---------------- DRIVER LOGS ----------------
  Future<void> upsertDriverMinute({
    required String docId,
    required Map<String, dynamic> payload,
  }) async {
    await db.insert('driver_logs_new', {
      'id'            : docId,
      'name'          : payload['name'],
      'username'      : payload['username'],
      'session'       : payload['session'],
      'minute_start'  : payload['minute_start'],
      'minute_end'    : payload['minute_end'],
      'frames_seen'   : payload['frames_seen'],
      'avg_latitude'  : payload['avg_latitude'],
      'avg_longitude' : payload['avg_longitude'],
      'gps_samples'   : payload['gps_samples'],
      'counts_json'   : (payload['counts'] ?? {}).toString(),
      'client_ts'     : payload['client_ts'],
    }, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  // -------- helpers --------
  String _nowIso() => DateTime.now().toIso8601String();
  String? _composeName(String? f, String? l) {
    final F = f?.trim() ?? '';
    final L = l?.trim() ?? '';
    final s = [F, L].where((x) => x.isNotEmpty).join(' ');
    return s.isEmpty ? null : s;
  }

  String _randomId() {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    final rnd = Random.secure();
    return List.generate(20, (_) => chars[rnd.nextInt(chars.length)]).join();
  }
}

/// ---------- Sinks plugged into your existing utilities ----------

/// DriverLogs -> SQLite
abstract class DriverLogsSink {
  Future<void> writeMinute({required String docId, required Map<String, dynamic> payload});
}
class SqliteDriverLogsSink implements DriverLogsSink {
  @override
  Future<void> writeMinute({required String docId, required Map<String, dynamic> payload}) {
    return LocalAppDb.I.upsertDriverMinute(docId: docId, payload: payload);
  }
}

/// Metrics -> SQLite
abstract class MetricsSink {
  Future<void> insert(Map<String, dynamic> payload);
}
class SqliteMetricsSink implements MetricsSink {
  @override
  Future<void> insert(Map<String, dynamic> payload) => LocalAppDb.I.insertMetric(payload);
}
