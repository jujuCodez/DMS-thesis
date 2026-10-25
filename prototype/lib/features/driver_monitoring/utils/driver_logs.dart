// lib/features/driver_monitoring/utils/driver_logs.dart
import 'dart:async';
import 'dart:math';
import 'package:cloud_firestore/cloud_firestore.dart';

// Optional: used only by clearLocal() to wipe local SQLite cache of driver logs.
import '../../../data/local_app_db.dart' show LocalAppDb;

/// Per-minute aggregates written only when there's activity.
/// Identity is frozen per session so it never flips mid-run.
class DriverLogs {
  DriverLogs._();
  static final DriverLogs I = DriverLogs._();

  // ------------ pluggable sink ------------
  /// When set, writes minutes to this sink; otherwise defaults to Firestore.
  DriverLogsSink? _sink; // null => default to Firestore
  void setSink(DriverLogsSink? s) => _sink = s;

  // ------------ identity (mutable) ------------
  String? _fullName;        // latest provided/learned full name
  String? _username;        // latest provided username
  bool get _hasIdentity => _username != null && _username!.trim().isNotEmpty;

  // ------------ identity (frozen per session) ------------
  String? _sessionUsername; // frozen at _startSession
  String? _sessionFullName; // can be filled in if username matches

  void setIdentity({
    String? fullName,
    String? username,
    bool flushIfChanged = true,
  }) {
    final newUser = (username ?? '').trim();
    final curUser = (_username ?? '').trim();
    final sessUser = (_sessionUsername ?? '').trim();

    bool usernameAccepted = false;
    if (username != null) {
      // If a session is running with frozen user, ignore attempts to switch.
      if (_sessionId != null &&
          sessUser.isNotEmpty &&
          newUser.isNotEmpty &&
          newUser != sessUser) {
        // ignore
      } else {
        if (newUser != curUser) {
          _username = newUser;
          usernameAccepted = true;
        }
      }
    }

    if (fullName != null) {
      final f = fullName.trim();
      if (f != _fullName) _fullName = f;
      if (_sessionId != null &&
          sessUser.isNotEmpty &&
          (newUser.isEmpty || newUser == sessUser)) {
        _sessionFullName = f;
      }
    }

    if (flushIfChanged && usernameAccepted && _bucket != null) {
      _flush(close: false);
    }

    if ((_fullName == null || _fullName!.trim().isEmpty) && _hasIdentity) {
      _maybeResolveNameFromProfile(_username!);
    }
  }

  void clearIdentity() {
    _flush(close: false);
    _fullName = null;
    _username = null;
    _sessionUsername = null;
    _sessionFullName = null;
    _attemptedNameLookupThisSession = false;
  }

  // ------------ enable/disable (controls Firestore uploads) ------------
  bool _enabled = false;
  bool get enabled => _enabled;

  void setEnabled(bool v) {
    if (_enabled == v) return;
    _enabled = v;
    if (v) {
      _startSession();
    } else {
      _stopSession();
    }
  }

  // ------------ session + timer ------------
  String? _sessionId;
  Timer? _tickTimer;
  _Bucket? _bucket;
  bool _attemptedNameLookupThisSession = false;

  void _startSession() {
    _sessionId = _makeSessionId();
    _bucket = _Bucket.forNow();
    _attemptedNameLookupThisSession = false;

    if (_hasIdentity) {
      _sessionUsername = _username!.trim();
      _sessionFullName =
          (_fullName ?? '').trim().isEmpty ? null : _fullName!.trim();
    } else {
      _sessionUsername = null;
      _sessionFullName = null;
    }

    if ((_sessionFullName == null || _sessionFullName!.isEmpty) &&
        (_sessionUsername != null && _sessionUsername!.isNotEmpty)) {
      _maybeResolveNameFromProfile(_sessionUsername!);
    }

    _tickTimer?.cancel();
    _tickTimer = Timer.periodic(const Duration(seconds: 5), (_) => _tick());
  }

  void _stopSession() {
    _tickTimer?.cancel();
    _tickTimer = null;
    _flush(close: true);
    _bucket = null;
    _sessionId = null;
    _sessionUsername = null;
    _sessionFullName = null;
    _attemptedNameLookupThisSession = false;
  }

  void dispose() => _stopSession();

  // ======================================================================
  // Public API to increment ONLY when audio actually happens
  // ======================================================================

  /// Call this **right after** the warn beep successfully starts playing.
  /// Example: DriverLogs.I.markWarnPlayed('looking_away');
  void markWarnPlayed(String category) {
    _increment(level: 'warn', category: category);
  }

  /// Call this **when the alert TTS initial line actually finishes**.
  /// Example: DriverLogs.I.markAlertSpoken('drowsy');
  void markAlertSpoken(String category) {
    _increment(level: 'alert', category: category);
  }

  // Optional GPS averaging (unchanged)
  void updateGps(double lat, double lon) {
    if (!_enabled || _bucket == null) return;
    if (_sessionUsername == null || _sessionUsername!.isEmpty) return;
    _rollMinuteIfNeeded();
    _bucket!.addGps(lat, lon);
  }

  // ------------ internals ------------
  final _db = FirebaseFirestore.instance;

  void _increment({required String level, required String category}) {
    if (!_enabled || _bucket == null) return;
    if (_sessionUsername == null || _sessionUsername!.isEmpty) return;

    _rollMinuteIfNeeded();

    final cat = _normalizeCategory(category);
    _bucket!.bump(cat, level);
  }

  /// Normalize any incoming label/category into the canonical server keys:
  /// yawning | using_phone | drowsy | looking_away | no_face
  String _normalizeCategory(String raw) {
    final r = raw.toLowerCase().trim();
    if (r.contains('yawn')) return 'yawning';
    if (r.contains('phone')) return 'using_phone';
    if (r.contains('drowsy') || r.contains('sleep')) return 'drowsy';
    if (r.contains('no_face') || r.contains('noface') || r.contains('no face')) {
      return 'no_face';
    }
    if (r.contains('look') || r.contains('away') || r.contains('pose') || r.contains('distract')) {
      return 'looking_away';
    }
    // default any unknown -> looking_away
    return 'looking_away';
  }

  void _tick() {
    if (!_enabled) return;
    _rollMinuteIfNeeded();

    if ((_sessionFullName == null || _sessionFullName!.trim().isEmpty) &&
        (_sessionUsername != null && _sessionUsername!.isNotEmpty) &&
        !_attemptedNameLookupThisSession) {
      _maybeResolveNameFromProfile(_sessionUsername!);
    }
  }

  void _rollMinuteIfNeeded() {
    final nowUtc = DateTime.now().toUtc();
    final minuteStart = DateTime.utc(
      nowUtc.year,
      nowUtc.month,
      nowUtc.day,
      nowUtc.hour,
      nowUtc.minute,
    );
    if (_bucket == null ||
        !_bucket!.minuteStart.isAtSameMomentAs(minuteStart)) {
      _flush(close: false);
      _bucket = _Bucket(minuteStart);
    }
  }

  bool _bucketHasActivity(_Bucket b) {
    if (b.gpsSamples > 0) return true;
    for (final levels in b.counts.values) {
      if ((levels['warn'] ?? 0) > 0 || (levels['alert'] ?? 0) > 0) {
        return true;
      }
    }
    return false;
  }

  Future<void> _flush({required bool close}) async {
    final b = _bucket;
    final sid = _sessionId;
    if (b == null || sid == null) return;
    if (_sessionUsername == null || _sessionUsername!.isEmpty) return;

    // Skip empty minutes unless closing the session.
    if (!close && !_bucketHasActivity(b)) return;

    final username = _sessionUsername!;
    final name = _resolveName(_sessionFullName ?? _fullName, username);

    final minuteEnd = b.minuteStart.add(const Duration(minutes: 1));
    final avgLat = b.gpsSamples > 0 ? (b.latSum / b.gpsSamples) : null;
    final avgLon = b.gpsSamples > 0 ? (b.lonSum / b.gpsSamples) : null;

    final payload = <String, dynamic>{
      "name": name,
      "username": username,

      "session": sid,
      "server_ts": FieldValue.serverTimestamp(),
      "client_ts": DateTime.now().toUtc().toIso8601String(),
      "minute_start": _isoMinute(b.minuteStart),
      "minute_end": _isoMinute(minuteEnd),

      // kept for schema compatibility, we never increment it here
      "frames_seen": b.framesSeen,
      "avg_latitude": avgLat,
      "avg_longitude": avgLon,
      "gps_samples": b.gpsSamples,
      "counts": {
        "yawning": {
          "warn": b.cnt("yawning", "warn"),
          "alert": b.cnt("yawning", "alert"),
        },
        "using_phone": {
          "warn": b.cnt("using_phone", "warn"),
          "alert": b.cnt("using_phone", "alert"),
        },
        "drowsy": {
          "warn": b.cnt("drowsy", "warn"),
          "alert": b.cnt("drowsy", "alert"),
        },
        "looking_away": {
          "warn": b.cnt("looking_away", "warn"),
          "alert": b.cnt("looking_away", "alert"),
        },
        "no_face": {
          "warn": b.cnt("no_face", "warn"),
          "alert": b.cnt("no_face", "alert"),
        },
      },
    };

    final docId = _buildDocId(
      username: username,
      session: sid,
      minuteStartUtc: b.minuteStart,
    );

    try {
      if (_sink != null) {
        await _sink!.writeMinute(docId: docId, payload: payload);
      } else {
        await _db
            .collection('driver_logs_new')
            .doc(docId)
            .set(payload, SetOptions(merge: true));
      }
    } catch (_) {
      // swallow
    }
  }

  Future<void> _maybeResolveNameFromProfile(String username) async {
    if (_attemptedNameLookupThisSession) return;
    _attemptedNameLookupThisSession = true;
    try {
      final q = await _db
          .collection('drivers')
          .where('username', isEqualTo: username)
          .limit(1)
          .get();
      if (q.docs.isNotEmpty) {
        final d = q.docs.first.data();
        String s(Object? x) => (x ?? '').toString().trim();
        final full = s(d['full_name']);
        if (full.isNotEmpty) {
          _fullName = full;
          if (_sessionUsername == username) _sessionFullName = full;
        } else {
          final first = s(d['first_name']);
          final last = s(d['last_name']);
          final composed =
              [first, last].where((v) => v.isNotEmpty).join(' ');
          if (composed.isNotEmpty) {
            _fullName = composed;
            if (_sessionUsername == username) _sessionFullName = composed;
          }
        }
      }
    } catch (_) {}
  }

  String _makeSessionId() {
    final now = DateTime.now()
        .toUtc()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('/', '-');
    final rnd = 10000 + Random().nextInt(90000);
    return '${now}_$rnd';
  }

  String _buildDocId({
    required String username,
    required String session,
    required DateTime minuteStartUtc,
  }) {
    String sanitize(String s) =>
        s.replaceAll(RegExp(r'[^A-Za-z0-9_\-\.]'), '_');
    final u = sanitize(username);
    final sess = sanitize(session);
    final minute = _isoMinute(minuteStartUtc).replaceAll(':', '-');
    return '${u}_${sess}_$minute';
  }

  String _isoMinute(DateTime dt) {
    final base = dt.toUtc().toIso8601String();
    final noMs = base.contains('.')
        ? '${base.split('.').first}Z'
        : base.endsWith('Z')
            ? base
            : '${base}Z';
    return noMs;
  }

  String _resolveName(String? fullName, String username) {
    final fn = fullName?.trim();
    if (fn != null && fn.isNotEmpty) return fn;
    return username;
  }

  // ======================================================================
  // Local SQLite management (for Settings → "Delete local driver logs")
  // ======================================================================

  /// Deletes **local** per-minute driver logs stored in the app's SQLite cache.
  ///
  /// - This does **not** affect Firestore.
  /// - Safe to call even if the local table doesn't exist yet.
  Future<void> clearLocal() async {
    try {
      final db = LocalAppDb.I.db;
      // By convention we store per-minute driver logs in this table.
      // If your app uses a different table name, adjust here.
      await db.delete('driver_logs_local');
    } catch (_) {
      // swallow (e.g., table doesn't exist)
    }
  }
}

/// Sink interface (Firestore/SQLite/anything can implement).
abstract class DriverLogsSink {
  Future<void> writeMinute({
    required String docId,
    required Map<String, dynamic> payload,
  });
}

class FirestoreDriverLogsSink implements DriverLogsSink {
  final FirebaseFirestore _db;
  FirestoreDriverLogsSink({FirebaseFirestore? db})
      : _db = db ?? FirebaseFirestore.instance;

  @override
  Future<void> writeMinute({
    required String docId,
    required Map<String, dynamic> payload,
  }) async {
    await _db
        .collection('driver_logs_new')
        .doc(docId)
        .set(payload, SetOptions(merge: true));
  }
}

class _Bucket {
  _Bucket(this.minuteStart);
  _Bucket.forNow()
      : minuteStart = DateTime.utc(
          DateTime.now().toUtc().year,
          DateTime.now().toUtc().month,
          DateTime.now().toUtc().day,
          DateTime.now().toUtc().hour,
          DateTime.now().toUtc().minute,
        );

  final DateTime minuteStart;

  // Kept for schema compatibility only.
  int framesSeen = 0;

  double latSum = 0, lonSum = 0;
  int gpsSamples = 0;
  void addGps(double lat, double lon) {
    latSum += lat;
    lonSum += lon;
    gpsSamples++;
  }

  /// Canonical keys expected by the server.
  final Map<String, Map<String, int>> counts = {
    'yawning':      {'warn': 0, 'alert': 0},
    'using_phone':  {'warn': 0, 'alert': 0},
    'drowsy':       {'warn': 0, 'alert': 0},
    'looking_away': {'warn': 0, 'alert': 0},
    'no_face':      {'warn': 0, 'alert': 0},
  };

  void bump(String category, String level) {
    final map = counts[category];
    if (map == null) return;
    map[level] = (map[level] ?? 0) + 1;
  }

  int cnt(String category, String level) => counts[category]?[level] ?? 0;
}
