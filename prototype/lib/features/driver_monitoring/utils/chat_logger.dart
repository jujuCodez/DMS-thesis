import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/foundation.dart' show kIsWeb;

import '../../../data/local_app_db.dart';
import 'metrics.dart' show Metrics;

enum Masking { none, truncate, hash }

class ChatLogger {
  ChatLogger._();
  static final ChatLogger I = ChatLogger._();

  bool enabled = true;
  void setEnabled(bool v) => enabled = v;

  Masking masking = Masking.none;
  void setMasking(Masking m) => masking = m;

  String? _convId;
  int _turnSeq = 0;

  String get sessionId => Metrics.I.sessionId;
  String get device => Metrics.I.device;
  String? get username => Metrics.I.username;
  String? get appVer => Metrics.I.appVer;

  Future<void> _ensureTables() async {
    final db = LocalAppDb.I.db;

    await db.execute('''
      CREATE TABLE IF NOT EXISTS chat_turns (
        client_ts TEXT,
        session   TEXT,
        conv_id   TEXT,
        turn_seq  INTEGER,
        role      TEXT,
        source    TEXT,
        phase     TEXT,
        cause     TEXT,
        label     TEXT,
        text      TEXT,
        text_len  INTEGER,
        meta_json TEXT
      );
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS llm_calls (
        client_ts     TEXT,
        session       TEXT,
        conv_id       TEXT,
        turn_seq      INTEGER,
        phase         TEXT,
        provider      TEXT,
        model         TEXT,
        endpoint      TEXT,
        temperature   REAL,
        top_p         REAL,
        max_tokens    INTEGER,
        http_status   INTEGER,
        error         TEXT,
        dur_ms        REAL,
        ttft_ms       REAL,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        total_tokens  INTEGER,
        request_bytes INTEGER,
        response_bytes INTEGER,
        cached_hit    INTEGER,
        retries       INTEGER,
        meta_json     TEXT
      );
    ''');

    await db.execute('''
      CREATE TABLE IF NOT EXISTS tts_events (
        client_ts TEXT,
        session   TEXT,
        conv_id   TEXT,
        turn_seq  INTEGER,
        utter_id  TEXT,
        action    TEXT,
        language  TEXT,
        rate      REAL,
        pitch     REAL,
        engine    TEXT,
        text_len  INTEGER,
        dur_ms    REAL,
        meta_json TEXT
      );
    ''');

    await db.execute('CREATE INDEX IF NOT EXISTS idx_chat_conv ON chat_turns(conv_id, turn_seq);');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_llm_conv ON llm_calls(conv_id, turn_seq);');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_tts_conv ON tts_events(conv_id, turn_seq);');
  }

  Future<String> ensureConversation({String? convId}) async {
    if (!enabled) return _convId ??= _newConvId();
    await _ensureTables();
    if (convId != null) _convId = convId;
    _convId ??= _newConvId();
    return _convId!;
  }

  void resetConversation() {
    _convId = null;
    _turnSeq = 0;
  }

  String _newConvId() {
    final ts = DateTime.now().toUtc().toIso8601String().replaceAll(':', '-');
    final rnd = (math.Random().nextInt(900000) + 100000).toString();
    return '${ts}_$rnd';
  }

  int nextTurnSeq() => ++_turnSeq;

  String _mask(String text) {
    switch (masking) {
      case Masking.none:
        return text;
      case Masking.truncate:
        final t = text.trim();
        if (t.length <= 120) return t;
        return '${t.substring(0, 120)}…';
      case Masking.hash:
        int h = 0;
        for (var c in text.codeUnits) {
          h = (h * 31 + c) & 0x7fffffff;
        }
        return '[[hash:${h.toRadixString(16)} len=${text.length}]]';
    }
  }

  Map<String, dynamic> _env() => {
        'username': username,
        'device': device,
        'app_ver': appVer,
        'platform': kIsWeb ? 'web' : 'mobile',
      };

  Future<void> logTurn({
    required String role,      // user | assistant | system | tool
    required String source,    // stt | app | llm | tts | tool | system
    required int turnSeq,
    required String phase,     // initial | followup | chat
    String? cause,
    String? label,
    required String text,
    Map<String, Object?>? meta,
  }) async {
    if (!enabled) return;
    await _ensureTables();
    final db = LocalAppDb.I.db;
    await db.insert('chat_turns', {
      'client_ts': DateTime.now().toUtc().toIso8601String(),
      'session'  : sessionId,
      'conv_id'  : await ensureConversation(),
      'turn_seq' : turnSeq,
      'role'     : role,
      'source'   : source,
      'phase'    : phase,
      'cause'    : cause,
      'label'    : label,
      'text'     : _mask(text),
      'text_len' : text.length,
      'meta_json': meta == null ? null : jsonEncode({..._env(), ...meta}),
    });
  }

  Future<void> logLlmCall({
    required int turnSeq,
    required String phase,
    required String provider,
    required String model,
    required String endpoint,
    double? temperature,
    double? topP,
    int? maxTokens,
    required int httpStatus,
    String? error,
    required double durMs,
    double? ttftMs,
    int? inputTokens,
    int? outputTokens,
    int? totalTokens,
    int? requestBytes,
    int? responseBytes,
    bool? cacheHit,
    int? retries,
    Map<String, Object?>? meta,
  }) async {
    if (!enabled) return;
    await _ensureTables();
    final db = LocalAppDb.I.db;
    await db.insert('llm_calls', {
      'client_ts'     : DateTime.now().toUtc().toIso8601String(),
      'session'       : sessionId,
      'conv_id'       : await ensureConversation(),
      'turn_seq'      : turnSeq,
      'phase'         : phase,
      'provider'      : provider,
      'model'         : model,
      'endpoint'      : endpoint,
      'temperature'   : temperature,
      'top_p'         : topP,
      'max_tokens'    : maxTokens,
      'http_status'   : httpStatus,
      'error'         : error,
      'dur_ms'        : durMs,
      'ttft_ms'       : ttftMs,
      'input_tokens'  : inputTokens,
      'output_tokens' : outputTokens,
      'total_tokens'  : totalTokens,
      'request_bytes' : requestBytes,
      'response_bytes': responseBytes,
      'cached_hit'    : cacheHit == null ? null : (cacheHit ? 1 : 0),
      'retries'       : retries,
      'meta_json'     : meta == null ? null : jsonEncode({..._env(), ...meta}),
    });
  }

  Future<void> logTtsEvent({
    required int turnSeq,
    required String utterId,
    required String action,  // start | end
    required String language,
    required double rate,
    required double pitch,
    String engine = 'flutter_tts',
    int? textLen,
    double? durMs,
    Map<String, Object?>? meta,
  }) async {
    if (!enabled) return;
    await _ensureTables();
    final db = LocalAppDb.I.db;
    await db.insert('tts_events', {
      'client_ts' : DateTime.now().toUtc().toIso8601String(),
      'session'   : sessionId,
      'conv_id'   : await ensureConversation(),
      'turn_seq'  : turnSeq,
      'utter_id'  : utterId,
      'action'    : action,
      'language'  : language,
      'rate'      : rate,
      'pitch'     : pitch,
      'engine'    : engine,
      'text_len'  : textLen,
      'dur_ms'    : durMs,
      'meta_json' : meta == null ? null : jsonEncode({..._env(), ...meta}),
    });
  }
}
