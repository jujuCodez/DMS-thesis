import 'dart:convert';
import 'dart:io' show Platform, Directory, File;
import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter_file_dialog/flutter_file_dialog.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';

import '../../../data/local_app_db.dart';

class ChatLogsPage extends StatefulWidget {
  const ChatLogsPage({super.key});
  @override
  State<ChatLogsPage> createState() => _ChatLogsPageState();
}

class _ChatLogsPageState extends State<ChatLogsPage> {
  String? _convFilter; // null => latest 10 convs
  List<String> _recentConvs = [];
  List<Map<String, Object?>> _turns = [];
  List<Map<String, Object?>> _llm = [];
  List<Map<String, Object?>> _tts = [];
  List<Map<String, Object?>> _stt = []; // from metrics (name='stt_latency')

  // quick summary (recomputed per conversation load)
  late Map<String, num> _summaryLlmdur;
  late Map<String, num> _summaryLlmTtft;
  late Map<String, num> _summaryTtsdur;
  late Map<String, num> _summarySttdur;

  // single vertical scroll for the whole page (prevents nested-scroll weirdness)
  final ScrollController _listCtrl = ScrollController();

  @override
  void initState() {
    super.initState();
    _summaryLlmdur = _zerosSummary();
    _summaryLlmTtft = _zerosSummary();
    _summaryTtsdur = _zerosSummary();
    _summarySttdur = _zerosSummary();
    _loadRecent();
  }

  @override
  void dispose() {
    _listCtrl.dispose();
    super.dispose();
  }

  // ---------- load helpers ----------

  Future<void> _loadRecent() async {
    final db = LocalAppDb.I.db;
    final convs = await db.rawQuery('''
      SELECT conv_id, MAX(client_ts) as last_ts
      FROM chat_turns
      GROUP BY conv_id
      ORDER BY last_ts DESC
      LIMIT 10
    ''');
    setState(() => _recentConvs =
        convs.map((r) => (r['conv_id'] ?? '').toString()).toList());
    if (_recentConvs.isNotEmpty) {
      _convFilter ??= _recentConvs.first;
      await _loadConv(_convFilter!);
    }
  }

  Future<void> _loadConv(String convId) async {
    final db = LocalAppDb.I.db;

    final turns = await db.query(
      'chat_turns',
      where: 'conv_id = ?',
      whereArgs: [convId],
      orderBy: 'turn_seq ASC, client_ts ASC',
    );

    final llm = await db.query(
      'llm_calls',
      where: 'conv_id = ?',
      whereArgs: [convId],
      orderBy: 'turn_seq ASC, client_ts ASC',
    );

    final tts = await db.query(
      'tts_events',
      where: 'conv_id = ?',
      whereArgs: [convId],
      orderBy: 'turn_seq ASC, client_ts ASC',
    );

    // ---- STT rows from metrics, matched by session + time window ----
    final sessions = <String>{
      ...turns
          .map((r) => (r['session'] ?? '').toString())
          .where((s) => s.isNotEmpty),
      ...llm
          .map((r) => (r['session'] ?? '').toString())
          .where((s) => s.isNotEmpty),
      ...tts
          .map((r) => (r['session'] ?? '').toString())
          .where((s) => s.isNotEmpty),
    };

    DateTime? minTs, maxTs;
    for (final r in turns) {
      final ts = DateTime.tryParse((r['client_ts'] ?? '').toString());
      if (ts == null) continue;
      minTs = (minTs == null || ts.isBefore(minTs!)) ? ts : minTs;
      maxTs = (maxTs == null || ts.isAfter(maxTs!)) ? ts : maxTs;
    }
    for (final r in [...llm, ...tts]) {
      final ts = DateTime.tryParse((r['client_ts'] ?? '').toString());
      if (ts == null) continue;
      minTs = (minTs == null || ts.isBefore(minTs!)) ? ts : minTs;
      maxTs = (maxTs == null || ts.isAfter(maxTs!)) ? ts : maxTs;
    }

    List<Map<String, Object?>> sttRows = [];
    if (minTs != null && maxTs != null) {
      final startIso =
          minTs!.subtract(const Duration(seconds: 10)).toUtc().toIso8601String();
      final endIso =
          maxTs!.add(const Duration(seconds: 20)).toUtc().toIso8601String();

      if (sessions.isNotEmpty) {
        final placeholders =
            List.generate(sessions.length, (_) => '?').join(',');
        sttRows = await db.rawQuery(
          '''
            SELECT * FROM metrics
            WHERE name = 'stt_latency'
              AND client_ts BETWEEN ? AND ?
              AND session IN ($placeholders)
            ORDER BY client_ts ASC
          ''',
          [startIso, endIso, ...sessions],
        );
      } else {
        sttRows = await db.rawQuery(
          '''
            SELECT * FROM metrics
            WHERE name = 'stt_latency'
              AND client_ts BETWEEN ? AND ?
            ORDER BY client_ts ASC
          ''',
          [startIso, endIso],
        );
      }
    }

    setState(() {
      _convFilter = convId;
      _turns = turns;
      _llm = llm;
      _tts = tts;
      _stt = sttRows;
      _recomputeSummary();
    });
  }

  // ---------- PURGE (clears turns, llm_calls, tts_events, associated metrics) ----------

  Future<void> _confirmAndPurgeCurrent() async {
    if (_convFilter == null) return;
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Purge conversation?'),
        content: const Text(
          'This will permanently delete chat turns, LLM calls, TTS events, and matching STT metrics for this conversation.',
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton.tonal(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Purge')),
        ],
      ),
    );
    if (ok != true) return;

    final convId = _convFilter!;
    final db = LocalAppDb.I.db;

    await db.transaction((txn) async {
      // sessions tied to this conversation
      final sessRows = await txn.rawQuery('''
        SELECT DISTINCT session FROM (
          SELECT session FROM chat_turns WHERE conv_id = ?
          UNION ALL
          SELECT session FROM llm_calls WHERE conv_id = ?
          UNION ALL
          SELECT session FROM tts_events WHERE conv_id = ?
        ) WHERE session IS NOT NULL AND session != ''
      ''', [convId, convId, convId]);

      final sessions = sessRows
          .map((r) => (r['session'] ?? '').toString())
          .where((s) => s.isNotEmpty)
          .toList();

      if (sessions.isNotEmpty) {
        final placeholders =
            List.generate(sessions.length, (_) => '?').join(',');
        await txn.rawDelete(
          'DELETE FROM metrics WHERE session IN ($placeholders)',
          sessions,
        );
      }

      await txn.delete('llm_calls', where: 'conv_id = ?', whereArgs: [convId]);
      await txn.delete('tts_events', where: 'conv_id = ?', whereArgs: [convId]);
      await txn.delete('chat_turns', where: 'conv_id = ?', whereArgs: [convId]);
    });

    _toast('Purged');
    await _loadRecent();
  }

  // ---------- summary / percentiles ----------

  Map<String, num> _zerosSummary() => {
        'count': 0,
        'p50': 0,
        'p95': 0,
        'p99': 0,
      };

  void _recomputeSummary() {
    List<double> _vec(List<Map<String, Object?>> rows, String k) => rows
        .map((r) => (r[k] as num?)?.toDouble())
        .where((v) => v != null && v! > 0)
        .cast<double>()
        .toList();

    Map<String, num> _pct(List<double> xs) {
      if (xs.isEmpty) return _zerosSummary();
      xs.sort();
      double pick(double p) {
        final i = (p * (xs.length - 1)).round().clamp(0, xs.length - 1);
        return xs[i];
      }
      return {
        'count': xs.length,
        'p50': pick(0.50),
        'p95': pick(0.95),
        'p99': pick(0.99),
      };
    }

    _summaryLlmdur = _pct(_vec(_llm, 'dur_ms'));
    _summaryLlmTtft = _pct(_vec(_llm, 'ttft_ms'));
    _summaryTtsdur = _pct(_vec(_tts, 'dur_ms'));
    _summarySttdur = _pct(_vec(_stt, 'dur_ms'));
  }

  // ---------- CSV builders & save helpers ----------

  static Uint8List _csvBytes({
    required List<Map<String, Object?>> rows,
    required List<String> cols,
  }) {
    String esc(v) => '"${(v ?? '').toString().replaceAll('"', '""')}"';
    final sb = StringBuffer()..writeln(cols.join(','));
    for (final r in rows) {
      sb.writeln(cols.map((k) => esc(r[k])).join(','));
    }
    return Uint8List.fromList(utf8.encode(sb.toString()));
  }

  static Uint8List _csvSummaryBytes({
    required String convId,
    required Map<String, num> llmDur,
    required Map<String, num> llmTtft,
    required Map<String, num> ttsDur,
    required Map<String, num> sttDur,
  }) {
    final cols = ['conv_id', 'metric', 'count', 'p50', 'p95', 'p99'];
    String esc(v) => '"${(v ?? '').toString().replaceAll('"', '""')}"';
    final sb = StringBuffer()..writeln(cols.join(','));

    void add(String metric, Map<String, num> m) {
      sb.writeln([
        convId,
        metric,
        (m['count'] ?? 0).toString(),
        (m['p50'] ?? 0).toString(),
        (m['p95'] ?? 0).toString(),
        (m['p99'] ?? 0).toString(),
      ].map(esc).join(','));
    }

    add('llm_dur_ms', llmDur);
    add('llm_ttft_ms', llmTtft);
    add('tts_dur_ms', ttsDur);
    add('stt_ms', sttDur);

    return Uint8List.fromList(utf8.encode(sb.toString()));
  }

  Future<String?> _saveSmart({
    required Uint8List bytes,
    required String fileName,
    required String mimeType,
    required String saveLabel,
    required String extension,
  }) async {
    // 1) Android: SAF “Save to…”
    if (!kIsWeb && Platform.isAndroid) {
      try {
        final params = SaveFileDialogParams(
          data: bytes,
          fileName: fileName,
          mimeTypesFilter: [mimeType],
        );
        final savedUri = await FlutterFileDialog.saveFile(params: params);
        return savedUri == null ? null : 'Saved via Android system picker';
      } catch (_) {
        // Fall through to desktop/save/share.
      }
    }

    // 2) Desktop (and Android/iOS fallback): native Save As…
    try {
      final location = await getSaveLocation(
        suggestedName: fileName,
        acceptedTypeGroups: [
          XTypeGroup(label: saveLabel, extensions: [extension])
        ],
      );
      if (location != null) {
        final xf = XFile.fromData(bytes, name: fileName);
        await xf.saveTo(location.path);
        return location.path;
      }
    } catch (_) {
      // Ignore; try share.
    }

    // 3) Share sheet fallback
    final tmpPath = await _writeTemp(bytes, fileName);
    try {
      await Share.shareXFiles([XFile(tmpPath)]);
      return 'Shared from temp: $tmpPath';
    } catch (_) {
      return tmpPath; // at least we left a file behind
    }
  }

  Future<String> _writeTemp(Uint8List bytes, String fileName) async {
    final dir = kIsWeb
        ? (await Directory.systemTemp.createTemp('chatlogs_web_'))
        : Directory.systemTemp;
    final path = p.join(dir.path, fileName);
    final f = File(path);
    await f.writeAsBytes(bytes, flush: true);
    return path;
  }

  // ---------- exporters ----------

  Future<void> _exportTurnsCsv() async {
    if (_turns.isEmpty) return;
    const cols = [
      'client_ts',
      'session',
      'conv_id',
      'turn_seq',
      'role',
      'source',
      'phase',
      'cause',
      'label',
      'text_len',
      'text',
      'meta_json'
    ];
    final bytes = _csvBytes(rows: _turns, cols: cols);
    final name = 'chat_turns_${_shortId(_convFilter)}.csv';
    final res = await _saveSmart(
      bytes: bytes,
      fileName: name,
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
    if (!mounted) return;
    _toast(res);
  }

  Future<void> _exportLlmCsv() async {
    if (_llm.isEmpty) return;
    const cols = [
      'client_ts',
      'session',
      'conv_id',
      'turn_seq',
      'phase',
      'provider',
      'model',
      'endpoint',
      'temperature',
      'top_p',
      'max_tokens',
      'http_status',
      'error',
      'dur_ms',
      'ttft_ms',
      'input_tokens',
      'output_tokens',
      'total_tokens',
      'request_bytes',
      'response_bytes',
      'cached_hit',
      'retries',
      'meta_json'
    ];
    final bytes = _csvBytes(rows: _llm, cols: cols);
    final name = 'llm_calls_${_shortId(_convFilter)}.csv';
    final res = await _saveSmart(
      bytes: bytes,
      fileName: name,
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
    if (!mounted) return;
    _toast(res);
  }

  Future<void> _exportTtsCsv() async {
    if (_tts.isEmpty) return;
    const cols = [
      'client_ts',
      'session',
      'conv_id',
      'turn_seq',
      'utter_id',
      'action',
      'language',
      'rate',
      'pitch',
      'engine',
      'text_len',
      'dur_ms',
      'meta_json'
    ];
    final bytes = _csvBytes(rows: _tts, cols: cols);
    final name = 'tts_events_${_shortId(_convFilter)}.csv';
    final res = await _saveSmart(
      bytes: bytes,
      fileName: name,
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
    if (!mounted) return;
    _toast(res);
  }

  Future<void> _exportSttCsv() async {
    if (_stt.isEmpty) return;
    const cols = [
      'client_ts',
      'session',
      'name',
      'type',
      'dur_ms',
      'had_user_reply',
      'text_len'
    ];
    final bytes = _csvBytes(rows: _stt, cols: cols);
    final name = 'stt_latency_${_shortId(_convFilter)}.csv';
    final res = await _saveSmart(
      bytes: bytes,
      fileName: name,
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
    if (!mounted) return;
    _toast(res);
  }

  Future<void> _exportSummaryCsv() async {
    if ((_llm.isEmpty && _tts.isEmpty && _stt.isEmpty) || _convFilter == null) {
      return;
    }
    final bytes = _csvSummaryBytes(
      convId: _convFilter!,
      llmDur: _summaryLlmdur,
      llmTtft: _summaryLlmTtft,
      ttsDur: _summaryTtsdur,
      sttDur: _summarySttdur,
    );
    final name = 'chat_summary_${_shortId(_convFilter)}.csv';
    final res = await _saveSmart(
      bytes: bytes,
      fileName: name,
      mimeType: 'text/csv',
      saveLabel: 'CSV',
      extension: 'csv',
    );
    if (!mounted) return;
    _toast(res);
  }

  Future<void> _exportAllJson() async {
    if (_convFilter == null) return;
    final all = {
      'chat_turns': _turns,
      'llm_calls': _llm,
      'tts_events': _tts,
      'stt_latency': _stt,
    };
    final bytes =
        Uint8List.fromList(utf8.encode(const JsonEncoder.withIndent('  ').convert(all)));
    final res = await _saveSmart(
      bytes: bytes,
      fileName: 'chat_${_shortId(_convFilter)}.json',
      mimeType: 'application/json',
      saveLabel: 'JSON',
      extension: 'json',
    );
    if (!mounted) return;
    _toast(res);
  }

  void _toast(String? res) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
          content:
              Text(res == null ? 'Export cancelled' : 'Exported: $res')),
    );
  }

  String _shortId(String? id) {
    if (id == null || id.isEmpty) return 'unknown';
    final s = id.replaceAll(':', '').replaceAll('-', '');
    return s.length <= 8 ? s : s.substring(0, 8);
  }

  // ---------- UI ----------

  @override
  Widget build(BuildContext context) {
    final convs = _recentConvs;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat Logs'),
        actions: [
          IconButton(
            tooltip: 'Purge conversation',
            onPressed: _convFilter == null ? null : _confirmAndPurgeCurrent,
            icon: const Icon(Icons.delete_forever_outlined),
          ),
        ],
      ),
      // Pull-to-refresh + single primary scroll (fixes "can’t swipe down")
      body: RefreshIndicator(
        onRefresh: () async {
          if (_convFilter != null) {
            await _loadConv(_convFilter!);
          } else {
            await _loadRecent();
          }
        },
        // Always scrollable so swipe-down works even with short content
        child: Scrollbar(
          controller: _listCtrl,
          child: CustomScrollView(
            controller: _listCtrl,
            physics: const AlwaysScrollableScrollPhysics(),
            slivers: [
              SliverPadding(
                padding: const EdgeInsets.all(12),
                sliver: SliverList.list(
                  children: [
                    // Top bar: conversation picker (no overflow)
                    Row(children: [
                      const Text('Conversation: '),
                      const SizedBox(width: 8),
                      Expanded(
                        child: DropdownButton<String>(
                          isExpanded: true,
                          value: _convFilter,
                          items: [
                            for (final id in convs)
                              DropdownMenuItem(
                                value: id,
                                child: Text(
                                  id,
                                  overflow: TextOverflow.ellipsis,
                                  maxLines: 1,
                                  softWrap: false,
                                ),
                              ),
                          ],
                          onChanged: (v) => v == null ? null : _loadConv(v),
                        ),
                      ),
                    ]),
                    const SizedBox(height: 10),

                    // Export buttons row (wraps)
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          OutlinedButton.icon(
                            icon: const Icon(Icons.download_outlined),
                            label: const Text('Turns CSV'),
                            onPressed:
                                _turns.isEmpty ? null : _exportTurnsCsv,
                          ),
                          OutlinedButton.icon(
                            icon: const Icon(Icons.download_outlined),
                            label: const Text('LLM CSV'),
                            onPressed: _llm.isEmpty ? null : _exportLlmCsv,
                          ),
                          OutlinedButton.icon(
                            icon: const Icon(Icons.download_outlined),
                            label: const Text('TTS CSV'),
                            onPressed: _tts.isEmpty ? null : _exportTtsCsv,
                          ),
                          OutlinedButton.icon(
                            icon: const Icon(Icons.download_outlined),
                            label: const Text('STT CSV'),
                            onPressed: _stt.isEmpty ? null : _exportSttCsv,
                          ),
                          FilledButton.icon(
                            icon: const Icon(Icons.assessment_outlined),
                            label: const Text('Summary CSV'),
                            onPressed: (_llm.isEmpty &&
                                    _tts.isEmpty &&
                                    _stt.isEmpty)
                                ? null
                                : _exportSummaryCsv,
                          ),
                          OutlinedButton.icon(
                            icon: const Icon(Icons.save_alt),
                            label: const Text('Export JSON'),
                            onPressed:
                                _convFilter == null ? null : _exportAllJson,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 10),

                    // Summary card (chips wrap)
                    Card(
                      child: Padding(
                        padding:
                            const EdgeInsets.fromLTRB(14, 12, 14, 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Summary (per conversation)',
                              style: TextStyle(
                                  fontWeight: FontWeight.w700),
                            ),
                            const SizedBox(height: 10),
                            _SummaryRow(
                                title: 'LLM latency (dur_ms)',
                                m: _summaryLlmdur),
                            _SummaryRow(
                                title: 'LLM TTFT (ms)',
                                m: _summaryLlmTtft),
                            _SummaryRow(
                                title: 'TTS duration (ms)',
                                m: _summaryTtsdur),
                            _SummaryRow(
                                title: 'STT time-to-first (ms)',
                                m: _summarySttdur),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),

                    // Tables (each has its own horizontal scrollbar)
                    const Text('Turns',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    _MiniTable(
                      rows: _turns,
                      cols: const [
                        'turn_seq',
                        'role',
                        'source',
                        'phase',
                        'cause',
                        'label',
                        'text_len',
                        'text'
                      ],
                    ),
                    const SizedBox(height: 14),

                    const Text('LLM Calls',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    _MiniTable(
                      rows: _llm,
                      cols: const [
                        'turn_seq',
                        'provider',
                        'model',
                        'http_status',
                        'dur_ms',
                        'ttft_ms',
                        'input_tokens',
                        'output_tokens',
                        'error'
                      ],
                    ),
                    const SizedBox(height: 14),

                    const Text('TTS Events',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    _MiniTable(
                      rows: _tts,
                      cols: const [
                        'turn_seq',
                        'utter_id',
                        'action',
                        'language',
                        'rate',
                        'pitch',
                        'text_len',
                        'dur_ms'
                      ],
                    ),
                    const SizedBox(height: 14),

                    const Text('STT Latency (from metrics)',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 6),
                    _MiniTable(
                      rows: _stt,
                      cols: const [
                        'client_ts',
                        'type',
                        'dur_ms',
                        'had_user_reply',
                        'text_len'
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SummaryRow extends StatelessWidget {
  final String title;
  final Map<String, num> m;
  const _SummaryRow({required this.title, required this.m});

  String _fmt(num n) =>
      (n == n.roundToDouble()) ? n.toStringAsFixed(0) : n.toStringAsFixed(1);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            overflow: TextOverflow.ellipsis,
            maxLines: 1,
            softWrap: false,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              _Chip('count', (m['count'] ?? 0).toString()),
              _Chip('p50', '${_fmt((m['p50'] ?? 0) as num)} ms'),
              _Chip('p95', '${_fmt((m['p95'] ?? 0) as num)} ms'),
              _Chip('p99', '${_fmt((m['p99'] ?? 0) as num)} ms'),
            ],
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final String value;
  const _Chip(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: cs.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: cs.outlineVariant),
      ),
      child: Text(
        '$label $value',
        overflow: TextOverflow.ellipsis,
        maxLines: 1,
        softWrap: false,
        style: const TextStyle(fontWeight: FontWeight.w600),
      ),
    );
  }
}

/// Data table with its own horizontal scrollbar & ellipsized cells.
class _MiniTable extends StatefulWidget {
  final List<Map<String, Object?>> rows;
  final List<String> cols;
  const _MiniTable({required this.rows, required this.cols});

  @override
  State<_MiniTable> createState() => _MiniTableState();
}

class _MiniTableState extends State<_MiniTable> {
  late final ScrollController _hCtrl;

  @override
  void initState() {
    super.initState();
    _hCtrl = ScrollController();
  }

  @override
  void dispose() {
    _hCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.rows.isEmpty) return const Text('—');

    return Scrollbar(
      controller: _hCtrl,
      thumbVisibility: true,
      // Only react to horizontal scroll notifications for this scrollbar.
      notificationPredicate: (notif) => notif.metrics.axis == Axis.horizontal,
      child: SingleChildScrollView(
        controller: _hCtrl,
        scrollDirection: Axis.horizontal,
        child: DataTable(
          headingRowHeight: 32,
          dataRowMinHeight: 28,
          dataRowMaxHeight: 28,
          columnSpacing: 20,
          columns: [
            for (final c in widget.cols)
              DataColumn(
                label: SizedBox(
                  width: _headerWidthFor(c),
                  child: Text(
                    c,
                    overflow: TextOverflow.ellipsis,
                    maxLines: 1,
                    softWrap: false,
                  ),
                ),
              ),
          ],
          rows: [
            for (final r in widget.rows)
              DataRow(
                cells: [
                  for (final c in widget.cols)
                    DataCell(
                      SizedBox(
                        width: _cellWidthFor(c),
                        child: Text(
                          (r[c]?.toString() ?? ''),
                          overflow: TextOverflow.ellipsis,
                          maxLines: 1,
                          softWrap: false,
                        ),
                      ),
                    ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  double _cellWidthFor(String col) {
    switch (col) {
      case 'text':
      case 'meta_json':
      case 'model':
        return 420;
      default:
        return 200;
    }
  }

  double _headerWidthFor(String col) {
    switch (col) {
      case 'text':
      case 'meta_json':
      case 'model':
        return 260;
      default:
        return 140;
    }
  }
}
