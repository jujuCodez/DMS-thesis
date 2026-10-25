import 'dart:async';
import 'dart:math' as math;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:permission_handler/permission_handler.dart';

import 'gemini_service.dart';
import 'metrics.dart' show Metrics;
import 'chat_logger.dart'; // LLM/Chat/TTS logging

class TTSService {
  final FlutterTts _tts = FlutterTts();
  final stt.SpeechToText _stt = stt.SpeechToText();
  final GeminiService _gemini = GeminiService();
  final math.Random _rand = math.Random();

  bool _speaking = false;
  bool get isInteractionActive => _speaking;

  bool enabled = true;
  bool useGemini = true;

  final Duration cooldown;
  // ↑ Give users time to say a full sentence
  Duration listenWindow = const Duration(seconds: 8);
  // Short pause that ends dictation after the user stops talking briefly
  Duration sttPauseFor = const Duration(seconds: 2);
  Duration geminiTimeoutInitial = const Duration(milliseconds: 1400);
  Duration geminiTimeoutFollowup = const Duration(milliseconds: 1400);

  DateTime? _lastSpokenAt;

  double _rate;
  double _pitch;
  String _language;

  bool _sttReady = false;
  int _listenSession = 0;

  final List<String> _recentLines = <String>[];
  static const int _recentKeep = 6;

  TTSService({
    this.cooldown = const Duration(seconds: 6),
    String language = "en-US",
    double rate = 0.60,
    double pitch = 1.0,
  })  : _rate = rate,
        _pitch = pitch,
        _language = language {
    _tts.setLanguage(_language);
    _tts.setSpeechRate(_rate);
    _tts.setPitch(_pitch);
    _tts.awaitSpeakCompletion(true);
  }

  // ----------------- Public controls -----------------

  Future<void> setRate(double rate) async {
    _rate = (rate.clamp(0.1, 1.0) as double);
    await _tts.setSpeechRate(_rate);
  }

  Future<void> setPitch(double pitch) async {
    _pitch = (pitch.clamp(0.5, 2.0) as double);
    await _tts.setPitch(_pitch);
  }

  Future<void> setLanguage(String languageCode) async {
    _language = languageCode;
    await _tts.setLanguage(_language);
  }

  /// Low-level speak with cooldown.
  Future<void> speak(
    String message, {
    bool bypassCooldown = false,
    double? rate,
    double? pitch,
  }) async {
    if (!enabled) return;

    final now = DateTime.now();
    if (!bypassCooldown &&
        _lastSpokenAt != null &&
        now.difference(_lastSpokenAt!) < cooldown) {
      return;
    }

    if (rate != null) {
      final r = (rate.clamp(0.1, 1.0) as double);
      await _tts.setSpeechRate(r);
    } else {
      await _tts.setSpeechRate(_rate);
    }
    if (pitch != null) {
      final p = (pitch.clamp(0.5, 2.0) as double);
      await _tts.setPitch(p);
    } else {
      await _tts.setPitch(_pitch);
    }

    await _tts.stop();
    final line = message.trim();

    // TTS start/end logs (utterance-level)
    final utterId = 'utt_${DateTime.now().millisecondsSinceEpoch}';
    final tStart = DateTime.now();
    await ChatLogger.I.logTtsEvent(
      turnSeq: ChatLogger.I.nextTurnSeq(),
      utterId: utterId,
      action: 'start',
      language: _language,
      rate: _rate,
      pitch: _pitch,
      textLen: line.length,
      meta: {'phase': 'chat'},
    );

    await _tts.speak(line);
    final durMs = DateTime.now().difference(tStart).inMicroseconds / 1000.0;

    await ChatLogger.I.logTtsEvent(
      turnSeq: ChatLogger.I.nextTurnSeq(),
      utterId: utterId,
      action: 'end',
      language: _language,
      rate: _rate,
      pitch: _pitch,
      textLen: line.length,
      durMs: durMs,
      meta: {'phase': 'chat'},
    );

    _lastSpokenAt = DateTime.now();
    _remember(line);
  }

  Future<void> stop() async {
    try {
      await _stt.stop();
    } catch (_) {}
    try {
      await _tts.stop();
    } catch (_) {}
    _speaking = false;
  }

  // ----------------- ALERT ENTRY (returns after initial line) -----------------

  Future<bool> speakAlert({
    required String causeKey, // 'noface' | 'pose' | 'cls_X'
    String? label,            // e.g., "drowsy", "yawning", "using phone"
    bool bypassCooldown = true,
  }) async {
    final initialDone = Completer<bool>();

    _runAlertFlow(
      causeKey: causeKey,
      label: label,
      bypassCooldown: bypassCooldown,
      initialSignal: initialDone,
    );

    return initialDone.future;
  }

  // ----------------- Actual flow (initial + listen + follow-up) -----------------
  Future<void> _runAlertFlow({
    required String causeKey,
    String? label,
    required bool bypassCooldown,
    required Completer<bool> initialSignal,
  }) async {
    if (!enabled) {
      if (!initialSignal.isCompleted) initialSignal.complete(false);
      return;
    }
    if (_speaking) {
      if (!initialSignal.isCompleted) initialSignal.complete(false);
      return;
    }

    final now = DateTime.now();
    if (!bypassCooldown &&
        _lastSpokenAt != null &&
        now.difference(_lastSpokenAt!) < cooldown) {
      if (!initialSignal.isCompleted) initialSignal.complete(false);
      return;
    }

    _speaking = true;
    final session = ++_listenSession;

    try {
      await ChatLogger.I.ensureConversation();

      // 1) Initial line (Gemini)
      final tGem0 = DateTime.now();
      final initial = await _dynamicInitialLine(
        causeKey: causeKey,
        label: label,
      );
      final llmMsInitial =
          DateTime.now().difference(tGem0).inMicroseconds / 1000.0;

      Metrics.I.noteTtsFlow(
        phase: 'initial',
        llmMs: llmMsInitial,
        llmToTtsMs: 0.0,
        hadUserReply: null,
        textLen: initial.length,
      );

      // TTS logs around initial speak
      final utterId0 = 'utt_init_${DateTime.now().millisecondsSinceEpoch}';
      final t0 = DateTime.now();
      await ChatLogger.I.logTtsEvent(
        turnSeq: ChatLogger.I.nextTurnSeq(),
        utterId: utterId0,
        action: 'start',
        language: _language,
        rate: _rate,
        pitch: _pitch,
        textLen: initial.length,
        meta: {'phase': 'initial', 'cause': causeKey, 'label': label},
      );

      await speak(initial, bypassCooldown: true);

      await ChatLogger.I.logTtsEvent(
        turnSeq: ChatLogger.I.nextTurnSeq(),
        utterId: utterId0,
        action: 'end',
        language: _language,
        rate: _rate,
        pitch: _pitch,
        textLen: initial.length,
        durMs: DateTime.now().difference(t0).inMicroseconds / 1000.0,
        meta: {'phase': 'initial', 'cause': causeKey, 'label': label},
      );

      if (!initialSignal.isCompleted) initialSignal.complete(true);

      await Future.delayed(const Duration(milliseconds: 220));

      // 2) Listen for up to [listenWindow]
      String? userReply;
      if (await _ensureSttReady()) {
        userReply = await _listenForReply(
          session: session,
          listenFor: listenWindow,
          pauseFor: sttPauseFor,
        );
      } else {
        await Future.delayed(listenWindow);
        userReply = null;
      }

      // If user replied, log as chat turn (source=stt)
      if (userReply != null && userReply.trim().isNotEmpty) {
        await ChatLogger.I.logTurn(
          role: 'user',
          source: 'stt',
          turnSeq: ChatLogger.I.nextTurnSeq(),
          phase: 'followup',
          cause: causeKey,
          label: label,
          text: userReply.trim(),
          meta: const {},
        );
      }

      // 3) Follow-up (Gemini)
      final tGem1 = DateTime.now();
      final followup = await _dynamicFollowupLine(
        causeKey: causeKey,
        label: label,
        userReply: userReply,
      );
      final llmMsFollow =
          DateTime.now().difference(tGem1).inMicroseconds / 1000.0;

      Metrics.I.noteTtsFlow(
        phase: 'followup',
        llmMs: llmMsFollow,
        llmToTtsMs: 0.0,
        hadUserReply: userReply != null && userReply.isNotEmpty,
        textLen: followup.length,
      );

      final utterId1 = 'utt_follow_${DateTime.now().millisecondsSinceEpoch}';
      final t1 = DateTime.now();
      await ChatLogger.I.logTtsEvent(
        turnSeq: ChatLogger.I.nextTurnSeq(),
        utterId: utterId1,
        action: 'start',
        language: _language,
        rate: _rate,
        pitch: _pitch,
        textLen: followup.length,
        meta: {
          'phase': 'followup',
          'cause': causeKey,
          'label': label,
          'had_user_reply': (userReply ?? '').isNotEmpty
        },
      );

      await speak(followup, bypassCooldown: true);

      await ChatLogger.I.logTtsEvent(
        turnSeq: ChatLogger.I.nextTurnSeq(),
        utterId: utterId1,
        action: 'end',
        language: _language,
        rate: _rate,
        pitch: _pitch,
        textLen: followup.length,
        durMs: DateTime.now().difference(t1).inMicroseconds / 1000.0,
        meta: {
          'phase': 'followup',
          'cause': causeKey,
          'label': label,
          'had_user_reply': (userReply ?? '').isNotEmpty
        },
      );
    } catch (_) {
      if (!initialSignal.isCompleted) initialSignal.complete(false);
    } finally {
      await Future.delayed(const Duration(milliseconds: 60));
      _speaking = false;
    }
  }

  // ----------------- Dynamic initial line (Gemini) -----------------

  Future<String> _dynamicInitialLine({
    required String causeKey,
    String? label,
  }) async {
    final type = _canonicalType(causeKey: causeKey, label: label);

    if (!useGemini) return _initialFallback(type);

    final lc = (label ?? '').toLowerCase().trim();
    final tone = _pickTone();
    final chip = _initialChip();
    final avoid = _recentAvoidList();

    final prompt = [
      'You are a friendly in-car co-driver. A driver monitoring alert fired.',
      'The VERY FIRST sentence MUST be exactly: "$type detected."',
      if (lc.isNotEmpty) 'Label hint: $lc.',
      'Immediately follow with ONE short question that invites a brief reply. It must end with "?".',
      'Keep the whole thing ~12–26 words total across both sentences. Use a $tone tone. $chip',
      if (avoid.isNotEmpty) 'Avoid these phrases verbatim: $avoid',
      'Sound conversational with contractions; vary rhythm; no corporate tone.',
      'No emojis, no disclaimers, no “as an AI”.',
    ].join(' ');

    try {
      final g = await _gemini
          .generate(
            prompt,
            meta: {'phase': 'initial', 'cause': causeKey, 'label': label ?? ''},
          )
          .timeout(geminiTimeoutInitial);
      final out = (g ?? '').trim();
      if (out.isNotEmpty && _startsWithType(out, type) && out.contains('?')) {
        return out;
      }
    } catch (_) {}

    return _initialFallback(type);
  }

  String _initialFallback(String type) {
    switch (type) {
      case 'Drowsiness':
        return 'Drowsiness detected. Do you need a quick break or some fresh air?';
      case 'Yawning':
        return 'Yawning detected. Feeling sleepy at all?';
      case 'Phone usage':
        return 'Phone usage detected. Can you set the phone aside for now?';
      case 'Looking away':
        return 'Looking away detected. Is everything okay ahead?';
      case 'Driver not detected':
        return 'Driver not detected. Are you back in view?';
      default:
        return 'Distraction detected. Doing alright up there?';
    }
  }

  // ----------------- Dynamic follow-up (Gemini) -----------------

  Future<String> _dynamicFollowupLine({
    required String causeKey,
    String? label,
    String? userReply,
  }) async {
    final type = _canonicalType(causeKey: causeKey, label: label);

    if (!useGemini) {
      return (userReply != null && userReply.isNotEmpty)
          ? _localCoachingForReply(causeKey, userReply)
          : _localClosingFor(causeKey);
    }

    final lc = (label ?? '').toLowerCase().trim();
    final tone = _pickTone();
    final chip = (userReply != null && userReply.isNotEmpty)
        ? _coachingChip()
        : _closingChip();
    final avoid = _recentAvoidList();

    final prompt = [
      'You are a supportive in-car co-driver.',
      'Detection type: $type.',
      if (lc.isNotEmpty) 'Label hint: $lc.',
      if (userReply != null && userReply.isNotEmpty)
        'The driver said: "${_summ(userReply, max: 80)}". '
            'Mirror 1–3 of their keywords naturally and respond with ONE short coaching message '
            '(12–24 words), empathetic, specific, and practical, in a $tone tone. $chip'
      else
        'The driver was silent. Provide ONE short closing reminder (10–18 words), supportive and calm, in a $tone tone. $chip',
      if (avoid.isNotEmpty) 'Avoid these phrases verbatim: $avoid',
      'Use contractions and vary sentence length a bit. No emojis, no disclaimers, no lists.',
    ].join(' ');

    try {
      final g = await _gemini
          .generate(
            prompt,
            meta: {
              'phase': 'followup',
              'cause': causeKey,
              'label': label ?? '',
              'had_user_reply': (userReply ?? '').isNotEmpty
            },
          )
          .timeout(geminiTimeoutFollowup);
      final text = (g ?? '').trim();
      if (text.isNotEmpty) return text;
    } catch (_) {}

    return (userReply != null && userReply.isNotEmpty)
        ? _localCoachingForReply(causeKey, userReply)
        : _localClosingFor(causeKey);
  }

  // ----------------- Local fallbacks -----------------

  String _localCoachingForReply(String causeKey, String reply) {
    final r = reply.toLowerCase();
    if (r.contains('sleep') || r.contains('tired') || r.contains('sleepy')) {
      return 'Thanks for sharing—if it’s safe, take a short break and sip some water.';
    }
    if (r.contains('ok') ||
        r.contains('okay') ||
        r.contains('fine') ||
        r.contains('alright')) {
      return 'Got it—let’s keep attention forward and spacing steady.';
    }
    if (causeKey == 'pose') {
      return 'Understood—keep your gaze centered and use mirrors smoothly.';
    }
    if (causeKey == 'noface') {
      return 'Thanks—return to view when it’s safe to do so.';
    }
    return 'Appreciate the update—let’s stay focused and drive smoothly.';
  }

  String _localClosingFor(String causeKey) {
    if (causeKey == 'noface') {
      return 'Please return to view when possible and keep a steady pace.';
    }
    if (causeKey == 'pose') {
      return 'Let’s keep your gaze forward and hands relaxed on the wheel.';
    }
    return 'Let’s stay focused and keep a safe following distance.';
  }

  // ----------------- STT -----------------

  Future<bool> _ensureSttReady() async {
    final mic = await Permission.microphone.status;
    if (!mic.isGranted) {
      final granted = await Permission.microphone.request();
      if (!granted.isGranted) return false;
    }
    if (_sttReady) return true;

    _sttReady = await _stt.initialize(
      onStatus: (_) {},
      onError: (_) {},
    );
    return _sttReady;
  }

  /// Dictation-style listener that returns the full sentence.
  /// - Uses ListenMode.dictation (vs confirmation) so the recognizer streams words continuously.
  /// - Accumulates partials in `lastText`, completes on `finalResult`. If the overall window
  ///   expires, returns the best-known `lastText` (or null if none).
  Future<String?> _listenForReply({
    required int session,
    required Duration listenFor,
    required Duration pauseFor,
  }) async {
    final mySession = session;
    final completer = Completer<String?>();
    bool done = false;

    String lastText = '';
    bool firstNoted = false; // metrics: log first word latency once
    final sttStart = DateTime.now();

    void finish([String? text]) async {
      if (done) return;
      done = true;
      try {
        await _stt.stop();
      } catch (_) {}
      // If we never logged the first token, log as no-reply or reply now.
      if (!firstNoted) {
        final ms = DateTime.now().difference(sttStart).inMicroseconds / 1000.0;
        Metrics.I.noteStt(
          hadReply: (text != null && text.isNotEmpty),
          msToFirst: (text != null && text.isNotEmpty) ? ms : listenFor.inMilliseconds.toDouble(),
          textLen: (text ?? '').length,
        );
      }
      if (!completer.isCompleted) {
        final t = (text ?? '').trim();
        completer.complete(t.isEmpty ? null : t);
      }
    }

    // Force-complete after the window; prefer the longest text we have
    Timer(listenFor, () {
      if (done) return;
      finish(lastText.isNotEmpty ? lastText : null);
    });

    await _stt.listen(
      listenFor: listenFor,
      pauseFor: pauseFor,
      partialResults: true,
      cancelOnError: false,
      listenMode: stt.ListenMode.dictation, // <- continuous dictation mode
      localeId: null, // use system default
      onResult: (res) async {
        if (done || _listenSession != mySession) return;

        final text = res.recognizedWords.trim();
        if (text.isNotEmpty) {
          lastText = text; // accumulate the best-so-far sentence

          if (!firstNoted) {
            final ms =
                DateTime.now().difference(sttStart).inMicroseconds / 1000.0;
            Metrics.I.noteStt(
              hadReply: true,
              msToFirst: ms,
              textLen: text.length,
            );
            firstNoted = true;

            // After we’ve heard speech, shorten the silence needed to finalize.
            try {
              _stt.changePauseFor(const Duration(milliseconds: 800));
            } catch (_) {}
          }
        }

        if (res.finalResult) {
          // Recognizer says we're done: return the full sentence we have.
          finish(lastText.isNotEmpty ? lastText : null);
        }
      },
    );

    return completer.future;
  }

  // ----------------- Utilities -----------------

  void _remember(String line) {
    if (line.isEmpty) return;
    _recentLines.add(line);
    if (_recentLines.length > _recentKeep) {
      _recentLines.removeRange(0, _recentLines.length - _recentKeep);
    }
  }

  String _recentAvoidList() {
    if (_recentLines.isEmpty) return '';
    final quoted =
        _recentLines.map((s) => '"${s.replaceAll('"', '')}"').join(', ');
    return '[$quoted]';
  }

  String _canonicalType({required String causeKey, String? label}) {
    final l = (label ?? '').toLowerCase();
    if (l.contains('drowsy')) return 'Drowsiness';
    if (l.contains('yawn')) return 'Yawning';
    if (l.contains('phone')) return 'Phone usage';

    if (causeKey == 'pose') return 'Looking away';
    if (causeKey == 'noface') return 'Driver not detected';

    final m = RegExp(r'cls_(\d+)').firstMatch(causeKey);
    if (m != null) {
      switch (int.tryParse(m.group(1) ?? '')) {
        case 0:
          return 'Drowsiness';
        case 2:
          return 'Phone usage';
        case 3:
          return 'Yawning';
      }
    }
    return 'Distraction';
  }

  bool _startsWithType(String text, String type) {
    final t = text.trimLeft();
    final expect = '${type} detected';
    return t.toLowerCase().startsWith(expect.toLowerCase());
  }

  String _summ(String s, {int max = 80}) =>
      (s.length <= max) ? s : s.substring(0, max - 1) + '…';

  String _pickTone() {
    const tones = [
      'friendly',
      'calm',
      'reassuring',
      'encouraging',
      'conversational',
      'supportive',
      'warm but concise',
      'steady and confident',
      'matter-of-fact but kind',
    ];
    return tones[_rand.nextInt(tones.length)];
  }

  String _initialChip() {
    const chips = [
      'Use a contraction and a gentle, concrete tip (e.g., stretch, breathe, sip water).',
      'Keep it conversational; vary sentence length slightly; avoid starting with “Please”.',
      'Add one practical nudge (window, posture, hydration) and keep it human.',
      'Sound like a real co-driver: light touch, not bossy, not robotic.',
      'Avoid cliché phrasing; pick a fresh, natural wording.',
    ];
    return chips[_rand.nextInt(chips.length)];
  }

  String _coachingChip() {
    const chips = [
      'Mirror 1–3 of their words; offer one specific next step.',
      'Acknowledge their feeling, then give a simple, doable action.',
      'Keep it gentle and practical; no lecturing; one clear tip.',
      'Use a contraction and a warm tone; keep it concrete.',
      'Avoid repeating the initial line; move the conversation forward.',
    ];
    return chips[_rand.nextInt(chips.length)];
  }

  String _closingChip() {
    const chips = [
      'End with a calm reminder and one light suggestion.',
      'Short and steady; one supportive cue is enough.',
      'Reassure, then nudge attention forward; keep it concise.',
      'Gentle, human tone; vary the cadence slightly.',
      'Avoid templated closings; pick a natural phrasing.',
    ];
    return chips[_rand.nextInt(chips.length)];
  }
}
