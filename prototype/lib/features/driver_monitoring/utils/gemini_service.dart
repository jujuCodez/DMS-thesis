import 'dart:convert';
import 'package:http/http.dart' as http;

import 'chat_logger.dart'; // logs chat turns + llm calls

/// Ultra-light Gemini caller with logging.
/// - Minimal chat history
/// - Returns `null` on network/API errors
class GeminiService {
  // Supply a personal development key at build time; never commit it.
  static const String _apiKey = String.fromEnvironment('GEMINI_API_KEY');

  final String model;

  static const String _base = "https://generativelanguage.googleapis.com"; // v1beta

  // Mutable chat history (kept small)
  final List<Map<String, dynamic>> _chatHistory = <Map<String, dynamic>>[];

  // System prompt
  static const String _systemPrompt =
      "You are the Driver Monitoring Assistant. "
      "Detect drowsiness and distraction. "
      "Keep replies short, light, and not repetitive.";

  GeminiService({this.model = "gemini-2.5-flash-lite-preview-06-17"});

  /// Ask Gemini for a short response.
  /// Returns reply text, or `null` on error.
  /// `meta` can include: phase ('initial'|'followup'|'chat'), cause, label, etc.
  Future<String?> generate(String userMessage, {Map<String, Object?>? meta}) async {
    if (_apiKey.isEmpty) return null;
    final url = Uri.parse(
      "$_base/v1beta/models/$model:generateContent?key=$_apiKey",
    );

    final body = {
      "systemInstruction": {
        "parts": [
          {"text": _systemPrompt}
        ]
      },
      "generationConfig": {
        "maxOutputTokens": 64,
        "temperature": 0.7,
        "topP": 0.95,
      },
      "contents": [
        ..._chatHistory,
        {
          "role": "user",
          "parts": [
            {"text": userMessage}
          ]
        }
      ],
    };

    // ----- logging: prompt turn -----
    await ChatLogger.I.ensureConversation();
    final turnSeq = ChatLogger.I.nextTurnSeq();
    await ChatLogger.I.logTurn(
      role: "user",
      source: "app",
      turnSeq: turnSeq,
      phase: (meta?["phase"]?.toString() ?? "chat"),
      cause: meta?["cause"]?.toString(),
      label: meta?["label"]?.toString(),
      text: userMessage,
      meta: {
        "provider": "google",
        "model": model,
        "kind": "prompt",
      },
    );

    // ----- request -----
    final t0 = DateTime.now();
    int httpStatus = 0;
    String? error;
    int? reqBytes;
    int? respBytes;
    int? inputTok;
    int? outputTok;
    int? totalTok;

    try {
      final reqStr = jsonEncode(body);
      reqBytes = utf8.encode(reqStr).length;

      final res = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: reqStr,
      );

      httpStatus = res.statusCode;
      respBytes = res.bodyBytes.length;

      if (res.statusCode != 200) {
        error = "HTTP_${res.statusCode}";
        return null;
      }

      final data = jsonDecode(res.body) as Map<String, dynamic>;

      // usageMetadata may be present: {promptTokenCount, candidatesTokenCount, totalTokenCount}
      final usage = data["usageMetadata"] as Map<String, dynamic>?;
      inputTok = usage?["promptTokenCount"] as int?;
      outputTok = usage?["candidatesTokenCount"] as int?;
      totalTok = usage?["totalTokenCount"] as int?;

      final reply = (data["candidates"]?[0]?["content"]?["parts"]?[0]?["text"] ?? "")
          .toString()
          .trim();

      if (reply.isEmpty) return null;

      // history (keep tiny)
      _chatHistory.add({"role": "user", "parts": [{"text": userMessage}]});
      _chatHistory.add({"role": "model", "parts": [{"text": reply}]});
      if (_chatHistory.length > 6) {
        _chatHistory.removeRange(0, 2);
      }

      // ----- logging: assistant turn -----
      await ChatLogger.I.logTurn(
        role: "assistant",
        source: "llm",
        turnSeq: turnSeq,
        phase: (meta?["phase"]?.toString() ?? "chat"),
        cause: meta?["cause"]?.toString(),
        label: meta?["label"]?.toString(),
        text: reply,
        meta: {
          "provider": "google",
          "model": model,
          "kind": "reply",
        },
      );

      return reply;
    } catch (e) {
      error = e.runtimeType.toString();
      return null;
    } finally {
      final durMs = DateTime.now().difference(t0).inMicroseconds / 1000.0;

      // ----- logging: llm call summary -----
      await ChatLogger.I.logLlmCall(
        turnSeq: turnSeq,
        phase: (meta?["phase"]?.toString() ?? "chat"),
        provider: "google",
        model: model,
        endpoint: "/v1beta/models/$model:generateContent",
        temperature: 0.7,
        topP: 0.95,
        maxTokens: 64,
        httpStatus: httpStatus,
        error: error,
        durMs: durMs,
        ttftMs: null, // non-streaming
        inputTokens: inputTok,
        outputTokens: outputTok,
        totalTokens: totalTok,
        requestBytes: reqBytes,
        responseBytes: respBytes,
        cacheHit: null,
        retries: 0,
        meta: meta,
      );
    }
  }

  void resetHistory() => _chatHistory.clear();
}
