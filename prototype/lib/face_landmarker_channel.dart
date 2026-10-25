// lib/native/face_landmarker_channel.dart
import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';

class FaceLandmarkerChannel {
  static const MethodChannel _ch = MethodChannel('face_landmarker');

  Future<bool> initialize({
    String modelAssetPath = 'face_landmarker.task',
    int numFaces = 1,
    double minFaceDetectionConfidence = 0.5,
    double minFacePresenceConfidence = 0.5,
    double minTrackingConfidence = 0.5,
    bool outputBlendshapes = true,
    bool outputTransformationMatrices = true,
  }) async {
    final ok = await _ch.invokeMethod<bool>('initialize', {
      'modelAssetPath': modelAssetPath,
      'numFaces': numFaces,
      'minFaceDetectionConfidence': minFaceDetectionConfidence,
      'minFacePresenceConfidence': minFacePresenceConfidence,
      'minTrackingConfidence': minTrackingConfidence,
      'outputBlendshapes': outputBlendshapes,
      'outputTransformationMatrices': outputTransformationMatrices,
    });
    return ok ?? false;
  }

  Future<FaceResult?> detect(Uint8List jpegBytes, {int? timestampMs}) async {
    final m = await _ch.invokeMethod<Map<dynamic, dynamic>>('detect', {
      'jpegBytes': jpegBytes,
      'timestampMs': timestampMs ?? DateTime.now().millisecondsSinceEpoch,
    });
    if (m == null) return null;
    return FaceResult.fromMap(Map<String, dynamic>.from(m));
  }

  Future<void> close() => _ch.invokeMethod('close');
}

/// Parsed result for convenience (only first face is commonly used).
class FaceResult {
  final int timestampMs;
  final List<List<Landmark>> faces; // normalized [0..1] x,y, z in image space
  final List<List<Blendshape>> blendshapes;
  final List<List<double>> matrices; // 4x4 row-major per face (length 16)

  bool get hasFace => faces.isNotEmpty;

  FaceResult({
    required this.timestampMs,
    required this.faces,
    required this.blendshapes,
    required this.matrices,
  });

  factory FaceResult.fromMap(Map<String, dynamic> m) {
    final faces = <List<Landmark>>[];
    for (final f in (m['faces'] as List? ?? const [])) {
      faces.add((f as List)
          .map((lm) => Landmark(
                x: (lm['x'] as num).toDouble(),
                y: (lm['y'] as num).toDouble(),
                z: (lm['z'] as num?)?.toDouble() ?? 0.0,
              ))
          .toList());
    }

    final blends = <List<Blendshape>>[];
    for (final b in (m['blendshapes'] as List? ?? const [])) {
      blends.add((b as List)
          .map((c) => Blendshape(
                index: (c['index'] as num).toInt(),
                score: (c['score'] as num).toDouble(),
                categoryName: (c['categoryName'] as String?) ?? '',
                displayName: (c['displayName'] as String?) ?? '',
              ))
          .toList());
    }

    final mats = <List<double>>[];
    for (final mat in (m['facialTransformationMatrices'] as List? ?? const [])) {
      mats.add((mat as List).map((e) => (e as num).toDouble()).toList());
    }

    return FaceResult(
      timestampMs: (m['timestampMs'] as num).toInt(),
      faces: faces,
      blendshapes: blends,
      matrices: mats,
    );
  }

  /// Convenience: yaw/pitch/roll (degrees) from first 4x4 matrix (ZYX order).
  /// Assumes row-major 4x4 with rotation in upper-left 3x3.
  /// yaw=Y, pitch=X, roll=Z (right-handed); see standard ZYX extraction.
  Ypr? firstFaceYpr() {
    if (matrices.isEmpty) return null;
    final a = matrices.first;
    if (a.length < 12) return null;

    // Rotation submatrix (row-major)
    final r00 = a[0],  r01 = a[1],  r02 = a[2];
    final r10 = a[4],  r11 = a[5],  r12 = a[6];
    final r20 = a[8],  r21 = a[9],  r22 = a[10];

    final pitch = math.asin((-r20).clamp(-1.0, 1.0));
    final roll  = math.atan2(r21, r22);
    final yaw   = math.atan2(r10, r00);

    const toDeg = 180.0 / math.pi;
    return Ypr(
      yawDeg:  yaw * toDeg,
      pitchDeg: pitch * toDeg,
      rollDeg:  roll * toDeg,
    );
  }
}

class Landmark {
  final double x, y, z;
  const Landmark({required this.x, required this.y, required this.z});
}

class Blendshape {
  final int index;
  final double score;
  final String categoryName;
  final String displayName;
  const Blendshape({
    required this.index,
    required this.score,
    required this.categoryName,
    required this.displayName,
  });
}

class Ypr {
  final double yawDeg, pitchDeg, rollDeg;
  const Ypr({required this.yawDeg, required this.pitchDeg, required this.rollDeg});
}
