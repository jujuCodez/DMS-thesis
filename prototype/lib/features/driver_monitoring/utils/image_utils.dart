// lib/features/driver_monitoring/utils/image_utils.dart
import 'dart:typed_data';
import 'dart:ui' show Rect;
import 'package:image/image.dart' as img;

/// Classification labels for the ONNX model outputs.
/// Keep the order exactly in sync with the model.
const List<String> kClassNames = <String>[
  'drowsy',
  'focused',
  'using phone',
  'yawning',
];

/// Convert an `image` package RGB image to NV21 (Y + interleaved VU).
/// UV is set to 128 (neutral chroma), effectively grayscale.
/// Used for ML Kit on Android.
Uint8List rgbToNv21Gray(img.Image rgb) {
  final int w = rgb.width;
  final int h = rgb.height;

  final int ySize = w * h;
  final int uvSize = ySize ~/ 2; // NV21 has half as many chroma bytes
  final out = Uint8List(ySize + uvSize);

  // --- Y plane (luma) ---
  int yi = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final p = rgb.getPixel(x, y);

      final int r = (p.r as num).toInt();
      final int g = (p.g as num).toInt();
      final int b = (p.b as num).toInt();

      // BT.601 luma (integer approximation)
      int Y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      if (Y < 0) Y = 0;
      if (Y > 255) Y = 255;

      out[yi++] = Y;
    }
  }

  // --- Interleaved VU plane (neutral chroma = 128) ---
  final int startUV = ySize;
  for (int i = 0; i < uvSize; i += 2) {
    out[startUV + i] = 128;     // V
    out[startUV + i + 1] = 128; // U
  }

  return out;
}

/// Clamp a [Rect] into image bounds [w]x[h].
Rect clampRect(Rect r, int w, int h) {
  final left = r.left.clamp(0.0, w.toDouble());
  final top = r.top.clamp(0.0, h.toDouble());
  final right = r.right.clamp(0.0, w.toDouble());
  final bottom = r.bottom.clamp(0.0, h.toDouble());
  return Rect.fromLTRB(left, top, right, bottom);
}

/// Super-cheap brightness estimate (0..255) from a JPEG byte stream.
/// Samples bytes evenly across the payload — enough to detect exposure swings.
int fastApproxLuma(Uint8List jpg) {
  if (jpg.length < 2048) {
    int sum = 0, cnt = 0;
    for (int i = 2; i < jpg.length; i += 64) { // skip SOI bytes at 0..1
      sum += jpg[i];
      cnt++;
    }
    return cnt == 0 ? 128 : (sum ~/ cnt);
  }

  const samples = 512; // tune (256–1024 is fine)
  final step = jpg.length ~/ samples;
  int sum = 0;
  for (int i = 2, k = 0; k < samples && i < jpg.length; i += step, k++) {
    sum += jpg[i];
  }
  return sum ~/ samples; // ~0..255
}

/// Convert an RGB image to CHW Float32 with mean/std normalization (RGB).
Float32List imageToCHWFloat32(
  img.Image rgb, {
  required List<double> mean,
  required List<double> std,
}) {
  final w = rgb.width, h = rgb.height;
  final floats = Float32List(3 * h * w);
  final cR = 0, cG = h * w, cB = 2 * h * w;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final px = rgb.getPixel(x, y);
      final r = (px.r as num).toDouble() / 255.0;
      final g = (px.g as num).toDouble() / 255.0;
      final b = (px.b as num).toDouble() / 255.0;
      final i = y * w + x;

      floats[cR + i] = (r - mean[0]) / std[0];
      floats[cG + i] = (g - mean[1]) / std[1];
      floats[cB + i] = (b - mean[2]) / std[2];
    }
  }
  return floats;
}

/// Convert to **grayscale replicated to 3 channels** (CHW float32).
/// - Computes 8-bit luma fast (integer BT.601 approx).
/// - Replicates the same normalized value to R, G, B channels.
/// - Keeps your model's 3-channel input contract but with gray content.
Float32List imageToCHWGray3Float32(
  img.Image im, {
  List<double> mean = const [0.485, 0.456, 0.406],
  List<double> std  = const [0.229, 0.224, 0.225],
}) {
  final w = im.width, h = im.height, n = w * h;
  final out = Float32List(3 * n);

  // Channel offsets (CHW)
  final offR = 0;
  final offG = n;
  final offB = 2 * n;

  var i = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final p = im.getPixel(x, y);

      final r = (p.r as num).toInt();
      final g = (p.g as num).toInt();
      final b = (p.b as num).toInt();

      // Integer BT.601 luma: (77*R + 150*G + 29*B) >> 8
      final l = (77 * r + 150 * g + 29 * b) >> 8; // 0..255
      final f = l / 255.0;

      out[offR + i] = (f - mean[0]) / std[0];
      out[offG + i] = (f - mean[1]) / std[1];
      out[offB + i] = (f - mean[2]) / std[2];
      i++;
    }
  }
  return out;
}

/// Argmax helper.
int argMax(List<double> xs) {
  var bestI = 0;
  var bestV = -double.infinity;
  for (int i = 0; i < xs.length; i++) {
    final v = xs[i];
    if (v > bestV) {
      bestV = v;
      bestI = i;
    }
  }
  return bestI;
}

/* -------------------------------------------------------------------------- */
/* Optional deprecated wrappers to avoid churn if older code calls them.      */
/* -------------------------------------------------------------------------- */

@Deprecated('Use rgbToNv21Gray')
Uint8List _rgbToNv21Gray(img.Image rgb) => rgbToNv21Gray(rgb);

@Deprecated('Use clampRect')
Rect _clampRect(Rect r, int w, int h) => clampRect(r, w, h);

@Deprecated('Use fastApproxLuma')
int _fastApproxLuma(Uint8List jpg) => fastApproxLuma(jpg);

@Deprecated('Use imageToCHWFloat32')
Float32List _imageToCHWFloat32(
  img.Image rgb, {
  required List<double> mean,
  required List<double> std,
}) =>
    imageToCHWFloat32(rgb, mean: mean, std: std);

@Deprecated('Use argMax')
int _argMax(List<double> xs) => argMax(xs);
