// lib/features/driver_monitoring/ui/widgets/thresh_chip.dart
import 'package:flutter/material.dart';

/// Tiny UI chip for adjusting a numeric threshold with +/- controls.
///
/// Example:
/// ThreshChip(
///   label: 'Pitch',
///   value: 20,
///   onDec: () => setState(() => _pitchThresh = (_pitchThresh - 2).clamp(5, 45)),
///   onInc: () => setState(() => _pitchThresh = (_pitchThresh + 2).clamp(5, 45)),
/// )
class ThreshChip extends StatelessWidget {
  final String label;
  final double value;
  final VoidCallback onDec;
  final VoidCallback onInc;

  const ThreshChip({
    super.key,
    required this.label,
    required this.value,
    required this.onDec,
    required this.onInc,
  });

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme.bodySmall;

    return Container(
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.grey.shade200,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$label ${value.toStringAsFixed(0)}', style: text),
          const SizedBox(width: 6),
          _IconBtn(
            icon: Icons.remove,
            tooltip: 'Decrease $label',
            onTap: onDec,
          ),
          const SizedBox(width: 4),
          _IconBtn(
            icon: Icons.add,
            tooltip: 'Increase $label',
            onTap: onInc,
          ),
        ],
      ),
    );
  }
}

class _IconBtn extends StatelessWidget {
  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;

  const _IconBtn({
    required this.icon,
    required this.tooltip,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkResponse(
      onTap: onTap,
      radius: 18,
      child: Tooltip(
        message: tooltip,
        child: Icon(icon, size: 18),
      ),
    );
  }
}
