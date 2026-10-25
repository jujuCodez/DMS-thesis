import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

/// Profile editor that:
/// - loads the first document in `drivers` where `username == widget.username`
/// - if none exists, shows empty fields
/// - saves using an **auto ID** (via `add`) for first save
/// - updates existing doc afterwards
/// - stores created_at / updated_at as ISO-8601 strings (local time)
class ProfilePage extends StatefulWidget {
  final String username; // pass from SettingsPage/AppShell (AuthController.current.username)

  const ProfilePage({super.key, required this.username});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final _first = TextEditingController();
  final _last = TextEditingController();
  final _brand = TextEditingController();
  final _model = TextEditingController();

  bool _loading = true;
  DocumentReference<Map<String, dynamic>>? _docRef;

  CollectionReference<Map<String, dynamic>> get _coll =>
      FirebaseFirestore.instance.collection('drivers');

  @override
  void initState() {
    super.initState();
    _loadExisting();
  }

  @override
  void dispose() {
    _first.dispose();
    _last.dispose();
    _brand.dispose();
    _model.dispose();
    super.dispose();
  }

  String _nowIso() => DateTime.now().toIso8601String(); // e.g. 2025-09-10T06:05:49.143102

  Future<void> _loadExisting() async {
    if (widget.username.isEmpty) {
      setState(() => _loading = false);
      return;
    }

    try {
      final q = await _coll
          .where('username', isEqualTo: widget.username)
          .limit(1)
          .get();
      if (q.docs.isNotEmpty) {
        final doc = q.docs.first;
        _docRef = doc.reference;
        final d = doc.data();
        _first.text = (d['first_name'] ?? '').toString();
        _last.text = (d['last_name'] ?? '').toString();
        _brand.text = (d['car_brand'] ?? '').toString();
        _model.text = (d['car_model'] ?? '').toString();
      }
    } catch (_) {
      // ignore – show empty fields
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _save() async {
    if (widget.username.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No username available – please sign in')),
      );
      return;
    }

    final first = _first.text.trim();
    final last = _last.text.trim();
    final data = <String, dynamic>{
      'username': widget.username,
      'first_name': first,
      'last_name': last,
      'full_name': '${first.isEmpty ? '' : first}${last.isEmpty ? '' : ' $last'}',
      'car_brand': _brand.text.trim(),
      'car_model': _model.text.trim(),
      'updated_at': _nowIso(), // ISO string
    };

    try {
      if (_docRef == null) {
        // First save => auto ID + created_at
        data['created_at'] = _nowIso(); // ISO string
        final created = await _coll.add(data); // <-- auto-generated doc id
        _docRef = created;
      } else {
        await _docRef!.set(data, SetOptions(merge: true));
      }

      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Profile saved')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Save failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // If there’s no username from AuthController, show a friendly nudge
    if (widget.username.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('Profile')),
        body: const Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Text(
              'Please sign in to edit your profile.',
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  TextField(
                    controller: _first,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      labelText: 'First name',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _last,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      labelText: 'Last name',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _brand,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      labelText: 'Car brand',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _model,
                    textInputAction: TextInputAction.done,
                    decoration: const InputDecoration(
                      labelText: 'Car model',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton(
                    onPressed: _save,
                    child: const Text('Save'),
                  ),
                ],
              ),
      ),
    );
  }
}
