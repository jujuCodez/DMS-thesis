// lib/auth_controller.dart
import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'data/local_app_db.dart';

import 'features/driver_monitoring/utils/metrics.dart';
import 'features/driver_monitoring/utils/driver_logs.dart';

class UserAcc {
  final String id;
  final String username;
  const UserAcc({required this.id, required this.username});
}

class AuthController extends ChangeNotifier {
  UserAcc? current;

  // Live watcher so DriverLogs name updates when profile is edited.
  StreamSubscription<QuerySnapshot<Map<String, dynamic>>>? _profileSub;

  static String _isoWithMicros(DateTime dt) {
    // Format: 2025-09-10T06:05:49.031843  (no 'Z')
    String two(int n) => n.toString().padLeft(2, '0');
    final micro = dt.microsecond.toString().padLeft(6, '0');
    return "${dt.year.toString().padLeft(4, '0')}-"
        "${two(dt.month)}-${two(dt.day)}T"
        "${two(dt.hour)}:${two(dt.minute)}:${two(dt.second)}.$micro";
  }

  CollectionReference<Map<String, dynamic>> get _col =>
      FirebaseFirestore.instance.collection('useracc');

  // ---------- Helpers ----------

  void _stopProfileWatcher() {
    _profileSub?.cancel();
    _profileSub = null;
  }

  String _resolveFullNameFromDoc(Map<String, dynamic> d, String username) {
    String s(Object? x) => (x ?? '').toString().trim();
    final full = s(d['full_name']);
    if (full.isNotEmpty) return full;
    final first = s(d['first_name']);
    final last = s(d['last_name']);
    if (first.isNotEmpty || last.isNotEmpty) {
      return [first, last].where((v) => v.isNotEmpty).join(' ');
    }
    return username; // final fallback
  }

  Future<void> _prefetchProfileAndApply(String username) async {
    try {
      final q = await FirebaseFirestore.instance
          .collection('drivers')
          .where('username', isEqualTo: username)
          .limit(1)
          .get();

      final name = q.docs.isNotEmpty
          ? _resolveFullNameFromDoc(q.docs.first.data(), username)
          : username;

      Metrics.I.setUsername(username);
      DriverLogs.I.setIdentity(fullName: name, username: username);
    } catch (_) {
      // On error, still tag by username so logging continues.
      Metrics.I.setUsername(username);
      DriverLogs.I.setIdentity(fullName: username, username: username);
    }
  }

  void _startProfileWatcher(String username) {
    _stopProfileWatcher();
    _profileSub = FirebaseFirestore.instance
        .collection('drivers')
        .where('username', isEqualTo: username)
        .limit(1)
        .snapshots()
        .listen((qs) {
      final name = (qs.docs.isNotEmpty)
          ? _resolveFullNameFromDoc(qs.docs.first.data(), username)
          : username;
      Metrics.I.setUsername(username);
      DriverLogs.I.setIdentity(fullName: name, username: username);
    });
  }

  Future<void> _applyIdentityAsync() async {
    final u = current?.username;

    if (u == null) {
      _stopProfileWatcher();
      Metrics.I.setUsername(null);
      DriverLogs.I.clearIdentity();
      return;
    }

    await _prefetchProfileAndApply(u);
    _startProfileWatcher(u);
  }

  String _friendlyFirebaseError(FirebaseException e, String ctx) {
    switch (e.code) {
      case 'permission-denied':
        return '$ctx failed: permission denied (check Firestore rules for "useracc" and "drivers").';
      case 'unavailable':
        return '$ctx failed: Firestore unavailable/offline.';
      case 'deadline-exceeded':
        return '$ctx failed: network timeout.';
      default:
        return '$ctx failed: ${e.message ?? e.code}.';
    }
  }

  // ---------- Auth API ----------

  Future<String?> signUp({
    required String username,
    required String password,
  }) async {
    try {
      // Ensure unique username
      final exist =
          await _col.where('username', isEqualTo: username).limit(1).get();
      if (exist.docs.isNotEmpty) return 'Username already taken';

      final createdAt = _isoWithMicros(DateTime.now());
      final doc = await _col.add({
        'username': username,
        'password': password, // NOTE: plaintext per your request
        'created_at': createdAt, // as a STRING in your exact format
      });

      current = UserAcc(id: doc.id, username: username);
      await _applyIdentityAsync();
      notifyListeners();
      return null; // success
    } on FirebaseException catch (e) {
      return _friendlyFirebaseError(e, 'Sign up');
    } catch (e) {
      return 'Sign up failed: $e';
    }
  }

  Future<String?> signIn({
    required String username,
    required String password,
  }) async {
    try {
      // Query by username only (avoid composite index), check password in app
      final qs =
          await _col.where('username', isEqualTo: username).limit(1).get();
      if (qs.docs.isEmpty) return 'Account not found';

      final data = qs.docs.first.data();
      if (data['password'] != password) return 'Wrong password';

      current =
          UserAcc(id: qs.docs.first.id, username: data['username'] as String);
      await _applyIdentityAsync();
      notifyListeners();
      return null; // success
    } on FirebaseException catch (e) {
      return _friendlyFirebaseError(e, 'Sign in');
    } catch (e) {
      return 'Sign in failed: $e';
    }
  }

  Future<void> signOut() async {
    current = null;
    await _applyIdentityAsync(); // clears Metrics + DriverLogs identity & stops watcher
    notifyListeners();
  }

  @override
  void dispose() {
    _stopProfileWatcher();
    super.dispose();
  }
}
