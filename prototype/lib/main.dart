// lib/main.dart
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';

import 'firebase_options.dart';

// Settings (theme, monitoring/metrics toggles, camera URL)
import 'features/driver_monitoring/utils/app_settings.dart';

// GPS → DriverLogs bridge (auto-starts only when monitoring is ON)
import 'features/driver_monitoring/utils/gps_bridge.dart';

// Auth & screens
import 'auth_controller.dart';
import 'login_signup_page.dart';
import 'features/driver_monitoring/ui/camera_page.dart';
import 'features/driver_monitoring/ui/settings_page.dart';

// ---- Local sinks (SQLite) ----
import 'data/local_app_db.dart'; // Local DB + sinks
import 'features/driver_monitoring/utils/driver_logs.dart' show DriverLogs;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // Load persisted toggles (monitoring, metrics), theme, camera URL
  await AppSettings.I.init();

  // Wire local SQLite sinks once on boot (Android/iOS only).
  // DriverLogs will write per-minute aggregates to SQLite instead of Firestore.
  await _wireLocalSinksOnBoot();

  // Start GPS bridge (it listens to AppSettings and runs only when monitoring = ON)
  GpsBridge.I.init();

  runApp(const MyApp());
}

Future<void> _wireLocalSinksOnBoot() async {
  if (kIsWeb) return; // skip on web
  try {
    // Open (or create) the app-local SQLite database.
    await LocalAppDb.I.openDefault();

    // Plug DriverLogs into the SQLite sink.
    // (Cast to dynamic to avoid interface name clashes across files.)
    // ignore: invalid_use_of_protected_member, invalid_use_of_visible_for_testing_member
    (DriverLogs.I as dynamic).setSink(SqliteDriverLogsSink());

    // If you later add a pluggable sink to Metrics, wire it here similarly:
    // (Metrics.I as dynamic).setSink(SqliteMetricsSink());

    debugPrint('[boot] Local SQLite sinks wired. DB at: ${LocalAppDb.I.activePath}');
  } catch (e) {
    debugPrint('[boot] Local sink wiring skipped: $e');
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final AuthController _auth = AuthController();

  @override
  void initState() {
    super.initState();
    _auth.addListener(_onAuthChanged);
  }

  @override
  void dispose() {
    _auth.removeListener(_onAuthChanged);
    super.dispose();
  }

  void _onAuthChanged() => setState(() {});

  @override
  Widget build(BuildContext context) {
    // Rebuild MaterialApp whenever settings change (e.g., theme toggle)
    return AnimatedBuilder(
      animation: AppSettings.I,
      builder: (context, _) {
        final signedIn = _auth.current != null;

        return MaterialApp(
          debugShowCheckedModeBanner: false,
          title: 'Driver Monitoring',

          // Light theme
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.indigo,
              brightness: Brightness.light,
            ),
          ),

          // Dark theme
          darkTheme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.indigo,
              brightness: Brightness.dark,
            ),
          ),

          // Follow AppSettings (system / light / dark)
          themeMode: AppSettings.I.themeMode,

          home: signedIn ? AppShell(auth: _auth) : LoginSignupPage(auth: _auth),
        );
      },
    );
  }
}

class AppShell extends StatefulWidget {
  final AuthController auth;
  const AppShell({super.key, required this.auth});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _ix = 0; // 0=Monitor, 1=Settings
  final _navKey = GlobalKey<ScaffoldMessengerState>();

  @override
  void initState() {
    super.initState();
    AppSettings.I.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    AppSettings.I.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    // If monitoring just turned ON while user is on Settings, push them back to Monitor.
    if (AppSettings.I.monitoringEnabled && _ix == 1) {
      setState(() => _ix = 0);
      final messenger = ScaffoldMessenger.maybeOf(context);
      messenger?.hideCurrentSnackBar();
      messenger?.showSnackBar(
        const SnackBar(
          content: Text('Turn monitoring OFF to open Settings.'),
          duration: Duration(seconds: 2),
        ),
      );
    } else {
      setState(() {}); // refresh icons/labels if theme or other toggles changed
    }
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      const CameraFaceOnnxPage(), // shows the on-screen Monitoring toggle itself
      SettingsPage(
        onLogout: () async => widget.auth.signOut(),
        currentUsername: widget.auth.current?.username ?? '',
      ),
    ];

    final settingsBlocked = AppSettings.I.monitoringEnabled;

    return Scaffold(
      key: _navKey,
      body: pages[_ix],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _ix,
        onDestinationSelected: (i) {
          // Block navigating to Settings when monitoring is ON
          if (i == 1 && settingsBlocked) {
            final messenger = ScaffoldMessenger.maybeOf(context);
            messenger?.hideCurrentSnackBar();
            messenger?.showSnackBar(
              const SnackBar(
                content: Text('Turn monitoring OFF to open Settings.'),
                duration: Duration(seconds: 2),
              ),
            );
            return;
          }
          setState(() => _ix = i);
        },
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam),
            label: 'Monitor',
          ),
          NavigationDestination(
            // Visually hint that Settings is disabled while monitoring is ON
            icon: Icon(
              Icons.settings_outlined,
              color: settingsBlocked
                  ? Theme.of(context).colorScheme.onSurface.withOpacity(0.38)
                  : null,
            ),
            selectedIcon: Icon(
              Icons.settings,
              color: settingsBlocked
                  ? Theme.of(context).colorScheme.onSurface.withOpacity(0.38)
                  : null,
            ),
            label: settingsBlocked ? 'Settings (locked)' : 'Settings',
          ),
        ],
      ),
    );
  }
}
