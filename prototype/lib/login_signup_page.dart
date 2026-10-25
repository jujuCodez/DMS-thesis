// lib/login_signup_page.dart
import 'package:flutter/material.dart';
import 'auth_controller.dart';

class LoginSignupPage extends StatefulWidget {
  final AuthController auth;
  const LoginSignupPage({super.key, required this.auth});

  @override
  State<LoginSignupPage> createState() => _LoginSignupPageState();
}

class _LoginSignupPageState extends State<LoginSignupPage>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 2, vsync: this);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 420),
          child: Card(
            margin: const EdgeInsets.all(24),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                const Text('Welcome', style: TextStyle(
                  fontSize: 24, fontWeight: FontWeight.w600)),
                const SizedBox(height: 12),
                TabBar(controller: _tabs, tabs: const [
                  Tab(text: 'Login'),
                  Tab(text: 'Sign up'),
                ]),
                const SizedBox(height: 12),
                SizedBox(
                  height: 250,
                  child: TabBarView(
                    controller: _tabs,
                    children: [
                      _LoginForm(auth: widget.auth),
                      _SignupForm(auth: widget.auth),
                    ],
                  ),
                ),
              ]),
            ),
          ),
        ),
      ),
    );
  }
}

class _LoginForm extends StatefulWidget {
  final AuthController auth;
  const _LoginForm({required this.auth});
  @override
  State<_LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<_LoginForm> {
  final _form = GlobalKey<FormState>();
  final _user = TextEditingController();
  final _pass = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _form,
      child: Column(children: [
        TextFormField(
          controller: _user,
          decoration: const InputDecoration(labelText: 'Username'),
          validator: (v) => (v == null || v.isEmpty) ? 'Enter username' : null,
        ),
        const SizedBox(height: 8),
        TextFormField(
          controller: _pass,
          decoration: const InputDecoration(labelText: 'Password'),
          obscureText: true,
          validator: (v) => (v == null || v.isEmpty) ? 'Enter password' : null,
        ),
        const SizedBox(height: 8),
        if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
        const SizedBox(height: 8),
        FilledButton(
          onPressed: _busy ? null : () async {
            if (!_form.currentState!.validate()) return;
            setState(() { _busy = true; _error = null; });
            final err = await widget.auth.signIn(
              username: _user.text.trim(),
              password: _pass.text,
            );
            if (mounted) setState(() { _busy = false; _error = err; });
          },
          child: _busy
              ? const SizedBox(width: 18, height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : const Text('Login'),
        ),
      ]),
    );
  }
}

class _SignupForm extends StatefulWidget {
  final AuthController auth;
  const _SignupForm({required this.auth});
  @override
  State<_SignupForm> createState() => _SignupFormState();
}

class _SignupFormState extends State<_SignupForm> {
  final _form = GlobalKey<FormState>();
  final _user = TextEditingController();
  final _pass = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _form,
      child: Column(children: [
        TextFormField(
          controller: _user,
          decoration: const InputDecoration(labelText: 'Username'),
          validator: (v) => (v == null || v.isEmpty) ? 'Enter username' : null,
        ),
        const SizedBox(height: 8),
        TextFormField(
          controller: _pass,
          decoration: const InputDecoration(labelText: 'Password (min 4)'),
          obscureText: true,
          validator: (v) => (v == null || v.length < 4) ? 'Too short' : null,
        ),
        const SizedBox(height: 8),
        if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
        const SizedBox(height: 8),
        FilledButton(
          onPressed: _busy ? null : () async {
            if (!_form.currentState!.validate()) return;
            setState(() { _busy = true; _error = null; });
            final err = await widget.auth.signUp(
              username: _user.text.trim(),
              password: _pass.text,
            );
            if (mounted) setState(() { _busy = false; _error = err; });
          },
          child: _busy
              ? const SizedBox(width: 18, height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : const Text('Create account'),
        ),
      ]),
    );
  }
}
