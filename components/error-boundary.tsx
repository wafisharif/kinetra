import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { reportError } from '@/constants/crashReporting';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

/**
 * App-wide crash guard.
 *
 * The app's main screen (app/(tabs)/index.tsx) is a single ~17,000-line
 * component covering dozens of screens, and until now nothing anywhere in
 * the app caught a render error -- an uncaught exception in any one screen
 * would take the whole app down to a blank white screen with no way back
 * short of force-quitting. This catches that instead: it reports the error
 * (silently, only if crash reporting has been configured -- see
 * constants/crashReporting.ts, a true no-op otherwise) and shows a plain
 * recovery screen with a button that resets the boundary and re-renders.
 *
 * Scope, honestly stated: per React's error boundary contract, this only
 * catches errors thrown during rendering, in lifecycle methods, and in
 * constructors of the component tree below it. It does NOT catch errors
 * inside event handlers, timers, or async callbacks (a rejected fetch
 * inside a button's onPress, for example) -- which is exactly why the
 * app's two real network risk points (the /analyze and /ai-coach calls)
 * already have their own try/catch blocks and reportError() calls
 * alongside this, rather than relying on this boundary to catch those.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    reportError(error, {
      route: 'render-tree',
      componentStack: (info.componentStack ?? '').slice(0, 500),
    });
  }

  handleReset = () => {
    this.setState({ hasError: false });
  };

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <Text style={styles.title}>Something went wrong</Text>
          <Text style={styles.body}>
            Kinetra hit an unexpected error and had to stop that screen.
            Nothing stored on your phone -- your saved sessions, calibration,
            and streaks -- is affected by this.
          </Text>
          <Pressable
            style={styles.button}
            onPress={this.handleReset}
            accessibilityRole="button"
            accessibilityLabel="Try again"
          >
            <Text style={styles.buttonText}>Try Again</Text>
          </Pressable>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 28,
    paddingVertical: 40,
  },
  title: {
    color: '#f1f5f9',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
    textAlign: 'center',
  },
  body: {
    color: '#94a3b8',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
    marginBottom: 28,
    maxWidth: 320,
  },
  button: {
    backgroundColor: '#2563eb',
    paddingVertical: 13,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
});
