/**
 * Crash reporting -- OFF by default, same pattern as the AI Coach API key
 * (see server.py's ANTHROPIC_API_KEY handling): the feature reads an
 * environment variable, and if it's not set, everything in this file is a
 * true no-op. No account was created on your behalf, no data leaves the
 * device, and nothing here changes app behavior until you deliberately
 * turn it on.
 *
 * To turn it on:
 *   1. Create your own free account at https://sentry.io (or self-host --
 *      this is standard Sentry, not a Kinetra-specific service) and create
 *      a React Native project to get a DSN.
 *   2. Add it to `.env`:  EXPO_PUBLIC_SENTRY_DSN=https://...@...ingest.sentry.io/...
 *   3. Rebuild the app. That's it -- initSentry() below picks it up
 *      automatically.
 *
 * What this scaffold gives you once a DSN is set:
 *   - Unhandled JS errors and unhandled promise rejections are reported.
 *   - Explicit reportError() calls at the app's two real risk points
 *     (the /analyze and /ai-coach network calls) so a failure there shows
 *     up with real context instead of only a console.log the user never
 *     sees.
 *
 * What this scaffold deliberately does NOT include: the Expo config plugin
 * ("@sentry/react-native/expo") that enables automatic native crash
 * symbolication and release/source-map upload during `eas build`. That
 * plugin needs an org slug, project slug, and auth token, and -- more
 * importantly -- changes native build output in a way that can't be
 * verified without actually running a real EAS build, which this
 * environment cannot do. Wiring up JS-level error reporting first (safe,
 * fully testable) and adding the native plugin later once you have real
 * Sentry project details and can test an actual build is the more careful
 * order to do this in. See https://docs.sentry.io/platforms/react-native/manual-setup/expo/
 * when you're ready for that step.
 */
import * as Sentry from '@sentry/react-native';

const SENTRY_DSN = process.env.EXPO_PUBLIC_SENTRY_DSN;

let initialized = false;

/** Call once, as early as possible (see app/_layout.tsx). No-op with no DSN. */
export function initCrashReporting() {
  if (!SENTRY_DSN) {
    return;
  }
  try {
    Sentry.init({
      dsn: SENTRY_DSN,
      // Movement session data (scores/grades) never touches this SDK --
      // Sentry only ever sees error stack traces and the breadcrumbs
      // reportError() attaches below (a route name, an HTTP status code),
      // never video, pose data, or personally identifying fields.
      tracesSampleRate: 0,
      enableAutoSessionTracking: false,
      debug: __DEV__,
    });
    initialized = true;
  } catch (error) {
    // Crash reporting failing to initialize should never crash the app it's
    // meant to be protecting.
    console.log('Crash reporting failed to initialize:', error);
  }
}

/**
 * Report a caught error with optional non-sensitive context. Safe to call
 * unconditionally throughout the app -- it's a no-op until a DSN is set.
 */
export function reportError(error: unknown, context?: Record<string, string | number | boolean>) {
  if (!initialized) {
    return;
  }
  try {
    Sentry.captureException(error, context ? { extra: context } : undefined);
  } catch (reportingError) {
    console.log('Failed to report error to crash reporting:', reportingError);
  }
}
