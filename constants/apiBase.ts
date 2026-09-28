/**
 * The one shared source for Kinetra's backend base URL.
 *
 * Previously this lived only inside `app/(tabs)/index.tsx` as a local
 * const, which worked fine when it was the only file that ever talked to
 * the backend. `hooks/useDashboardSync.ts` needs the exact same URL (it
 * calls `/auth/*` and `/sync` on the same Flask host that `/analyze` and
 * `/ai-coach` already use), so this was pulled out into its own file
 * rather than duplicated -- two copies of "which URL is the backend"
 * would eventually drift out of sync with each other.
 *
 * Backend URLs are config-driven via environment variables (see `.env` /
 * `.env.example` at the project root) instead of being hardcoded here.
 * Expo automatically loads `.env` and inlines any `EXPO_PUBLIC_`-prefixed
 * variable into the JS bundle at build/export time -- this is the standard
 * Expo mechanism for build-time config that isn't secret (the value ends
 * up visible in the client bundle either way, exactly like a hardcoded
 * string would). This means rotating a backend host (e.g. replacing an
 * expiring ngrok URL, or pointing a preview build at a staging server) is
 * a one-line edit to `.env`, never a source-code change here.
 *
 * The literal strings below are fallbacks only, used if `.env` is ever
 * missing -- they keep local development working out of the box but
 * should not be relied on for anything shipped.
 */

export const LOCAL_API_BASE_URL =
  process.env.EXPO_PUBLIC_LOCAL_API_BASE_URL || 'http://192.168.1.163:5000';

// ⚠️  The fallback below is an ngrok free-tier URL and WILL EXPIRE. Set
// EXPO_PUBLIC_DEPLOYED_API_BASE_URL in `.env` to your real deployed backend
// URL (Render, Railway, Fly.io, your own server, etc.) before shipping a
// production or preview build.
export const DEPLOYED_API_BASE_URL =
  process.env.EXPO_PUBLIC_DEPLOYED_API_BASE_URL || 'https://unrest-busily-snort.ngrok-free.dev';

// __DEV__ is a React Native global: true in a development build, false in
// a production/release build.
export const API_BASE_URL = __DEV__ ? LOCAL_API_BASE_URL : DEPLOYED_API_BASE_URL;
