import { useCallback, useEffect, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { API_BASE_URL } from '@/constants/apiBase';

// ---------------------------------------------------------------------------
// Web dashboard sync (optional).
//
// The app itself has no accounts and keeps everything on-device -- see the
// Privacy Policy. This hook is the one bridge out of that: if someone signs
// in here with a web dashboard account (the same account system that
// powers kinetraapp.com/dashboard -- see accounts.py in this repo), their
// session history, Daily Check-Ins, and calibration get pushed to that
// account so the browser dashboard can show real data instead of the
// sample placeholder it shows everyone else.
//
// Signing in here is entirely optional and changes nothing about how the
// app behaves if it's skipped -- see the "Web Dashboard Sync" section in
// Settings, the only place this hook is used from.
//
// Sync strategy is deliberately simple: every push sends the app's full
// current sessions/check-ins/calibration and replaces whatever the server
// had stored for this account (see the backend's /sync endpoint in
// sync.py). There is no merge, no per-item diffing, and no conflict
// resolution. That is fine for what this actually is -- one person's own
// on-device data going to their own account, from what is realistically
// one device at a time -- and it avoids an entire category of sync bugs a
// fancier incremental design would introduce for no real benefit here.
// ---------------------------------------------------------------------------

const TOKEN_KEY = 'dashboard_sync_token_v1';
const USER_KEY = 'dashboard_sync_user_v1';

// Matches the web dashboard's own REQUEST_TIMEOUT_MS (js/dashboard.js in
// the kinetra-website repo) -- both talk to the same Render free-tier
// backend, which spins an idle instance down and can take 20-50s to wake
// back up on the first request after a quiet period (longer than most
// Render apps, since this one's cold start also has to import mediapipe/
// opencv/numpy before Flask can even start handling requests). A shorter
// timeout here was the actual bug behind "sign-in never works from another
// device" reports: the app was giving up well before a cold backend had
// even finished waking, on every single first attempt.
const REQUEST_TIMEOUT_MS = 45000;

export type DashboardUser = { id: string; name: string; email: string };

export type SyncStatus = 'idle' | 'syncing' | 'synced' | 'error';

export type SyncPayload = {
  sessions: unknown[];
  checkIns: unknown[];
  calibration: unknown;
};

class ApiError extends Error {}

async function apiRequest(
  path: string,
  options: { method?: string; token?: string | null; body?: unknown } = {}
) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (options.token) headers.Authorization = `Bearer ${options.token}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method || 'GET',
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    });
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw new ApiError(
        'The Kinetra server is taking too long to respond. It\'s probably waking up from being idle -- this can take up to a minute the first time. Wait a bit and try again.'
      );
    }
    throw new ApiError(`Could not reach the Kinetra server at ${API_BASE_URL}.`);
  } finally {
    clearTimeout(timeoutId);
  }

  let data: any = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    throw new ApiError(data.error || data.details || 'Something went wrong. Try again.');
  }
  return data;
}

export function useDashboardSync() {
  const [user, setUser] = useState<DashboardUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  // True only while the persisted session is being read from AsyncStorage
  // on mount -- lets the Settings screen avoid a one-frame flash of the
  // sign-in form for someone who is already signed in.
  const [authLoading, setAuthLoading] = useState(true);
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const [syncStatus, setSyncStatus] = useState<SyncStatus>('idle');
  const [syncError, setSyncError] = useState<string | null>(null);
  const [lastSyncedAt, setLastSyncedAt] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [savedToken, savedUserRaw] = await Promise.all([
          AsyncStorage.getItem(TOKEN_KEY),
          AsyncStorage.getItem(USER_KEY),
        ]);
        if (savedToken && savedUserRaw) {
          setToken(savedToken);
          setUser(JSON.parse(savedUserRaw));
        }
      } catch (error) {
        console.log('Failed to load dashboard sync session:', error);
      } finally {
        setAuthLoading(false);
      }
    })();
  }, []);

  const persistSession = useCallback(async (nextToken: string, nextUser: DashboardUser) => {
    setToken(nextToken);
    setUser(nextUser);
    try {
      await AsyncStorage.setItem(TOKEN_KEY, nextToken);
      await AsyncStorage.setItem(USER_KEY, JSON.stringify(nextUser));
    } catch (error) {
      console.log('Failed to save dashboard sync session:', error);
    }
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      setAuthError(null);
      setAuthBusy(true);
      try {
        const data = await apiRequest('/auth/login', { method: 'POST', body: { email, password } });
        await persistSession(data.token, data.user);
        return true;
      } catch (error: any) {
        setAuthError(error.message || 'Sign in failed.');
        return false;
      } finally {
        setAuthBusy(false);
      }
    },
    [persistSession]
  );

  const signUp = useCallback(
    async (name: string, email: string, password: string) => {
      setAuthError(null);
      setAuthBusy(true);
      try {
        const data = await apiRequest('/auth/signup', {
          method: 'POST',
          body: { name, email, password },
        });
        await persistSession(data.token, data.user);
        return true;
      } catch (error: any) {
        setAuthError(error.message || 'Sign up failed.');
        return false;
      } finally {
        setAuthBusy(false);
      }
    },
    [persistSession]
  );

  const signOut = useCallback(async () => {
    setToken(null);
    setUser(null);
    setLastSyncedAt(null);
    setSyncStatus('idle');
    setSyncError(null);
    try {
      await AsyncStorage.removeItem(TOKEN_KEY);
      await AsyncStorage.removeItem(USER_KEY);
    } catch (error) {
      console.log('Failed to clear dashboard sync session:', error);
    }
  }, []);

  const clearAuthError = useCallback(() => setAuthError(null), []);

  // Pushes the given local data to whichever account is currently signed
  // in. A no-op (resolves false, touches nothing) when signed out, so
  // every call site below can call this unconditionally after saving
  // local data instead of checking auth state itself. Runs silently by
  // design -- this fires automatically right after saving a session, a
  // check-in, or a calibration change, and a sync hiccup in the
  // background shouldn't interrupt whatever the user was just doing. The
  // Settings screen surfaces syncStatus/syncError for anyone who wants to
  // check.
  const syncNow = useCallback(
    async (payload: SyncPayload) => {
      if (!token) return false;
      setSyncStatus('syncing');
      setSyncError(null);
      try {
        const data = await apiRequest('/sync', {
          method: 'POST',
          token,
          body: {
            sessions: payload.sessions,
            checkins: payload.checkIns,
            calibration: payload.calibration,
          },
        });
        setLastSyncedAt(typeof data.synced_at === 'number' ? data.synced_at : Date.now() / 1000);
        setSyncStatus('synced');
        return true;
      } catch (error: any) {
        setSyncStatus('error');
        setSyncError(error.message || 'Sync failed.');
        return false;
      }
    },
    [token]
  );

  return {
    user,
    authLoading,
    authBusy,
    authError,
    syncStatus,
    syncError,
    lastSyncedAt,
    signIn,
    signUp,
    signOut,
    syncNow,
    clearAuthError,
  };
}
