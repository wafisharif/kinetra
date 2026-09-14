import { reportError } from '@/constants/crashReporting';
import { useOnDevicePose } from '@/hooks/useOnDevicePose';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Notifications from 'expo-notifications';
import { useVideoPlayer, VideoView } from 'expo-video';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  BackHandler,
  Platform,
  Pressable,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

// Local (on-device, no server) daily reminder notification. Foreground
// behavior only matters if the app happens to be open when the reminder
// fires -- still show it as a banner/list entry so it's not silently
// swallowed, matching what a user would expect from any reminder app.
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: false,
    shouldSetBadge: false,
  }),
});

const REMINDER_NOTIFICATION_IDENTIFIER = 'kinetra-daily-reminder';

// Backend URLs are config-driven via environment variables (see `.env` /
// `.env.example` at the project root) instead of being hardcoded here. Expo
// automatically loads `.env` and inlines any `EXPO_PUBLIC_`-prefixed variable
// into the JS bundle at build/export time -- this is the standard Expo
// mechanism for build-time config that isn't secret (the value ends up
// visible in the client bundle either way, exactly like a hardcoded string
// would). This means rotating a backend host (e.g. replacing an expiring
// ngrok URL, or pointing a preview build at a staging server) is a one-line
// edit to `.env`, never a source-code change to this 17,000-line file.
//
// The literal strings below are fallbacks only, used if `.env` is ever
// missing -- they keep local development working out of the box but should
// not be relied on for anything shipped.
const LOCAL_API_BASE_URL = process.env.EXPO_PUBLIC_LOCAL_API_BASE_URL || 'http://192.168.1.163:5000';

// ⚠️  The fallback below is an ngrok free-tier URL and WILL EXPIRE. Set
// EXPO_PUBLIC_DEPLOYED_API_BASE_URL in `.env` to your real deployed backend
// URL (Render, Railway, Fly.io, your own server, etc.) before shipping a
// production or preview build.
const DEPLOYED_API_BASE_URL = process.env.EXPO_PUBLIC_DEPLOYED_API_BASE_URL || 'https://unrest-busily-snort.ngrok-free.dev';

// __DEV__ is a React Native global: true in a development build, false in a
// production/release build. This used to be
// `DEPLOYED_API_BASE_URL || LOCAL_API_BASE_URL`, which ALWAYS picked the
// deployed URL -- a non-empty string is always truthy, so LOCAL_API_BASE_URL
// could never actually be reached, even when developing locally.
const API_BASE_URL = __DEV__ ? LOCAL_API_BASE_URL : DEPLOYED_API_BASE_URL;

const ANALYSIS_TIMEOUT_MS = 90000;
const SERVER_HEALTH_TIMEOUT_MS = 8000;

const APP_NAME = 'Kinetra';
const APP_TAGLINE = 'Understand Movement. Move Better.';
const APP_BETA_LABEL = 'Beta';
const APP_DESCRIPTION =
  'Camera-based movement intelligence for tracking control, stability, mobility, and change over time.';
const APP_SAFETY_NOTE =
  'Kinetra is for movement awareness only. It does not diagnose, treat, predict injury, estimate fall risk, or replace medical advice.';

// Single source of truth for the version shown in Settings and the What's
// New screen. Keep in sync with app.json's "version" field when it changes
// -- there is deliberately no build-time wiring between the two (Expo's
// app.json version isn't readable from inside the JS bundle without adding
// expo-constants just for this one string), so bumping a release means
// updating both by hand.
const APP_VERSION = '0.1.0';

// A real, append-only changelog. Each entry should describe what genuinely
// shipped, in plain language a non-technical user would understand -- never
// backfilled with invented past version numbers or dates for milestones
// that didn't actually ship as separate releases. Add a new entry at the
// TOP of this array when you cut a new version; never edit past entries.
type WhatsNewEntry = {
  version: string;
  date: string; // human-readable, e.g. "September 2026" -- intentionally not day-precise, since exact ship dates for a solo project aren't a meaningful signal
  highlights: string[];
};
const WHATS_NEW_ENTRIES: WhatsNewEntry[] = [
  {
    version: '0.1.0',
    date: 'September 2026',
    highlights: [
      'Per-arm calibration: rep-detection thresholds can now be personalized separately for your left and right arm, instead of one fixed range for everyone.',
      'Team Roster and Team Screening mode, for coaches or teachers checking movement quality across a group rather than one person at a time.',
      'A Transparency page that plainly explains what Kinetra measures, its real limitations, and exactly what happens to your data and video.',
      'AI Coach can now ask Claude for a short, personalized note based on your recent scores (opt-in on the backend; only numeric scores and grades are ever sent, never video).',
      'Consistency Streaks and 8 honest achievement badges -- every badge is based on showing up and testing fully, never on getting a high score.',
      'Local daily reminder notifications, scheduled entirely on your device -- no account or server involved.',
      'A consolidated Settings screen for reminders, calibration, and privacy info.',
      'A full pass to find and fix real layout bugs (text clipping instead of wrapping on a couple of summary screens) using an automated visual QA sweep across every screen.',
    ],
  },
];

type AnalysisResult = {
  mode: 'rep' | 'rehab' | 'lab' | 'daily';
  side?: 'left' | 'right';
  fps?: number;
  calibration_data?: {
    observed_min_angle: number | null;
    observed_max_angle: number | null;
    flex_threshold_used: number | null;
    extend_threshold_used: number | null;
    thresholds_calibrated: boolean;
  };
  daily_task?: DailyTask;
  daily_task_label?: string;
  daily_task_focus?: {
    primary_goal: string;
    tracked_region: string;
    current_status: string;
    future_metrics: string[];
  };

  task_analysis?: {
    task_type: string;
    status: string;
    right_knee_angle_mean?: number | null;
    right_hip_angle_mean?: number | null;
    torso_lean_mean?: number | null;
    lower_body_valid_frames?: number;
    lower_body_signal_score?: number;
    lower_body_signal_grade?: string;

    transition_detected?: boolean;
    transition_start_frame?: number | null;
    transition_end_frame?: number | null;
    transition_duration_sec?: number | null;
    knee_extension_range?: number | null;
    hip_extension_range?: number | null;
    max_torso_lean?: number | null;
    torso_control?: number | null;
    rise_stability_score?: number | null;
    rise_stability_grade?: string;
    reach_distance_max?: number | null;
    reach_distance_min?: number | null;
    reach_range?: number | null;
    reach_smoothness?: number | null;
    endpoint_steadiness?: number | null;
    reach_stability_score?: number | null;
    reach_stability_grade?: string;
    arm_raise_max?: number | null;
    arm_raise_min?: number | null;
    arm_raise_range?: number | null;
    arm_raise_smoothness?: number | null;
    top_steadiness?: number | null;
    arm_distance_mean?: number | null;
    elbow_angle_mean?: number | null;
    arm_raise_valid_frames?: number;
    arm_raise_signal_score?: number;
    arm_raise_stability_score?: number | null;
    arm_raise_stability_grade?: string;

    walking_valid_frames?: number;
    walking_signal_score?: number;
    walking_signal_grade?: string;
    estimated_step_cycles?: number;
    walking_duration_sec?: number | null;
    cadence_estimate?: number | null;
    hip_path_range?: number | null;
    hip_vertical_variability?: number | null;
    step_rhythm_variability?: number | null;
    left_knee_range?: number | null;
    right_knee_range?: number | null;
    knee_range_difference?: number | null;
    walking_stability_score?: number | null;
    walking_stability_grade?: string;

    balance_valid_frames?: number;
    balance_signal_score?: number;
    balance_signal_grade?: string;
    hip_sway_range?: number | null;
    shoulder_sway_range?: number | null;
    shoulder_vertical_variability?: number | null;
    torso_lean_max?: number | null;
    torso_lean_variability?: number | null;
    ankle_center_drift?: number | null;
    support_width_mean?: number | null;
    balance_stability_score?: number | null;
    balance_stability_grade?: string;

    tug_valid_frames?: number;
    tug_signal_score?: number;
    tug_signal_grade?: string;
    tug_duration_sec?: number | null;
    tug_path_range?: number | null;
    tug_return_pattern_detected?: boolean;
    tug_direction_changes?: number;
    tug_estimated_step_cycles?: number;
    tug_cadence_estimate?: number | null;
    tug_knee_range?: number | null;
    tug_hip_range?: number | null;
    tug_torso_lean_max?: number | null;
    tug_torso_lean_variability?: number | null;
    tug_hip_vertical_variability?: number | null;
    tug_mobility_score?: number | null;
    tug_mobility_grade?: string;

    summary?: string;
    task_insights?: string[];

    note: string;
  } | null;
  reps: number;
  transitions: number;
  total_frames: number;
  signal_quality?: {
    score: number;
    valid_frames: number;
    total_frames: number;
    grade: string;
    message: string;
  };
  max_extension_speed: number | null;
  max_flexion_speed: number | null;
  output_csv: string;
  video_path: string;
  fatigue_analysis: string;
  interpretation: {
    summary: string;
    focus: string;
    insight: string;
  };
  global_metrics: {
    smoothness: number | null;
    symmetry: number | null;
    control: number | null;
    efficiency: number | null;

    smoothness_grade: string;
    symmetry_grade: string;
    control_grade: string;
    efficiency_grade: string;
  };
  rep_analysis: Array<{
    rep: number;
    start: number;
    end: number;
    smoothness: number | null;
    symmetry: number | null;
    control: number | null;
    smoothness_grade: string;
    symmetry_grade: string;
    control_grade: string;
    score: number | null;
    score_grade: string;
  }>;

  best_rep?: {
    rep: number;
    start: number;
    end: number;
    smoothness: number | null;
    symmetry: number | null;
    control: number | null;
    smoothness_grade: string;
    symmetry_grade: string;
    control_grade: string;
    score: number | null;
    score_grade: string;
  };

  worst_rep?: {
    rep: number;
    start: number;
    end: number;
    smoothness: number | null;
    symmetry: number | null;
    control: number | null;
    smoothness_grade: string;
    symmetry_grade: string;
    control_grade: string;
    score: number | null;
    score_grade: string;
  };

  score_trend?: number[];
  performance_drop?: number | null;
  performance_summary?: string;
  key_insights?: string[];

  movement_health_score?: number | null;
  movement_health_grade?: string;

  movement_signature?: {
    label: string;
    description: string;
  };
};

type SavedSession = {
  id: string;
  timestamp: string;
  mode: 'rep' | 'rehab' | 'lab' | 'daily';
  daily_task?: DailyTask;
  daily_task_label?: string;
  primary_score: number | null;
  primary_grade: string;
  confidence_grade?: string;
  // Phase 2: which arm this session analyzed and whether personalized
  // calibration thresholds were active for it. Optional so old sessions
  // saved before this field existed still load and render fine.
  side?: 'left' | 'right';
  thresholds_calibrated?: boolean;
  // Phase 3: set only when Team Screening mode was on for this recording --
  // lets a coach or PE teacher tag a session with which athlete it belongs
  // to, without needing any separate roster-specific recording flow.
  athlete_name?: string;
};

type TesterNote = {
  id: string;
  timestamp: string;
  testerName: string;
  confusionPoint: string;
  bugFound: string;
  featureRequest: string;
  overallReaction: string;
};

type ServerHealthStatus = 'unchecked' | 'checking' | 'online' | 'offline';

type ServerHealthState = {
  status: ServerHealthStatus;
  message: string;
  apiUrl: string;
  checkedAt?: string;
};

type AnalysisBackend = 'mediapipe' | 'yolo';
type DailyTask = 'reach' | 'arm_raise' | 'sit_to_stand' | 'walking' | 'balance' | 'timed_up_and_go';

type MovementModuleId =
  | 'rep_quality'
  | 'rehab_consistency'
  | 'daily_reach'
  | 'daily_arm_raise'
  | 'daily_sit_to_stand'
  | 'daily_walking'
  | 'daily_balance'
  | 'daily_timed_up_and_go'
  | 'movement_lab'
  | 'walking_yolo'
  | 'balance_yolo'
  | 'timed_up_and_go_yolo';

type MovementModuleStatus = 'active' | 'foundation_ready' | 'future';

function getGradeColors(grade: string) {
  const lower = grade.toLowerCase();
  if (
    lower.includes('stable or improving') ||
    lower.includes('baseline trend engine ready') ||
    lower.includes('trend snapshot ready') ||
    lower.includes('within baseline') ||
    lower.includes('long-term improvement') ||
    lower.includes('30-day improvement')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('trend engine building') ||
    lower.includes('early trend') ||
    lower.includes('trend building') ||
    lower.includes('first check saved') ||
    lower.includes('baseline building')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('recheck needed') ||
    lower.includes('long-term watch') ||
    lower.includes('below baseline') ||
    lower.includes('30-day decline') ||
    lower.includes('clear decline') ||
    lower.includes('slight decline')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('stable movement passport') ||
    lower.includes('baseline passport ready') ||
    lower.includes('passport snapshot ready')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('passport building') ||
    lower.includes('early snapshot') ||
    lower.includes('shareable snapshot') ||
    lower.includes('stronger baseline snapshot')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('passport recheck needed') ||
    lower.includes('not ready to share yet')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('starter profile complete')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('profile started') ||
    lower.includes('almost ready') ||
    lower.includes('start here')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }
  if (
    lower.includes('stable mobility profile') ||
    lower.includes('functional mobility snapshot ready')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('functional mobility profile building') ||
    lower.includes('profile building') ||
    lower.includes('building baseline')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('recheck recommended') ||
    lower.includes('needs attention') ||
    lower.includes('needs recheck')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('yolo framework ready') ||
    lower.includes('framework ready') ||
    lower.includes('active')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('future')) {
    return {
      bg: 'rgba(71, 85, 105, 0.22)',
      border: 'rgba(148, 163, 184, 0.35)',
      text: '#cbd5e1',
    };
  }
  if (
    lower.includes('weekly movement stable')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('weekly report started') ||
    lower.includes('weekly profile building')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('no weekly data yet') ||
    lower.includes('one weekly area needs attention') ||
    lower.includes('multiple weekly areas need attention')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('stable or improving') ||
    lower.includes('positive change detected')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('movement profile building') ||
    lower.includes('build more baseline data') ||
    lower.includes('stable check')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('one area needs attention') ||
    lower.includes('multiple areas need attention') ||
    lower.includes('watch this result') ||
    lower.includes('retest before trusting this') ||
    lower.includes('start tracking first')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('stable or improving') ||
    lower.includes('mostly stable')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('building daily health profile') ||
    lower.includes('no daily health data yet')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('one area needs attention') ||
    lower.includes('multiple areas need attention')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }
  if (lower.includes('first analytics round complete')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('early patterns visible') ||
    lower.includes('tester data started')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (lower.includes('no tester data yet')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (lower.includes('ready for small beta launch')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('almost beta ready') ||
    lower.includes('internal testing stage')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (lower.includes('not launch ready')) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }
  if (lower.includes('guided test complete')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('testing in progress')) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (lower.includes('not started')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }
  if (
    lower.includes('first testing round complete') ||
    lower.includes('feedback pattern forming')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('ready for first private tester') ||
    lower.includes('private testing started') ||
    lower.includes('core modes need data') ||
    lower.includes('internal testing needed')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (
    lower.includes('no test data yet') ||
    lower.includes('fix bugs before more testing')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }
  if (
    lower.includes('ready to prioritize fixes') ||
    lower.includes('feature requests emerging')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('baseline confusion') ||
    lower.includes('trend confusion') ||
    lower.includes('score confusion') ||
    lower.includes('camera setup confusion') ||
    lower.includes('ux confusion pattern') ||
    lower.includes('feedback needs review')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (
    lower.includes('fix bugs first') ||
    lower.includes('no action plan yet')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }
  if (
    lower.includes('testing round complete') ||
    lower.includes('useful feedback collected')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('feedback started')) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (lower.includes('no feedback yet')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (lower.includes('ready for private testing')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('almost ready') ||
    lower.includes('needs more internal testing')
  ) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (lower.includes('not ready yet')) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (
    lower.includes('mostly stable') ||
    lower.includes('mostly stable or improving')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('building movement profile') ||
    lower.includes('building baseline') ||
    lower.includes('no movement data yet') ||
    lower.includes('no score yet')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('one area below baseline') ||
    lower.includes('multiple areas below baseline')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (lower.includes('high confidence')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('moderate confidence')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (
    lower.includes('consistency strongly improving') ||
    lower.includes('consistency improving') ||
    lower.includes('above rehab baseline') ||
    lower.includes('slightly above rehab baseline')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('consistency stable') ||
    lower.includes('within rehab baseline') ||
    lower.includes('building rehab baseline')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('consistency clearly decreasing') ||
    lower.includes('consistency slightly decreasing') ||
    lower.includes('below rehab baseline') ||
    lower.includes('slightly below rehab baseline')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (
    lower.includes('highly consistent') ||
    lower.includes('mostly consistent')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('somewhat consistent')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (
    lower.includes('consistency needs work') ||
    lower.includes('limited data')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (
    lower.includes('above baseline') ||
    lower.includes('slightly above baseline')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (
    lower.includes('within normal range') ||
    lower.includes('building baseline')
  ) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('below baseline') ||
    lower.includes('slightly below baseline')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (
    lower.includes('strong improvement') ||
    lower.includes('slight improvement')
  ) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('stable')) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (
    lower.includes('clear decline') ||
    lower.includes('slight decline')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (
    lower.includes('low confidence') ||
    lower.includes('retest') ||
    lower.includes('needs work') ||
    lower.includes('needs improvement')
  ) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  if (lower.includes('excellent')) {
    return {
      bg: 'rgba(21, 128, 61, 0.18)',
      border: 'rgba(34, 197, 94, 0.45)',
      text: '#86efac',
    };
  }

  if (lower.includes('good')) {
    return {
      bg: 'rgba(30, 64, 175, 0.18)',
      border: 'rgba(96, 165, 250, 0.45)',
      text: '#93c5fd',
    };
  }

  if (lower.includes('moderate')) {
    return {
      bg: 'rgba(161, 98, 7, 0.18)',
      border: 'rgba(250, 204, 21, 0.45)',
      text: '#fde68a',
    };
  }

  if (lower.includes('poor')) {
    return {
      bg: 'rgba(127, 29, 29, 0.18)',
      border: 'rgba(248, 113, 113, 0.45)',
      text: '#fca5a5',
    };
  }

  return {
    bg: 'rgba(71, 85, 105, 0.22)',
    border: 'rgba(148, 163, 184, 0.35)',
    text: '#cbd5e1',
  };
}

function getMovementModuleRegistry() {
  return [
    {
      id: 'daily_reach' as MovementModuleId,
      title: 'Daily Reach',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks reach control, smoothness, endpoint steadiness, and reach stability.',
      currentState: 'Already connected to the current Daily Reach analysis.',
    },
    {
      id: 'daily_arm_raise' as MovementModuleId,
      title: 'Daily Arm Raise',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks arm raise range, smoothness, top steadiness, and arm raise stability.',
      currentState: 'Already connected to the current Daily Arm Raise analysis.',
    },
    {
      id: 'daily_sit_to_stand' as MovementModuleId,
      title: 'Daily Sit-to-Stand',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks lower-body transition quality, torso control, and rise stability.',
      currentState: 'Already connected to the current Daily Sit-to-Stand analysis.',
    },
    {
      id: 'daily_walking' as MovementModuleId,
      title: 'Daily Walking',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks walking rhythm, lower-body motion, step consistency, and walking stability.',
      currentState: 'Walking V1 is connected as a MediaPipe-backed Daily movement check.',
    },
    {
      id: 'daily_balance' as MovementModuleId,
      title: 'Daily Balance',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks standing balance, visible sway, torso control, and posture stability.',
      currentState: 'Balance V1 is connected as a MediaPipe-backed Daily movement check.',
    },
    {
      id: 'daily_timed_up_and_go' as MovementModuleId,
      title: 'Daily Timed Up and Go',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks functional mobility sequence quality across standing, walking, turning/returning, and torso control.',
      currentState: 'Timed Up and Go V1 is connected as a MediaPipe-backed functional mobility check.',
    },
    {
      id: 'rehab_consistency' as MovementModuleId,
      title: 'Rehab Consistency',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks repeatability, consistency, and controlled movement quality.',
      currentState: 'Already connected to the current Rehab mode.',
    },
    {
      id: 'rep_quality' as MovementModuleId,
      title: 'Rep Quality',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Tracks repeated motion quality, rep scores, best/worst reps, and fatigue patterns.',
      currentState: 'Already connected to the current Rep mode.',
    },
    {
      id: 'movement_lab' as MovementModuleId,
      title: 'Movement Lab',
      backend: 'mediapipe' as AnalysisBackend,
      status: 'active' as MovementModuleStatus,
      purpose: 'Keeps detailed raw movement metrics and exploratory analysis available.',
      currentState: 'Already connected to the current Lab mode.',
    },
    {
      id: 'walking_yolo' as MovementModuleId,
      title: 'Walking Analysis',
      backend: 'yolo' as AnalysisBackend,
      status: 'foundation_ready' as MovementModuleStatus,
      purpose: 'Future module for visible walking rhythm, gait consistency, and side-to-side stability.',
      currentState: 'Framework ready, but backend YOLO analysis is not implemented yet.',
    },
    {
      id: 'balance_yolo' as MovementModuleId,
      title: 'Balance Analysis',
      backend: 'yolo' as AnalysisBackend,
      status: 'future' as MovementModuleStatus,
      purpose: 'Future module for standing balance, visible sway, and postural stability checks.',
      currentState: 'Planned after Walking Analysis V1.',
    },
    {
      id: 'timed_up_and_go_yolo' as MovementModuleId,
      title: 'Timed Up and Go',
      backend: 'yolo' as AnalysisBackend,
      status: 'future' as MovementModuleStatus,
      purpose: 'Future module for sit-to-stand, walking, turning, returning, and sitting sequence analysis.',
      currentState: 'Planned after Walking and Balance modules.',
    },
  ];
}

function getActiveMovementModule(
  mode: 'rep' | 'rehab' | 'lab' | 'daily',
  task: DailyTask
) {
  const modules = getMovementModuleRegistry();

  if (mode === 'daily' && task === 'reach') {
    return modules.find((module) => module.id === 'daily_reach')!;
  }

  if (mode === 'daily' && task === 'arm_raise') {
    return modules.find((module) => module.id === 'daily_arm_raise')!;
  }

  if (mode === 'daily' && task === 'sit_to_stand') {
    return modules.find((module) => module.id === 'daily_sit_to_stand')!;
  }

  if (mode === 'daily' && task === 'walking') {
    return modules.find((module) => module.id === 'daily_walking')!;
  }

  if (mode === 'daily' && task === 'balance') {
    return modules.find((module) => module.id === 'daily_balance')!;
  }

  if (mode === 'daily' && task === 'timed_up_and_go') {
    return modules.find((module) => module.id === 'daily_timed_up_and_go')!;
  }

  if (mode === 'rehab') {
    return modules.find((module) => module.id === 'rehab_consistency')!;
  }

  if (mode === 'rep') {
    return modules.find((module) => module.id === 'rep_quality')!;
  }

  return modules.find((module) => module.id === 'movement_lab')!;
}

function getBackendLabel(backend: AnalysisBackend) {
  if (backend === 'yolo') return 'YOLO';
  return 'MediaPipe';
}

function getModuleStatusLabel(status: MovementModuleStatus) {
  if (status === 'active') return 'Active';
  if (status === 'foundation_ready') return 'Framework Ready';
  return 'Future';
}

function getYoloFrameworkSummary() {
  const modules = getMovementModuleRegistry();
  const activeModules = modules.filter((module) => module.status === 'active');
  const yoloModules = modules.filter((module) => module.backend === 'yolo');
  const foundationReadyModules = modules.filter(
    (module) => module.status === 'foundation_ready'
  );

  return {
    status: 'YOLO Framework Ready',
    summary: 'The app can now organize MediaPipe and future YOLO modules under one movement-module system.',
    nextStep: 'Build Walking Analysis V1 as the first YOLO-backed module.',
    warning: 'Do not replace current MediaPipe modules. YOLO should extend the app, not break Daily/Rehab.',
    activeCount: activeModules.length,
    yoloCount: yoloModules.length,
    foundationReadyCount: foundationReadyModules.length,
    modules,
  };
}

function getModeLabel(mode: 'rep' | 'rehab' | 'lab' | 'daily') {
  if (mode === 'rep') return 'Rep Quality Coach';
  if (mode === 'rehab') return 'Rehab Tracker';
  if (mode === 'daily') return 'Daily Movement Health';
  return 'Movement Lab';
}

function getModeDescription(mode: 'rep' | 'rehab' | 'lab' | 'daily') {
  if (mode === 'rep') {
    return 'Analyze repeated movement quality, rep consistency, and mechanical efficiency.';
  }

  if (mode === 'rehab') {
    return 'Review controlled movement, stability, and consistency during recovery-style motion.';
  }

  if (mode === 'daily') {
    return 'Check everyday movement stability, control, efficiency, and overall movement health.';
  }

  return 'Explore raw movement signals, trends, and biomechanical metrics for deeper analysis.';
}

function getDailyTaskLabel(task: DailyTask) {
  if (task === 'reach') return 'Reach';
  if (task === 'arm_raise') return 'Arm Raise';
  if (task === 'walking') return 'Walking';
  if (task === 'balance') return 'Balance';
  if (task === 'timed_up_and_go') return 'Timed Up and Go';
  return 'Sit-to-Stand';
}

function getDailyTaskDescription(task: DailyTask) {
  if (task === 'reach') {
    return 'Record a simple reaching motion to check control, smoothness, and movement stability.';
  }

  if (task === 'arm_raise') {
    return 'Record a controlled arm raise to review movement stability and coordination.';
  }

  if (task === 'walking') {
    return 'Record a short walking clip to review walking rhythm, lower-body motion, and step consistency.';
  }

  if (task === 'balance') {
    return 'Record a short standing balance check to review visible sway, torso control, and posture stability.';
  }

  if (task === 'timed_up_and_go') {
    return 'Record a functional mobility sequence: sit, stand, walk, turn or return, and finish clearly in frame.';
  }

  return 'Record a sit-to-stand movement to check control, stability, and everyday movement quality.';
}

function getDailyTaskResultDescription(task: DailyTask) {
  if (task === 'reach') {
    return 'This check reviews smoothness, control, symmetry, and efficiency during a simple reaching movement.';
  }

  if (task === 'arm_raise') {
    return 'This check reviews control, coordination, and stability during a controlled arm raise.';
  }

  if (task === 'walking') {
    return 'This check reviews walking rhythm, lower-body visibility, hip motion, knee motion, and walking stability.';
  }

  if (task === 'balance') {
    return 'This check reviews standing balance, visible sway, torso lean, and posture stability.';
  }

  if (task === 'timed_up_and_go') {
    return 'This check reviews functional mobility across standing, walking, turning or returning, torso control, and sequence quality.';
  }

  return 'This check reviews stability, control, and everyday movement quality during a sit-to-stand motion.';
}

function getDailyTaskInstructions(task: DailyTask) {
  if (task === 'reach') {
    return [
      'Place the camera so your shoulder, elbow, and wrist stay visible.',
      'Reach forward or outward in one smooth controlled motion.',
      'Move naturally instead of rushing.',
      'Keep the camera stable and avoid cutting your arm out of frame.',
    ];
  }

  if (task === 'arm_raise') {
    return [
      'Place the camera so your shoulder, elbow, and wrist stay visible.',
      'Raise your arm slowly and lower it with control.',
      'Keep your torso mostly still so the arm motion is easier to measure.',
      'Use bright lighting and a stable camera angle.',
    ];
  }

  if (task === 'walking') {
    return [
      'Place the camera from the side if possible.',
      'Keep hips, knees, ankles, and feet visible.',
      'Walk naturally for several steps.',
      'Keep the camera stable and avoid cutting off your feet.',
    ];
  }

  if (task === 'balance') {
    return [
      'Place the camera so your full body is visible.',
      'Stand still in a comfortable position.',
      'Keep shoulders, hips, knees, ankles, and feet visible.',
      'Keep the camera stable and avoid moving the phone.',
    ];
  }

  if (task === 'timed_up_and_go') {
    return [
      'Start seated with your full body visible.',
      'Stand up, walk a few steps, turn or return, and finish clearly in frame.',
      'Keep hips, knees, ankles, feet, and torso visible as much as possible.',
      'Use a stable camera and avoid rushing the sequence.',
    ];
  }

  return [
    'Place the camera so your full body is visible from the side.',
    'Start seated, stand up naturally, then sit back down if comfortable.',
    'Keep your feet visible and avoid blocking your hips or knees.',
    'Use a stable camera angle with good lighting.',
  ];
}

function getRecordingQualityChecklist(task: DailyTask) {
  if (task === 'reach') {
    return [
      'Shoulder, elbow, wrist, and hand are visible.',
      'Camera is stable and not moving.',
      'Arm will stay in frame during the reach.',
      'Lighting is bright enough to see your arm clearly.',
    ];
  }

  if (task === 'arm_raise') {
    return [
      'Shoulder, elbow, wrist, and full arm path are visible.',
      'Camera is stable and not tilted too much.',
      'Torso stays mostly still.',
      'Lighting is bright enough to see the full arm.',
    ];
  }

  if (task === 'walking') {
    return [
      'Full lower body is visible.',
      'Hips, knees, ankles, and feet stay in frame.',
      'Camera is stable and not handheld.',
      'You have enough room to take several steps.',
    ];
  }

  if (task === 'balance') {
    return [
      'Full body is visible.',
      'Shoulders, hips, knees, ankles, and feet stay in frame.',
      'Camera is stable and not handheld.',
      'You can stand still safely without needing support.',
    ];
  }

  if (task === 'timed_up_and_go') {
    return [
      'Chair and full body are visible at the start.',
      'Walking path stays mostly in frame.',
      'Hips, knees, ankles, feet, and torso are visible.',
      'Camera is stable and the full sequence is recorded.',
    ];
  }

  return [
    'Full body is visible from the side.',
    'Hips, knees, ankles, and feet are visible.',
    'Chair and floor area are clearly visible.',
    'Camera is stable and not held in your hand.',
  ];
}

function getCameraSetupGuideForDailyTask(task: DailyTask) {
  if (task === 'reach') {
    return {
      title: 'Reach Camera Setup',
      bestAngle: 'Place the camera in front of you or slightly to the side so your full arm stays visible.',
      bodyParts: [
        'Shoulder',
        'Elbow',
        'Wrist',
        'Hand',
      ],
      setupSteps: [
        'Put the phone on a stable surface instead of holding it.',
        'Step back until your shoulder, elbow, wrist, and hand are all visible.',
        'Start with your arm relaxed and visible.',
        'Reach forward or outward without moving out of frame.',
        'Hold the final reach position briefly so endpoint steadiness can be measured.',
      ],
      commonMistakes: [
        'Only the hand is visible but the shoulder is cut off.',
        'The camera moves while recording.',
        'The arm leaves the frame during the reach.',
        'The room is too dark for reliable tracking.',
      ],
    };
  }

  if (task === 'arm_raise') {
    return {
      title: 'Arm Raise Camera Setup',
      bestAngle: 'Place the camera in front of you or slightly to the side so the full arm path is visible.',
      bodyParts: [
        'Shoulder',
        'Elbow',
        'Wrist',
        'Full arm path',
      ],
      setupSteps: [
        'Put the phone on a stable surface at chest height if possible.',
        'Step back until your whole arm is visible from lowered position to raised position.',
        'Keep your torso mostly still.',
        'Raise your arm slowly and lower it with control.',
        'Avoid turning sideways halfway through the recording.',
      ],
      commonMistakes: [
        'The wrist disappears at the top of the raise.',
        'The torso moves too much, making arm motion harder to isolate.',
        'The camera is too close.',
        'The arm moves too fast for a clean check.',
      ],
    };
  }

  if (task === 'walking') {
    return {
      title: 'Walking Camera Setup',
      bestAngle: 'Place the camera to the side so hips, knees, ankles, feet, and several steps are visible.',
      bodyParts: [
        'Hips',
        'Knees',
        'Ankles',
        'Feet',
        'Walking path',
      ],
      setupSteps: [
        'Put the phone on a stable surface instead of holding it.',
        'Use a side view if possible.',
        'Step back far enough so your full lower body stays visible.',
        'Walk naturally for several steps.',
        'Avoid walking out of frame before the recording ends.',
      ],
      commonMistakes: [
        'Feet are cut off.',
        'Camera is too close.',
        'Only one step is visible.',
        'Camera moves while recording.',
        'Walking path is angled too sharply away from the camera.',
      ],
    };
  }

  if (task === 'balance') {
    return {
      title: 'Balance Camera Setup',
      bestAngle: 'Place the camera far enough away so your full body stays visible while standing still.',
      bodyParts: [
        'Shoulders',
        'Hips',
        'Torso',
        'Knees',
        'Ankles',
        'Feet',
      ],
      setupSteps: [
        'Put the phone on a stable surface instead of holding it.',
        'Stand where your full body is visible.',
        'Face the camera or stand at a slight angle.',
        'Stand still in a comfortable position for the recording.',
        'Stop if you feel unsafe or unstable.',
      ],
      commonMistakes: [
        'Feet are cut off.',
        'Camera is handheld.',
        'Only the upper body is visible.',
        'Lighting is too dark.',
        'User steps out of frame during the check.',
      ],
    };
  }

  if (task === 'timed_up_and_go') {
    return {
      title: 'Timed Up and Go Camera Setup',
      bestAngle: 'Place the camera far enough away so the chair, full body, and walking path stay visible.',
      bodyParts: [
        'Chair',
        'Torso',
        'Hips',
        'Knees',
        'Ankles',
        'Feet',
        'Walking path',
      ],
      setupSteps: [
        'Put the phone on a stable surface instead of holding it.',
        'Place the chair and walking path in frame.',
        'Start seated with your full body visible.',
        'Stand up, walk a few steps, turn or return, then finish clearly in frame.',
        'Keep the movement safe and stop if you feel unstable.',
      ],
      commonMistakes: [
        'Chair is not visible.',
        'Feet are cut off.',
        'User walks completely out of frame.',
        'Camera is too close.',
        'The turn or return part is not recorded.',
      ],
    };
  }

  return {
    title: 'Sit-to-Stand Camera Setup',
    bestAngle: 'Place the camera to your side so your hips, knees, ankles, feet, chair, and torso are visible.',
    bodyParts: [
      'Torso',
      'Hips',
      'Knees',
      'Ankles',
      'Feet',
      'Chair',
    ],
    setupSteps: [
      'Put the phone on a stable surface to your side.',
      'Step or sit far enough away so your full body is visible.',
      'Make sure the chair and floor are visible.',
      'Start seated, then stand naturally.',
      'Keep your feet, knees, hips, and torso visible during the entire motion.',
    ],
    commonMistakes: [
      'Camera is in front instead of from the side.',
      'Feet or knees are cut off.',
      'Chair is not visible.',
      'Camera is handheld and shakes.',
    ],
  };
}

function getCameraSetupGuideForMode(
  mode: 'rep' | 'rehab' | 'lab' | 'daily',
  task: DailyTask
) {
  if (mode === 'daily') {
    return getCameraSetupGuideForDailyTask(task);
  }

  if (mode === 'rehab') {
    return {
      title: 'Rehab Camera Setup',
      bestAngle: 'Place the camera so the moving joint stays visible for the whole repeated movement.',
      bodyParts: [
        'Moving joint',
        'Nearby limb segments',
        'Full movement path',
      ],
      setupSteps: [
        'Choose one slow repeated movement.',
        'Put the camera on a stable surface.',
        'Make sure the moving joint stays visible the entire time.',
        'Repeat the same movement several times without changing the motion halfway through.',
        'Move slowly enough that each cycle is clear.',
      ],
      commonMistakes: [
        'Changing the movement halfway through the recording.',
        'Moving too fast.',
        'Only part of the moving joint is visible.',
        'Camera shakes because someone is holding it.',
      ],
    };
  }

  if (mode === 'rep') {
    return {
      title: 'Rep Quality Camera Setup',
      bestAngle: 'Place the camera so the repeated movement and main joint are clearly visible.',
      bodyParts: [
        'Main moving joint',
        'Upper limb segment',
        'Lower limb segment',
        'Full repeated motion path',
      ],
      setupSteps: [
        'Put the phone on a stable surface.',
        'Make sure the main joint does not leave the frame.',
        'Do several clear repeated movements.',
        'Avoid changing speed dramatically unless that is part of the test.',
        'Keep the camera angle consistent across tests.',
      ],
      commonMistakes: [
        'Only part of the joint is visible.',
        'Reps are too small to detect clearly.',
        'Camera moves during the recording.',
        'The movement changes from rep to rep.',
      ],
    };
  }

  return {
    title: 'Movement Lab Camera Setup',
    bestAngle: 'Use a stable camera angle that clearly shows the body region you want to analyze.',
    bodyParts: [
      'Tracked joint',
      'Connected body segments',
      'Full movement path',
    ],
    setupSteps: [
      'Decide which body region you want to study.',
      'Place the camera so the whole region stays visible.',
      'Use consistent lighting.',
      'Record a short clean trial first.',
      'Use the same camera angle when comparing sessions.',
    ],
    commonMistakes: [
      'Changing the camera angle between sessions.',
      'Recording too far away or too close.',
      'Poor lighting.',
      'Body parts leaving the frame.',
    ],
  };
}

function getDailyConfidence(result: AnalysisResult | null) {
  if (!result) {
    return {
      grade: 'N/A',
      message: 'No analysis available yet.',
      nextAction: 'Record a movement check to generate feedback.',
    };
  }

  const signalGrade = result.signal_quality?.grade || 'N/A';
  const taskType = result.task_analysis?.task_type;

  if (signalGrade === 'Poor') {
    return {
      grade: 'Low Confidence',
      message: 'The camera could not reliably track enough of the movement.',
      nextAction: 'Re-record with better lighting, a stable camera, and the needed body parts fully visible.',
    };
  }

  if (taskType === 'sit_to_stand') {
    const lowerGrade = result.task_analysis?.lower_body_signal_grade;

    if (lowerGrade === 'Poor') {
      return {
        grade: 'Low Confidence',
        message: 'The lower body was not tracked well enough for a reliable sit-to-stand result.',
        nextAction: 'Re-record from the side with hips, knees, ankles, and feet visible.',
      };
    }

    return {
      grade: signalGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Sit-to-stand tracking was usable for this analysis.',
      nextAction: 'Review rise stability, torso lean, and transition duration.',
    };
  }

  if (taskType === 'reach') {
    return {
      grade: signalGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Reach tracking was usable for this analysis.',
      nextAction: 'Review reach stability, smoothness, and endpoint steadiness.',
    };
  }

  if (taskType === 'arm_raise') {
    return {
      grade: signalGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Arm raise tracking was usable for this analysis.',
      nextAction: 'Review arm raise range, smoothness, and top-position steadiness.',
    };
  }

  if (taskType === 'walking') {
    const walkingGrade = result.task_analysis?.walking_signal_grade;

    if (walkingGrade === 'Poor') {
      return {
        grade: 'Low Confidence',
        message: 'Walking tracking was not reliable enough because the lower body was not visible enough.',
        nextAction: 'Re-record from the side with hips, knees, ankles, and feet visible for several steps.',
      };
    }

    return {
      grade: walkingGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Walking tracking was usable for this analysis.',
      nextAction: 'Review walking rhythm, hip motion, knee motion, and walking stability.',
    };
  }

  if (taskType === 'balance') {
    const balanceGrade = result.task_analysis?.balance_signal_grade;

    if (balanceGrade === 'Poor') {
      return {
        grade: 'Low Confidence',
        message: 'Balance tracking was not reliable enough because the full body was not visible enough.',
        nextAction: 'Re-record with your full body visible and the camera stable.',
      };
    }

    return {
      grade: balanceGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Balance tracking was usable for this analysis.',
      nextAction: 'Review sway, torso lean, posture stability, and signal quality.',
    };
  }

  if (taskType === 'timed_up_and_go') {
    const tugGrade = result.task_analysis?.tug_signal_grade;

    if (tugGrade === 'Poor') {
      return {
        grade: 'Low Confidence',
        message: 'Timed Up and Go tracking was not reliable enough because the full sequence was not visible enough.',
        nextAction: 'Re-record with the chair, full body, and walking path visible.',
      };
    }

    return {
      grade: tugGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
      message: 'Timed Up and Go tracking was usable for this analysis.',
      nextAction: 'Review sequence duration, path movement, return pattern, step cycles, and torso control.',
    };
  }

  return {
    grade: signalGrade === 'Good' ? 'High Confidence' : 'Moderate Confidence',
    message: 'Movement tracking was usable for this analysis.',
    nextAction: 'Review the result and repeat with a stable camera for comparison.',
  };
}

function getPrimaryTaskScore(result: AnalysisResult | null) {
  if (!result) {
    return {
      score: null,
      grade: 'N/A',
      label: 'Movement Score',
    };
  }

  const taskType = result.task_analysis?.task_type;

  if (taskType === 'sit_to_stand') {
    return {
      score: result.task_analysis?.rise_stability_score ?? null,
      grade: result.task_analysis?.rise_stability_grade || 'N/A',
      label: 'Rise Stability',
    };
  }

  if (taskType === 'reach') {
    return {
      score: result.task_analysis?.reach_stability_score ?? null,
      grade: result.task_analysis?.reach_stability_grade || 'N/A',
      label: 'Reach Stability',
    };
  }

  if (taskType === 'arm_raise') {
    return {
      score: result.task_analysis?.arm_raise_stability_score ?? null,
      grade: result.task_analysis?.arm_raise_stability_grade || 'N/A',
      label: 'Arm Raise Stability',
    };
  }

  if (taskType === 'walking') {
    return {
      score: result.task_analysis?.walking_stability_score ?? null,
      grade: result.task_analysis?.walking_stability_grade || 'N/A',
      label: 'Walking Stability',
    };
  }

  if (taskType === 'balance') {
    return {
      score: result.task_analysis?.balance_stability_score ?? null,
      grade: result.task_analysis?.balance_stability_grade || 'N/A',
      label: 'Balance Stability',
    };
  }

  if (taskType === 'timed_up_and_go') {
    return {
      score: result.task_analysis?.tug_mobility_score ?? null,
      grade: result.task_analysis?.tug_mobility_grade || 'N/A',
      label: 'Functional Mobility',
    };
  }

  return {
    score: result.movement_health_score ?? null,
    grade: result.movement_health_grade || 'N/A',
    label: 'Movement Health',
  };
}

function getScoreChangeText(currentScore: number | null, previousScore: number | null | undefined) {
  if (currentScore === null || previousScore === null || previousScore === undefined) {
    return 'Not enough data yet';
  }

  const change = currentScore - previousScore;

  if (Math.abs(change) < 0.1) {
    return 'No meaningful change';
  }

  if (change > 0) {
    return `Improved by ${change.toFixed(1)} points`;
  }

  return `Decreased by ${Math.abs(change).toFixed(1)} points`;
}

function formatSessionDate(timestamp: string) {
  const date = new Date(timestamp);

  return date.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
  }) + ' • ' + date.toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
}

function getSavedSessionsForTask(
  sessions: SavedSession[],
  task: DailyTask
) {
  return sessions.filter(
    (session) => session.mode === 'daily' && session.daily_task === task
  );
}

function getTaskTrendScores(
  sessions: SavedSession[],
  task: DailyTask
) {
  return getSavedSessionsForTask(sessions, task)
    .slice()
    .reverse()
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);
}

function getLatestSessionForTask(
  sessions: SavedSession[],
  task: DailyTask
) {
  return getSavedSessionsForTask(sessions, task)[0] || null;
}

function getSavedRehabSessions(sessions: SavedSession[]) {
  return sessions.filter((session) => session.mode === 'rehab');
}

function getRehabTrendScores(sessions: SavedSession[]) {
  return getSavedRehabSessions(sessions)
    .slice()
    .reverse()
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);
}

// Phase 3: Team Screening roster. Groups saved sessions by athlete_name
// (sessions with no name -- i.e. Team Screening was off -- are excluded
// entirely, so a coach's roster never mixes in the coach's own solo
// testing). `savedSessions` is already newest-first (see saveAnalysisSession
// above), so the first session found per athlete is their most recent.
function isLowMovementGrade(grade: string) {
  const lower = grade.toLowerCase();
  return (
    lower.includes('poor') ||
    lower.includes('needs work') ||
    lower.includes('recheck needed') ||
    lower.includes('decline')
  );
}

function getTeamRoster(sessions: SavedSession[]) {
  const named = sessions.filter(
    (session): session is SavedSession & { athlete_name: string } =>
      !!session.athlete_name && session.athlete_name.trim().length > 0
  );

  const athleteNames = [...new Set(named.map((session) => session.athlete_name))];

  return athleteNames
    .map((name) => {
      const athleteSessions = named.filter((session) => session.athlete_name === name);
      const latestSession = athleteSessions[0];

      return {
        name,
        sessionCount: athleteSessions.length,
        latestSession,
        flagged: isLowMovementGrade(latestSession.primary_grade),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

// --- Consistency streak + badges (Phase 4: startup-shaped, honest gamification) ---
//
// Design intent: reward showing up, not gaming a score. A streak counts a
// calendar day as "done" the moment ANY session is saved that day -- one
// quick daily check and a full workout both count equally, so there's no
// incentive to over-record. Badges are similarly all effort/consistency
// based (never score-based), so nobody is rewarded for gaming a high score
// or penalized for a low one -- that would directly undercut the app's own
// honesty-first design elsewhere (Transparency screen, calibration, etc).

function getLocalDateKey(timestamp: string) {
  const date = new Date(timestamp);
  // Local calendar day, not UTC -- a session at 11pm and one at 1am the
  // "same night" should usually count as two different days for a streak,
  // matching how a person actually experiences "did I check in today".
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

function getConsistencyStreaks(sessions: SavedSession[]) {
  if (sessions.length === 0) {
    return { currentStreak: 0, longestStreak: 0, activeToday: false };
  }

  const uniqueDayKeys = [...new Set(sessions.map((s) => getLocalDateKey(s.timestamp)))];
  const dayTimestamps = uniqueDayKeys
    .map((key) => {
      const [year, month, day] = key.split('-').map(Number);
      return new Date(year, month, day).getTime();
    })
    .sort((a, b) => b - a); // newest first

  const ONE_DAY_MS = 24 * 60 * 60 * 1000;
  const todayKey = getLocalDateKey(new Date().toISOString());
  const activeToday = uniqueDayKeys.includes(todayKey);

  // Current streak: walk backward from today (or yesterday, if today has no
  // check-in yet but yesterday's streak is still "alive" until today ends).
  let currentStreak = 0;
  const startTime = dayTimestamps[0];
  const mostRecentGapDays = Math.round((new Date().setHours(0, 0, 0, 0) - startTime) / ONE_DAY_MS);

  if (mostRecentGapDays <= 1) {
    currentStreak = 1;
    for (let i = 1; i < dayTimestamps.length; i += 1) {
      const gap = Math.round((dayTimestamps[i - 1] - dayTimestamps[i]) / ONE_DAY_MS);
      if (gap === 1) {
        currentStreak += 1;
      } else {
        break;
      }
    }
  }

  // Longest streak ever, across all history.
  let longestStreak = 1;
  let running = 1;
  for (let i = 1; i < dayTimestamps.length; i += 1) {
    const gap = Math.round((dayTimestamps[i - 1] - dayTimestamps[i]) / ONE_DAY_MS);
    if (gap === 1) {
      running += 1;
    } else {
      longestStreak = Math.max(longestStreak, running);
      running = 1;
    }
  }
  longestStreak = Math.max(longestStreak, running);

  return { currentStreak, longestStreak, activeToday };
}

type Badge = {
  id: string;
  label: string;
  description: string;
  earned: boolean;
};

function getEarnedBadges(
  sessions: SavedSession[],
  hasAnyCalibration: boolean
): Badge[] {
  const { longestStreak } = getConsistencyStreaks(sessions);
  const dailyTasks: DailyTask[] = [
    'reach',
    'arm_raise',
    'sit_to_stand',
    'walking',
    'balance',
    'timed_up_and_go',
  ];
  const completedTaskCount = dailyTasks.filter(
    (task) => getSavedSessionsForTask(sessions, task).length > 0
  ).length;
  const hasLeftSide = sessions.some((s) => s.side === 'left');
  const hasRightSide = sessions.some((s) => s.side === 'right' || !s.side);
  const rehabCount = getSavedRehabSessions(sessions).length;
  const hasTeamScreening = sessions.some((s) => !!s.athlete_name);

  return [
    {
      id: 'first_checkin',
      label: 'First Check-In',
      description: 'Saved your first movement check.',
      earned: sessions.length >= 1,
    },
    {
      id: 'streak_3',
      label: '3-Day Streak',
      description: 'Checked in three days in a row.',
      earned: longestStreak >= 3,
    },
    {
      id: 'streak_7',
      label: '7-Day Streak',
      description: 'Checked in seven days in a row.',
      earned: longestStreak >= 7,
    },
    {
      id: 'calibrated',
      label: 'Personalized',
      description: 'Calibrated rep detection to your own range of motion.',
      earned: hasAnyCalibration,
    },
    {
      id: 'both_sides',
      label: 'Both Sides Tested',
      description: 'Recorded a check for both your left and right side.',
      earned: hasLeftSide && hasRightSide,
    },
    {
      id: 'full_battery',
      label: 'Full Battery',
      description: 'Tried all six daily movement checks at least once.',
      earned: completedTaskCount >= dailyTasks.length,
    },
    {
      id: 'rehab_consistency',
      label: 'Rehab Regular',
      description: 'Logged five or more rehab consistency checks.',
      earned: rehabCount >= 5,
    },
    {
      id: 'team_screener',
      label: 'Team Screener',
      description: 'Used Team Screening to check in at least one athlete.',
      earned: hasTeamScreening,
    },
  ];
}

function getLatestRehabSession(sessions: SavedSession[]) {
  return getSavedRehabSessions(sessions)[0] || null;
}

function getBaselineForTask(
  sessions: SavedSession[],
  task: DailyTask
) {
  const scores = getSavedSessionsForTask(sessions, task)
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);

  if (scores.length < 3) {
    return {
      baselineScore: null,
      sessionCount: scores.length,
      minScore: null,
      maxScore: null,
    };
  }

  const total = scores.reduce((sum, score) => sum + score, 0);
  const baselineScore = total / scores.length;

  return {
    baselineScore: Number(baselineScore.toFixed(1)),
    sessionCount: scores.length,
    minScore: Math.min(...scores),
    maxScore: Math.max(...scores),
  };
}

function getBaselineInterpretation(
  result: AnalysisResult | null,
  sessions: SavedSession[]
) {
  if (!result || result.mode !== 'daily' || !result.daily_task) {
    return {
      status: 'No Baseline Yet',
      summary: 'Baseline comparison is available for Daily movement checks.',
      detail: 'Record Daily checks over time to build a personal baseline.',
      nextStep: 'Complete a Daily movement check first.',
      baselineScore: null,
      difference: null,
      sessionCount: 0,
    };
  }

  const primary = getPrimaryTaskScore(result);
  const currentScore = primary.score;
  const task = result.daily_task;
  const taskLabel = result.daily_task_label || getDailyTaskLabel(task);
  const baseline = getBaselineForTask(sessions, task);

  if (currentScore === null) {
    return {
      status: 'No Score Available',
      summary: `${taskLabel} baseline could not be compared because this check does not have a valid score.`,
      detail: 'This usually happens when movement tracking quality is too low or the task was not clearly captured.',
      nextStep: 'Re-record with the needed body parts visible and a stable camera angle.',
      baselineScore: baseline.baselineScore,
      difference: null,
      sessionCount: baseline.sessionCount,
    };
  }

  if (baseline.baselineScore === null) {
    const remaining = Math.max(0, 3 - baseline.sessionCount);

    return {
      status: 'Building Baseline',
      summary: `${taskLabel} needs ${remaining} more saved check${remaining === 1 ? '' : 's'} to build a personal baseline.`,
      detail: 'The app uses at least 3 saved checks of the same movement to estimate your normal range.',
      nextStep: `Repeat the ${taskLabel.toLowerCase()} check with a similar camera setup.`,
      baselineScore: null,
      difference: null,
      sessionCount: baseline.sessionCount,
    };
  }

  const difference = Number((currentScore - baseline.baselineScore).toFixed(1));
  const absDifference = Math.abs(difference);

  if (absDifference < 4) {
    return {
      status: 'Within Normal Range',
      summary: `${taskLabel} is close to your personal baseline.`,
      detail: `Today’s score is ${Math.abs(difference).toFixed(1)} points from your usual score, which is a small difference.`,
      nextStep: 'Keep recording with the same setup to make your baseline more reliable.',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
    };
  }

  if (difference >= 8) {
    return {
      status: 'Above Baseline',
      summary: `${taskLabel} is clearly above your personal baseline today.`,
      detail: `Today’s score is ${difference.toFixed(1)} points higher than your usual score.`,
      nextStep: 'Repeat this check later to see whether this improvement stays consistent.',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
    };
  }

  if (difference >= 4) {
    return {
      status: 'Slightly Above Baseline',
      summary: `${taskLabel} is slightly above your personal baseline today.`,
      detail: `Today’s score is ${difference.toFixed(1)} points higher than your usual score.`,
      nextStep: 'Keep tracking this movement to confirm whether the improvement continues.',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
    };
  }

  if (difference <= -8) {
    return {
      status: 'Below Baseline',
      summary: `${taskLabel} is clearly below your personal baseline today.`,
      detail: `Today’s score is ${Math.abs(difference).toFixed(1)} points lower than your usual score. This could reflect movement change or recording quality.`,
      nextStep: 'Re-test with the same camera setup before treating this as a real drop.',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
    };
  }

  return {
    status: 'Slightly Below Baseline',
    summary: `${taskLabel} is slightly below your personal baseline today.`,
    detail: `Today’s score is ${Math.abs(difference).toFixed(1)} points lower than your usual score.`,
    nextStep: 'Repeat the check later with the same setup to see if the drop continues.',
    baselineScore: baseline.baselineScore,
    difference,
    sessionCount: baseline.sessionCount,
  };
}

function getDailyInsightV2(result: AnalysisResult | null, previousSession: SavedSession | null) {
  if (!result) {
    return {
      assessment: 'No Analysis Yet',
      strength: 'Record a movement check to generate an assessment.',
      limitation: 'No limitation available yet.',
      recommendedNextCheck: 'Complete a Daily movement check first.',
    };
  }

  const confidence = getDailyConfidence(result);
  const primary = getPrimaryTaskScore(result);
  const taskType = result.task_analysis?.task_type;
  const score = primary.score;

  if (confidence.grade === 'Low Confidence') {
    return {
      assessment: 'Retest Recommended',
      strength: 'The app detected some movement information, but the recording was not reliable enough.',
      limitation: 'Camera tracking quality was too low to trust the result confidently.',
      recommendedNextCheck: confidence.nextAction,
    };
  }

  let assessment = 'Movement Check Complete';

  if (score !== null) {
    if (score >= 85) {
      assessment = 'Excellent';
    } else if (score >= 70) {
      assessment = 'Good';
    } else if (score >= 55) {
      assessment = 'Needs Improvement';
    } else {
      assessment = 'Retest Recommended';
    }
  }

  let strength = 'Movement data was captured successfully.';
  let limitation = 'No major limitation was detected from the current result.';
  let recommendedNextCheck = 'Repeat this same check later using a similar camera setup.';

  if (taskType === 'reach') {
    const smoothness = result.task_analysis?.reach_smoothness;
    const steadiness = result.task_analysis?.endpoint_steadiness;
    const range = result.task_analysis?.reach_range;

    if (smoothness !== null && smoothness !== undefined && smoothness <= 0.03) {
      strength = 'Reach motion was relatively smooth and controlled.';
    } else if (steadiness !== null && steadiness !== undefined && steadiness <= 0.03) {
      strength = 'The hand stayed relatively steady near the end of the reach.';
    } else if (range !== null && range !== undefined && range >= 0.15) {
      strength = 'The app detected a clear reaching movement.';
    }

    if (steadiness !== null && steadiness !== undefined && steadiness > 0.03) {
      limitation = 'The hand became less steady near the end of the reach.';
    } else if (smoothness !== null && smoothness !== undefined && smoothness > 0.03) {
      limitation = 'Reach motion showed noticeable variability.';
    } else if (range !== null && range !== undefined && range < 0.15) {
      limitation = 'Reach range was limited or not clearly captured.';
    }

    recommendedNextCheck = 'Repeat the reach check with the same camera angle and try to hold the final reach position briefly.';
  }

  if (taskType === 'arm_raise') {
    const range = result.task_analysis?.arm_raise_range;
    const smoothness = result.task_analysis?.arm_raise_smoothness;
    const topSteadiness = result.task_analysis?.top_steadiness;

    if (topSteadiness !== null && topSteadiness !== undefined && topSteadiness <= 0.025) {
      strength = 'The arm stayed relatively steady near the top of the raise.';
    } else if (smoothness !== null && smoothness !== undefined && smoothness <= 0.025) {
      strength = 'Arm raise motion was smooth and controlled.';
    } else if (range !== null && range !== undefined && range >= 0.12) {
      strength = 'The app detected a clear arm raise range.';
    }

    if (topSteadiness !== null && topSteadiness !== undefined && topSteadiness > 0.025) {
      limitation = 'The arm became less steady near the top of the raise.';
    } else if (smoothness !== null && smoothness !== undefined && smoothness > 0.025) {
      limitation = 'Arm raise motion showed noticeable variability.';
    } else if (range !== null && range !== undefined && range < 0.12) {
      limitation = 'Arm raise range was limited or not clearly captured.';
    }

    recommendedNextCheck = 'Repeat the arm raise check with your shoulder, elbow, and wrist visible for the entire motion.';
  }

  if (taskType === 'walking') {
    const stepCycles = result.task_analysis?.estimated_step_cycles;
    const cadence = result.task_analysis?.cadence_estimate;
    const hipVariability = result.task_analysis?.hip_vertical_variability;
    const rhythmVariability = result.task_analysis?.step_rhythm_variability;
    const kneeDifference = result.task_analysis?.knee_range_difference;

    if (stepCycles !== null && stepCycles !== undefined && stepCycles >= 2) {
      strength = 'The app detected multiple walking cycles.';
    }

    if (cadence !== null && cadence !== undefined) {
      strength = `Estimated cadence was ${cadence.toFixed(1)} steps per minute.`;
    }

    if (rhythmVariability !== null && rhythmVariability !== undefined && rhythmVariability > 0.04) {
      limitation = 'Walking rhythm varied noticeably during the recording.';
    } else if (hipVariability !== null && hipVariability !== undefined && hipVariability > 0.04) {
      limitation = 'Hip height varied noticeably during walking.';
    } else if (kneeDifference !== null && kneeDifference !== undefined && kneeDifference > 15) {
      limitation = 'Left and right knee motion ranges differed noticeably.';
    } else if (stepCycles !== null && stepCycles !== undefined && stepCycles < 2) {
      limitation = 'Only a small number of walking cycles were detected.';
    }

    recommendedNextCheck = 'Repeat the walking check from the side with hips, knees, ankles, feet, and several steps visible.';
  }

  if (taskType === 'balance') {
    const hipSway = result.task_analysis?.hip_sway_range;
    const shoulderSway = result.task_analysis?.shoulder_sway_range;
    const torsoVariability = result.task_analysis?.torso_lean_variability;
    const torsoMax = result.task_analysis?.torso_lean_max;
    const ankleDrift = result.task_analysis?.ankle_center_drift;

    if (hipSway !== null && hipSway !== undefined && hipSway <= 0.06) {
      strength = 'Hip position stayed relatively steady during the balance check.';
    }

    if (shoulderSway !== null && shoulderSway !== undefined && shoulderSway <= 0.06) {
      strength = 'Shoulder position stayed relatively steady during the balance check.';
    }

    if (torsoVariability !== null && torsoVariability !== undefined && torsoVariability > 5) {
      limitation = 'Torso lean varied noticeably during the balance check.';
    } else if (hipSway !== null && hipSway !== undefined && hipSway > 0.06) {
      limitation = 'Hip center shifted noticeably during the balance check.';
    } else if (shoulderSway !== null && shoulderSway !== undefined && shoulderSway > 0.06) {
      limitation = 'Shoulder position drifted noticeably during the balance check.';
    } else if (torsoMax !== null && torsoMax !== undefined && torsoMax > 25) {
      limitation = 'Maximum torso lean was high during the check.';
    } else if (ankleDrift !== null && ankleDrift !== undefined && ankleDrift > 0.04) {
      limitation = 'Foot or ankle center appeared to drift during the check.';
    }

    recommendedNextCheck = 'Repeat the balance check with full body visible, stable camera, and the same standing position.';
  }

  if (taskType === 'timed_up_and_go') {
    const duration = result.task_analysis?.tug_duration_sec;
    const pathRange = result.task_analysis?.tug_path_range;
    const returnDetected = result.task_analysis?.tug_return_pattern_detected;
    const stepCycles = result.task_analysis?.tug_estimated_step_cycles;
    const torsoMax = result.task_analysis?.tug_torso_lean_max;
    const torsoVariability = result.task_analysis?.tug_torso_lean_variability;

    if (returnDetected) {
      strength = 'The app detected a return or direction-change pattern during the sequence.';
    }

    if (stepCycles !== null && stepCycles !== undefined && stepCycles >= 2) {
      strength = 'The app detected multiple walking cycles during the sequence.';
    }

    if (duration !== null && duration !== undefined) {
      strength = `Visible sequence duration was about ${duration.toFixed(1)} seconds.`;
    }

    if (pathRange !== null && pathRange !== undefined && pathRange < 0.08) {
      limitation = 'Body path movement was small, so the walking/return portion may not have been fully captured.';
    } else if (!returnDetected) {
      limitation = 'A clear turn or return pattern was not detected.';
    } else if (stepCycles !== null && stepCycles !== undefined && stepCycles < 2) {
      limitation = 'Only a small number of walking cycles were detected.';
    } else if (torsoMax !== null && torsoMax !== undefined && torsoMax > 35) {
      limitation = 'Torso lean was high during the sequence.';
    } else if (torsoVariability !== null && torsoVariability !== undefined && torsoVariability > 6) {
      limitation = 'Torso motion varied noticeably during the sequence.';
    }

    recommendedNextCheck = 'Repeat the Timed Up and Go check with the chair, full body, walking path, and return/turn visible.';
  }

  if (taskType === 'sit_to_stand') {
    const transitionDetected = result.task_analysis?.transition_detected;
    const torsoLean = result.task_analysis?.max_torso_lean;
    const torsoControl = result.task_analysis?.torso_control;
    const duration = result.task_analysis?.transition_duration_sec;

    if (transitionDetected) {
      strength = 'A sit-to-stand transition was detected and measured.';
    }

    if (duration !== null && duration !== undefined && duration >= 0.5 && duration <= 4.0) {
      strength = 'Transition timing was within a reasonable controlled range.';
    }

    if (torsoLean !== null && torsoLean !== undefined && torsoLean > 35) {
      limitation = 'Torso lean was high during the rise.';
    } else if (torsoControl !== null && torsoControl !== undefined && torsoControl > 8) {
      limitation = 'Torso motion varied noticeably during the transition.';
    } else if (!transitionDetected) {
      limitation = 'The sit-to-stand transition was not clearly detected.';
    }

    recommendedNextCheck = 'Repeat the sit-to-stand check from the side with hips, knees, ankles, and feet visible.';
  }

  if (previousSession && previousSession.primary_score !== null && score !== null) {
    const change = score - previousSession.primary_score;

    if (change > 3) {
      recommendedNextCheck = 'Repeat this check again later to confirm whether the improvement is consistent.';
    } else if (change < -3) {
      recommendedNextCheck = 'Repeat this check with the same setup to confirm whether the drop was real or caused by recording quality.';
    }
  }

  return {
    assessment,
    strength,
    limitation,
    recommendedNextCheck,
  };
}

function getTrendInterpretation(
  result: AnalysisResult | null,
  previousSession: SavedSession | null
) {
  if (!result) {
    return {
      status: 'No Trend Yet',
      summary: 'Analyze a movement check to begin tracking change.',
      detail: 'Trend interpretation appears after the app has a current result to compare.',
      nextStep: 'Record a Daily movement check first.',
    };
  }


  const primary = getPrimaryTaskScore(result);
  const currentScore = primary.score;
  const previousScore = previousSession?.primary_score;
  const taskLabel = result.daily_task_label || 'this movement';

  if (
    !previousSession ||
    previousScore === null ||
    previousScore === undefined ||
    currentScore === null
  ) {
    return {
      status: 'No Previous Check Yet',
      summary: `This is the first saved ${taskLabel.toLowerCase()} check available for comparison.`,
      detail: 'The app needs at least two saved checks of the same movement to interpret a trend.',
      nextStep: `Repeat the ${taskLabel.toLowerCase()} check later using a similar camera angle.`,
    };
  }

  const change = currentScore - previousScore;
  const absChange = Math.abs(change);

  if (absChange < 3) {
    return {
      status: 'Stable',
      summary: `${taskLabel} stayed about the same compared with your previous check.`,
      detail: 'The score changed only slightly, so this may just reflect normal recording variation.',
      nextStep: 'Keep using the same camera setup so future comparisons are more meaningful.',
    };
  }

  if (change >= 8) {
    return {
      status: 'Strong Improvement',
      summary: `${taskLabel} improved clearly compared with your previous check.`,
      detail: `The score increased by ${change.toFixed(1)} points, which is large enough to be worth monitoring.`,
      nextStep: 'Repeat this check again later to confirm the improvement is consistent.',
    };
  }

  if (change >= 3) {
    return {
      status: 'Slight Improvement',
      summary: `${taskLabel} improved slightly compared with your previous check.`,
      detail: `The score increased by ${change.toFixed(1)} points. This may be real improvement, but it should be confirmed with more checks.`,
      nextStep: 'Repeat the same movement with the same camera angle to see if the trend continues.',
    };
  }

  if (change <= -8) {
    return {
      status: 'Clear Decline',
      summary: `${taskLabel} decreased clearly compared with your previous check.`,
      detail: `The score dropped by ${absChange.toFixed(1)} points. This may reflect worse movement quality or a poorer recording setup.`,
      nextStep: 'Re-test with stable lighting and the same camera angle before treating this as a real decline.',
    };
  }

  return {
    status: 'Slight Decline',
    summary: `${taskLabel} decreased slightly compared with your previous check.`,
    detail: `The score dropped by ${absChange.toFixed(1)} points. Small drops can happen from camera angle, lighting, or natural movement variation.`,
    nextStep: 'Repeat the check later with the same setup to see if the decrease continues.',
  };
}

function getRehabConsistencyAnalysis(
  result: AnalysisResult | null,
  previousSession: SavedSession | null
) {
  if (!result) {
    return {
      score: null,
      grade: 'N/A',
      status: 'No Analysis Yet',
      repeatability: 'N/A',
      controlStability: 'N/A',
      summary: 'Record a controlled movement to generate a rehab consistency result.',
      limitation: 'No movement data is available yet.',
      previousComparison: 'No previous rehab check available yet.',
      nextStep: 'Record a slow, controlled movement with a stable camera angle.',
    };
  }

  const validRepScores = result.rep_analysis
    .map((rep) => rep.score)
    .filter((score): score is number => score !== null);

  const validControls = result.rep_analysis
    .map((rep) => rep.control)
    .filter((control): control is number => control !== null);

  if (validRepScores.length === 0) {
    return {
      score: result.movement_health_score ?? null,
      grade: 'Limited Data',
      status: 'Retest Recommended',
      repeatability: 'Not enough controlled cycles detected.',
      controlStability: 'Not enough control data available.',
      summary: 'The app could not detect enough repeatable movement cycles for a reliable consistency check.',
      limitation: 'Movement may have been too small, too fast, unclear, or partly out of frame.',
      previousComparison: 'No reliable current score available for comparison.',
      nextStep: 'Re-record with a larger but comfortable range of motion and keep the moving body part visible.',
    };
  }

  const averageScore =
    validRepScores.reduce((sum, score) => sum + score, 0) / validRepScores.length;

  const scoreVariance =
    validRepScores.reduce((sum, score) => {
      return sum + Math.pow(score - averageScore, 2);
    }, 0) / validRepScores.length;

  const scoreStd = Math.sqrt(scoreVariance);

  const averageControl =
    validControls.length > 0
      ? validControls.reduce((sum, value) => sum + value, 0) / validControls.length
      : null;

  let consistencyScore = averageScore;

  consistencyScore -= Math.min(25, scoreStd * 1.5);

  if (validRepScores.length < 3) {
    consistencyScore -= 10;
  }

  if (averageControl !== null && averageControl > 0.7) {
    consistencyScore -= 10;
  }

  consistencyScore = Math.max(0, Math.min(100, Number(consistencyScore.toFixed(1))));

  let grade = 'Needs Work';

  if (consistencyScore >= 85) {
    grade = 'Excellent';
  } else if (consistencyScore >= 70) {
    grade = 'Good';
  } else if (consistencyScore >= 55) {
    grade = 'Moderate';
  }

  let status = 'Consistency Needs Work';

  if (consistencyScore >= 85) {
    status = 'Highly Consistent';
  } else if (consistencyScore >= 70) {
    status = 'Mostly Consistent';
  } else if (consistencyScore >= 55) {
    status = 'Somewhat Consistent';
  }

  let repeatability = 'Movement cycles were detected and compared.';

  if (scoreStd < 6 && validRepScores.length >= 3) {
    repeatability = 'Movement cycles were highly repeatable across the recording.';
  } else if (scoreStd < 12) {
    repeatability = 'Movement cycles were reasonably repeatable, with some variation.';
  } else {
    repeatability = 'Movement cycles varied noticeably from one cycle to another.';
  }

  let controlStability = 'Control stability could not be fully determined.';

  if (averageControl !== null) {
    if (averageControl <= 0.35) {
      controlStability = 'Control stayed stable across the movement.';
    } else if (averageControl <= 0.7) {
      controlStability = 'Control was usable, but some instability appeared.';
    } else {
      controlStability = 'Control was inconsistent, suggesting unstable movement execution.';
    }
  }

  let limitation = 'No major consistency limitation was detected.';

  if (validRepScores.length < 3) {
    limitation = 'Only a small number of cycles were detected, so consistency is less certain.';
  } else if (scoreStd >= 12) {
    limitation = 'The biggest limitation was variation between movement cycles.';
  } else if (averageControl !== null && averageControl > 0.7) {
    limitation = 'The biggest limitation was unstable control during the movement.';
  }

  let previousComparison = 'No previous rehab check available yet.';

  if (previousSession && previousSession.primary_score !== null) {
    const change = consistencyScore - previousSession.primary_score;

    if (Math.abs(change) < 3) {
      previousComparison = 'Consistency stayed about the same compared with your previous rehab check.';
    } else if (change > 0) {
      previousComparison = `Consistency improved by ${change.toFixed(1)} points compared with your previous rehab check.`;
    } else {
      previousComparison = `Consistency decreased by ${Math.abs(change).toFixed(1)} points compared with your previous rehab check.`;
    }
  }

  return {
    score: consistencyScore,
    grade,
    status,
    repeatability,
    controlStability,
    summary: `Rehab consistency was calculated from ${validRepScores.length} detected movement cycle${validRepScores.length === 1 ? '' : 's'}.`,
    limitation,
    previousComparison,
    nextStep: 'Repeat the same controlled movement later with the same camera angle to track consistency over time.',
  };
}

function getRehabBaseline(sessions: SavedSession[]) {
  const scores = getSavedRehabSessions(sessions)
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);

  if (scores.length < 3) {
    return {
      baselineScore: null,
      sessionCount: scores.length,
      minScore: null,
      maxScore: null,
    };
  }

  const total = scores.reduce((sum, score) => sum + score, 0);
  const baselineScore = total / scores.length;

  return {
    baselineScore: Number(baselineScore.toFixed(1)),
    sessionCount: scores.length,
    minScore: Math.min(...scores),
    maxScore: Math.max(...scores),
  };
}

function getRehabBaselineInterpretation(
  result: AnalysisResult | null,
  sessions: SavedSession[]
) {
  if (!result || result.mode !== 'rehab') {
    return {
      status: 'No Rehab Baseline Yet',
      baselineScore: null,
      difference: null,
      sessionCount: 0,
      summary: 'Rehab baseline comparison is available after recording Rehab checks.',
      detail: 'The app needs repeated Rehab checks to estimate your normal consistency level.',
      nextStep: 'Complete a Rehab consistency check first.',
    };
  }

  const rehabAnalysis = getRehabConsistencyAnalysis(result, null);
  const currentScore = rehabAnalysis.score;
  const baseline = getRehabBaseline(sessions);

  if (currentScore === null) {
    return {
      status: 'No Rehab Score Available',
      baselineScore: baseline.baselineScore,
      difference: null,
      sessionCount: baseline.sessionCount,
      summary: 'This Rehab check does not have a valid consistency score.',
      detail: 'This usually happens when not enough repeatable movement cycles were detected.',
      nextStep: 'Re-record a slow repeated movement with the moving joint clearly visible.',
    };
  }

  if (baseline.baselineScore === null) {
    const remaining = Math.max(0, 3 - baseline.sessionCount);

    return {
      status: 'Building Rehab Baseline',
      baselineScore: null,
      difference: null,
      sessionCount: baseline.sessionCount,
      summary: `You need ${remaining} more Rehab check${remaining === 1 ? '' : 's'} to build a consistency baseline.`,
      detail: 'The app uses at least 3 saved Rehab checks to estimate your normal controlled-movement consistency.',
      nextStep: 'Repeat the same type of controlled movement with a similar camera setup.',
    };
  }

  const difference = Number((currentScore - baseline.baselineScore).toFixed(1));
  const absDifference = Math.abs(difference);

  if (absDifference < 4) {
    return {
      status: 'Within Rehab Baseline',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
      summary: 'Today’s consistency is close to your Rehab baseline.',
      detail: `Today’s score is ${absDifference.toFixed(1)} points from your usual Rehab consistency score.`,
      nextStep: 'Keep recording with the same setup to make your Rehab baseline more reliable.',
    };
  }

  if (difference >= 8) {
    return {
      status: 'Above Rehab Baseline',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
      summary: 'Today’s consistency is clearly above your Rehab baseline.',
      detail: `Today’s score is ${difference.toFixed(1)} points higher than your usual Rehab consistency score.`,
      nextStep: 'Repeat this Rehab check later to confirm the improvement is consistent.',
    };
  }

  if (difference >= 4) {
    return {
      status: 'Slightly Above Rehab Baseline',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
      summary: 'Today’s consistency is slightly above your Rehab baseline.',
      detail: `Today’s score is ${difference.toFixed(1)} points higher than your usual Rehab consistency score.`,
      nextStep: 'Keep tracking to confirm whether consistency is improving over time.',
    };
  }

  if (difference <= -8) {
    return {
      status: 'Below Rehab Baseline',
      baselineScore: baseline.baselineScore,
      difference,
      sessionCount: baseline.sessionCount,
      summary: 'Today’s consistency is clearly below your Rehab baseline.',
      detail: `Today’s score is ${absDifference.toFixed(1)} points lower than your usual Rehab consistency score.`,
      nextStep: 'Re-test with the same camera angle before treating this as a real drop.',
    };
  }

  return {
    status: 'Slightly Below Rehab Baseline',
    baselineScore: baseline.baselineScore,
    difference,
    sessionCount: baseline.sessionCount,
    summary: 'Today’s consistency is slightly below your Rehab baseline.',
    detail: `Today’s score is ${absDifference.toFixed(1)} points lower than your usual Rehab consistency score.`,
    nextStep: 'Repeat the check later with the same setup to see if the decrease continues.',
  };
}

function getRehabTrendInterpretation(
  result: AnalysisResult | null,
  previousSession: SavedSession | null
) {
  if (!result || result.mode !== 'rehab') {
    return {
      status: 'No Rehab Trend Yet',
      summary: 'Rehab trend interpretation appears after a Rehab check is analyzed.',
      detail: 'The app needs a current Rehab result to compare consistency over time.',
      nextStep: 'Record a Rehab consistency check first.',
    };
  }

  const rehabAnalysis = getRehabConsistencyAnalysis(result, previousSession);
  const currentScore = rehabAnalysis.score;
  const previousScore = previousSession?.primary_score;

  if (!previousSession || previousScore === null || previousScore === undefined || currentScore === null) {
    return {
      status: 'No Previous Rehab Check Yet',
      summary: 'This is the first saved Rehab check available for comparison.',
      detail: 'The app needs at least two Rehab checks to interpret consistency change.',
      nextStep: 'Repeat Rehab mode later using the same type of controlled movement.',
    };
  }

  const change = Number((currentScore - previousScore).toFixed(1));
  const absChange = Math.abs(change);

  if (absChange < 3) {
    return {
      status: 'Consistency Stable',
      summary: 'Rehab consistency stayed about the same compared with the previous check.',
      detail: 'The score changed only slightly, so this may reflect normal recording or movement variation.',
      nextStep: 'Keep using the same movement and camera setup to track consistency more accurately.',
    };
  }

  if (change >= 8) {
    return {
      status: 'Consistency Strongly Improving',
      summary: 'Rehab consistency improved clearly compared with the previous check.',
      detail: `The consistency score increased by ${change.toFixed(1)} points, which is a meaningful improvement.`,
      nextStep: 'Repeat the same Rehab check later to confirm the improvement is consistent.',
    };
  }

  if (change >= 3) {
    return {
      status: 'Consistency Improving',
      summary: 'Rehab consistency improved slightly compared with the previous check.',
      detail: `The consistency score increased by ${change.toFixed(1)} points.`,
      nextStep: 'Keep tracking this movement to see whether the improvement continues.',
    };
  }

  if (change <= -8) {
    return {
      status: 'Consistency Clearly Decreasing',
      summary: 'Rehab consistency dropped clearly compared with the previous check.',
      detail: `The consistency score decreased by ${absChange.toFixed(1)} points. This could reflect less repeatable movement or a worse recording setup.`,
      nextStep: 'Re-test with the same camera angle and movement before treating this as a real decline.',
    };
  }

  return {
    status: 'Consistency Slightly Decreasing',
    summary: 'Rehab consistency decreased slightly compared with the previous check.',
    detail: `The consistency score decreased by ${absChange.toFixed(1)} points.`,
    nextStep: 'Repeat the check later to see whether the decrease continues.',
  };
}

function getRehabSessionSummary(
  result: AnalysisResult | null,
  previousSession: SavedSession | null,
  sessions: SavedSession[]
) {
  if (!result || result.mode !== 'rehab') {
    return {
      headline: 'No Rehab Summary Yet',
      mainMessage: 'Record a Rehab check to generate a consistency summary.',
      focusArea: 'No focus area available yet.',
      nextCheck: 'Complete a Rehab consistency check first.',
    };
  }

  const rehabAnalysis = getRehabConsistencyAnalysis(result, previousSession);
  const baseline = getRehabBaselineInterpretation(result, sessions);
  const trend = getRehabTrendInterpretation(result, previousSession);

  let headline = rehabAnalysis.status;
  let mainMessage = rehabAnalysis.summary;
  let focusArea = rehabAnalysis.limitation;
  let nextCheck = rehabAnalysis.nextStep;

  if (rehabAnalysis.score === null) {
    return {
      headline: 'Retest Recommended',
      mainMessage: 'This Rehab check did not produce enough reliable consistency data.',
      focusArea: rehabAnalysis.limitation,
      nextCheck: rehabAnalysis.nextStep,
    };
  }

  if (
    trend.status === 'Consistency Strongly Improving' ||
    trend.status === 'Consistency Improving'
  ) {
    headline = 'Consistency Improving';
    mainMessage = trend.summary;
    nextCheck = trend.nextStep;
  }

  if (
    trend.status === 'Consistency Clearly Decreasing' ||
    trend.status === 'Consistency Slightly Decreasing'
  ) {
    headline = 'Consistency Decreasing';
    mainMessage = trend.summary;
    nextCheck = trend.nextStep;
  }

  if (
    baseline.status === 'Below Rehab Baseline' ||
    baseline.status === 'Slightly Below Rehab Baseline'
  ) {
    focusArea = baseline.detail;
  }

  if (
    baseline.status === 'Above Rehab Baseline' ||
    baseline.status === 'Slightly Above Rehab Baseline'
  ) {
    mainMessage = baseline.summary;
  }

  return {
    headline,
    mainMessage,
    focusArea,
    nextCheck,
  };
}

function getBaselineSnapshotLabel(
  score: number | null,
  baselineScore: number | null,
  sessionCount: number
) {
  if (score === null) {
    return {
      status: 'No Score Yet',
      detail: 'No valid score is available yet.',
      difference: null,
    };
  }

  if (baselineScore === null) {
    return {
      status: 'Building Baseline',
      detail: `${sessionCount}/3 checks saved for baseline.`,
      difference: null,
    };
  }

  const difference = Number((score - baselineScore).toFixed(1));
  const absDifference = Math.abs(difference);

  if (absDifference < 4) {
    return {
      status: 'Stable',
      detail: `${difference >= 0 ? '+' : ''}${difference} vs baseline.`,
      difference,
    };
  }

  if (difference >= 4) {
    return {
      status: 'Above Baseline',
      detail: `+${difference} vs baseline.`,
      difference,
    };
  }

  return {
    status: 'Below Baseline',
    detail: `${difference} vs baseline.`,
    difference,
  };
}

function getMovementOverview(sessions: SavedSession[]) {
  const dailyTasks = (['reach', 'arm_raise', 'sit_to_stand', 'walking', 'balance', 'timed_up_and_go'] as const).map((task) => {
    const latestSession = getLatestSessionForTask(sessions, task);
    const baseline = getBaselineForTask(sessions, task);

    const snapshot = getBaselineSnapshotLabel(
      latestSession?.primary_score ?? null,
      baseline.baselineScore,
      baseline.sessionCount
    );

    return {
      key: task,
      label: getDailyTaskLabel(task),
      score: latestSession?.primary_score ?? null,
      grade: latestSession?.primary_grade || 'N/A',
      baselineScore: baseline.baselineScore,
      sessionCount: baseline.sessionCount,
      status: snapshot.status,
      detail: snapshot.detail,
      difference: snapshot.difference,
    };
  });

  const latestRehab = getLatestRehabSession(sessions);
  const rehabBaseline = getRehabBaseline(sessions);

  const rehabSnapshot = getBaselineSnapshotLabel(
    latestRehab?.primary_score ?? null,
    rehabBaseline.baselineScore,
    rehabBaseline.sessionCount
  );

  const rehabItem = {
    key: 'rehab',
    label: 'Rehab Consistency',
    score: latestRehab?.primary_score ?? null,
    grade: latestRehab?.primary_grade || 'N/A',
    baselineScore: rehabBaseline.baselineScore,
    sessionCount: rehabBaseline.sessionCount,
    status: rehabSnapshot.status,
    detail: rehabSnapshot.detail,
    difference: rehabSnapshot.difference,
  };

  const items = [...dailyTasks, rehabItem];

  const scoredItems = items.filter((item) => item.score !== null);
  const baselineReadyItems = items.filter((item) => item.baselineScore !== null);
  const belowBaselineItems = items.filter((item) => item.status === 'Below Baseline');
  const aboveBaselineItems = items.filter((item) => item.status === 'Above Baseline');
  const stableItems = items.filter((item) => item.status === 'Stable');
  const buildingItems = items.filter((item) => item.status === 'Building Baseline');

  let status = 'Building Movement Profile';
  let summary = 'Keep recording checks to build a clearer picture of movement change over time.';
  let watch = 'Not enough baseline data yet.';
  let nextStep = 'Record each movement at least 3 times with a consistent camera setup.';

  if (scoredItems.length === 0) {
    return {
      status: 'No Movement Data Yet',
      summary: 'No saved movement checks are available yet.',
      watch: 'Nothing to compare yet.',
      nextStep: 'Start with a Daily Reach, Arm Raise, Sit-to-Stand, or Rehab check.',
      items,
    };
  }

  if (baselineReadyItems.length === 0) {
    return {
      status: 'Building Movement Profile',
      summary: `${scoredItems.length} saved check${scoredItems.length === 1 ? '' : 's'} found. Baselines are still being built.`,
      watch: `${buildingItems.length} area${buildingItems.length === 1 ? '' : 's'} still need more checks before baseline comparison is meaningful.`,
      nextStep: 'Repeat the same checks until each has at least 3 saved results.',
      items,
    };
  }

  if (belowBaselineItems.length >= 2) {
    status = 'Multiple Areas Below Baseline';
    summary = `${belowBaselineItems.length} movement areas are currently below their personal baseline.`;
    watch = `Watch: ${belowBaselineItems.map((item) => item.label).join(', ')}.`;
    nextStep = 'Re-test these areas with the same setup before treating the drop as meaningful.';
  } else if (belowBaselineItems.length === 1) {
    status = 'One Area Below Baseline';
    summary = `${belowBaselineItems[0].label} is currently below its personal baseline.`;
    watch = `Main area to watch: ${belowBaselineItems[0].label}.`;
    nextStep = `Repeat ${belowBaselineItems[0].label} with the same camera angle to confirm whether the drop is real.`;
  } else if (aboveBaselineItems.length >= 1 && stableItems.length >= 1) {
    status = 'Mostly Stable or Improving';
    summary = 'Current saved checks look stable or above baseline overall.';
    watch = buildingItems.length > 0
      ? `${buildingItems.length} area${buildingItems.length === 1 ? '' : 's'} still need more baseline data.`
      : 'No major below-baseline area is visible right now.';
    nextStep = 'Keep recording consistently to strengthen long-term tracking.';
  } else if (stableItems.length >= 1) {
    status = 'Mostly Stable';
    summary = 'Current saved checks are mostly close to personal baseline.';
    watch = buildingItems.length > 0
      ? `${buildingItems.length} area${buildingItems.length === 1 ? '' : 's'} still need more checks.`
      : 'No major below-baseline area is visible right now.';
    nextStep = 'Continue checking the same movements over time.';
  }

  return {
    status,
    summary,
    watch,
    nextStep,
    items,
  };
}

function getDailyHealthV2Overview(sessions: SavedSession[]) {
  const movementOverview = getMovementOverview(sessions);

  const items = movementOverview.items;
  const scoredItems = items.filter((item) => item.score !== null);
  const baselineReadyItems = items.filter((item) => item.baselineScore !== null);
  const buildingItems = items.filter((item) => item.status === 'Building Baseline');
  const belowBaselineItems = items.filter((item) => item.status === 'Below Baseline');
  const aboveBaselineItems = items.filter((item) => item.status === 'Above Baseline');
  const stableItems = items.filter((item) => item.status === 'Stable');

  const sortedScoredItems = scoredItems
    .slice()
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  const bestArea = sortedScoredItems[0] || null;

  let areaToWatch: (typeof items)[number] | null =
    belowBaselineItems[0] ||
    buildingItems[0] ||
    sortedScoredItems[sortedScoredItems.length - 1] ||
    null;

  let overallStatus = movementOverview.status;
  let headline = 'Build Your Movement Profile';
  let summary = movementOverview.summary;
  let nextCheck = movementOverview.nextStep;
  let healthFocus = movementOverview.watch;

  const completenessPercent = Math.round((scoredItems.length / items.length) * 100);
  const baselinePercent = Math.round((baselineReadyItems.length / items.length) * 100);

  if (scoredItems.length === 0) {
    overallStatus = 'No Daily Health Data Yet';
    headline = 'Start Your First Movement Check';
    summary = 'No movement checks are saved yet, so the app cannot summarize your movement health profile.';
    healthFocus = 'No movement area has been checked yet.';
    nextCheck = 'Start with Daily Reach or Sit-to-Stand.';
    areaToWatch = null;
  } else if (baselineReadyItems.length === 0) {
    overallStatus = 'Building Daily Health Profile';
    headline = 'Baseline Still Building';
    summary = `${scoredItems.length} movement area${scoredItems.length === 1 ? '' : 's'} have saved data, but baselines are still being built.`;
    healthFocus = `${buildingItems.length} area${buildingItems.length === 1 ? '' : 's'} still need repeated checks.`;
    nextCheck = areaToWatch
      ? `Repeat ${areaToWatch.label} until it has at least 3 saved checks.`
      : 'Repeat your main Daily checks with the same camera setup.';
  } else if (belowBaselineItems.length >= 2) {
    overallStatus = 'Multiple Areas Need Attention';
    headline = 'Several Areas Are Below Baseline';
    summary = `${belowBaselineItems.length} movement areas are currently below their personal baseline.`;
    healthFocus = `Watch these areas: ${belowBaselineItems.map((item) => item.label).join(', ')}.`;
    nextCheck = `Re-test ${belowBaselineItems[0].label} first with the same camera setup.`;
  } else if (belowBaselineItems.length === 1) {
    overallStatus = 'One Area Needs Attention';
    headline = `${belowBaselineItems[0].label} Needs a Recheck`;
    summary = `${belowBaselineItems[0].label} is currently below its personal baseline.`;
    healthFocus = `Main area to watch: ${belowBaselineItems[0].label}.`;
    nextCheck = `Repeat ${belowBaselineItems[0].label} with the same camera angle to confirm whether this is real.`;
  } else if (aboveBaselineItems.length >= 1 && stableItems.length >= 1) {
    overallStatus = 'Stable or Improving';
    headline = 'Movement Profile Looks Stable';
    summary = 'Current saved movement checks look stable or above baseline overall.';
    healthFocus = bestArea
      ? `Strongest area right now: ${bestArea.label}.`
      : 'No major below-baseline area is visible right now.';
    nextCheck = buildingItems.length > 0
      ? `Keep building baseline for ${buildingItems[0].label}.`
      : 'Keep recording consistently to strengthen long-term tracking.';
  } else if (stableItems.length >= 1) {
    overallStatus = 'Mostly Stable';
    headline = 'Movement Profile Looks Mostly Stable';
    summary = 'Current saved movement checks are mostly close to personal baseline.';
    healthFocus = bestArea
      ? `Most stable area right now: ${bestArea.label}.`
      : 'No major below-baseline area is visible right now.';
    nextCheck = buildingItems.length > 0
      ? `Repeat ${buildingItems[0].label} to keep building baseline.`
      : 'Continue checking the same movements over time.';
  }

  return {
    overallStatus,
    headline,
    summary,
    healthFocus,
    nextCheck,
    bestArea,
    areaToWatch,
    completenessPercent,
    baselinePercent,
    totalAreas: items.length,
    scoredAreaCount: scoredItems.length,
    baselineReadyCount: baselineReadyItems.length,
    items,
  };
}

function getLatestMovementReport(sessions: SavedSession[]) {
  const dailyHealth = getDailyHealthV2Overview(sessions);
  const generatedAt = new Date().toISOString();

  return {
    generatedAt,
    headline: dailyHealth.headline,
    status: dailyHealth.overallStatus,
    summary: dailyHealth.summary,
    focus: dailyHealth.healthFocus,
    nextCheck: dailyHealth.nextCheck,
    completenessPercent: dailyHealth.completenessPercent,
    baselinePercent: dailyHealth.baselinePercent,
    items: dailyHealth.items,
  };
}

function getMovementReportText(sessions: SavedSession[]) {
  const report = getLatestMovementReport(sessions);

  const areaLines = report.items.map((item) => {
    const scoreText = item.score !== null ? `${item.score}/100` : 'No score yet';
    const baselineText =
      item.baselineScore !== null ? `baseline ${item.baselineScore}/100` : 'baseline still building';

    return `- ${item.label}: ${scoreText} • ${item.status} • ${baselineText} • ${item.detail}`;
  });

  return `Movement Health Report

Generated:
${formatSessionDate(report.generatedAt)}

Overall Status:
${report.status}

Summary:
${report.summary}

Main Focus:
${report.focus}

Recommended Next Check:
${report.nextCheck}

Profile Completion:
${report.completenessPercent}% movement profile complete
${report.baselinePercent}% baselines built

Movement Areas:
${areaLines.join('\n')}

Important:
This report is for movement awareness and tracking only. It does not diagnose, treat, or replace medical advice.`;
}

function getPassportTrendSummaryForTask(sessions: SavedSession[], task: DailyTask) {
  const taskSessions = getSavedSessionsForTask(sessions, task);
  const scores = getTaskTrendScores(sessions, task);
  const latest = getLatestSessionForTask(sessions, task);
  const baseline = getBaselineForTask(sessions, task);

  if (!latest || latest.primary_score === null) {
    return {
      task,
      label: getDailyTaskLabel(task),
      status: 'No Data Yet',
      summary: `${getDailyTaskLabel(task)} has not been checked yet.`,
      trend: 'No trend yet',
      latestScore: null,
      previousScore: null,
      change: null,
      baselineScore: baseline.baselineScore,
      sessionCount: taskSessions.length,
      recommendation: `Record a ${getDailyTaskLabel(task)} check to start this section of your passport.`,
    };
  }

  const latestScore = latest.primary_score;
  const previousScore = scores.length >= 2 ? scores[scores.length - 2] : null;
  const change = previousScore !== null ? Number((latestScore - previousScore).toFixed(1)) : null;

  let status = 'Profile Building';
  let trend = 'Not enough repeated checks yet';
  let summary = `${getDailyTaskLabel(task)} has ${taskSessions.length} saved check${taskSessions.length === 1 ? '' : 's'}.`;
  let recommendation = `Repeat ${getDailyTaskLabel(task)} with the same camera setup to build a stronger trend.`;

  if (change !== null) {
    if (Math.abs(change) < 3) {
      status = 'Stable';
      trend = 'Stable';
      summary = `${getDailyTaskLabel(task)} stayed about the same compared with the previous saved check.`;
    } else if (change >= 8) {
      status = 'Strong Improvement';
      trend = `Improved by ${change.toFixed(1)} points`;
      summary = `${getDailyTaskLabel(task)} improved clearly compared with the previous saved check.`;
    } else if (change >= 3) {
      status = 'Slight Improvement';
      trend = `Improved by ${change.toFixed(1)} points`;
      summary = `${getDailyTaskLabel(task)} improved slightly compared with the previous saved check.`;
    } else if (change <= -8) {
      status = 'Clear Decline';
      trend = `Dropped by ${Math.abs(change).toFixed(1)} points`;
      summary = `${getDailyTaskLabel(task)} dropped clearly compared with the previous saved check. Re-test before treating this as a real decline.`;
      recommendation = `Recheck ${getDailyTaskLabel(task)} with the same camera setup.`;
    } else {
      status = 'Slight Decline';
      trend = `Dropped by ${Math.abs(change).toFixed(1)} points`;
      summary = `${getDailyTaskLabel(task)} decreased slightly compared with the previous saved check.`;
      recommendation = `Repeat ${getDailyTaskLabel(task)} later to see if the decrease continues.`;
    }
  }

  if (baseline.baselineScore !== null && latestScore !== null) {
    const baselineChange = Number((latestScore - baseline.baselineScore).toFixed(1));

    if (baselineChange <= -8) {
      status = 'Below Baseline';
      summary = `${getDailyTaskLabel(task)} is clearly below its personal baseline by ${Math.abs(baselineChange).toFixed(1)} points.`;
      recommendation = `Re-record ${getDailyTaskLabel(task)} before trusting this as a real change.`;
    } else if (baselineChange >= 8) {
      status = 'Above Baseline';
      summary = `${getDailyTaskLabel(task)} is clearly above its personal baseline by ${baselineChange.toFixed(1)} points.`;
      recommendation = `Repeat ${getDailyTaskLabel(task)} later to confirm the improvement.`;
    }
  }

  return {
    task,
    label: getDailyTaskLabel(task),
    status,
    summary,
    trend,
    latestScore,
    previousScore,
    change,
    baselineScore: baseline.baselineScore,
    sessionCount: taskSessions.length,
    recommendation,
  };
}

function getMovementPassport(sessions: SavedSession[]) {
  const dailyTasks = [
    'reach',
    'arm_raise',
    'sit_to_stand',
    'walking',
    'balance',
    'timed_up_and_go',
  ] as DailyTask[];

  const generatedAt = new Date().toISOString();
  const movementReport = getLatestMovementReport(sessions);
  const mobilityProfile = getFunctionalMobilityProfile(sessions);
  const weeklyReport = getWeeklyHealthReport(sessions);
  const coach = getMovementCoachPlan(sessions);
  const longitudinalTrends = getLongitudinalTrendEngine(sessions);

  const taskTrends = dailyTasks.map((task) =>
    getPassportTrendSummaryForTask(sessions, task)
  );

  const startedTasks = taskTrends.filter((item) => item.latestScore !== null);
  const baselineReadyTasks = taskTrends.filter((item) => item.baselineScore !== null);
  const improvingTasks = taskTrends.filter(
    (item) =>
      item.status === 'Strong Improvement' ||
      item.status === 'Slight Improvement' ||
      item.status === 'Above Baseline'
  );
  const watchTasks = taskTrends.filter(
    (item) =>
      item.status === 'Clear Decline' ||
      item.status === 'Slight Decline' ||
      item.status === 'Below Baseline'
  );

  const allScores = startedTasks
    .map((item) => item.latestScore)
    .filter((score): score is number => score !== null);

  const averageScore =
    allScores.length > 0
      ? Math.round(allScores.reduce((sum, score) => sum + score, 0) / allScores.length)
      : null;

  let passportStatus = 'No Passport Data Yet';
  let passportHeadline = 'Start Your Movement Passport';
  let passportSummary = 'Save movement checks to generate a shareable movement passport.';
  let movementFingerprint = 'Not enough data yet';
  let passportReadiness = 'Not ready to share yet';
  let nextAction = 'Complete the guided starter checks first.';

  if (startedTasks.length > 0) {
    passportStatus = 'Passport Building';
    passportHeadline = 'Movement Passport Building';
    passportSummary = 'Your saved checks are starting to form a personal movement profile.';
    movementFingerprint = mobilityProfile.profileType;
    passportReadiness = 'Early snapshot';
    nextAction = movementReport.nextCheck;
  }

  if (startedTasks.length >= 3) {
    passportStatus = 'Passport Snapshot Ready';
    passportHeadline = 'Movement Passport Snapshot Ready';
    passportSummary = 'You have enough saved checks to summarize your movement profile across multiple areas.';
    passportReadiness = 'Shareable snapshot';
    nextAction = mobilityProfile.recommendedNext;
  }

  if (baselineReadyTasks.length >= 2) {
    passportStatus = 'Baseline Passport Ready';
    passportHeadline = 'Baseline Movement Passport Ready';
    passportSummary = 'Your passport includes multiple baseline-ready movement areas.';
    passportReadiness = 'Stronger baseline snapshot';
  }

  if (watchTasks.length > 0) {
    passportStatus = 'Passport Recheck Needed';
    passportHeadline = 'Movement Passport Needs Recheck';
    passportSummary = `${watchTasks[0].label} needs a recheck before this passport should be treated as stable.`;
    nextAction = watchTasks[0].recommendation;
  }

  if (baselineReadyTasks.length >= 3 && watchTasks.length === 0) {
    passportStatus = 'Stable Movement Passport';
    passportHeadline = 'Stable Movement Passport';
    passportSummary = 'Your current saved checks suggest a stable movement profile across the tracked areas.';
    passportReadiness = 'Strong shareable snapshot';
  }

  const strongestTask =
    startedTasks
      .slice()
      .sort((a, b) => (b.latestScore ?? 0) - (a.latestScore ?? 0))[0] || null;

  const weakestTask =
    startedTasks
      .slice()
      .sort((a, b) => (a.latestScore ?? 0) - (b.latestScore ?? 0))[0] || null;

  const snapshotCards = [
    {
      label: 'Passport Score',
      value: averageScore !== null ? `${averageScore}/100` : 'N/A',
      detail: 'Average of saved movement area scores.',
    },
    {
      label: 'Checks Started',
      value: `${startedTasks.length}/${dailyTasks.length}`,
      detail: 'Movement areas with at least one saved check.',
    },
    {
      label: 'Baselines Built',
      value: `${baselineReadyTasks.length}/${dailyTasks.length}`,
      detail: 'Areas with at least 3 saved checks.',
    },
    {
      label: 'Watch Areas',
      value: `${watchTasks.length}`,
      detail: 'Areas that may need rechecking.',
    },
  ];

  const trendSummary = {
    improvingCount: improvingTasks.length,
    watchCount: watchTasks.length,
    stableCount: taskTrends.filter((item) => item.status === 'Stable').length,
    noDataCount: taskTrends.filter((item) => item.status === 'No Data Yet').length,
    improvingTasks,
    watchTasks,
  };

  return {
    generatedAt,
    passportStatus,
    passportHeadline,
    passportSummary,
    movementFingerprint,
    passportReadiness,
    nextAction,
    averageScore,
    strongestTask,
    weakestTask,
    taskTrends,
    snapshotCards,
    trendSummary,
    mobilityProfile,
    weeklyReport,
    coach,
    longitudinalTrends,
  };
}

function getMovementPassportText(sessions: SavedSession[]) {
  const passport = getMovementPassport(sessions);

  const trendLines = passport.taskTrends.map((item) => {
    const latestText = item.latestScore !== null ? `${item.latestScore}/100` : 'No score yet';
    const baselineText =
      item.baselineScore !== null ? `baseline ${item.baselineScore}/100` : 'baseline building';

    return `- ${item.label}: ${latestText} • ${item.status} • ${item.trend} • ${baselineText}`;
  });

  const watchText =
    passport.trendSummary.watchTasks.length > 0
      ? passport.trendSummary.watchTasks.map((item) => `- ${item.label}: ${item.recommendation}`).join('\n')
      : '- No watch areas currently flagged.';

  const improvingText =
    passport.trendSummary.improvingTasks.length > 0
      ? passport.trendSummary.improvingTasks.map((item) => `- ${item.label}: ${item.summary}`).join('\n')
      : '- No clear improvement areas yet.';

  return `Kinetra Passport

Generated:
${formatSessionDate(passport.generatedAt)}

Passport Status:
${passport.passportStatus}

Passport Score:
${passport.averageScore !== null ? `${passport.averageScore}/100` : 'Not enough data yet'}

Movement Fingerprint:
${passport.movementFingerprint}

Summary:
${passport.passportSummary}

Strongest Area:
${passport.strongestTask ? `${passport.strongestTask.label} (${passport.strongestTask.latestScore}/100)` : 'Not available yet'}

Main Watch Area:
${passport.weakestTask ? `${passport.weakestTask.label} (${passport.weakestTask.latestScore}/100)` : 'Not available yet'}

Recommended Next Action:
${passport.nextAction}

Profile Snapshot:
- Readiness: ${passport.passportReadiness}
- Profile completion: ${passport.mobilityProfile.completionPercent}%
- Baselines built: ${passport.mobilityProfile.baselinePercent}%
- Weekly status: ${passport.weeklyReport.weeklyStatus}
- Coach focus: ${passport.coach.whatNeedsAttention}

Trend Summary:
- Improving areas: ${passport.trendSummary.improvingCount}
- Stable areas: ${passport.trendSummary.stableCount}
- Watch areas: ${passport.trendSummary.watchCount}
- Missing areas: ${passport.trendSummary.noDataCount}

Longitudinal Trend Engine:
- Status: ${passport.longitudinalTrends.engineStatus}
- Trend confidence: ${passport.longitudinalTrends.trendConfidence}
- Main finding: ${passport.longitudinalTrends.mainFinding}
- Recommended action: ${passport.longitudinalTrends.nextAction}

Improving Areas:
${improvingText}

Watch / Recheck Areas:
${watchText}

Movement Areas:
${trendLines.join('\n')}

Important:
${APP_SAFETY_NOTE} It does not diagnose, treat, predict injury, estimate fall risk, or replace medical advice.`;
}

function getCoreDailyTasks() {
  return [
    'reach',
    'arm_raise',
    'sit_to_stand',
    'walking',
    'balance',
    'timed_up_and_go',
  ] as DailyTask[];
}

function getSessionsWithinDays(sessions: SavedSession[], days: number) {
  const now = Date.now();
  const windowMs = days * 24 * 60 * 60 * 1000;

  return sessions.filter((session) => {
    const time = new Date(session.timestamp).getTime();

    if (Number.isNaN(time)) {
      return false;
    }

    return now - time <= windowMs;
  });
}

function getScoreStats(scores: number[]) {
  if (scores.length === 0) {
    return {
      average: null,
      min: null,
      max: null,
      first: null,
      latest: null,
      change: null,
      volatility: null,
    };
  }

  const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const first = scores[0];
  const latest = scores[scores.length - 1];
  const change = latest - first;

  const volatility =
    scores.length >= 2
      ? Math.sqrt(
        scores.reduce((sum, score) => sum + Math.pow(score - average, 2), 0) /
        scores.length
      )
      : 0;

  return {
    average: Number(average.toFixed(1)),
    min: Number(min.toFixed(1)),
    max: Number(max.toFixed(1)),
    first: Number(first.toFixed(1)),
    latest: Number(latest.toFixed(1)),
    change: Number(change.toFixed(1)),
    volatility: Number(volatility.toFixed(1)),
  };
}

function getTaskLongitudinalTrend(sessions: SavedSession[], task: DailyTask) {
  const taskSessions = getSavedSessionsForTask(sessions, task).slice().reverse();

  const scoredSessions = taskSessions.filter(
    (session) => session.primary_score !== null
  );

  const allScores = scoredSessions
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);

  const sevenDaySessions = getSessionsWithinDays(scoredSessions, 7);
  const thirtyDaySessions = getSessionsWithinDays(scoredSessions, 30);

  const sevenDayScores = sevenDaySessions
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);

  const thirtyDayScores = thirtyDaySessions
    .map((session) => session.primary_score)
    .filter((score): score is number => score !== null);

  const allStats = getScoreStats(allScores);
  const sevenDayStats = getScoreStats(sevenDayScores);
  const thirtyDayStats = getScoreStats(thirtyDayScores);

  const latestSession = scoredSessions[scoredSessions.length - 1] || null;
  const previousSession = scoredSessions[scoredSessions.length - 2] || null;
  const baseline = getBaselineForTask(sessions, task);

  const latestScore = latestSession?.primary_score ?? null;
  const previousScore = previousSession?.primary_score ?? null;

  const latestChange =
    latestScore !== null && previousScore !== null
      ? Number((latestScore - previousScore).toFixed(1))
      : null;

  const baselineChange =
    latestScore !== null && baseline.baselineScore !== null
      ? Number((latestScore - baseline.baselineScore).toFixed(1))
      : null;

  let direction = 'No Trend Yet';
  let status = 'No Data Yet';
  let confidence = 'Low';
  let summary = `${getDailyTaskLabel(task)} has not been checked yet.`;
  let flag = 'Start tracking';
  let recommendation = `Record a ${getDailyTaskLabel(task)} check to start this trend.`;

  if (allScores.length === 1) {
    direction = 'First Check Saved';
    status = 'Trend Building';
    confidence = 'Low';
    summary = `${getDailyTaskLabel(task)} has one saved check. More checks are needed to identify a real trend.`;
    flag = 'Needs repeat check';
    recommendation = `Repeat ${getDailyTaskLabel(task)} with the same camera setup.`;
  }

  if (allScores.length === 2) {
    direction = latestChange !== null && latestChange >= 0 ? 'Early Improving' : 'Early Declining';
    status = 'Early Trend';
    confidence = 'Low';
    summary = `${getDailyTaskLabel(task)} has two saved checks. The app can compare them, but this is still early.`;
    flag = 'Early signal only';
    recommendation = `Save one more ${getDailyTaskLabel(task)} check to build a baseline.`;
  }

  if (allScores.length >= 3) {
    confidence = 'Moderate';
    status = 'Trend Active';

    if (latestChange !== null && Math.abs(latestChange) < 3) {
      direction = 'Stable';
      summary = `${getDailyTaskLabel(task)} is currently stable compared with the previous saved check.`;
      flag = 'Stable';
      recommendation = 'Keep recording with the same setup to strengthen the trend.';
    } else if (latestChange !== null && latestChange >= 8) {
      direction = 'Strong Improvement';
      summary = `${getDailyTaskLabel(task)} improved clearly compared with the previous saved check.`;
      flag = 'Improving';
      recommendation = 'Repeat later to confirm this improvement is consistent.';
    } else if (latestChange !== null && latestChange >= 3) {
      direction = 'Slight Improvement';
      summary = `${getDailyTaskLabel(task)} improved slightly compared with the previous saved check.`;
      flag = 'Slight improvement';
      recommendation = 'Keep tracking to confirm the improvement continues.';
    } else if (latestChange !== null && latestChange <= -8) {
      direction = 'Clear Decline';
      summary = `${getDailyTaskLabel(task)} dropped clearly compared with the previous saved check.`;
      flag = 'Recheck recommended';
      recommendation = `Recheck ${getDailyTaskLabel(task)} with the same camera setup before treating this as real decline.`;
    } else if (latestChange !== null && latestChange <= -3) {
      direction = 'Slight Decline';
      summary = `${getDailyTaskLabel(task)} dropped slightly compared with the previous saved check.`;
      flag = 'Watch';
      recommendation = `Repeat ${getDailyTaskLabel(task)} later to see if the drop continues.`;
    }
  }

  if (baselineChange !== null) {
    confidence = allScores.length >= 5 ? 'High' : 'Moderate';

    if (baselineChange <= -8) {
      status = 'Below Baseline';
      direction = 'Below Baseline';
      summary = `${getDailyTaskLabel(task)} is ${Math.abs(baselineChange).toFixed(1)} points below personal baseline.`;
      flag = 'Below baseline';
      recommendation = `Re-record ${getDailyTaskLabel(task)} with the same camera setup before trusting this as real change.`;
    } else if (baselineChange >= 8) {
      status = 'Above Baseline';
      direction = 'Above Baseline';
      summary = `${getDailyTaskLabel(task)} is ${baselineChange.toFixed(1)} points above personal baseline.`;
      flag = 'Above baseline';
      recommendation = 'Repeat later to confirm this improvement remains stable.';
    } else if (Math.abs(baselineChange) < 4) {
      status = 'Within Baseline';
      direction = 'Stable Baseline';
      summary = `${getDailyTaskLabel(task)} is close to personal baseline.`;
      flag = 'Within baseline';
      recommendation = 'Keep tracking consistently.';
    }
  }

  if (
    thirtyDayStats.change !== null &&
    Math.abs(thirtyDayStats.change) >= 10 &&
    thirtyDayScores.length >= 3
  ) {
    if (thirtyDayStats.change > 0) {
      direction = '30-Day Improvement';
      summary = `${getDailyTaskLabel(task)} improved by ${thirtyDayStats.change.toFixed(1)} points over the 30-day window.`;
      flag = 'Long-term improvement';
    } else {
      direction = '30-Day Decline';
      status = 'Long-Term Watch';
      summary = `${getDailyTaskLabel(task)} dropped by ${Math.abs(thirtyDayStats.change).toFixed(1)} points over the 30-day window.`;
      flag = 'Long-term recheck';
      recommendation = `Recheck ${getDailyTaskLabel(task)} and compare with the same camera setup.`;
    }
  }

  return {
    task,
    label: getDailyTaskLabel(task),
    status,
    direction,
    confidence,
    summary,
    flag,
    recommendation,
    latestScore,
    previousScore,
    latestChange,
    baselineScore: baseline.baselineScore,
    baselineChange,
    sessionCount: allScores.length,
    sevenDayStats,
    thirtyDayStats,
    allStats,
  };
}

function getLongitudinalTrendEngine(sessions: SavedSession[]) {
  const tasks = getCoreDailyTasks();
  const taskTrends = tasks.map((task) => getTaskLongitudinalTrend(sessions, task));

  const activeTrends = taskTrends.filter((trend) => trend.latestScore !== null);
  const baselineReadyTrends = taskTrends.filter((trend) => trend.baselineScore !== null);

  const improvingTrends = taskTrends.filter(
    (trend) =>
      trend.direction.includes('Improvement') ||
      trend.direction.includes('Above Baseline')
  );

  const declineTrends = taskTrends.filter(
    (trend) =>
      trend.direction.includes('Decline') ||
      trend.status.includes('Below Baseline') ||
      trend.status.includes('Long-Term Watch')
  );

  const stableTrends = taskTrends.filter(
    (trend) =>
      trend.direction.includes('Stable') ||
      trend.status.includes('Within Baseline')
  );

  const highConfidenceTrends = taskTrends.filter((trend) => trend.confidence === 'High');
  const moderateConfidenceTrends = taskTrends.filter((trend) => trend.confidence === 'Moderate');

  let engineStatus = 'No Longitudinal Data Yet';
  let headline = 'Start Long-Term Movement Tracking';
  let summary = 'Save repeated movement checks so the app can detect stability, improvement, decline, and baseline changes.';
  let mainFinding = 'No long-term trend is available yet.';
  let nextAction = 'Start with the guided starter checks, then repeat them with the same camera setup.';
  let trendConfidence = 'Low';

  if (activeTrends.length > 0) {
    engineStatus = 'Trend Engine Building';
    headline = 'Longitudinal Trends Are Building';
    summary = 'The app has started tracking movement change over time.';
    mainFinding = `${activeTrends.length} movement area${activeTrends.length === 1 ? '' : 's'} have saved trend data.`;
    nextAction = activeTrends[0].recommendation;
    trendConfidence = 'Early';
  }

  if (activeTrends.length >= 3) {
    engineStatus = 'Trend Snapshot Ready';
    headline = 'Movement Trend Snapshot Ready';
    summary = 'The app can now summarize change across multiple movement areas.';
    mainFinding = 'Several movement areas have enough data for an early trend snapshot.';
    trendConfidence = 'Moderate';
  }

  if (baselineReadyTrends.length >= 2) {
    engineStatus = 'Baseline Trend Engine Ready';
    headline = 'Baseline Trend Tracking Is Active';
    summary = 'Multiple movement areas now have baselines, so the app can compare current movement against personal normal ranges.';
    mainFinding = `${baselineReadyTrends.length} movement areas are baseline-ready.`;
    trendConfidence = highConfidenceTrends.length > 0 ? 'High' : 'Moderate';
  }

  if (declineTrends.length > 0) {
    engineStatus = 'Recheck Needed';
    headline = 'One or More Trends Need Rechecking';
    summary = 'At least one movement area shows a drop, below-baseline result, or long-term watch flag.';
    mainFinding = `${declineTrends[0].label} is the top recheck area.`;
    nextAction = declineTrends[0].recommendation;
    trendConfidence = declineTrends[0].confidence;
  } else if (improvingTrends.length > 0 && stableTrends.length > 0) {
    engineStatus = 'Stable or Improving';
    headline = 'Trends Look Stable or Improving';
    summary = 'Current saved movement trends look stable or improving across tracked areas.';
    mainFinding = `${improvingTrends.length} movement area${improvingTrends.length === 1 ? '' : 's'} are improving or above baseline.`;
    nextAction = 'Keep recording consistently to confirm the trend.';
  }

  const overallTrendScore =
    activeTrends.length > 0
      ? Math.round(
        activeTrends.reduce((sum, trend) => sum + (trend.latestScore ?? 0), 0) /
        activeTrends.length
      )
      : null;

  return {
    engineStatus,
    headline,
    summary,
    mainFinding,
    nextAction,
    trendConfidence,
    overallTrendScore,
    taskTrends,
    activeTrends,
    baselineReadyTrends,
    improvingTrends,
    declineTrends,
    stableTrends,
    highConfidenceTrends,
    moderateConfidenceTrends,
  };
}

function getLongitudinalTrendText(sessions: SavedSession[]) {
  const engine = getLongitudinalTrendEngine(sessions);

  const taskLines = engine.taskTrends.map((trend) => {
    const latestText = trend.latestScore !== null ? `${trend.latestScore}/100` : 'No score yet';
    const baselineText =
      trend.baselineScore !== null ? `baseline ${trend.baselineScore}/100` : 'baseline building';
    const changeText =
      trend.latestChange !== null
        ? `latest change ${trend.latestChange > 0 ? '+' : ''}${trend.latestChange.toFixed(1)}`
        : 'latest change unavailable';

    return `- ${trend.label}: ${latestText} • ${trend.status} • ${trend.direction} • ${changeText} • ${baselineText} • confidence ${trend.confidence}`;
  });

  const declineLines =
    engine.declineTrends.length > 0
      ? engine.declineTrends.map((trend) => `- ${trend.label}: ${trend.recommendation}`).join('\n')
      : '- No decline/recheck trends currently flagged.';

  const improvementLines =
    engine.improvingTrends.length > 0
      ? engine.improvingTrends.map((trend) => `- ${trend.label}: ${trend.summary}`).join('\n')
      : '- No clear improvement trends yet.';

  return `Longitudinal Trend Report

Status:
${engine.engineStatus}

Overall Trend Score:
${engine.overallTrendScore !== null ? `${engine.overallTrendScore}/100` : 'Not enough data yet'}

Trend Confidence:
${engine.trendConfidence}

Summary:
${engine.summary}

Main Finding:
${engine.mainFinding}

Recommended Next Action:
${engine.nextAction}

Trend Counts:
- Active trend areas: ${engine.activeTrends.length}
- Baseline-ready areas: ${engine.baselineReadyTrends.length}
- Improving areas: ${engine.improvingTrends.length}
- Stable areas: ${engine.stableTrends.length}
- Recheck/watch areas: ${engine.declineTrends.length}

Improving Trends:
${improvementLines}

Recheck / Watch Trends:
${declineLines}

Movement Area Trends:
${taskLines.join('\n')}

Important:
This trend report is for movement awareness and tracking only. It does not diagnose, treat, predict injury, estimate fall risk, or replace medical advice.`;
}

function getMovementCoachPlan(sessions: SavedSession[]) {
  const dailyHealth = getDailyHealthV2Overview(sessions);
  const report = getLatestMovementReport(sessions);

  const checkedItems = report.items.filter((item) => item.score !== null);
  const belowBaselineItems = report.items.filter((item) => item.status === 'Below Baseline');
  const buildingItems = report.items.filter((item) => item.status === 'Building Baseline');
  const stableItems = report.items.filter((item) => item.status === 'Stable');
  const aboveBaselineItems = report.items.filter((item) => item.status === 'Above Baseline');

  let coachStatus = 'Start Tracking First';
  let coachSummary = 'You do not have enough saved movement data yet for a useful coach summary.';
  let whatLooksGood = 'No clear strength is available yet.';
  let whatNeedsAttention = 'No specific limitation is available yet.';
  let trustLevel = 'Low';
  let nextAction = 'Start with a Daily Reach check, then repeat it later with the same camera setup.';
  let actionPlan = [
    'Record one Daily Reach check.',
    'Use the Camera Setup Guide before recording.',
    'Repeat the same check later to start building a baseline.',
  ];

  if (checkedItems.length > 0) {
    coachStatus = 'Movement Profile Building';
    coachSummary = dailyHealth.summary;
    trustLevel = dailyHealth.baselineReadyCount > 0 ? 'Moderate' : 'Early';
    nextAction = dailyHealth.nextCheck;

    whatLooksGood = dailyHealth.bestArea
      ? `${dailyHealth.bestArea.label} is currently your strongest recorded area.`
      : 'At least one movement area has been recorded successfully.';

    whatNeedsAttention = dailyHealth.areaToWatch
      ? `${dailyHealth.areaToWatch.label} is the main area to watch right now.`
      : 'No clear problem area is visible yet. Keep recording consistently.';

    actionPlan = [
      dailyHealth.nextCheck,
      'Use the same camera angle each time so changes are more meaningful.',
      'Build at least 3 saved checks for each important movement area.',
    ];
  }

  if (buildingItems.length > 0 && checkedItems.length > 0) {
    coachStatus = 'Build More Baseline Data';
    coachSummary = 'Your movement profile is started, but the app still needs repeated checks before it can compare most areas confidently.';
    whatNeedsAttention = `${buildingItems.length} area${buildingItems.length === 1 ? '' : 's'} still need baseline data.`;
    trustLevel = 'Early';
    nextAction = `Repeat ${buildingItems[0].label} with the same setup.`;
    actionPlan = [
      `Record ${buildingItems[0].label} again.`,
      'Keep the camera angle and lighting as similar as possible.',
      'Do not overreact to one score before a baseline exists.',
    ];
  }

  if (belowBaselineItems.length === 1) {
    coachStatus = 'One Area Needs Attention';
    coachSummary = `${belowBaselineItems[0].label} is below its personal baseline.`;
    whatNeedsAttention = `${belowBaselineItems[0].label} should be rechecked before treating this as a real change.`;
    trustLevel = 'Moderate';
    nextAction = `Re-test ${belowBaselineItems[0].label} with the same camera setup.`;
    actionPlan = [
      `Repeat ${belowBaselineItems[0].label}.`,
      'Use the same camera angle, distance, and lighting.',
      'If the score stays low across repeated checks, keep watching that movement area.',
    ];
  }

  if (belowBaselineItems.length >= 2) {
    coachStatus = 'Multiple Areas Need Attention';
    coachSummary = `${belowBaselineItems.length} movement areas are below baseline.`;
    whatNeedsAttention = `Watch these areas: ${belowBaselineItems.map((item) => item.label).join(', ')}.`;
    trustLevel = 'Moderate';
    nextAction = `Start by re-testing ${belowBaselineItems[0].label}.`;
    actionPlan = [
      `Re-test ${belowBaselineItems[0].label} first.`,
      'Do not change multiple things at once.',
      'Confirm whether the drop repeats before assuming it is real.',
    ];
  }

  if (aboveBaselineItems.length > 0 && belowBaselineItems.length === 0 && stableItems.length > 0) {
    coachStatus = 'Stable or Improving';
    coachSummary = 'Your saved movement profile looks stable or above baseline overall.';
    whatLooksGood = `${aboveBaselineItems[0].label} is above baseline right now.`;
    whatNeedsAttention = buildingItems.length > 0
      ? `${buildingItems[0].label} still needs more baseline data.`
      : 'No major below-baseline area is visible right now.';
    trustLevel = dailyHealth.baselineReadyCount >= 2 ? 'Good' : 'Moderate';
    nextAction = buildingItems.length > 0
      ? `Keep building baseline for ${buildingItems[0].label}.`
      : 'Keep recording consistently to strengthen long-term tracking.';
    actionPlan = [
      'Keep the same recording setup.',
      'Repeat the same movement checks over time.',
      'Use History and Movement Report to compare changes.',
    ];
  }

  return {
    coachStatus,
    coachSummary,
    whatLooksGood,
    whatNeedsAttention,
    trustLevel,
    nextAction,
    actionPlan,
    profileComplete: dailyHealth.completenessPercent,
    baselinesBuilt: dailyHealth.baselinePercent,
  };
}

function getCurrentResultCoachAdvice(
  result: AnalysisResult | null,
  previousSession: SavedSession | null,
  sessions: SavedSession[]
) {
  if (!result) {
    return {
      status: 'No Current Result',
      summary: 'Analyze a movement check to get coach advice.',
      trust: 'No result available yet.',
      focus: 'Record a clear movement check first.',
      nextAction: 'Start with Daily Reach or Sit-to-Stand.',
    };
  }

  const confidence = getDailyConfidence(result);
  const primary = getPrimaryTaskScore(result);
  const trend = result.mode === 'rehab'
    ? getRehabTrendInterpretation(result, previousSession)
    : getTrendInterpretation(result, previousSession);

  const baseline = result.mode === 'rehab'
    ? getRehabBaselineInterpretation(result, sessions)
    : getBaselineInterpretation(result, sessions);

  let status = primary.grade || 'Movement Check Complete';
  let summary = `This check produced a ${primary.label.toLowerCase()} score of ${primary.score !== null ? `${primary.score}/100` : 'N/A'}.`;
  let trust = confidence.message;
  let focus = 'Review the score, confidence, baseline, and trend before making conclusions.';
  let nextAction = confidence.nextAction;

  if (confidence.grade === 'Low Confidence') {
    status = 'Retest Before Trusting This';
    summary = 'The recording quality was not strong enough to confidently interpret the result.';
    trust = 'Low confidence: the camera likely missed important movement information.';
    focus = 'Improve camera setup first. Do not treat this score as meaningful yet.';
    nextAction = confidence.nextAction;
  } else if (
    baseline.status.includes('Below') ||
    trend.status.includes('Decline') ||
    trend.status.includes('Decreasing')
  ) {
    status = 'Watch This Result';
    summary = 'This result may be lower than your recent or baseline movement pattern.';
    trust = `${confidence.grade}: ${confidence.message}`;
    focus = 'The main question is whether this repeats with the same camera setup.';
    nextAction = baseline.nextStep || trend.nextStep;
  } else if (
    baseline.status.includes('Above') ||
    trend.status.includes('Improvement') ||
    trend.status.includes('Improving')
  ) {
    status = 'Positive Change Detected';
    summary = 'This result looks better than your recent or baseline movement pattern.';
    trust = `${confidence.grade}: ${confidence.message}`;
    focus = 'The main question is whether this improvement stays consistent.';
    nextAction = trend.nextStep || baseline.nextStep;
  } else {
    status = 'Stable Check';
    summary = 'This result looks generally stable based on the available comparison data.';
    trust = `${confidence.grade}: ${confidence.message}`;
    focus = 'Keep tracking with the same setup to make future comparisons stronger.';
    nextAction = trend.nextStep || baseline.nextStep;
  }

  return {
    status,
    summary,
    trust,
    focus,
    nextAction,
  };
}

function getCoachShareText(sessions: SavedSession[]) {
  const coach = getMovementCoachPlan(sessions);

  return `Movement Coach Summary

Coach Status:
${coach.coachStatus}

Summary:
${coach.coachSummary}

What Looks Good:
${coach.whatLooksGood}

What Needs Attention:
${coach.whatNeedsAttention}

Trust Level:
${coach.trustLevel}

Next Action:
${coach.nextAction}

Action Plan:
${coach.actionPlan.map((item, index) => `${index + 1}. ${item}`).join('\n')}

Profile:
${coach.profileComplete}% movement profile complete
${coach.baselinesBuilt}% baselines built

Important:
This coach summary is for movement awareness and tracking only. It does not diagnose, treat, or replace medical advice.`;
}

function getSessionsInLastDays(sessions: SavedSession[], days: number) {
  const now = Date.now();
  const cutoff = now - days * 24 * 60 * 60 * 1000;

  return sessions.filter((session) => {
    const sessionTime = new Date(session.timestamp).getTime();
    return sessionTime >= cutoff;
  });
}

function getWeeklyAreaCount(
  sessions: SavedSession[],
  areaKey: string
) {
  if (areaKey === 'rehab') {
    return sessions.filter((session) => session.mode === 'rehab').length;
  }

  return sessions.filter(
    (session) =>
      session.mode === 'daily' &&
      session.daily_task === areaKey
  ).length;
}

function getWeeklyHealthReport(sessions: SavedSession[]) {
  const recentSessions = getSessionsInLastDays(sessions, 7);
  const dailyHealth = getDailyHealthV2Overview(sessions);
  const movementOverview = getMovementOverview(sessions);

  const weeklyItems = movementOverview.items.map((item) => {
    const weeklyCount = getWeeklyAreaCount(recentSessions, item.key);

    return {
      ...item,
      weeklyCount,
    };
  });

  const checkedThisWeek = weeklyItems.filter((item) => item.weeklyCount > 0);
  const belowBaselineItems = weeklyItems.filter(
    (item) => item.weeklyCount > 0 && item.status === 'Below Baseline'
  );
  const stableItems = weeklyItems.filter(
    (item) => item.weeklyCount > 0 && item.status === 'Stable'
  );
  const aboveBaselineItems = weeklyItems.filter(
    (item) => item.weeklyCount > 0 && item.status === 'Above Baseline'
  );
  const buildingItems = weeklyItems.filter(
    (item) => item.status === 'Building Baseline'
  );

  const mostActiveArea =
    weeklyItems
      .slice()
      .sort((a, b) => b.weeklyCount - a.weeklyCount)[0] || null;

  const bestArea =
    checkedThisWeek
      .slice()
      .filter((item) => item.score !== null)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0] || null;

  const areaToWatch =
    belowBaselineItems[0] ||
    buildingItems[0] ||
    checkedThisWeek
      .slice()
      .filter((item) => item.score !== null)
      .sort((a, b) => (a.score ?? 0) - (b.score ?? 0))[0] ||
    null;

  let weeklyStatus = 'No Weekly Data Yet';
  let headline = 'Start This Week’s Movement Report';
  let summary = 'No movement checks were recorded in the last 7 days.';
  let focus = 'Record at least one movement check to start a weekly report.';
  let nextAction = 'Start with Daily Reach or Sit-to-Stand.';

  if (recentSessions.length > 0) {
    weeklyStatus = 'Weekly Report Started';
    headline = 'Weekly Movement Tracking Started';
    summary = `${recentSessions.length} movement check${recentSessions.length === 1 ? '' : 's'} recorded in the last 7 days.`;
    focus = dailyHealth.healthFocus;
    nextAction = dailyHealth.nextCheck;
  }

  if (recentSessions.length >= 3) {
    weeklyStatus = 'Weekly Profile Building';
    headline = 'Weekly Movement Profile Building';
    summary = `${recentSessions.length} checks this week. This is enough to start seeing weekly patterns, but more repeated checks will make it stronger.`;
    focus = areaToWatch
      ? `Main area to watch: ${areaToWatch.label}.`
      : 'Keep building weekly consistency across the main movement areas.';
    nextAction = areaToWatch
      ? `Repeat ${areaToWatch.label} with the same setup.`
      : 'Repeat your most important Daily check.';
  }

  if (belowBaselineItems.length === 1) {
    weeklyStatus = 'One Weekly Area Needs Attention';
    headline = `${belowBaselineItems[0].label} Needs a Recheck`;
    summary = `${belowBaselineItems[0].label} was checked this week and is below baseline.`;
    focus = `Watch ${belowBaselineItems[0].label} before assuming this is a real change.`;
    nextAction = `Re-test ${belowBaselineItems[0].label} with the same camera setup.`;
  }

  if (belowBaselineItems.length >= 2) {
    weeklyStatus = 'Multiple Weekly Areas Need Attention';
    headline = 'Several Areas Need Rechecking';
    summary = `${belowBaselineItems.length} checked areas are below baseline this week.`;
    focus = `Watch: ${belowBaselineItems.map((item) => item.label).join(', ')}.`;
    nextAction = `Start by re-testing ${belowBaselineItems[0].label}.`;
  }

  if (
    recentSessions.length >= 3 &&
    belowBaselineItems.length === 0 &&
    (stableItems.length > 0 || aboveBaselineItems.length > 0)
  ) {
    weeklyStatus = 'Weekly Movement Stable';
    headline = 'This Week Looks Stable';
    summary = 'The movement checks recorded this week look stable or above baseline overall.';
    focus = bestArea
      ? `Strongest area this week: ${bestArea.label}.`
      : 'No major below-baseline area appeared this week.';
    nextAction = buildingItems.length > 0
      ? `Keep building baseline for ${buildingItems[0].label}.`
      : 'Keep recording consistently next week.';
  }

  return {
    weeklyStatus,
    headline,
    summary,
    focus,
    nextAction,
    totalChecksThisWeek: recentSessions.length,
    areasCheckedThisWeek: checkedThisWeek.length,
    profileComplete: dailyHealth.completenessPercent,
    baselinesBuilt: dailyHealth.baselinePercent,
    mostActiveArea,
    bestArea,
    areaToWatch,
    weeklyItems,
  };
}

function getWeeklyHealthReportText(sessions: SavedSession[]) {
  const weekly = getWeeklyHealthReport(sessions);

  const areaLines = weekly.weeklyItems.map((item) => {
    const scoreText = item.score !== null ? `${item.score}/100` : 'No score yet';

    return `- ${item.label}: ${item.weeklyCount} check${item.weeklyCount === 1 ? '' : 's'} this week • ${scoreText} • ${item.status}`;
  });

  return `Weekly Movement Health Report

  Status:
  ${weekly.weeklyStatus}

  Headline:
  ${weekly.headline}

  Summary:
  ${weekly.summary}

  Focus:
  ${weekly.focus}

  Recommended Next Action:
  ${weekly.nextAction}

  This Week:
  ${weekly.totalChecksThisWeek} total movement check${weekly.totalChecksThisWeek === 1 ? '' : 's'}
  ${weekly.areasCheckedThisWeek}/6 movement areas checked

  Profile:
  ${weekly.profileComplete}% movement profile complete
  ${weekly.baselinesBuilt}% baselines built

  Movement Areas:
  ${areaLines.join('\n')}

  Important:
  This weekly report is for movement awareness and tracking only. It does not diagnose, treat, or replace medical advice.`;
}

function getFunctionalMobilityProfile(sessions: SavedSession[]) {
  const overview = getMovementOverview(sessions);

  const domainItems = [
    {
      key: 'upper_body_control',
      label: 'Upper-Body Control',
      description: 'Reach and arm raise control, steadiness, and smoothness.',
      tasks: ['reach', 'arm_raise'] as DailyTask[],
    },
    {
      key: 'lower_body_transition',
      label: 'Lower-Body Transition',
      description: 'Sit-to-stand control, rise stability, and trunk control.',
      tasks: ['sit_to_stand'] as DailyTask[],
    },
    {
      key: 'gait_rhythm',
      label: 'Gait Rhythm',
      description: 'Walking rhythm, cadence, lower-body motion, and step consistency.',
      tasks: ['walking'] as DailyTask[],
    },
    {
      key: 'postural_stability',
      label: 'Postural Stability',
      description: 'Standing balance, visible sway, torso lean, and posture control.',
      tasks: ['balance'] as DailyTask[],
    },
    {
      key: 'functional_mobility',
      label: 'Functional Mobility',
      description: 'Timed Up and Go sequence quality across standing, walking, turning, and returning.',
      tasks: ['timed_up_and_go'] as DailyTask[],
    },
  ];

  const scoredDomains = domainItems.map((domain) => {
    const matchingItems = overview.items.filter((item) =>
      domain.tasks.includes(item.key as DailyTask)
    );

    const scoredItems = matchingItems.filter((item) => item.score !== null);
    const baselineReadyItems = matchingItems.filter((item) => item.baselineScore !== null);
    const belowBaselineItems = matchingItems.filter((item) => item.status === 'Below Baseline');
    const buildingItems = matchingItems.filter((item) => item.status === 'Building Baseline');

    const averageScore =
      scoredItems.length > 0
        ? Math.round(
          scoredItems.reduce((sum, item) => sum + (item.score ?? 0), 0) /
          scoredItems.length
        )
        : null;

    let status = 'No Data Yet';
    let summary = 'No checks have been saved for this domain yet.';

    if (averageScore !== null) {
      status = 'Profile Building';
      summary = `${domain.label} has ${scoredItems.length} saved check${scoredItems.length === 1 ? '' : 's'}.`;
    }

    if (buildingItems.length > 0 && scoredItems.length > 0) {
      status = 'Building Baseline';
      summary = `${domain.label} has data, but still needs repeated checks for a stronger baseline.`;
    }

    if (baselineReadyItems.length > 0 && belowBaselineItems.length === 0) {
      status = 'Stable';
      summary = `${domain.label} is currently stable or within baseline based on saved checks.`;
    }

    if (belowBaselineItems.length === 1) {
      status = 'Needs Recheck';
      summary = `${belowBaselineItems[0].label} is below baseline and should be rechecked.`;
    }

    if (belowBaselineItems.length >= 2) {
      status = 'Needs Attention';
      summary = `${belowBaselineItems.length} checks in this domain are below baseline.`;
    }

    return {
      ...domain,
      score: averageScore,
      status,
      summary,
      checkedCount: scoredItems.length,
      totalTasks: matchingItems.length,
      baselineReadyCount: baselineReadyItems.length,
      belowBaselineCount: belowBaselineItems.length,
      items: matchingItems,
    };
  });

  const scoredOnly = scoredDomains.filter((domain) => domain.score !== null);
  const belowBaselineDomains = scoredDomains.filter((domain) => domain.belowBaselineCount > 0);
  const buildingDomains = scoredDomains.filter(
    (domain) => domain.status === 'Building Baseline' || domain.status === 'Profile Building'
  );

  const overallScore =
    scoredOnly.length > 0
      ? Math.round(
        scoredOnly.reduce((sum, domain) => sum + (domain.score ?? 0), 0) /
        scoredOnly.length
      )
      : null;

  let profileLabel = 'No Mobility Profile Yet';
  let headline = 'Start Your Functional Mobility Profile';
  let summary = 'Record movement checks to build a full picture of upper-body control, lower-body transition, walking, balance, and functional mobility.';
  let recommendedNext = 'Start with Reach, Sit-to-Stand, Walking, Balance, or Timed Up and Go.';
  let profileType = 'Not enough data yet';

  if (overallScore !== null) {
    profileLabel = 'Functional Mobility Profile Building';
    headline = 'Your Mobility Profile Is Building';
    summary = 'The app is combining your saved movement checks into one functional mobility profile.';
    recommendedNext = 'Keep recording the missing or baseline-building areas.';
    profileType = 'Mixed mobility profile';
  }

  if (scoredOnly.length >= 3) {
    profileLabel = 'Functional Mobility Snapshot Ready';
    headline = 'Your Functional Mobility Snapshot Is Ready';
    summary = 'You have enough saved checks for the app to summarize your movement profile across multiple domains.';
    recommendedNext = 'Use the weakest domain and below-baseline checks to decide what to recheck next.';
  }

  if (belowBaselineDomains.length > 0) {
    profileLabel = 'Recheck Recommended';
    headline = 'One or More Mobility Domains Need Rechecking';
    summary = 'At least one movement domain has a below-baseline result. Recheck before treating it as a real decline.';
    recommendedNext = `Recheck ${belowBaselineDomains[0].label}.`;
  }

  if (scoredOnly.length >= 4 && belowBaselineDomains.length === 0) {
    profileLabel = 'Stable Mobility Profile';
    headline = 'Your Mobility Profile Looks Stable';
    summary = 'Your saved checks suggest a stable functional mobility profile across the areas currently tracked.';
    recommendedNext = 'Keep recording consistently to strengthen long-term trend tracking.';
  }

  const strongestDomain =
    scoredOnly
      .slice()
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0] || null;

  const weakestDomain =
    scoredOnly
      .slice()
      .sort((a, b) => (a.score ?? 0) - (b.score ?? 0))[0] || null;

  if (strongestDomain && weakestDomain && strongestDomain.key !== weakestDomain.key) {
    profileType = `${strongestDomain.label}-dominant profile with ${weakestDomain.label.toLowerCase()} as the main watch area`;
  }

  const urgentRecheck =
    belowBaselineDomains[0] ||
    buildingDomains[0] ||
    weakestDomain ||
    null;

  const completionPercent = Math.round(
    (scoredOnly.length / Math.max(domainItems.length, 1)) * 100
  );

  const baselinePercent = Math.round(
    (
      scoredDomains.reduce((sum, domain) => sum + domain.baselineReadyCount, 0) /
      Math.max(
        scoredDomains.reduce((sum, domain) => sum + domain.totalTasks, 0),
        1
      )
    ) * 100
  );

  return {
    overallScore,
    profileLabel,
    headline,
    summary,
    recommendedNext,
    profileType,
    strongestDomain,
    weakestDomain,
    urgentRecheck,
    completionPercent,
    baselinePercent,
    domains: scoredDomains,
  };
}

function getFunctionalMobilityProfileText(sessions: SavedSession[]) {
  const profile = getFunctionalMobilityProfile(sessions);

  const domainLines = profile.domains.map((domain) => {
    const scoreText = domain.score !== null ? `${domain.score}/100` : 'No score yet';

    return `- ${domain.label}: ${scoreText} • ${domain.status} • ${domain.summary}`;
  });

  return `Functional Mobility Profile

Status:
${profile.profileLabel}

Overall Score:
${profile.overallScore !== null ? `${profile.overallScore}/100` : 'Not enough data yet'}

Profile Type:
${profile.profileType}

Summary:
${profile.summary}

Strongest Domain:
${profile.strongestDomain ? profile.strongestDomain.label : 'Not available yet'}

Main Watch Area:
${profile.weakestDomain ? profile.weakestDomain.label : 'Not available yet'}

Recommended Next Action:
${profile.recommendedNext}

Profile Completion:
${profile.completionPercent}% complete
${profile.baselinePercent}% baselines built

Domains:
${domainLines.join('\n')}

Important:
This profile is for movement awareness and tracking only. It does not diagnose, treat, predict injury, or replace medical advice.`;
}

function getOnboardingProgress(sessions: SavedSession[]) {
  const firstChecks = [
    {
      task: 'reach' as DailyTask,
      label: 'Reach',
      reason: 'Starts your upper-body control profile.',
    },
    {
      task: 'sit_to_stand' as DailyTask,
      label: 'Sit-to-Stand',
      reason: 'Starts your lower-body transition profile.',
    },
    {
      task: 'balance' as DailyTask,
      label: 'Balance',
      reason: 'Starts your postural stability profile.',
    },
  ];

  const steps = firstChecks.map((item) => {
    const hasSavedCheck = getSavedSessionsForTask(sessions, item.task).length > 0;

    return {
      ...item,
      complete: hasSavedCheck,
    };
  });

  const completedCount = steps.filter((step) => step.complete).length;
  const completionPercent = Math.round((completedCount / steps.length) * 100);

  let status = 'Start Here';
  let headline = 'Build Your First Mobility Profile';
  let summary = 'Complete three starter checks so the app can begin building your movement profile.';
  let nextTask = steps.find((step) => !step.complete)?.task || 'reach';
  let nextTaskLabel = steps.find((step) => !step.complete)?.label || 'Reach';

  if (completedCount === 1) {
    status = 'Profile Started';
    headline = 'Good Start — Keep Going';
    summary = 'You completed your first starter check. Two more checks will make the profile more useful.';
  }

  if (completedCount === 2) {
    status = 'Almost Ready';
    headline = 'Almost Done With Setup';
    summary = 'One more starter check will complete your first mobility profile setup.';
  }

  if (completedCount === 3) {
    status = 'Starter Profile Complete';
    headline = 'Your Starter Mobility Profile Is Ready';
    summary = 'You completed the core setup checks. Now you can review your Mobility Profile and keep building baselines.';
    nextTask = 'timed_up_and_go';
    nextTaskLabel = 'Timed Up and Go';
  }

  return {
    status,
    headline,
    summary,
    steps,
    completedCount,
    totalCount: steps.length,
    completionPercent,
    nextTask,
    nextTaskLabel,
  };
}

function getSuggestedNextProfileTask(sessions: SavedSession[]) {
  const priorityTasks = [
    'reach',
    'sit_to_stand',
    'balance',
    'walking',
    'timed_up_and_go',
    'arm_raise',
  ] as DailyTask[];

  const missingTask = priorityTasks.find(
    (task) => getSavedSessionsForTask(sessions, task).length === 0
  );

  if (missingTask) {
    return {
      task: missingTask,
      label: getDailyTaskLabel(missingTask),
      reason: 'This check has not been saved yet, so it will expand your mobility profile.',
    };
  }

  const baselineTask = priorityTasks.find(
    (task) => getBaselineForTask(sessions, task).baselineScore === null
  );

  if (baselineTask) {
    const baseline = getBaselineForTask(sessions, baselineTask);
    const remaining = Math.max(0, 3 - baseline.sessionCount);

    return {
      task: baselineTask,
      label: getDailyTaskLabel(baselineTask),
      reason: `${getDailyTaskLabel(baselineTask)} needs ${remaining} more saved check${remaining === 1 ? '' : 's'} to build a baseline.`,
    };
  }

  return {
    task: 'timed_up_and_go' as DailyTask,
    label: 'Timed Up and Go',
    reason: 'All starter areas have data. Timed Up and Go is the best full functional mobility recheck.',
  };
}

function getTestingReadiness(sessions: SavedSession[]) {
  const dailyTasks = [
    'reach',
    'arm_raise',
    'sit_to_stand',
    'walking',
    'balance',
    'timed_up_and_go',
  ] as DailyTask[];

  const dailyTaskCounts = dailyTasks.map((task) => ({
    task,
    label: getDailyTaskLabel(task),
    count: getSavedSessionsForTask(sessions, task).length,
    baselineBuilt: getBaselineForTask(sessions, task).baselineScore !== null,
  }));

  const dailyTaskCount = dailyTaskCounts.reduce(
    (sum, item) => sum + item.count,
    0
  );

  const dailyTasksStarted = dailyTaskCounts.filter((item) => item.count > 0).length;
  const dailyBaselinesBuilt = dailyTaskCounts.filter((item) => item.baselineBuilt).length;

  const rehabCount = getSavedRehabSessions(sessions).length;
  const rehabBaselineBuilt = getRehabBaseline(sessions).baselineScore !== null;

  let readinessScore = 20;

  readinessScore += Math.min(25, dailyTaskCount * 2);
  readinessScore += Math.min(20, dailyTasksStarted * 4);
  readinessScore += Math.min(15, rehabCount * 5);
  readinessScore += dailyBaselinesBuilt * 6;

  if (rehabBaselineBuilt) {
    readinessScore += 10;
  }

  readinessScore = Math.min(100, readinessScore);

  let status = 'Not Ready Yet';
  let summary = 'The app can run, but you should collect more of your own test recordings before handing it to other people.';
  let nextStep = 'Record each core Daily movement check and Rehab mode at least once before private testing.';

  if (readinessScore >= 80) {
    status = 'Ready for Private Testing';
    summary = 'The app has enough core functionality and saved-session behavior to test with real users.';
    nextStep = 'Test with 3–5 people and watch silently for confusion.';
  } else if (readinessScore >= 55) {
    status = 'Almost Ready';
    summary = 'The app is close to private testing, but more saved checks will make baselines and history easier to demonstrate.';
    nextStep = 'Record more Daily movement checks and Rehab checks before testing.';
  } else if (readinessScore >= 35) {
    status = 'Needs More Internal Testing';
    summary = 'The core app is built, but you should test more yourself before giving it to others.';
    nextStep = 'Run each mode yourself and confirm the results make sense.';
  }

  return {
    readinessScore,
    status,
    summary,
    nextStep,
    dailyTaskCount,
    dailyTasksStarted,
    dailyTaskCounts,
    rehabCount,
    dailyBaselinesBuilt,
    rehabBaselineBuilt,
  };
}

function getPrivateTestingTasks() {
  return [
    {
      title: 'First Impression',
      goal: 'See whether the user understands what the app does without you explaining it.',
      instructions: [
        'Hand them the app on the home screen.',
        'Do not explain the app first.',
        'Ask them what they think the app is for.',
        'Watch whether they notice How It Works, Daily, Rehab, and History.',
      ],
    },
    {
      title: 'Daily Reach Test',
      goal: 'See whether they can record a simple Daily movement correctly.',
      instructions: [
        'Ask them to choose Daily mode.',
        'Ask them to choose Reach.',
        'Watch whether they understand the camera setup.',
        'Do not correct them unless they are completely stuck.',
        'See whether the result makes sense to them.',
      ],
    },
    {
      title: 'Daily Sit-to-Stand Test',
      goal: 'See whether the full-body instructions are clear enough.',
      instructions: [
        'Ask them to choose Sit-to-Stand.',
        'Watch whether they know to place the camera from the side.',
        'Watch whether they keep hips, knees, ankles, and feet visible.',
        'Check whether they understand the result after analysis.',
      ],
    },
    {
      title: 'Rehab Consistency Test',
      goal: 'See whether Rehab mode feels distinct from Rep mode.',
      instructions: [
        'Ask them to choose Rehab.',
        'Tell them only to perform a slow repeated controlled movement.',
        'Watch whether they understand repeatability and consistency.',
        'Check whether the Rehab Session Summary is understandable.',
      ],
    },
    {
      title: 'History + Movement Overview',
      goal: 'See whether users understand tracking over time.',
      instructions: [
        'Ask them to open View History.',
        'Watch whether Movement Change Overview makes sense.',
        'Ask what they think “baseline” means.',
        'Ask what they would record next based on the overview.',
      ],
    },
  ];
}

function getGuidedTestWorkflowSteps() {
  return [
    {
      key: 'first_impression',
      title: 'First Impression',
      goal: 'See whether the tester understands what the app is for without you explaining it.',
      actionLabel: 'Start at Home',
      mode: null,
      task: null,
      instructions: [
        'Hand the app to the tester on the home screen.',
        'Do not explain the app first.',
        'Ask: “What do you think this app does?”',
        'Watch whether they notice Daily, Rehab, History, Camera Setup, and Rollout tools.',
      ],
    },
    {
      key: 'daily_reach',
      title: 'Daily Reach Check',
      goal: 'Test whether a user can complete a simple Daily movement check.',
      actionLabel: 'Launch Reach',
      mode: 'daily',
      task: 'reach',
      instructions: [
        'Ask the tester to record a Reach check.',
        'Watch whether Camera Setup makes sense.',
        'Watch whether they keep shoulder, elbow, wrist, and hand visible.',
        'After results appear, ask what they think the score means.',
      ],
    },
    {
      key: 'daily_sit_to_stand',
      title: 'Daily Sit-to-Stand Check',
      goal: 'Test whether full-body setup instructions are clear enough.',
      actionLabel: 'Launch Sit-to-Stand',
      mode: 'daily',
      task: 'sit_to_stand',
      instructions: [
        'Ask the tester to record Sit-to-Stand.',
        'Watch whether they place the camera from the side.',
        'Check whether hips, knees, ankles, feet, chair, and floor stay visible.',
        'After results appear, ask whether the result feels understandable.',
      ],
    },
    {
      key: 'rehab_consistency',
      title: 'Rehab Consistency Check',
      goal: 'Test whether Rehab mode feels different from Daily mode.',
      actionLabel: 'Launch Rehab',
      mode: 'rehab',
      task: null,
      instructions: [
        'Ask the tester to perform one slow repeated movement.',
        'Do not over-explain consistency.',
        'Watch whether they understand repeated controlled motion.',
        'After results appear, ask what “consistency” means to them.',
      ],
    },
    {
      key: 'history_overview',
      title: 'History + Movement Overview',
      goal: 'Test whether the tester understands tracking over time.',
      actionLabel: 'Open History',
      mode: 'history',
      task: null,
      instructions: [
        'Ask the tester to open History.',
        'Watch whether Movement Change Overview makes sense.',
        'Ask what they think baseline means.',
        'Ask what movement they would record next.',
      ],
    },
    {
      key: 'feedback_notes',
      title: 'Save Tester Feedback',
      goal: 'Record the tester’s confusion immediately.',
      actionLabel: 'Open Feedback Notes',
      mode: 'feedback',
      task: null,
      instructions: [
        'Save where the tester got confused.',
        'Save any bug or timeout.',
        'Save any feature request.',
        'Save their overall reaction in their own words.',
      ],
    },
  ];
}

function getGuidedTestProgress(completedSteps: string[]) {
  const steps = getGuidedTestWorkflowSteps();
  const completedCount = steps.filter((step) =>
    completedSteps.includes(step.key)
  ).length;

  const totalCount = steps.length;
  const percent = Math.round((completedCount / totalCount) * 100);

  let status = 'Not Started';
  let summary = 'No guided testing steps have been completed yet.';
  let nextStep = 'Start with First Impression before explaining the app.';

  if (completedCount > 0) {
    status = 'Testing In Progress';
    summary = `${completedCount} of ${totalCount} guided testing steps completed.`;
    nextStep = 'Continue the next unchecked step and save tester feedback afterward.';
  }

  if (completedCount === totalCount) {
    status = 'Guided Test Complete';
    summary = 'All guided testing steps are complete for this test round.';
    nextStep = 'Review Feedback Notes and use the Feedback Action Plan to decide what to fix first.';
  }

  return {
    status,
    summary,
    nextStep,
    completedCount,
    totalCount,
    percent,
  };
}

function getTestingObservationPrompts() {
  return [
    'Where did the user pause or look confused?',
    'Did they understand what each mode was for?',
    'Did they know where to put the camera?',
    'Did they understand the score?',
    'Did they understand confidence?',
    'Did they understand baseline?',
    'Did they understand trend?',
    'Did they know what to do next after seeing results?',
    'Did any screen feel too crowded?',
    'Did any wording feel too technical?',
  ];
}

function getFeedbackSummary(notes: TesterNote[]) {
  if (notes.length === 0) {
    return {
      status: 'No Feedback Yet',
      summary: 'No tester feedback has been saved yet.',
      nextStep: 'Test the app with one person and save their confusion points.',
    };
  }

  const confusionCount = notes.filter((note) => note.confusionPoint.trim().length > 0).length;
  const bugCount = notes.filter((note) => note.bugFound.trim().length > 0).length;
  const requestCount = notes.filter((note) => note.featureRequest.trim().length > 0).length;

  let status = 'Feedback Started';
  let summary = `${notes.length} tester note${notes.length === 1 ? '' : 's'} saved.`;
  let nextStep = 'Review the notes and fix repeated confusion before testing more people.';

  if (notes.length >= 5) {
    status = 'Testing Round Complete';
    summary = `${notes.length} tester notes saved. This is enough for a first private testing round.`;
    nextStep = 'Group the repeated issues and fix the biggest confusion points first.';
  } else if (notes.length >= 3) {
    status = 'Useful Feedback Collected';
    summary = `${notes.length} tester notes saved. Patterns may start becoming visible.`;
    nextStep = 'Test 1–2 more people before making major design decisions.';
  }

  return {
    status,
    summary,
    nextStep,
    confusionCount,
    bugCount,
    requestCount,
  };
}

function getFeedbackActionPlan(notes: TesterNote[]) {
  if (notes.length === 0) {
    return {
      status: 'No Action Plan Yet',
      topPriority: 'Test the app with one person first.',
      reason: 'There are no tester notes yet, so there is nothing to prioritize.',
      fixFirst: 'Run one private test and save the tester’s confusion points.',
      testNext: 'Start with Daily Reach, Sit-to-Stand, Rehab, and History.',
      priorityItems: [],
    };
  }

  const confusionNotes = notes.filter(
    (note) => note.confusionPoint.trim().length > 0
  );

  const bugNotes = notes.filter(
    (note) => note.bugFound.trim().length > 0
  );

  const requestNotes = notes.filter(
    (note) => note.featureRequest.trim().length > 0
  );

  const baselineConfusions = notes.filter((note) =>
    note.confusionPoint.toLowerCase().includes('baseline')
  );

  const trendConfusions = notes.filter((note) =>
    note.confusionPoint.toLowerCase().includes('trend')
  );

  const cameraConfusions = notes.filter((note) =>
    note.confusionPoint.toLowerCase().includes('camera') ||
    note.confusionPoint.toLowerCase().includes('angle') ||
    note.confusionPoint.toLowerCase().includes('record')
  );

  const scoreConfusions = notes.filter((note) =>
    note.confusionPoint.toLowerCase().includes('score') ||
    note.confusionPoint.toLowerCase().includes('grade')
  );

  const priorityItems = [
    ...bugNotes.map((note) => ({
      type: 'Bug',
      testerName: note.testerName,
      text: note.bugFound,
    })),
    ...confusionNotes.map((note) => ({
      type: 'Confusion',
      testerName: note.testerName,
      text: note.confusionPoint,
    })),
    ...requestNotes.map((note) => ({
      type: 'Feature Request',
      testerName: note.testerName,
      text: note.featureRequest,
    })),
  ].slice(0, 6);

  let status = 'Feedback Needs Review';
  let topPriority = 'Review tester notes manually.';
  let reason = 'The app has feedback, but no dominant issue pattern is obvious yet.';
  let fixFirst = 'Look for repeated confusion across testers.';
  let testNext = 'Test with another person and compare whether the same confusion appears.';

  if (bugNotes.length > 0) {
    status = 'Fix Bugs First';
    topPriority = 'Fix technical issues before adding new features.';
    reason = `${bugNotes.length} bug-related note${bugNotes.length === 1 ? '' : 's'} found. Bugs block trust faster than confusing wording.`;
    fixFirst = 'Fix the most repeated or most severe bug before testing more users.';
    testNext = 'Re-test the same flow that caused the bug.';
  } else if (cameraConfusions.length >= 2) {
    status = 'Camera Setup Confusion';
    topPriority = 'Make recording setup impossible to miss.';
    reason = `${cameraConfusions.length} tester notes mention camera, recording, or angle confusion.`;
    fixFirst = 'Send users through the Camera Setup Guide before Daily and Rehab recordings, then test whether fewer bad recordings happen.';
    testNext = 'Test Daily Reach, Sit-to-Stand, and Rehab again with a new user without explaining the setup out loud.';
  } else if (baselineConfusions.length >= 2) {
    status = 'Baseline Confusion';
    topPriority = 'Explain baseline more clearly.';
    reason = `${baselineConfusions.length} tester notes mention baseline confusion.`;
    fixFirst = 'Rewrite baseline wording so users understand it means their normal score over repeated checks.';
    testNext = 'Ask the next tester what they think baseline means before explaining it.';
  } else if (trendConfusions.length >= 2) {
    status = 'Trend Confusion';
    topPriority = 'Clarify trend interpretation.';
    reason = `${trendConfusions.length} tester notes mention trend confusion.`;
    fixFirst = 'Make trend wording shorter and more action-focused.';
    testNext = 'Ask the next tester what changed compared with the previous check.';
  } else if (scoreConfusions.length >= 2) {
    status = 'Score Confusion';
    topPriority = 'Explain scores and grades better.';
    reason = `${scoreConfusions.length} tester notes mention score or grade confusion.`;
    fixFirst = 'Add clearer score meaning: higher score means more stable or more consistent movement.';
    testNext = 'Ask the next tester what they think the score means.';
  } else if (confusionNotes.length >= 3) {
    status = 'UX Confusion Pattern';
    topPriority = 'Simplify the most confusing screen.';
    reason = `${confusionNotes.length} confusion note${confusionNotes.length === 1 ? '' : 's'} saved.`;
    fixFirst = 'Identify which screen appears most often in confusion notes and simplify it.';
    testNext = 'Run one more test without explaining the app and watch where they pause.';
  } else if (requestNotes.length >= 3) {
    status = 'Feature Requests Emerging';
    topPriority = 'Review feature requests, but do not build them yet.';
    reason = `${requestNotes.length} feature request${requestNotes.length === 1 ? '' : 's'} saved.`;
    fixFirst = 'Only build a requested feature if multiple users ask for the same thing.';
    testNext = 'Ask future testers what they expected the app to do next.';
  } else if (notes.length >= 5) {
    status = 'Ready to Prioritize Fixes';
    topPriority = 'Group repeated feedback and choose the highest-impact fix.';
    reason = 'A full first testing round has enough notes to begin prioritizing changes.';
    fixFirst = 'Fix repeated confusion before adding new features.';
    testNext = 'Run a second testing round after fixes.';
  }

  return {
    status,
    topPriority,
    reason,
    fixFirst,
    testNext,
    priorityItems,
  };
}

function getTesterIssueCategory(text: string) {
  const lower = text.toLowerCase();

  if (
    lower.includes('camera') ||
    lower.includes('angle') ||
    lower.includes('record') ||
    lower.includes('frame') ||
    lower.includes('visible') ||
    lower.includes('lighting')
  ) {
    return 'Camera Setup';
  }

  if (
    lower.includes('baseline') ||
    lower.includes('normal range') ||
    lower.includes('usual score')
  ) {
    return 'Baseline Understanding';
  }

  if (
    lower.includes('trend') ||
    lower.includes('change') ||
    lower.includes('improvement') ||
    lower.includes('decline')
  ) {
    return 'Trend Understanding';
  }

  if (
    lower.includes('score') ||
    lower.includes('grade') ||
    lower.includes('points') ||
    lower.includes('number')
  ) {
    return 'Score Understanding';
  }

  if (
    lower.includes('history') ||
    lower.includes('overview') ||
    lower.includes('dashboard') ||
    lower.includes('screen')
  ) {
    return 'Navigation / Screen Layout';
  }

  if (
    lower.includes('slow') ||
    lower.includes('lag') ||
    lower.includes('timeout') ||
    lower.includes('network') ||
    lower.includes('error') ||
    lower.includes('crash') ||
    lower.includes('analyze')
  ) {
    return 'Technical Reliability';
  }

  if (
    lower.includes('rehab') ||
    lower.includes('consistency') ||
    lower.includes('repeat') ||
    lower.includes('controlled')
  ) {
    return 'Rehab Understanding';
  }

  return 'Other';
}

function countCategories(items: string[]) {
  const counts: Record<string, number> = {};

  items.forEach((item) => {
    if (item.trim().length === 0) return;

    const category = getTesterIssueCategory(item);
    counts[category] = (counts[category] || 0) + 1;
  });

  return Object.entries(counts)
    .map(([category, count]) => ({
      category,
      count,
    }))
    .sort((a, b) => b.count - a.count);
}

function getTesterAnalytics(notes: TesterNote[]) {
  const confusionTexts = notes
    .map((note) => note.confusionPoint)
    .filter((text) => text.trim().length > 0);

  const bugTexts = notes
    .map((note) => note.bugFound)
    .filter((text) => text.trim().length > 0);

  const requestTexts = notes
    .map((note) => note.featureRequest)
    .filter((text) => text.trim().length > 0);

  const reactionTexts = notes
    .map((note) => note.overallReaction)
    .filter((text) => text.trim().length > 0);

  const confusionCategories = countCategories(confusionTexts);
  const bugCategories = countCategories(bugTexts);
  const requestCategories = countCategories(requestTexts);

  const topConfusion = confusionCategories[0]?.category || 'None yet';
  const topBug = bugCategories[0]?.category || 'None yet';
  const topRequest = requestCategories[0]?.category || 'None yet';

  let status = 'No Tester Data Yet';
  let summary = 'No tester notes have been saved yet.';
  let fixFocus = 'Run one guided test and save feedback notes.';
  let nextTest = 'Start with one tester and watch where they get confused.';

  if (notes.length > 0) {
    status = 'Tester Data Started';
    summary = `${notes.length} tester note${notes.length === 1 ? '' : 's'} saved.`;
    fixFocus = 'Keep collecting notes until patterns repeat.';
    nextTest = 'Test another person and compare whether the same issues appear.';
  }

  if (notes.length >= 3) {
    status = 'Early Patterns Visible';
    summary = `${notes.length} tester notes saved. Early confusion patterns may be useful.`;

    if (bugTexts.length > 0) {
      fixFocus = `Fix technical reliability first. Top bug category: ${topBug}.`;
    } else if (confusionTexts.length > 0) {
      fixFocus = `Reduce confusion first. Top confusion category: ${topConfusion}.`;
    } else if (requestTexts.length > 0) {
      fixFocus = `Review repeated feature requests. Top request category: ${topRequest}.`;
    }

    nextTest = 'Run 2 more tests after fixing the biggest repeated issue.';
  }

  if (notes.length >= 5) {
    status = 'First Analytics Round Complete';
    summary = `${notes.length} tester notes saved. This is enough to prioritize the first cleanup pass.`;

    if (bugTexts.length > 0) {
      fixFocus = `Fix bugs before adding features. Most common bug area: ${topBug}.`;
    } else if (confusionTexts.length > 0) {
      fixFocus = `Simplify the most confusing area: ${topConfusion}.`;
    } else {
      fixFocus = 'No major bug or confusion pattern dominates yet. Keep testing before major changes.';
    }

    nextTest = 'After fixing the top issue, run another guided test with a new person.';
  }

  return {
    status,
    summary,
    fixFocus,
    nextTest,
    totalNotes: notes.length,
    confusionCount: confusionTexts.length,
    bugCount: bugTexts.length,
    requestCount: requestTexts.length,
    reactionCount: reactionTexts.length,
    topConfusion,
    topBug,
    topRequest,
    confusionCategories,
    bugCategories,
    requestCategories,
  };
}

function getRolloutDashboardStatus(
  sessions: SavedSession[],
  notes: TesterNote[]
) {
  const testingReadiness = getTestingReadiness(sessions);
  const feedbackSummary = getFeedbackSummary(notes);
  const feedbackActionPlan = getFeedbackActionPlan(notes);
  const movementOverview = getMovementOverview(sessions);

  const totalMovementChecks = sessions.length;
  const totalFeedbackNotes = notes.length;

  const hasDailyData =
    getSavedSessionsForTask(sessions, 'reach').length > 0 ||
    getSavedSessionsForTask(sessions, 'arm_raise').length > 0 ||
    getSavedSessionsForTask(sessions, 'sit_to_stand').length > 0 ||
    getSavedSessionsForTask(sessions, 'walking').length > 0 ||
    getSavedSessionsForTask(sessions, 'balance').length > 0 ||
    getSavedSessionsForTask(sessions, 'timed_up_and_go').length > 0;

  const hasRehabData = getSavedRehabSessions(sessions).length > 0;

  const hasFeedback = notes.length > 0;
  const hasEnoughFeedback = notes.length >= 3;
  const hasFirstTestingRound = notes.length >= 5;

  let status = 'Internal Testing Needed';
  let summary = 'The app has core features, but it still needs structured internal testing before wider rollout.';
  let nextAction = 'Run each main movement mode yourself and confirm the flow feels clear.';
  let warning = 'Do not launch publicly until camera setup, scoring, baseline, and feedback flow are understandable to testers.';

  if (totalMovementChecks === 0) {
    status = 'No Test Data Yet';
    summary = 'No saved movement checks are available yet.';
    nextAction = 'Record Daily Reach, Daily Sit-to-Stand, and Rehab before testing with other people.';
    warning = 'Without saved checks, History, baseline, and Movement Overview cannot be demonstrated.';
  } else if (!hasDailyData || !hasRehabData) {
    status = 'Core Modes Need Data';
    summary = 'Some core modes have data, but not all important testing flows are covered yet.';
    nextAction = 'Record at least one Daily movement check and one Rehab check.';
    warning = 'Do not test externally until both Daily and Rehab flows have been checked.';
  } else if (!hasFeedback) {
    status = 'Ready for First Private Tester';
    summary = 'Core movement flows have data. The next step is to test with one real person.';
    nextAction = 'Use the Testing Guide with one tester and save their feedback notes.';
    warning = 'Do not explain every screen while they test. Watch where they get confused.';
  } else if (!hasEnoughFeedback) {
    status = 'Private Testing Started';
    summary = `${totalFeedbackNotes} tester note${totalFeedbackNotes === 1 ? '' : 's'} saved. More feedback is needed before making major product decisions.`;
    nextAction = 'Test 2–3 more people and look for repeated confusion patterns.';
    warning = 'Do not build new major features yet. Keep collecting feedback.';
  } else if (!hasFirstTestingRound) {
    status = 'Feedback Pattern Forming';
    summary = `${totalFeedbackNotes} tester notes saved. Early patterns may be visible.`;
    nextAction = feedbackActionPlan.fixFirst;
    warning = 'Fix only the clearest repeated issue. Do not chase every single suggestion.';
  } else {
    status = 'First Testing Round Complete';
    summary = 'You have enough feedback for a first private testing cycle.';
    nextAction = feedbackActionPlan.fixFirst;
    warning = 'Prioritize bugs and repeated confusion before adding new capabilities.';
  }

  if (feedbackActionPlan.status === 'Fix Bugs First') {
    status = 'Fix Bugs Before More Testing';
    summary = 'Tester feedback includes at least one technical issue.';
    nextAction = feedbackActionPlan.fixFirst;
    warning = 'Bugs destroy trust faster than missing features. Fix them before expanding testing.';
  }

  return {
    status,
    summary,
    nextAction,
    warning,
    totalMovementChecks,
    totalFeedbackNotes,
    testingReadiness,
    feedbackSummary,
    feedbackActionPlan,
    movementOverview,
  };
}

function getBetaLaunchReadiness(
  sessions: SavedSession[],
  notes: TesterNote[],
  completedSteps: string[]
) {
  const rollout = getRolloutDashboardStatus(sessions, notes);
  const guidedProgress = getGuidedTestProgress(completedSteps);

  const hasMovementData = sessions.length > 0;
  const hasDailyData =
    getSavedSessionsForTask(sessions, 'reach').length > 0 ||
    getSavedSessionsForTask(sessions, 'arm_raise').length > 0 ||
    getSavedSessionsForTask(sessions, 'sit_to_stand').length > 0;

  const hasRehabData = getSavedRehabSessions(sessions).length > 0;
  const hasFeedback = notes.length > 0;
  const hasTestingRound = notes.length >= 5;
  const guidedComplete = guidedProgress.percent === 100;

  let score = 10;

  if (hasMovementData) score += 15;
  if (hasDailyData) score += 15;
  if (hasRehabData) score += 15;
  if (hasFeedback) score += 15;
  if (notes.length >= 3) score += 10;
  if (hasTestingRound) score += 10;
  if (guidedComplete) score += 10;

  score = Math.min(100, score);

  let status = 'Not Launch Ready';
  let summary = 'The app needs more private testing before you should push it broadly.';
  let nextStep = 'Run the Guided Test with at least one tester and save feedback notes.';
  let launchAdvice = 'Do not chase downloads yet. First prove that strangers can understand the app without you explaining it.';

  if (score >= 80) {
    status = 'Ready for Small Beta Launch';
    summary = 'The app has enough testing structure and feedback tracking to share with a small group.';
    nextStep = 'Share the beta invite with 5–10 people and collect feedback after each test.';
    launchAdvice = 'Keep the beta small. You want useful feedback before trying to get lots of downloads.';
  } else if (score >= 55) {
    status = 'Almost Beta Ready';
    summary = 'The app is close to beta-ready, but you still need more tester feedback or guided test progress.';
    nextStep = 'Complete one full Guided Test and save at least 3 tester notes.';
    launchAdvice = 'You can start recruiting testers, but do not advertise it like a finished product yet.';
  } else if (score >= 35) {
    status = 'Internal Testing Stage';
    summary = 'The app has strong features, but the rollout proof is still thin.';
    nextStep = 'Use the Guided Test Workflow with one person and write down exactly where they get confused.';
    launchAdvice = 'Your goal right now is learning, not downloads.';
  }

  return {
    score,
    status,
    summary,
    nextStep,
    launchAdvice,
    rolloutStatus: rollout.status,
    guidedProgressPercent: guidedProgress.percent,
    testerNoteCount: notes.length,
    movementCheckCount: sessions.length,
  };
}

function getBetaInviteText() {
  return `I'm testing Kinetra Beta.

Kinetra uses your phone camera to review simple movements like reaching, sit-to-stand, balance, walking, and controlled repeated motion. The goal is to help people understand movement control, stability, consistency, mobility, and change over time.

It is not medical advice and does not diagnose anything. I'm mainly looking for feedback on whether Kinetra is understandable, easy to use, and useful.

If you're willing to test it, I would ask you to:
1. Try one Daily movement check
2. Try one Rehab Consistency check
3. Open the Mobility Profile or Kinetra Passport
4. Tell me what felt confusing or useful

The whole test should only take a few minutes.`;
}

function getShortBetaPitch() {
  return 'Kinetra is a camera-based movement intelligence app that helps users track movement stability, control, mobility, and change over time.';
}

function getIdealTesterGroups() {
  return [
    {
      title: 'General users',
      reason: 'They test whether the app makes sense without technical explanation.',
    },
    {
      title: 'Students or athletes',
      reason: 'They may care about movement quality, consistency, and performance tracking.',
    },
    {
      title: 'Rehab-adjacent users',
      reason: 'They can test whether controlled movement tracking feels useful without medical claims.',
    },
    {
      title: 'Coaches, teachers, or mentors',
      reason: 'They can judge whether the explanations and graphs are understandable.',
    },
  ];
}

function getBetaLaunchChecklist() {
  return [
    'Test the app yourself before handing it to someone.',
    'Make sure your Flask backend and Expo app are both running.',
    'Use the Camera Setup Guide before recording.',
    'Do not explain every screen while the tester uses it.',
    'Save tester confusion in Feedback Notes immediately.',
    'Use the Feedback Action Plan before adding more features.',
  ];
}

function getFriendlyAnalysisError(errorMessage: string, mode: 'rep' | 'rehab' | 'lab' | 'daily') {
  const lower = errorMessage.toLowerCase();

  if (
    lower.includes('aborted') ||
    lower.includes('timeout') ||
    lower.includes('network request timed out')
  ) {
    return mode === 'daily'
      ? 'Analysis took too long. Re-record a shorter, clearer video and make sure your backend server is running.'
      : 'Analysis took too long. Try a shorter recording and make sure your backend server is running.';
  }

  if (
    lower.includes('network request failed') ||
    lower.includes('failed to fetch') ||
    lower.includes('network')
  ) {
    return `Could not connect to the analysis server at ${API_BASE_URL}. If you are testing locally, make sure your phone and computer are on the same Wi-Fi and Flask is running. If you are rolling out, use a deployed backend URL.`;
  }

  if (
    lower.includes('analysis timeout') ||
    lower.includes('abort') ||
    lower.includes('aborted') ||
    lower.includes('timed out')
  ) {
    return 'Analysis took too long. Try a shorter video first. If this keeps happening, the backend may need deployment, more compute, or a smaller upload size.';
  }

  if (
    lower.includes('json') ||
    lower.includes('unexpected end') ||
    lower.includes('raw_output')
  ) {
    return 'The backend returned an incomplete result. Restart the Flask server and try again with a shorter video.';
  }

  return mode === 'daily'
    ? 'Analysis could not finish. Try a shorter, clearer recording with the needed body parts visible.'
    : 'Analysis could not finish. Try again with a clearer recording.';
}

function getOverallLabel(result: AnalysisResult | null) {
  if (!result) return 'No Analysis Yet';

  const grades = [
    result.global_metrics.smoothness_grade,
    result.global_metrics.symmetry_grade,
    result.global_metrics.control_grade,
  ].join(' ').toLowerCase();

  if (grades.includes('poor')) return 'Needs Work';
  if (grades.includes('moderate')) return 'Decent';
  if (grades.includes('good') || grades.includes('excellent')) return 'Strong';

  return 'Analyzed';
}

function formatMetric(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'N/A';
  }
  return value.toFixed(digits);
}

function formatTaskMetric(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'N/A';
  }
  return value.toFixed(digits);
}

function getServerHealthBadgeLabel(status: ServerHealthStatus) {
  if (status === 'online') return 'Server Online';
  if (status === 'offline') return 'Server Offline';
  if (status === 'checking') return 'Checking Server';
  return 'Server Not Checked';
}

function getServerHealthSummary(status: ServerHealthStatus) {
  if (status === 'online') {
    return 'The app can reach the analysis backend. Video uploads should be able to run.';
  }

  if (status === 'offline') {
    return 'The app cannot reach the analysis backend. Video analysis will fail until the server is running or deployed.';
  }

  if (status === 'checking') {
    return 'The app is checking whether the backend is reachable.';
  }

  return 'Check the server before testing uploads or giving the app to someone else.';
}

function SummaryPill({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <View style={styles.summaryPill}>
      <Text style={styles.summaryPillLabel}>{label}</Text>
      <Text style={styles.summaryPillValue}>{value}</Text>
    </View>
  );
}

function MetricCard({
  title,
  grade,
  value,
}: {
  title: string;
  grade: string;
  value: string;
}) {
  const colors = getGradeColors(grade);

  return (
    <View style={styles.metricCard}>
      <Text style={styles.metricTitle}>{title}</Text>
      <View
        style={[
          styles.gradeBadge,
          {
            backgroundColor: colors.bg,
            borderColor: colors.border,
          },
        ]}
      >
        <Text style={[styles.gradeBadgeText, { color: colors.text }]}>{grade}</Text>
      </View>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

export default function HomeScreen() {
  const [started, setStarted] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatusMessage, setAnalysisStatusMessage] = useState('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [showResultsPanel, setShowResultsPanel] = useState(true);
  const [serverHealth, setServerHealth] = useState<ServerHealthState>({
    status: 'unchecked',
    message: 'Server has not been checked yet.',
    apiUrl: API_BASE_URL,
  });
  const [showServerDiagnostics, setShowServerDiagnostics] = useState(false);
  const [mode, setMode] = useState<'rep' | 'rehab' | 'lab' | 'daily'>('rep');
  const [dailyTask, setDailyTask] = useState<DailyTask>('reach');
  // Phase 1: which arm the backend should track. Defaults to 'right', which
  // is exactly the arm the backend always tracked before this selector
  // existed -- so a user who never touches this gets identical behavior.
  const [selectedSide, setSelectedSide] = useState<'left' | 'right'>('right');
  // Phase 2: optional personalized rep-detection thresholds, stored per arm
  // (a real, common case for this app's rehab audience: someone's affected
  // and unaffected arm can have genuinely different ranges of motion, so one
  // shared calibration for both sides would silently misgrade whichever arm
  // it wasn't measured from). null for a side means "use the backend's
  // built-in default range" -- also identical to pre-calibration-feature
  // behavior. Persisted to AsyncStorage so it survives an app restart;
  // loaded once on mount by loadCalibration() below.
  const [calibratedThresholdsBySide, setCalibratedThresholdsBySide] = useState<{
    left: { flex: number; extend: number } | null;
    right: { flex: number; extend: number } | null;
  }>({ left: null, right: null });
  // The calibration that actually applies to whichever arm is selected
  // right now -- this is what upload/display code should read.
  const calibratedThresholds = calibratedThresholdsBySide[selectedSide];
  // True only for the one recording immediately after the user taps
  // "Calibrate My Range" -- tells the result handler to turn that
  // recording's observed angle range into a calibration suggestion instead
  // of just showing normal results.
  const [isCalibrating, setIsCalibrating] = useState(false);
  // `side` records which arm was actually selected during the calibration
  // recording, so confirming "Use These" always saves to the correct arm's
  // slot even if the user switches the arm selector before confirming.
  const [pendingCalibrationSuggestion, setPendingCalibrationSuggestion] = useState<{
    side: 'left' | 'right';
    flex: number;
    extend: number;
    observedMin: number;
    observedMax: number;
  } | null>(null);
  // Only show the pending suggestion card while the selector is still on the
  // arm it was actually measured for -- switching arms hides (not discards)
  // it, so it reappears correctly if the user switches back.
  const relevantPendingCalibrationSuggestion =
    pendingCalibrationSuggestion && pendingCalibrationSuggestion.side === selectedSide
      ? pendingCalibrationSuggestion
      : null;
  // Separate from analysisError on purpose: a calibration recording that
  // didn't show enough range of motion is not an analysis failure -- the
  // video analyzed fine and its result is still shown normally. This just
  // explains why no calibration suggestion appeared.
  const [calibrationMessage, setCalibrationMessage] = useState<string | null>(null);
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>([]);
  const [comparisonSession, setComparisonSession] = useState<SavedSession | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [showTestingGuide, setShowTestingGuide] = useState(false);
  const [showFeedbackNotes, setShowFeedbackNotes] = useState(false);
  const [showCameraSetupGuide, setShowCameraSetupGuide] = useState(false);
  const [showTransparency, setShowTransparency] = useState(false);
  // Phase 3: Team Screening. When enabled, the athlete name typed below is
  // attached to the next saved session -- reusing the exact same
  // single-user recording flow underneath, not a separate pipeline. Team
  // Roster (below) then aggregates saved sessions by athlete_name.
  const [teamModeEnabled, setTeamModeEnabled] = useState(false);
  const [athleteNameInput, setAthleteNameInput] = useState('');
  const [showTeamRoster, setShowTeamRoster] = useState(false);
  const [showRolloutDashboard, setShowRolloutDashboard] = useState(false);
  const [showGuidedTestWorkflow, setShowGuidedTestWorkflow] = useState(false);
  const [completedGuidedSteps, setCompletedGuidedSteps] = useState<string[]>([]);
  const [showBetaLaunchKit, setShowBetaLaunchKit] = useState(false);
  const [showTesterAnalytics, setShowTesterAnalytics] = useState(false);
  const [showDailyHealthOverview, setShowDailyHealthOverview] = useState(false);
  const [showBuilderTools, setShowBuilderTools] = useState(false);
  const [showReportExport, setShowReportExport] = useState(false);
  const [showAiCoach, setShowAiCoach] = useState(false);
  // What's New / changelog. `lastSeenWhatsNewVersion` drives the small
  // unread dot on the Home nav chip -- it's set to the current APP_VERSION
  // the moment the screen is opened, so the dot disappears immediately
  // rather than needing an explicit "dismiss" action.
  const [showWhatsNew, setShowWhatsNew] = useState(false);
  const [lastSeenWhatsNewVersion, setLastSeenWhatsNewVersion] = useState<string | null>(null);
  // Phase 4 (startup pivot): a REAL LLM-generated note, layered onto the
  // existing (rule-based) AI Coach screen. This is intentionally separate
  // from `coach` (getMovementCoachPlan) above -- that stays fast, free, and
  // fully offline; this is an optional, explicit, one-tap request that
  // calls a Claude API proxy on our own backend (never the model directly
  // from the phone -- see server.py's /ai-coach route for why). Only
  // numeric scores/grades for the latest session are ever sent -- never
  // video, never raw pose data.
  const [llmCoachNote, setLlmCoachNote] = useState<string | null>(null);
  const [llmCoachLoading, setLlmCoachLoading] = useState(false);
  const [llmCoachError, setLlmCoachError] = useState<string | null>(null);
  const [llmCoachUnavailableReason, setLlmCoachUnavailableReason] = useState<string | null>(null);
  // Local, on-device daily reminder (no server, no push service -- just the
  // OS's own notification scheduler). `reminderHour` is in 24-hour local
  // time. Persisted so the schedule survives an app restart; re-scheduled
  // from scratch on load (Android/iOS clear scheduled local notifications
  // are NOT guaranteed to survive things like an app update, so treating
  // AsyncStorage as the source of truth and re-registering is more robust
  // than trusting the OS to have kept it).
  const [reminderEnabled, setReminderEnabled] = useState(false);
  const [reminderHour, setReminderHour] = useState(18);
  const [reminderStatusMessage, setReminderStatusMessage] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showWeeklyReport, setShowWeeklyReport] = useState(false);
  const [showYoloFramework, setShowYoloFramework] = useState(false);
  const [showMobilityProfile, setShowMobilityProfile] = useState(false);
  const [showTrendEngine, setShowTrendEngine] = useState(false);
  const [showGuidedOnboarding, setShowGuidedOnboarding] = useState(false);
  const [testerNotes, setTesterNotes] = useState<TesterNote[]>([]);
  const [testerName, setTesterName] = useState('');
  const [confusionPoint, setConfusionPoint] = useState('');
  const [bugFound, setBugFound] = useState('');
  const [featureRequest, setFeatureRequest] = useState('');
  const [overallReaction, setOverallReaction] = useState('');

  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  // Tracks the latest in-flight server check so stale responses from
  // earlier checks can't overwrite a newer result. Each call to
  // checkAnalysisServer bumps the counter; responses only apply state if
  // their captured counter is still current.
  const healthCheckTokenRef = useRef(0);
  // Same idea for the video upload: each upload is tagged with a token, and
  // the .then/.catch only applies state if the token still matches the
  // latest call. This prevents an older upload from clobbering the
  // analysis result of a newer recording.
  const uploadTokenRef = useRef(0);

  // ----- On-device pose (Step 1) -------------------------------------------
  // The hook loads MediaPipe Tasks Vision once at mount and exposes a
  // `detect(base64Jpeg)` function we can call against a still frame pulled
  // from the live camera. We surface only a tiny counter in the UI; the
  // real stick-figure drawing lands in Step 2.
  const onDevicePose = useOnDevicePose();
  const OnDevicePoseWorker = onDevicePose.Worker;
  const [onDeviceJointCount, setOnDeviceJointCount] = useState<number | null>(null);

  const loadSavedSessions = async () => {
    try {
      const raw = await AsyncStorage.getItem('movement_sessions_v1');
      const parsed = raw ? JSON.parse(raw) : [];
      setSavedSessions(parsed);
    } catch (error) {
      console.log('Failed to load saved sessions:', error);
    }
  };

  // Phase 2: load any previously-confirmed calibration for either arm so it
  // survives an app restart. Shape on disk: { left: {flex,extend}|null,
  // right: {flex,extend}|null }. Any parse failure or unexpected shape falls
  // back to "no calibration for either side" rather than crashing -- the
  // app is fully usable with default thresholds either way.
  const loadCalibration = async () => {
    try {
      const raw = await AsyncStorage.getItem('calibrated_thresholds_v1');
      if (!raw) return;

      const parsed = JSON.parse(raw);
      const isValidEntry = (entry: any) =>
        entry === null ||
        (entry && typeof entry.flex === 'number' && typeof entry.extend === 'number');

      if (parsed && isValidEntry(parsed.left) && isValidEntry(parsed.right)) {
        setCalibratedThresholdsBySide({
          left: parsed.left ?? null,
          right: parsed.right ?? null,
        });
      }
    } catch (error) {
      console.log('Failed to load calibration:', error);
    }
  };

  // Persists one side's calibration (or clears it with value=null) to both
  // state and AsyncStorage in one place, so the two call sites below can't
  // drift out of sync with each other.
  const persistCalibrationForSide = async (
    side: 'left' | 'right',
    value: { flex: number; extend: number } | null
  ) => {
    const updated = { ...calibratedThresholdsBySide, [side]: value };
    setCalibratedThresholdsBySide(updated);
    try {
      await AsyncStorage.setItem('calibrated_thresholds_v1', JSON.stringify(updated));
    } catch (error) {
      console.log('Failed to save calibration:', error);
    }
  };

  const loadReminderSettings = async () => {
    try {
      const raw = await AsyncStorage.getItem('reminder_settings_v1');
      if (!raw) return;
      const parsed = JSON.parse(raw);
      const hour = typeof parsed?.hour === 'number' && parsed.hour >= 0 && parsed.hour <= 23
        ? parsed.hour
        : 18;
      setReminderHour(hour);

      if (parsed?.enabled) {
        // Re-registering on every app start (rather than trusting the OS to
        // have kept a previous schedule) is deliberate -- see the comment
        // on the state declaration above.
        const granted = await ensureNotificationPermission();
        if (granted) {
          await scheduleReminderNotification(hour);
          setReminderEnabled(true);
        } else {
          // Permission was revoked since it was last enabled (e.g. in OS
          // settings) -- reflect that honestly instead of claiming it's on.
          setReminderEnabled(false);
        }
      }
    } catch (error) {
      console.log('Failed to load reminder settings:', error);
    }
  };

  const ensureNotificationPermission = async (): Promise<boolean> => {
    if (Platform.OS === 'web') {
      // expo-notifications does not support scheduled local notifications
      // on web -- this app's web export exists for development/testing,
      // not as a real target platform, so fail closed with a clear reason
      // rather than silently pretending it worked.
      setReminderStatusMessage('Reminders are not supported in the web preview -- try this on a real phone.');
      return false;
    }

    try {
      const existing = await Notifications.getPermissionsAsync();
      if (existing.granted) return true;

      const requested = await Notifications.requestPermissionsAsync();
      if (requested.granted) return true;

      setReminderStatusMessage(
        'Notification permission was not granted. Enable notifications for Kinetra in your phone settings, then try again.'
      );
      return false;
    } catch (error) {
      console.log('Failed to request notification permission:', error);
      setReminderStatusMessage('Could not request notification permission on this device.');
      return false;
    }
  };

  const scheduleReminderNotification = async (hour: number) => {
    try {
      await Notifications.cancelScheduledNotificationAsync(REMINDER_NOTIFICATION_IDENTIFIER).catch(() => {});

      await Notifications.scheduleNotificationAsync({
        identifier: REMINDER_NOTIFICATION_IDENTIFIER,
        content: {
          title: 'Movement check-in',
          body: 'Keep your streak going -- record a quick Kinetra check today.',
        },
        trigger: {
          type: Notifications.SchedulableTriggerInputTypes.DAILY,
          hour,
          minute: 0,
        },
      });
    } catch (error) {
      console.log('Failed to schedule reminder notification:', error);
      throw error;
    }
  };

  const setReminderPreference = async (enabled: boolean, hour: number) => {
    setReminderStatusMessage(null);

    if (!enabled) {
      setReminderEnabled(false);
      try {
        await Notifications.cancelScheduledNotificationAsync(REMINDER_NOTIFICATION_IDENTIFIER).catch(() => {});
        await AsyncStorage.setItem(
          'reminder_settings_v1',
          JSON.stringify({ enabled: false, hour })
        );
      } catch (error) {
        console.log('Failed to disable reminder:', error);
      }
      return;
    }

    const granted = await ensureNotificationPermission();
    if (!granted) {
      setReminderEnabled(false);
      return;
    }

    try {
      await scheduleReminderNotification(hour);
      setReminderEnabled(true);
      setReminderHour(hour);
      await AsyncStorage.setItem(
        'reminder_settings_v1',
        JSON.stringify({ enabled: true, hour })
      );
    } catch (error) {
      console.log('Failed to enable reminder:', error);
      setReminderStatusMessage('Could not schedule the reminder on this device. Try again.');
      setReminderEnabled(false);
    }
  };

  const findPreviousSessionForResult = (result: AnalysisResult) => {
    return savedSessions.find((session) => {
      if (session.mode !== result.mode) return false;

      if (result.mode === 'daily') {
        return session.daily_task === result.daily_task;
      }

      return true;
    }) || null;
  };

  const saveAnalysisSession = async (result: AnalysisResult) => {
    try {
      const primary =
        result.mode === 'rehab'
          ? (() => {
            const rehabAnalysis = getRehabConsistencyAnalysis(result, null);

            return {
              score: rehabAnalysis.score,
              grade: rehabAnalysis.status,
              label: 'Rehab Consistency',
            };
          })()
          : getPrimaryTaskScore(result);

      const confidence = getDailyConfidence(result);

      const newSession: SavedSession = {
        id: `${Date.now()}`,
        timestamp: new Date().toISOString(),
        mode: result.mode,
        daily_task: result.daily_task,
        daily_task_label: result.daily_task_label,
        primary_score: primary.score,
        primary_grade: primary.grade,
        confidence_grade: confidence.grade,
        side: result.side,
        thresholds_calibrated: result.calibration_data?.thresholds_calibrated,
        athlete_name:
          teamModeEnabled && athleteNameInput.trim() ? athleteNameInput.trim() : undefined,
      };

      const updatedSessions = [newSession, ...savedSessions].slice(0, 30);

      setSavedSessions(updatedSessions);
      await AsyncStorage.setItem(
        'movement_sessions_v1',
        JSON.stringify(updatedSessions)
      );

      // A new session just became "the latest" -- clear any AI Coach note
      // from a previous session so it can't be mistaken for feedback on
      // this new result.
      setLlmCoachNote(null);
      setLlmCoachError(null);
      setLlmCoachUnavailableReason(null);
    } catch (error) {
      console.log('Failed to save session:', error);
    }
  };

  const clearSavedSessions = async () => {
    try {
      setSavedSessions([]);
      setComparisonSession(null);
      await AsyncStorage.removeItem('movement_sessions_v1');
    } catch (error) {
      console.log('Failed to clear saved sessions:', error);
    }
  };

  const checkAnalysisServer = async () => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), SERVER_HEALTH_TIMEOUT_MS);
    // Tag this in-flight check. A newer call to checkAnalysisServer bumps
    // the ref, and any in-flight fetch that resolves later than the newer
    // one is ignored.
    const token = ++healthCheckTokenRef.current;

    setServerHealth({
      status: 'checking',
      message: 'Checking analysis server...',
      apiUrl: API_BASE_URL,
    });

    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (token !== healthCheckTokenRef.current) return null;

      if (response.ok) {
        setServerHealth({
          status: 'online',
          message: 'Backend health endpoint responded successfully.',
          apiUrl: API_BASE_URL,
          checkedAt: new Date().toISOString(),
        });

        return true;
      }

      if (response.status === 404) {
        setServerHealth({
          status: 'online',
          message: 'Server is reachable, but /health is not added yet. Add a Flask /health route before deployment.',
          apiUrl: API_BASE_URL,
          checkedAt: new Date().toISOString(),
        });

        return true;
      }

      setServerHealth({
        status: 'offline',
        message: `Server responded with status ${response.status}. The backend is reachable but not healthy.`,
        apiUrl: API_BASE_URL,
        checkedAt: new Date().toISOString(),
      });

      return false;
    } catch (error: any) {
      clearTimeout(timeoutId);

      if (token !== healthCheckTokenRef.current) return null;

      const message =
        error?.name === 'AbortError'
          ? 'Server check timed out. The backend may be asleep, overloaded, or unreachable.'
          : 'Could not reach the analysis server. Make sure Flask is running or use a deployed backend URL.';

      setServerHealth({
        status: 'offline',
        message,
        apiUrl: API_BASE_URL,
        checkedAt: new Date().toISOString(),
      });

      return false;
    }
  };

  const loadTesterNotes = async () => {
    try {
      const raw = await AsyncStorage.getItem('tester_notes_v1');
      const parsed = raw ? JSON.parse(raw) : [];
      setTesterNotes(parsed);
    } catch (error) {
      console.log('Failed to load tester notes:', error);
    }
  };

  const saveTesterNote = async () => {
    try {
      const hasAnyText =
        testerName.trim().length > 0 ||
        confusionPoint.trim().length > 0 ||
        bugFound.trim().length > 0 ||
        featureRequest.trim().length > 0 ||
        overallReaction.trim().length > 0;

      if (!hasAnyText) {
        return;
      }

      const newNote: TesterNote = {
        id: `${Date.now()}`,
        timestamp: new Date().toISOString(),
        testerName: testerName.trim() || 'Unnamed tester',
        confusionPoint: confusionPoint.trim(),
        bugFound: bugFound.trim(),
        featureRequest: featureRequest.trim(),
        overallReaction: overallReaction.trim(),
      };

      const updatedNotes = [newNote, ...testerNotes].slice(0, 30);

      setTesterNotes(updatedNotes);

      await AsyncStorage.setItem(
        'tester_notes_v1',
        JSON.stringify(updatedNotes)
      );

      setTesterName('');
      setConfusionPoint('');
      setBugFound('');
      setFeatureRequest('');
      setOverallReaction('');
    } catch (error) {
      console.log('Failed to save tester note:', error);
    }
  };

  const clearTesterNotes = async () => {
    try {
      setTesterNotes([]);
      await AsyncStorage.removeItem('tester_notes_v1');
    } catch (error) {
      console.log('Failed to clear tester notes:', error);
    }
  };

  const loadGuidedTestProgress = async () => {
    try {
      const raw = await AsyncStorage.getItem('guided_test_progress_v1');
      const parsed = raw ? JSON.parse(raw) : [];
      setCompletedGuidedSteps(parsed);
    } catch (error) {
      console.log('Failed to load guided test progress:', error);
    }
  };

  const toggleGuidedStepComplete = async (stepKey: string) => {
    try {
      const alreadyComplete = completedGuidedSteps.includes(stepKey);

      const updatedSteps = alreadyComplete
        ? completedGuidedSteps.filter((key) => key !== stepKey)
        : [...completedGuidedSteps, stepKey];

      setCompletedGuidedSteps(updatedSteps);

      await AsyncStorage.setItem(
        'guided_test_progress_v1',
        JSON.stringify(updatedSteps)
      );
    } catch (error) {
      console.log('Failed to update guided test progress:', error);
    }
  };

  const clearGuidedTestProgress = async () => {
    try {
      setCompletedGuidedSteps([]);
      await AsyncStorage.removeItem('guided_test_progress_v1');
    } catch (error) {
      console.log('Failed to clear guided test progress:', error);
    }
  };

  const shareBetaInvite = async () => {
    try {
      await Share.share({
        message: getBetaInviteText(),
      });
    } catch (error) {
      console.log('Failed to share beta invite:', error);
    }
  };

  const shareMovementReport = async () => {
    try {
      await Share.share({
        message: getMovementPassportText(savedSessions),
      });
    } catch (error) {
      console.log('Failed to share movement passport:', error);
    }
  };

  const shareCoachSummary = async () => {
    try {
      await Share.share({
        message: getCoachShareText(savedSessions),
      });
    } catch (error) {
      console.log('Failed to share coach summary:', error);
    }
  };

  const shareWeeklyReport = async () => {
    try {
      await Share.share({
        message: getWeeklyHealthReportText(savedSessions),
      });
    } catch (error) {
      console.log('Failed to share weekly report:', error);
    }
  };

  const shareMobilityProfile = async () => {
    try {
      await Share.share({
        message: getFunctionalMobilityProfileText(savedSessions),
      });
    } catch (error) {
      console.log('Failed to share mobility profile:', error);
    }
  };

  const shareLongitudinalTrendReport = async () => {
    try {
      await Share.share({
        message: getLongitudinalTrendText(savedSessions),
      });
    } catch (error) {
      console.log('Failed to share longitudinal trend report:', error);
    }
  };

  const requestLlmCoachNote = async () => {
    const latest = savedSessions[0] || null;

    if (!latest) {
      setLlmCoachError('Record and save at least one check first, then ask again.');
      return;
    }

    setLlmCoachLoading(true);
    setLlmCoachError(null);
    setLlmCoachUnavailableReason(null);
    setLlmCoachNote(null);

    const taskLabel =
      latest.daily_task_label ||
      (latest.daily_task ? getDailyTaskLabel(latest.daily_task) : null) ||
      (latest.mode === 'rehab' ? 'Rehab Consistency Check' : 'Movement Check');

    const recentScores = (
      latest.daily_task
        ? getSavedSessionsForTask(savedSessions, latest.daily_task)
        : latest.mode === 'rehab'
          ? getSavedRehabSessions(savedSessions)
          : [latest]
    )
      .slice(0, 6)
      .map((session) => session.primary_score)
      .filter((score): score is number => typeof score === 'number')
      .reverse(); // oldest -> newest, matching the backend prompt's expectation

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
      const response = await fetch(`${API_BASE_URL}/ai-coach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_label: taskLabel,
          mode: latest.mode,
          primary_score: latest.primary_score,
          primary_grade: latest.primary_grade,
          confidence_grade: latest.confidence_grade,
          side: latest.side,
          thresholds_calibrated: latest.thresholds_calibrated,
          recent_scores: recentScores,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const body = await response.json().catch(() => null);

      if (response.status === 503) {
        setLlmCoachUnavailableReason(
          "The app's owner hasn't turned AI Coach on for this server yet -- it needs an API key configured on the backend. Everything else in the app still works normally."
        );
        return;
      }

      if (!response.ok || !body?.feedback) {
        setLlmCoachError(
          body?.details || 'AI Coach could not respond right now. Try again in a moment.'
        );
        return;
      }

      setLlmCoachNote(body.feedback);
    } catch (error: any) {
      clearTimeout(timeoutId);
      reportError(error, { route: '/ai-coach' });
      setLlmCoachError(
        error?.name === 'AbortError'
          ? 'AI Coach took too long to respond. Try again shortly.'
          : `Could not reach the AI Coach service at ${API_BASE_URL}.`
      );
    } finally {
      setLlmCoachLoading(false);
    }
  };

  const loadOnboardingStatus = async () => {
    try {
      const hasSeenOnboarding = await AsyncStorage.getItem('has_seen_onboarding_v1');

      if (!hasSeenOnboarding) {
        setShowOnboarding(true);
      }
    } catch (error) {
      console.log('Failed to load onboarding status:', error);
    }
  };

  const finishOnboarding = async () => {
    try {
      await AsyncStorage.setItem('has_seen_onboarding_v1', 'true');
      setShowOnboarding(false);
    } catch (error) {
      console.log('Failed to save onboarding status:', error);
      setShowOnboarding(false);
    }
  };

  const loadWhatsNewStatus = async () => {
    try {
      const raw = await AsyncStorage.getItem('last_seen_whats_new_version_v1');
      setLastSeenWhatsNewVersion(raw);
    } catch (error) {
      console.log('Failed to load What\'s New status:', error);
    }
  };

  const openWhatsNew = async () => {
    setShowWhatsNew(true);
    try {
      await AsyncStorage.setItem('last_seen_whats_new_version_v1', APP_VERSION);
      setLastSeenWhatsNewVersion(APP_VERSION);
    } catch (error) {
      console.log('Failed to save What\'s New status:', error);
    }
  };

  useEffect(() => {
    requestPermission();
    loadSavedSessions();
    loadTesterNotes();
    loadGuidedTestProgress();
    loadOnboardingStatus();
    loadCalibration();
    loadReminderSettings();
    loadWhatsNewStatus();
  }, [requestPermission]);

  // ----- On-device pose snapshot loop (Step 1) -----------------------------
  // While the user is recording, grab a still frame from the camera once a
  // second and ask the on-device pose model how many joints it can see.
  // We deliberately don't try to be faster than ~1 fps here: takePictureAsync
  // and PoseLandmarker.detect are both heavy on a phone, and Step 1 only
  // needs to prove the brain is alive. Step 2 will swap this for true
  // per-frame processing.
  useEffect(() => {
    if (!recording || !cameraReady || !onDevicePose.ready) return;

    let cancelled = false;
    const tick = async () => {
      try {
        if (!cameraRef.current || cancelled) return;
        // Use a higher JPEG quality (0.7) and skipProcessing so the model
        // gets a clean, recognizable image. With quality 0.3 the model was
        // returning 0 landmarks even on clear subjects.
        const snapshot = await cameraRef.current.takePictureAsync({
          quality: 0.7,
          base64: true,
          skipProcessing: true,
        });
        if (cancelled || !snapshot?.base64) return;
        const dataUri = `data:image/jpeg;base64,${snapshot.base64}`;
        const count = await onDevicePose.detect(dataUri);
        if (!cancelled) {
          console.log('[onDevicePose] tick count:', count);
          setOnDeviceJointCount(count);
        }
      } catch (err) {
        // Snapshot can occasionally race with the camera teardown on stop.
        // Swallow it; the loop will try again on the next tick or stop.
        console.log('On-device pose tick failed:', err);
      }
    };

    const intervalId = setInterval(tick, 1000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [recording, cameraReady, onDevicePose.ready, onDevicePose.detect]);

  // Android hardware back button: if the user is mid-recording or mid-
  // analysis, popping out of the screen would lose work. Confirm with a
  // dialog before letting the back action through. On iOS there's no
  // hardware back button so this is a no-op.
  useEffect(() => {
    if (!started) {
      return;
    }
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (recording) {
        Alert.alert(
          'Recording in progress',
          'Stop the recording and leave the camera?',
          [
            { text: 'Stay', style: 'cancel' },
            {
              text: 'Stop & Leave',
              style: 'destructive',
              onPress: () => {
                stopRecording();
                setStarted(false);
              },
            },
          ],
        );
        return true;
      }
      if (isAnalyzing) {
        Alert.alert(
          'Analysis in progress',
          'Leave while the analysis is still running? The result will be discarded.',
          [
            { text: 'Stay', style: 'cancel' },
            {
              text: 'Leave',
              style: 'destructive',
              onPress: () => setStarted(false),
            },
          ],
        );
        return true;
      }
      setStarted(false);
      return true;
    });
    return () => sub.remove();
  }, [started, recording, isAnalyzing]);

  const startRecording = async () => {
    if (!cameraReady || !cameraRef.current) {
      console.log('Camera not ready yet');
      return;
    }

    try {
      setRecording(true);
      setAnalysisResult(null);
      setAnalysisError(null);
      setAnalysisStatusMessage('');
      setComparisonSession(null);
      setShowDetails(false);
      setShowResultsPanel(true);

      const result = await cameraRef.current.recordAsync({
        maxDuration: mode === 'daily' || mode === 'rehab' ? 12 : 30,
      });

      setRecording(false);

      if (result?.uri) {
        setVideoUri(result.uri);
        setShowResultsPanel(true);
      }
    } catch (error) {
      setRecording(false);
      console.log('Recording error:', error);
    }
  };

  const stopRecording = () => {
    if (cameraRef.current) {
      cameraRef.current.stopRecording();
    }
  };

  const replayVideo = () => {
    setShowResultsPanel(false);

    try {
      player.seekBy(-9999);
      player.play();
    } catch (error) {
      console.log('Replay error:', error);
    }
  };

  const uploadVideo = async (uri: string) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);
    // Tag this upload. If a newer upload starts (the user records again
    // before this one resolves), our late state updates are dropped so the
    // newer recording's result wins.
    const token = ++uploadTokenRef.current;

    try {
      setIsAnalyzing(true);
      setAnalysisError(null);
      setAnalysisResult(null);
      setCalibrationMessage(null);
      setAnalysisStatusMessage('Checking analysis server...');

      const serverReady = await checkAnalysisServer();

      if (token !== uploadTokenRef.current) return null;

      if (!serverReady) {
        setAnalysisError(
          `Cannot analyze yet because the backend is not reachable at ${API_BASE_URL}. Check Server Diagnostics before trying again.`
        );
        setAnalysisStatusMessage('');
        return null;
      }

      setAnalysisStatusMessage('Preparing video upload...');

      const formData = new FormData();

      formData.append('video', {
        uri,
        name: 'video.mp4',
        type: 'video/mp4',
      } as any);

      formData.append('mode', mode);
      formData.append('daily_task', dailyTask);
      formData.append('side', selectedSide);

      // Only send calibration thresholds once the user has actually
      // confirmed a calibration suggestion. Omitting these fields entirely
      // when calibratedThresholds is null matches server.py's own default
      // (None -> use the backend's built-in range), so a user who never
      // calibrates gets byte-identical rep detection to before this existed.
      if (calibratedThresholds) {
        formData.append('flex_threshold', String(calibratedThresholds.flex));
        formData.append('extend_threshold', String(calibratedThresholds.extend));
      }

      console.log('Selected daily task:', dailyTask, 'side:', selectedSide);
      console.log('Uploading to API:', `${API_BASE_URL}/analyze`);

      setAnalysisStatusMessage('Uploading video to analysis server...');

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
        // Deliberately no Content-Type header here. fetch/FormData sets it
        // itself, including the multipart `boundary=...` that has to match
        // how the body was actually encoded -- manually setting a bare
        // 'multipart/form-data' (no boundary) is a known source of
        // intermittent upload failures on React Native.
        signal: controller.signal,
      });

      setAnalysisStatusMessage('Reading analysis results...');

      let data: any = null;

      try {
        data = await response.json();
      } catch (jsonError) {
        throw new Error(
          `Backend returned a non-JSON response with status ${response.status}. Restart Flask or check backend logs.`
        );
      }

      if (token !== uploadTokenRef.current) return null;

      if (!response.ok || data.error) {
        throw new Error(
          data.details || JSON.stringify(data.raw_output) || data.error || `Analysis failed with status ${response.status}`
        );
      }

      const typedResult = data as AnalysisResult;

      // If the user tapped "Calibrate My Range" before this recording, turn
      // the angle range we just observed into a suggested personalized
      // rep-detection window instead of touching the result shown below.
      // A 15% margin keeps the flex/extend boundaries safely inside the
      // observed extremes (noise near the very top/bottom of a real range
      // of motion shouldn't false-trigger a transition); a range narrower
      // than 20 degrees is too little motion to calibrate from safely, so
      // we silently skip the suggestion rather than lock in a degenerate
      // window -- the user keeps the default range in that case.
      if (isCalibrating) {
        const cal = typedResult.calibration_data;
        const min = cal?.observed_min_angle;
        const max = cal?.observed_max_angle;

        if (typeof min === 'number' && typeof max === 'number' && max - min >= 20) {
          const margin = (max - min) * 0.15;
          setPendingCalibrationSuggestion({
            side: selectedSide,
            flex: Math.round((min + margin) * 10) / 10,
            extend: Math.round((max - margin) * 10) / 10,
            observedMin: Math.round(min * 10) / 10,
            observedMax: Math.round(max * 10) / 10,
          });
          setCalibrationMessage(null);
        } else {
          setPendingCalibrationSuggestion(null);
          setCalibrationMessage(
            'Not enough movement range in that recording to calibrate. Move through your full comfortable range of motion and try again, or skip calibration.'
          );
        }
        setIsCalibrating(false);
      }

      const previousSession = findPreviousSessionForResult(typedResult);
      setComparisonSession(previousSession);

      setAnalysisResult(typedResult);
      await saveAnalysisSession(typedResult);

      setAnalysisStatusMessage('');

      return typedResult;
    } catch (error: any) {
      if (token !== uploadTokenRef.current) return null;

      console.error('Upload failed:', error);
      reportError(error, { route: '/analyze', mode });

      const rawMessage =
        error?.name === 'AbortError'
          ? 'Analysis timeout'
          : error?.message || 'Upload failed';

      setAnalysisError(getFriendlyAnalysisError(rawMessage, mode));
      setAnalysisStatusMessage('');

      return null;
    } finally {
      clearTimeout(timeoutId);
      if (token === uploadTokenRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const player = useVideoPlayer(videoUri ?? null, (player) => {
    player.loop = false;
  });

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.centeredContainer}>
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>{getModeLabel(mode)}</Text>
        </View>
        <Text style={styles.title}>Camera Permission Required</Text>
        <Text style={styles.subtitle}>
          Allow camera access so the app can capture movement and analyze motion quality biomechanically.
        </Text>
        <Pressable style={styles.mainButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow Camera</Text>
        </Pressable>
      </View>
    );
  }

  if (showOnboarding) {
    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.onboardingContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Movement Intelligence</Text>
        </View>

        <Text style={styles.title}>Check How Your Body Moves</Text>

        <Text style={styles.subtitle}>
          Use your phone camera to review movement stability, control, and consistency across simple daily movements.
        </Text>

        <View style={styles.onboardingCard}>
          <Text style={styles.onboardingStepNumber}>1</Text>
          <View style={styles.onboardingStepTextBlock}>
            <Text style={styles.onboardingStepTitle}>Choose a movement check</Text>
            <Text style={styles.onboardingStepText}>
              Start with Reach, Arm Raise, or Sit-to-Stand. Each check looks at different movement qualities.
            </Text>
          </View>
        </View>

        <View style={styles.onboardingCard}>
          <Text style={styles.onboardingStepNumber}>2</Text>
          <View style={styles.onboardingStepTextBlock}>
            <Text style={styles.onboardingStepTitle}>Record with a clear camera angle</Text>
            <Text style={styles.onboardingStepText}>
              Keep the important body parts visible. Better video quality means more trustworthy results.
            </Text>
          </View>
        </View>

        <View style={styles.onboardingCard}>
          <Text style={styles.onboardingStepNumber}>3</Text>
          <View style={styles.onboardingStepTextBlock}>
            <Text style={styles.onboardingStepTitle}>Review your result</Text>
            <Text style={styles.onboardingStepText}>
              Get a score, confidence level, biggest strength, biggest limitation, and next action.
            </Text>
          </View>
        </View>

        <View style={styles.onboardingCard}>
          <Text style={styles.onboardingStepNumber}>4</Text>
          <View style={styles.onboardingStepTextBlock}>
            <Text style={styles.onboardingStepTitle}>Track change over time</Text>
            <Text style={styles.onboardingStepText}>
              Saved checks build baselines, show trends, and help you see how movement quality changes over time.
            </Text>
          </View>
        </View>

        <View style={styles.onboardingNoteCard}>
          <Text style={styles.onboardingNoteTitle}>Important</Text>
          <Text style={styles.onboardingNoteText}>
            This app is for movement awareness and tracking. It does not diagnose, treat, or replace medical advice.
          </Text>
        </View>

        <Pressable style={styles.mainButton} onPress={finishOnboarding}>
          <Text style={styles.buttonText}>Start Movement Check</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showTestingGuide) {
    const testingReadiness = getTestingReadiness(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.testingGuideContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Private Testing</Text>
        </View>

        <Text style={styles.title}>Testing Guide</Text>

        <Text style={styles.subtitle}>
          Use this guide to test the app with real people and find confusing screens before rollout.
        </Text>

        <View style={styles.testingReadinessCard}>
          <Text style={styles.sectionTitle}>Testing Readiness</Text>

          <Text style={styles.testingReadinessScore}>
            {testingReadiness.readinessScore}/100
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(testingReadiness.status).bg,
                borderColor: getGradeColors(testingReadiness.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(testingReadiness.status).text },
              ]}
            >
              {testingReadiness.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {testingReadiness.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {testingReadiness.nextStep}
          </Text>

          <View style={styles.testingStatsGrid}>
            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{testingReadiness.dailyTasksStarted}</Text>
              <Text style={styles.testingStatLabel}>Daily Areas Started</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{testingReadiness.dailyTaskCount}</Text>
              <Text style={styles.testingStatLabel}>Daily Checks</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{testingReadiness.dailyBaselinesBuilt}</Text>
              <Text style={styles.testingStatLabel}>Daily Baselines</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{testingReadiness.rehabCount}</Text>
              <Text style={styles.testingStatLabel}>Rehab Checks</Text>
            </View>
          </View>
        </View>

        <View style={styles.testingNoteCard}>
          <Text style={styles.testingNoteTitle}>Testing Rule</Text>
          <Text style={styles.testingNoteText}>
            Do not explain every screen while the person is testing. Watch silently first. Confusion is the data.
          </Text>

          <Pressable
            style={styles.testingGuideInlineButton}
            onPress={() => {
              setShowTestingGuide(false);
              setShowCameraSetupGuide(true);
            }}
          >
            <Text style={styles.testingGuideButtonText}>Open Camera Setup Guide</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>Tester Task Sequence</Text>

        {getPrivateTestingTasks().map((task, index) => (
          <View key={task.title} style={styles.testingTaskCard}>
            <Text style={styles.testingTaskNumber}>Step {index + 1}</Text>
            <Text style={styles.testingTaskTitle}>{task.title}</Text>
            <Text style={styles.testingTaskGoal}>{task.goal}</Text>

            {task.instructions.map((instruction, instructionIndex) => (
              <Text key={instructionIndex} style={styles.testingTaskInstruction}>
                • {instruction}
              </Text>
            ))}
          </View>
        ))}

        <Text style={styles.sectionTitle}>Observation Checklist</Text>

        <View style={styles.testingTaskCard}>
          {getTestingObservationPrompts().map((prompt, index) => (
            <Text key={index} style={styles.testingTaskInstruction}>
              □ {prompt}
            </Text>
          ))}
        </View>

        <View style={styles.testingNoteCard}>
          <Text style={styles.testingNoteTitle}>After Each Test</Text>
          <Text style={styles.testingNoteText}>
            Write down the exact screen where the tester got confused, what they expected to happen, and what they actually did.
          </Text>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowTestingGuide(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showFeedbackNotes) {
    const feedbackSummary = getFeedbackSummary(testerNotes);
    const feedbackActionPlan = getFeedbackActionPlan(testerNotes);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.feedbackContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Tester Feedback</Text>
        </View>

        <Text style={styles.title}>Feedback Notes</Text>

        <Text style={styles.subtitle}>
          Save what testers actually say, where they get confused, and what needs to be fixed before rollout.
        </Text>

        <View style={styles.feedbackSummaryCard}>
          <Text style={styles.sectionTitle}>Feedback Summary</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(feedbackSummary.status).bg,
                borderColor: getGradeColors(feedbackSummary.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(feedbackSummary.status).text },
              ]}
            >
              {feedbackSummary.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackSummary.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackSummary.nextStep}
          </Text>

          <View style={styles.testingStatsGrid}>
            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{testerNotes.length}</Text>
              <Text style={styles.testingStatLabel}>Notes</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{feedbackSummary.confusionCount}</Text>
              <Text style={styles.testingStatLabel}>Confusions</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{feedbackSummary.bugCount}</Text>
              <Text style={styles.testingStatLabel}>Bugs</Text>
            </View>

            <View style={styles.testingStatBox}>
              <Text style={styles.testingStatValue}>{feedbackSummary.requestCount}</Text>
              <Text style={styles.testingStatLabel}>Requests</Text>
            </View>
          </View>
        </View>

        <View style={styles.feedbackActionPlanCard}>
          <Text style={styles.sectionTitle}>Feedback Action Plan</Text>

          <Text style={styles.metricLabelSmall}>Current Priority</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(feedbackActionPlan.status).bg,
                borderColor: getGradeColors(feedbackActionPlan.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(feedbackActionPlan.status).text },
              ]}
            >
              {feedbackActionPlan.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Top Priority
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackActionPlan.topPriority}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Why
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackActionPlan.reason}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Fix First
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackActionPlan.fixFirst}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Test Next
          </Text>
          <Text style={styles.metricValueLarge}>
            {feedbackActionPlan.testNext}
          </Text>

          {feedbackActionPlan.priorityItems.length > 0 ? (
            <>
              <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                Priority Notes
              </Text>

              {feedbackActionPlan.priorityItems.map((item, index) => (
                <View key={`${item.type}-${index}`} style={styles.priorityFeedbackItem}>
                  <Text style={styles.priorityFeedbackType}>
                    {item.type} — {item.testerName}
                  </Text>
                  <Text style={styles.priorityFeedbackText}>
                    {item.text}
                  </Text>
                </View>
              ))}
            </>
          ) : null}
        </View>

        <View style={styles.feedbackFormCard}>
          <Text style={styles.sectionTitle}>Add Tester Note</Text>

          <Text style={styles.inputLabel}>Tester Name</Text>
          <TextInput
            style={styles.feedbackInput}
            value={testerName}
            onChangeText={setTesterName}
            placeholder="Example: Ali, Tester 1, Mom"
            placeholderTextColor="#64748b"
          />

          <Text style={styles.inputLabel}>Where did they get confused?</Text>
          <TextInput
            style={styles.feedbackTextArea}
            value={confusionPoint}
            onChangeText={setConfusionPoint}
            placeholder="Example: They did not understand what baseline meant."
            placeholderTextColor="#64748b"
            multiline
          />

          <Text style={styles.inputLabel}>Bug or technical issue</Text>
          <TextInput
            style={styles.feedbackTextArea}
            value={bugFound}
            onChangeText={setBugFound}
            placeholder="Example: Analyze button felt slow, app timed out, camera angle issue."
            placeholderTextColor="#64748b"
            multiline
          />

          <Text style={styles.inputLabel}>Feature request or suggestion</Text>
          <TextInput
            style={styles.feedbackTextArea}
            value={featureRequest}
            onChangeText={setFeatureRequest}
            placeholder="Example: They wanted a clearer camera setup picture."
            placeholderTextColor="#64748b"
            multiline
          />

          <Text style={styles.inputLabel}>Overall reaction</Text>
          <TextInput
            style={styles.feedbackTextArea}
            value={overallReaction}
            onChangeText={setOverallReaction}
            placeholder="Example: They liked the history graph but ignored the Movement Overview."
            placeholderTextColor="#64748b"
            multiline
          />

          <Pressable style={styles.mainButton} onPress={saveTesterNote}>
            <Text style={styles.buttonText}>Save Tester Note</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>Saved Notes</Text>

        {testerNotes.length === 0 ? (
          <View style={styles.feedbackNoteCard}>
            <Text style={styles.feedbackNoteTitle}>No notes saved yet</Text>
            <Text style={styles.feedbackNoteText}>
              Test the app with one person, then save what confused them.
            </Text>
          </View>
        ) : (
          testerNotes.map((note) => (
            <View key={note.id} style={styles.feedbackNoteCard}>
              <Text style={styles.feedbackNoteTitle}>
                {note.testerName}
              </Text>

              <Text style={styles.feedbackNoteDate}>
                {formatSessionDate(note.timestamp)}
              </Text>

              {note.confusionPoint.length > 0 ? (
                <>
                  <Text style={styles.inputLabel}>Confusion</Text>
                  <Text style={styles.feedbackNoteText}>{note.confusionPoint}</Text>
                </>
              ) : null}

              {note.bugFound.length > 0 ? (
                <>
                  <Text style={styles.inputLabel}>Bug</Text>
                  <Text style={styles.feedbackNoteText}>{note.bugFound}</Text>
                </>
              ) : null}

              {note.featureRequest.length > 0 ? (
                <>
                  <Text style={styles.inputLabel}>Feature Request</Text>
                  <Text style={styles.feedbackNoteText}>{note.featureRequest}</Text>
                </>
              ) : null}

              {note.overallReaction.length > 0 ? (
                <>
                  <Text style={styles.inputLabel}>Overall Reaction</Text>
                  <Text style={styles.feedbackNoteText}>{note.overallReaction}</Text>
                </>
              ) : null}
            </View>
          ))
        )}

        {testerNotes.length > 0 ? (
          <Pressable style={styles.clearHistoryButton} onPress={clearTesterNotes}>
            <Text style={styles.clearHistoryButtonText}>Clear Tester Notes</Text>
          </Pressable>
        ) : null}

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowFeedbackNotes(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showTransparency) {
    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.cameraSetupContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Transparency</Text>
        </View>

        <Text style={styles.title}>How Kinetra Actually Measures Movement</Text>

        <Text style={styles.subtitle}>
          {APP_SAFETY_NOTE}
        </Text>

        <View style={styles.cameraSetupHeroCard}>
          <Text style={styles.sectionTitle}>What It Does</Text>
          <Text style={styles.metricValueLarge}>
            Kinetra uses your phone camera and Google MediaPipe pose-estimation model to
            estimate the 2D position of your joints in each video frame, then computes
            angles, speeds, and consistency from those positions. There is no depth sensor,
            no motion-capture markers, and no wearable hardware involved -- just video.
          </Text>
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Known Limitations</Text>

          <Text style={styles.cameraMistakeText}>
            • A single 2D camera cannot fully resolve depth, so movement directly toward or
            away from the camera is measured less reliably than side-to-side movement.
          </Text>
          <Text style={styles.cameraMistakeText}>
            • Poor lighting, loose clothing, or a joint leaving the frame all reduce
            tracking accuracy -- the Camera Setup Guide exists because setup genuinely
            affects results.
          </Text>
          <Text style={styles.cameraMistakeText}>
            • Rep-detection thresholds are either a general default range or a range you
            personally calibrated -- calibration is a simple geometric estimate from your
            own recorded motion, not a clinically validated procedure.
          </Text>
          <Text style={styles.cameraMistakeText}>
            • Kinetra has not been validated against a clinical, marker-based motion-capture
            system. Camera-only movement analysis is an active, credible area of research
            (single-camera pose estimation has been used in published gait-assessment
            studies), but that is a statement about the general approach, not a claim that
            Kinetra itself has been independently validated.
          </Text>
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>What Has Been Fixed So Far</Text>

          <Text style={styles.cameraMistakeText}>
            • Scores used to shift depending on the phone recording frame rate for the exact
            same movement. Velocity is now normalized to a fixed reference rate, so results
            are comparable across devices.
          </Text>
          <Text style={styles.cameraMistakeText}>
            • The backend used to silently assume everyone was moving their right arm, with
            no warning if that was not true. There is now an explicit arm selector, threaded
            through the entire analysis.
          </Text>
          <Text style={styles.cameraMistakeText}>
            • Rep-detection used one fixed range of motion for every person. You can now
            calibrate it to your own observed range instead, separately for each arm.
          </Text>
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Your Data</Text>
          <Text style={styles.metricValueLarge}>
            Recorded video is uploaded to the analysis server only for the duration of one
            analysis request and is deleted from the server immediately afterward. Session
            results (scores, not video) are saved only on your own device. Kinetra does not
            currently share, sell, or publish your movement data anywhere.
          </Text>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowTransparency(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showWhatsNew) {
    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.cameraSetupContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>What&apos;s New</Text>
        </View>

        <Text style={styles.title}>What&apos;s New in {APP_NAME}</Text>

        <Text style={styles.subtitle}>
          A plain-language record of what&apos;s actually shipped, in order. Nothing here is
          backfilled or invented -- if a change isn&apos;t listed, it hasn&apos;t shipped yet.
        </Text>

        {WHATS_NEW_ENTRIES.map((entry) => (
          <View key={entry.version} style={styles.cameraSetupCard}>
            <Text style={styles.sectionTitle}>
              Version {entry.version} -- {entry.date}
            </Text>
            {entry.highlights.map((highlight, index) => (
              <Text key={index} style={styles.cameraMistakeText}>
                • {highlight}
              </Text>
            ))}
          </View>
        ))}

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowWhatsNew(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showSettings) {
    const reminderHourOptions = [
      { hour: 8, label: '8:00 AM' },
      { hour: 12, label: '12:00 PM' },
      { hour: 18, label: '6:00 PM' },
      { hour: 20, label: '8:00 PM' },
    ];

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.cameraSetupContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Settings</Text>
        </View>

        <Text style={styles.title}>Settings</Text>

        <Text style={styles.subtitle}>
          Everything that changes how Kinetra behaves for you, in one place.
        </Text>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Daily Reminder</Text>
          <Text style={styles.dailyTaskDescription}>
            A local notification from your phone -- not from a server, and not tied to any
            account. Nothing is sent anywhere to schedule this.
          </Text>

          <Pressable
            style={[
              styles.dailyTaskChip,
              { marginTop: 10, flex: 0, alignSelf: 'flex-start', paddingHorizontal: 14 },
              reminderEnabled && styles.dailyTaskChipActive,
            ]}
            onPress={() => setReminderPreference(!reminderEnabled, reminderHour)}
            accessibilityRole="switch"
            accessibilityState={{ checked: reminderEnabled }}
            accessibilityLabel="Daily reminder notification"
          >
            <Text
              style={[
                styles.dailyTaskChipText,
                reminderEnabled && styles.dailyTaskChipTextActive,
              ]}
            >
              {reminderEnabled ? 'Daily Reminder: On' : 'Daily Reminder: Off'}
            </Text>
          </Pressable>

          {reminderEnabled ? (
            <View style={{ marginTop: 14 }}>
              <Text style={styles.inputLabel}>Remind Me At</Text>
              <View style={styles.dailyTaskRow}>
                {reminderHourOptions.map((option) => (
                  <Pressable
                    key={option.hour}
                    style={[
                      styles.dailyTaskChip,
                      reminderHour === option.hour && styles.dailyTaskChipActive,
                    ]}
                    onPress={() => setReminderPreference(true, option.hour)}
                    accessibilityRole="radio"
                    accessibilityState={{ checked: reminderHour === option.hour }}
                    accessibilityLabel={`Remind me at ${option.label}`}
                  >
                    <Text
                      style={[
                        styles.dailyTaskChipText,
                        reminderHour === option.hour && styles.dailyTaskChipTextActive,
                      ]}
                    >
                      {option.label}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>
          ) : null}

          {reminderStatusMessage ? (
            <View style={[styles.cameraMistakeCard, { marginTop: 12 }]}>
              <Text style={styles.cameraMistakeText}>{reminderStatusMessage}</Text>
            </View>
          ) : null}
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Calibration</Text>
          <Text style={styles.dailyTaskDescription}>
            Right arm: {calibratedThresholdsBySide.right ? 'personalized' : 'using default range'}.
            {'\n'}Left arm: {calibratedThresholdsBySide.left ? 'personalized' : 'using default range'}.
          </Text>

          {calibratedThresholdsBySide.left || calibratedThresholdsBySide.right ? (
            <Pressable
              style={[styles.secondaryButton, { marginTop: 10 }]}
              onPress={() => {
                persistCalibrationForSide('left', null);
                persistCalibrationForSide('right', null);
              }}
            >
              <Text style={styles.secondaryButtonText}>Reset Both to Default Range</Text>
            </Pressable>
          ) : null}
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Privacy & Data</Text>
          <Text style={styles.dailyTaskDescription}>
            What Kinetra measures, its known limitations, and exactly what happens to your
            data and video.
          </Text>
          <Pressable
            style={[styles.secondaryButton, { marginTop: 10 }]}
            onPress={() => {
              setShowSettings(false);
              setShowTransparency(true);
            }}
          >
            <Text style={styles.secondaryButtonText}>Open Transparency Page</Text>
          </Pressable>
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>About</Text>
          <Text style={styles.dailyTaskDescription}>
            {APP_NAME} {APP_BETA_LABEL} -- version {APP_VERSION}{'\n'}
            Analysis server: {API_BASE_URL}
          </Text>
          <Pressable
            style={[styles.secondaryButton, { marginTop: 10 }]}
            onPress={() => {
              setShowSettings(false);
              openWhatsNew();
            }}
          >
            <Text style={styles.secondaryButtonText}>What&apos;s New</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowSettings(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showTeamRoster) {
    const roster = getTeamRoster(savedSessions);
    const flaggedCount = roster.filter((athlete) => athlete.flagged).length;

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.cameraSetupContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Team Screening</Text>
        </View>

        <Text style={styles.title}>Team Roster</Text>

        <Text style={styles.subtitle}>
          Every recording saved with an athlete name shows up here. This is a screening tool,
          not a diagnosis -- it just tells you who is worth a closer look.
        </Text>

        {roster.length === 0 ? (
          <View style={styles.cameraSetupCard}>
            <Text style={styles.sectionTitle}>No Athletes Yet</Text>
            <Text style={styles.historyEmptyText}>
              Turn on Team Screening from the home screen, enter a name before each recording,
              and athletes will start appearing here after their first saved result.
            </Text>
          </View>
        ) : (
          <>
            <View style={styles.cameraSetupHeroCard}>
              <Text style={styles.sectionTitle}>Roster Summary</Text>
              <Text style={styles.metricValueLarge}>
                {roster.length} athlete{roster.length === 1 ? '' : 's'} screened
                {flaggedCount > 0
                  ? `, ${flaggedCount} flagged for follow-up`
                  : ', none currently flagged'}
              </Text>
            </View>

            {flaggedCount > 0 ? (
              <View style={styles.cameraMistakeCard}>
                <Text style={styles.sectionTitle}>Needs a Closer Look</Text>
                <Text style={styles.cameraMistakeText}>
                  A flagged athlete&apos;s most recent result graded poorly or asked for a
                  recheck. That does not mean an injury -- it means a real person (coach,
                  athletic trainer, or clinician) should take a look before this athlete
                  continues normal activity.
                </Text>
              </View>
            ) : null}

            {roster.map((athlete) => {
              const gradeColors = getGradeColors(athlete.latestSession.primary_grade);

              return (
                <View
                  key={athlete.name}
                  style={[
                    styles.historyTaskCard,
                    athlete.flagged
                      ? { borderColor: 'rgba(248, 113, 113, 0.45)' }
                      : null,
                  ]}
                >
                  <Text style={styles.historyTaskTitle}>{athlete.name}</Text>

                  <Text style={styles.historyTaskSubtitle}>
                    {athlete.sessionCount} saved check{athlete.sessionCount === 1 ? '' : 's'}
                    {athlete.latestSession.side
                      ? ` -- most recent: ${athlete.latestSession.side} side`
                      : ''}
                  </Text>

                  <Text style={styles.metricLabelSmall}>Most Recent Result</Text>

                  <Text style={styles.historyLatestScore}>
                    {athlete.latestSession.primary_score !== null
                      ? `${athlete.latestSession.primary_score}/100`
                      : 'N/A'}
                  </Text>

                  <View
                    style={[
                      styles.gradeBadge,
                      {
                        backgroundColor: gradeColors.bg,
                        borderColor: gradeColors.border,
                      },
                    ]}
                  >
                    <Text style={[styles.gradeBadgeText, { color: gradeColors.text }]}>
                      {athlete.latestSession.primary_grade}
                    </Text>
                  </View>

                  {athlete.flagged ? (
                    <Text style={[styles.cameraMistakeText, { marginTop: 10 }]}>
                      ⚠ Flagged for follow-up
                    </Text>
                  ) : null}
                </View>
              );
            })}
          </>
        )}

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowTeamRoster(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showCameraSetupGuide) {
    const setupGuide = getCameraSetupGuideForMode(mode, dailyTask);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.cameraSetupContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Recording Setup</Text>
        </View>

        <Text style={styles.title}>Camera Setup Guide</Text>

        <Text style={styles.subtitle}>
          Use this before recording so the app can track the movement clearly and give a more trustworthy result.
        </Text>

        <View style={styles.cameraSetupHeroCard}>
          <Text style={styles.sectionTitle}>{setupGuide.title}</Text>

          <Text style={styles.metricLabelSmall}>Best Camera Angle</Text>
          <Text style={styles.metricValueLarge}>
            {setupGuide.bestAngle}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Body Parts That Must Stay Visible
          </Text>

          <View style={styles.setupChipWrap}>
            {setupGuide.bodyParts.map((part) => (
              <View key={part} style={styles.setupChip}>
                <Text style={styles.setupChipText}>{part}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Setup Steps</Text>

          {setupGuide.setupSteps.map((step, index) => (
            <View key={index} style={styles.setupStepRow}>
              <View style={styles.setupStepNumber}>
                <Text style={styles.setupStepNumberText}>{index + 1}</Text>
              </View>

              <Text style={styles.setupStepText}>{step}</Text>
            </View>
          ))}
        </View>

        <View style={styles.cameraMistakeCard}>
          <Text style={styles.sectionTitle}>Common Mistakes</Text>

          {setupGuide.commonMistakes.map((mistake, index) => (
            <Text key={index} style={styles.cameraMistakeText}>
              ✕ {mistake}
            </Text>
          ))}
        </View>

        <View style={styles.cameraSetupCard}>
          <Text style={styles.sectionTitle}>Quick Rule</Text>

          <Text style={styles.metricValueLarge}>
            If the important joint leaves the frame, the result becomes less reliable. Stable camera first, movement second.
          </Text>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => {
            setShowCameraSetupGuide(false);
            setShowRolloutDashboard(false);
            setStarted(true);
            setVideoUri(null);
            setRecording(false);
            setCameraReady(false);
            setAnalysisResult(null);
            setAnalysisError(null);
            setAnalysisStatusMessage('');
            setComparisonSession(null);
            setShowDetails(false);
          }}
        >
          <Text style={styles.buttonText}>Start Recording</Text>
        </Pressable>

        <Pressable
          style={styles.secondaryButton}
          onPress={() => setShowCameraSetupGuide(false)}
        >
          <Text style={styles.secondaryButtonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showRolloutDashboard) {
    const rollout = getRolloutDashboardStatus(savedSessions, testerNotes);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.rolloutDashboardContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Rollout</Text>
        </View>

        <Text style={styles.title}>Rollout Dashboard</Text>

        <Text style={styles.subtitle}>
          Use this screen to decide what to test, what to fix, and whether the app is ready for more users.
        </Text>

        <View style={styles.rolloutHeroCard}>
          <Text style={styles.sectionTitle}>Current Rollout Status</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(rollout.status).bg,
                borderColor: getGradeColors(rollout.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(rollout.status).text },
              ]}
            >
              {rollout.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Action
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.nextAction}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Warning
          </Text>
          <Text style={styles.rolloutWarningText}>
            {rollout.warning}
          </Text>
        </View>

        <View style={styles.rolloutStatsGrid}>
          <View style={styles.rolloutStatBox}>
            <Text style={styles.rolloutStatValue}>{rollout.totalMovementChecks}</Text>
            <Text style={styles.rolloutStatLabel}>Movement Checks</Text>
          </View>

          <View style={styles.rolloutStatBox}>
            <Text style={styles.rolloutStatValue}>{rollout.totalFeedbackNotes}</Text>
            <Text style={styles.rolloutStatLabel}>Tester Notes</Text>
          </View>

          <View style={styles.rolloutStatBox}>
            <Text style={styles.rolloutStatValue}>
              {rollout.testingReadiness.readinessScore}
            </Text>
            <Text style={styles.rolloutStatLabel}>Readiness</Text>
          </View>

          <View style={styles.rolloutStatBox}>
            <Text style={styles.rolloutStatValue}>
              {rollout.testingReadiness.dailyBaselinesBuilt + (rollout.testingReadiness.rehabBaselineBuilt ? 1 : 0)}
            </Text>
            <Text style={styles.rolloutStatLabel}>Baselines Built</Text>
          </View>
        </View>

        <View style={styles.rolloutSectionCard}>
          <Text style={styles.sectionTitle}>Testing Readiness</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(rollout.testingReadiness.status).bg,
                borderColor: getGradeColors(rollout.testingReadiness.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(rollout.testingReadiness.status).text },
              ]}
            >
              {rollout.testingReadiness.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What This Means
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.testingReadiness.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.testingReadiness.nextStep}
          </Text>
        </View>

        <View style={styles.rolloutSectionCard}>
          <Text style={styles.sectionTitle}>Feedback Priority</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(rollout.feedbackActionPlan.status).bg,
                borderColor: getGradeColors(rollout.feedbackActionPlan.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(rollout.feedbackActionPlan.status).text },
              ]}
            >
              {rollout.feedbackActionPlan.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Fix First
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.feedbackActionPlan.fixFirst}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Test Next
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.feedbackActionPlan.testNext}
          </Text>
        </View>

        <View style={styles.rolloutSectionCard}>
          <Text style={styles.sectionTitle}>Movement Profile</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(rollout.movementOverview.status).bg,
                borderColor: getGradeColors(rollout.movementOverview.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(rollout.movementOverview.status).text },
              ]}
            >
              {rollout.movementOverview.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.movementOverview.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What To Watch
          </Text>
          <Text style={styles.metricValueLarge}>
            {rollout.movementOverview.watch}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Quick Actions</Text>

        <View style={styles.rolloutActionGrid}>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowGuidedOnboarding(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Start Here</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowReportExport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>View Report</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowBuilderTools(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Builder Tools</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowDailyHealthOverview(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Daily Overview</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowTesterAnalytics(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Tester Analytics</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowBetaLaunchKit(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Beta Launch</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowGuidedTestWorkflow(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Guided Test</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowTestingGuide(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Testing Guide</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowFeedbackNotes(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Feedback Notes</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowCameraSetupGuide(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Camera Setup</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowRolloutDashboard(false);
              setShowHistory(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>History</Text>
          </Pressable>
        </View>

        <View style={styles.rolloutChecklistCard}>
          <Text style={styles.sectionTitle}>Before Wider Rollout</Text>

          <Text style={styles.rolloutChecklistText}>
            □ Test with at least 5 people.
          </Text>
          <Text style={styles.rolloutChecklistText}>
            □ Fix bugs before adding features.
          </Text>
          <Text style={styles.rolloutChecklistText}>
            □ Confirm users understand Camera Setup.
          </Text>
          <Text style={styles.rolloutChecklistText}>
            □ Confirm users understand baseline and trend.
          </Text>
          <Text style={styles.rolloutChecklistText}>
            □ Confirm users know what to do after seeing results.
          </Text>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowRolloutDashboard(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showGuidedTestWorkflow) {
    const guidedProgress = getGuidedTestProgress(completedGuidedSteps);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.guidedWorkflowContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Private Beta</Text>
        </View>

        <Text style={styles.title}>Guided Test Workflow</Text>

        <Text style={styles.subtitle}>
          Use this checklist while testing the app with one person. Complete the steps in order and save feedback afterward.
        </Text>

        <View style={styles.guidedProgressCard}>
          <Text style={styles.sectionTitle}>Test Progress</Text>

          <Text style={styles.guidedProgressPercent}>
            {guidedProgress.percent}%
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(guidedProgress.status).bg,
                borderColor: getGradeColors(guidedProgress.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(guidedProgress.status).text },
              ]}
            >
              {guidedProgress.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {guidedProgress.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {guidedProgress.nextStep}
          </Text>
        </View>

        {getGuidedTestWorkflowSteps().map((step, index) => {
          const isComplete = completedGuidedSteps.includes(step.key);

          return (
            <View key={step.key} style={styles.guidedStepCard}>
              <View style={styles.guidedStepHeader}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.guidedStepNumber}>
                    Step {index + 1}
                  </Text>
                  <Text style={styles.guidedStepTitle}>
                    {step.title}
                  </Text>
                </View>

                <Pressable
                  style={[
                    styles.guidedCheckButton,
                    isComplete ? styles.guidedCheckButtonDone : null,
                  ]}
                  onPress={() => toggleGuidedStepComplete(step.key)}
                >
                  <Text style={styles.guidedCheckButtonText}>
                    {isComplete ? '✓ Done' : 'Mark Done'}
                  </Text>
                </Pressable>
              </View>

              <Text style={styles.guidedStepGoal}>
                {step.goal}
              </Text>

              {step.instructions.map((instruction, instructionIndex) => (
                <Text key={instructionIndex} style={styles.guidedInstructionText}>
                  • {instruction}
                </Text>
              ))}

              <Pressable
                style={styles.guidedLaunchButton}
                onPress={() => {
                  if (step.mode === 'daily') {
                    setMode('daily');

                    if (step.task === 'reach' || step.task === 'sit_to_stand' || step.task === 'arm_raise') {
                      setDailyTask(step.task);
                    }

                    setShowGuidedTestWorkflow(false);
                    setShowCameraSetupGuide(true);
                    setStarted(false);
                  } else if (step.mode === 'rehab') {
                    setMode('rehab');
                    setShowGuidedTestWorkflow(false);
                    setShowCameraSetupGuide(true);
                    setStarted(false);
                  } else if (step.mode === 'history') {
                    setShowGuidedTestWorkflow(false);
                    setShowHistory(true);
                  } else if (step.mode === 'feedback') {
                    setShowGuidedTestWorkflow(false);
                    setShowFeedbackNotes(true);
                  } else {
                    setShowGuidedTestWorkflow(false);
                  }
                }}
              >
                <Text style={styles.guidedLaunchButtonText}>
                  {step.actionLabel}
                </Text>
              </Pressable>
            </View>
          );
        })}

        <View style={styles.guidedReminderCard}>
          <Text style={styles.sectionTitle}>Testing Reminder</Text>
          <Text style={styles.guidedReminderText}>
            Do not explain too much. If the tester gets confused, that is useful data. Save it in Feedback Notes.
          </Text>
        </View>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedTestWorkflow(false);
              setShowFeedbackNotes(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Feedback Notes</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedTestWorkflow(false);
              setShowRolloutDashboard(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Rollout Dashboard</Text>
          </Pressable>
        </View>

        {completedGuidedSteps.length > 0 ? (
          <Pressable
            style={styles.clearHistoryButton}
            onPress={clearGuidedTestProgress}
          >
            <Text style={styles.clearHistoryButtonText}>Reset Guided Test Progress</Text>
          </Pressable>
        ) : null}

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowGuidedTestWorkflow(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showBetaLaunchKit) {
    const betaLaunch = getBetaLaunchReadiness(
      savedSessions,
      testerNotes,
      completedGuidedSteps
    );

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.betaLaunchContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Beta Launch</Text>
        </View>

        <Text style={styles.title}>Beta Launch Kit</Text>

        <Text style={styles.subtitle}>
          Use this screen to recruit testers, explain the app clearly, and prepare for a small private beta.
        </Text>

        <View style={styles.betaLaunchHeroCard}>
          <Text style={styles.sectionTitle}>Beta Readiness</Text>

          <Text style={styles.betaLaunchScore}>
            {betaLaunch.score}/100
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(betaLaunch.status).bg,
                borderColor: getGradeColors(betaLaunch.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(betaLaunch.status).text },
              ]}
            >
              {betaLaunch.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {betaLaunch.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {betaLaunch.nextStep}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Hard Truth
          </Text>
          <Text style={styles.betaLaunchWarningText}>
            {betaLaunch.launchAdvice}
          </Text>
        </View>

        <View style={styles.betaLaunchStatsGrid}>
          <View style={styles.betaLaunchStatBox}>
            <Text style={styles.betaLaunchStatValue}>{betaLaunch.movementCheckCount}</Text>
            <Text style={styles.betaLaunchStatLabel}>Movement Checks</Text>
          </View>

          <View style={styles.betaLaunchStatBox}>
            <Text style={styles.betaLaunchStatValue}>{betaLaunch.testerNoteCount}</Text>
            <Text style={styles.betaLaunchStatLabel}>Tester Notes</Text>
          </View>

          <View style={styles.betaLaunchStatBox}>
            <Text style={styles.betaLaunchStatValue}>{betaLaunch.guidedProgressPercent}%</Text>
            <Text style={styles.betaLaunchStatLabel}>Guided Test</Text>
          </View>

          <View style={styles.betaLaunchStatBox}>
            <Text style={styles.betaLaunchStatValue}>
              {betaLaunch.rolloutStatus}
            </Text>
            <Text style={styles.betaLaunchStatLabel}>Rollout Status</Text>
          </View>
        </View>

        <View style={styles.betaLaunchCard}>
          <Text style={styles.sectionTitle}>One-Sentence Pitch</Text>
          <Text style={styles.metricValueLarge}>
            {getShortBetaPitch()}
          </Text>
        </View>

        <View style={styles.betaLaunchCard}>
          <Text style={styles.sectionTitle}>Shareable Beta Invite</Text>

          <Text style={styles.betaInviteText}>
            {getBetaInviteText()}
          </Text>

          <Pressable
            style={styles.shareBetaButton}
            onPress={shareBetaInvite}
          >
            <Text style={styles.buttonText}>Share Beta Invite</Text>
          </Pressable>
        </View>

        <View style={styles.betaLaunchCard}>
          <Text style={styles.sectionTitle}>Best First Testers</Text>

          {getIdealTesterGroups().map((group) => (
            <View key={group.title} style={styles.betaTesterGroup}>
              <Text style={styles.betaTesterGroupTitle}>
                {group.title}
              </Text>
              <Text style={styles.betaTesterGroupText}>
                {group.reason}
              </Text>
            </View>
          ))}
        </View>

        <View style={styles.betaLaunchChecklistCard}>
          <Text style={styles.sectionTitle}>Before You Share</Text>

          {getBetaLaunchChecklist().map((item, index) => (
            <Text key={index} style={styles.betaChecklistText}>
              □ {item}
            </Text>
          ))}
        </View>

        <Text style={styles.sectionTitle}>Launch Actions</Text>

        <View style={styles.rolloutActionGrid}>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowBuilderTools(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Builder Tools</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowTesterAnalytics(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Tester Analytics</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowGuidedTestWorkflow(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Guided Test</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowFeedbackNotes(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Feedback Notes</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowRolloutDashboard(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Rollout Dashboard</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowBetaLaunchKit(false);
              setShowCameraSetupGuide(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Camera Setup</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowBetaLaunchKit(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showTesterAnalytics) {
    const testerAnalytics = getTesterAnalytics(testerNotes);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.testerAnalyticsContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Tester Analytics</Text>
        </View>

        <Text style={styles.title}>Tester Analytics</Text>

        <Text style={styles.subtitle}>
          Use this screen to see what testers are confused by, what bugs repeat, and what should be fixed first.
        </Text>

        <View style={styles.testerAnalyticsHeroCard}>
          <Text style={styles.sectionTitle}>Analytics Status</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(testerAnalytics.status).bg,
                borderColor: getGradeColors(testerAnalytics.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(testerAnalytics.status).text },
              ]}
            >
              {testerAnalytics.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {testerAnalytics.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Fix Focus
          </Text>
          <Text style={styles.metricValueLarge}>
            {testerAnalytics.fixFocus}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Test
          </Text>
          <Text style={styles.metricValueLarge}>
            {testerAnalytics.nextTest}
          </Text>
        </View>

        <View style={styles.testerAnalyticsStatsGrid}>
          <View style={styles.testerAnalyticsStatBox}>
            <Text style={styles.testerAnalyticsStatValue}>{testerAnalytics.totalNotes}</Text>
            <Text style={styles.testerAnalyticsStatLabel}>Total Notes</Text>
          </View>

          <View style={styles.testerAnalyticsStatBox}>
            <Text style={styles.testerAnalyticsStatValue}>{testerAnalytics.confusionCount}</Text>
            <Text style={styles.testerAnalyticsStatLabel}>Confusions</Text>
          </View>

          <View style={styles.testerAnalyticsStatBox}>
            <Text style={styles.testerAnalyticsStatValue}>{testerAnalytics.bugCount}</Text>
            <Text style={styles.testerAnalyticsStatLabel}>Bugs</Text>
          </View>

          <View style={styles.testerAnalyticsStatBox}>
            <Text style={styles.testerAnalyticsStatValue}>{testerAnalytics.requestCount}</Text>
            <Text style={styles.testerAnalyticsStatLabel}>Requests</Text>
          </View>
        </View>

        <View style={styles.testerAnalyticsCard}>
          <Text style={styles.sectionTitle}>Top Issues</Text>

          <Text style={styles.analyticsIssueLabel}>Top Confusion</Text>
          <Text style={styles.analyticsIssueValue}>
            {testerAnalytics.topConfusion}
          </Text>

          <Text style={styles.analyticsIssueLabel}>Top Bug Area</Text>
          <Text style={styles.analyticsIssueValue}>
            {testerAnalytics.topBug}
          </Text>

          <Text style={styles.analyticsIssueLabel}>Top Feature Request Area</Text>
          <Text style={styles.analyticsIssueValue}>
            {testerAnalytics.topRequest}
          </Text>
        </View>

        <View style={styles.testerAnalyticsCard}>
          <Text style={styles.sectionTitle}>Confusion Categories</Text>

          {testerAnalytics.confusionCategories.length === 0 ? (
            <Text style={styles.analyticsEmptyText}>
              No confusion notes saved yet.
            </Text>
          ) : (
            testerAnalytics.confusionCategories.map((item) => (
              <View key={item.category} style={styles.analyticsCategoryRow}>
                <Text style={styles.analyticsCategoryName}>
                  {item.category}
                </Text>
                <Text style={styles.analyticsCategoryCount}>
                  {item.count}
                </Text>
              </View>
            ))
          )}
        </View>

        <View style={styles.testerAnalyticsCard}>
          <Text style={styles.sectionTitle}>Bug Categories</Text>

          {testerAnalytics.bugCategories.length === 0 ? (
            <Text style={styles.analyticsEmptyText}>
              No bug notes saved yet.
            </Text>
          ) : (
            testerAnalytics.bugCategories.map((item) => (
              <View key={item.category} style={styles.analyticsCategoryRow}>
                <Text style={styles.analyticsCategoryName}>
                  {item.category}
                </Text>
                <Text style={styles.analyticsCategoryCount}>
                  {item.count}
                </Text>
              </View>
            ))
          )}
        </View>

        <View style={styles.testerAnalyticsCard}>
          <Text style={styles.sectionTitle}>Feature Request Categories</Text>

          {testerAnalytics.requestCategories.length === 0 ? (
            <Text style={styles.analyticsEmptyText}>
              No feature request notes saved yet.
            </Text>
          ) : (
            testerAnalytics.requestCategories.map((item) => (
              <View key={item.category} style={styles.analyticsCategoryRow}>
                <Text style={styles.analyticsCategoryName}>
                  {item.category}
                </Text>
                <Text style={styles.analyticsCategoryCount}>
                  {item.count}
                </Text>
              </View>
            ))
          )}
        </View>

        <Text style={styles.sectionTitle}>Analytics Actions</Text>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTesterAnalytics(false);
              setShowFeedbackNotes(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Feedback Notes</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTesterAnalytics(false);
              setShowGuidedTestWorkflow(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Guided Test</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTesterAnalytics(false);
              setShowRolloutDashboard(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Rollout Dashboard</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTesterAnalytics(false);
              setShowBetaLaunchKit(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Beta Launch</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowTesterAnalytics(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showDailyHealthOverview) {
    const dailyHealth = getDailyHealthV2Overview(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.dailyHealthOverviewContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Daily Health</Text>
        </View>

        <Text style={styles.title}>Daily Health Overview</Text>

        <Text style={styles.subtitle}>
          See your overall movement profile across Daily checks and Rehab consistency.
        </Text>

        <View style={styles.dailyHealthHeroCard}>
          <Text style={styles.sectionTitle}>Movement Snapshot</Text>

          <Text style={styles.dailyHealthHeadline}>
            {dailyHealth.headline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(dailyHealth.overallStatus).bg,
                borderColor: getGradeColors(dailyHealth.overallStatus).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(dailyHealth.overallStatus).text },
              ]}
            >
              {dailyHealth.overallStatus}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {dailyHealth.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Main Focus
          </Text>
          <Text style={styles.metricValueLarge}>
            {dailyHealth.healthFocus}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Recommended Next Check
          </Text>
          <Text style={styles.metricValueLarge}>
            {dailyHealth.nextCheck}
          </Text>
        </View>

        <View style={styles.dailyHealthStatsGrid}>
          <View style={styles.dailyHealthStatBox}>
            <Text style={styles.dailyHealthStatValue}>
              {dailyHealth.completenessPercent}%
            </Text>
            <Text style={styles.dailyHealthStatLabel}>Profile Complete</Text>
          </View>

          <View style={styles.dailyHealthStatBox}>
            <Text style={styles.dailyHealthStatValue}>
              {dailyHealth.baselinePercent}%
            </Text>
            <Text style={styles.dailyHealthStatLabel}>Baselines Built</Text>
          </View>

          <View style={styles.dailyHealthStatBox}>
            <Text style={styles.dailyHealthStatValue}>
              {dailyHealth.scoredAreaCount}/{dailyHealth.totalAreas}
            </Text>
            <Text style={styles.dailyHealthStatLabel}>Areas Checked</Text>
          </View>

          <View style={styles.dailyHealthStatBox}>
            <Text style={styles.dailyHealthStatValue}>
              {dailyHealth.baselineReadyCount}/{dailyHealth.totalAreas}
            </Text>
            <Text style={styles.dailyHealthStatLabel}>Baseline Areas</Text>
          </View>
        </View>

        <View style={styles.dailyHealthCard}>
          <Text style={styles.sectionTitle}>Strongest Area</Text>

          {dailyHealth.bestArea ? (
            <>
              <Text style={styles.dailyHealthAreaTitle}>
                {dailyHealth.bestArea.label}
              </Text>
              <Text style={styles.dailyHealthAreaScore}>
                {dailyHealth.bestArea.score !== null ? `${dailyHealth.bestArea.score}/100` : 'N/A'}
              </Text>
              <Text style={styles.metricValueLarge}>
                {dailyHealth.bestArea.detail}
              </Text>
            </>
          ) : (
            <Text style={styles.metricValueLarge}>
              No strongest area yet. Record at least one movement check first.
            </Text>
          )}
        </View>

        <View style={styles.dailyHealthCard}>
          <Text style={styles.sectionTitle}>Area To Watch</Text>

          {dailyHealth.areaToWatch ? (
            <>
              <Text style={styles.dailyHealthAreaTitle}>
                {dailyHealth.areaToWatch.label}
              </Text>
              <Text style={styles.dailyHealthAreaScore}>
                {dailyHealth.areaToWatch.score !== null ? `${dailyHealth.areaToWatch.score}/100` : 'N/A'}
              </Text>
              <Text style={styles.metricValueLarge}>
                {dailyHealth.areaToWatch.detail}
              </Text>
            </>
          ) : (
            <Text style={styles.metricValueLarge}>
              No area to watch yet. Build your movement profile by recording Daily and Rehab checks.
            </Text>
          )}
        </View>

        <Text style={styles.sectionTitle}>Movement Areas</Text>

        {dailyHealth.items.map((item) => (
          <View key={item.key} style={styles.dailyHealthAreaCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.dailyHealthAreaName}>
                {item.label}
              </Text>

              <Text style={styles.dailyHealthAreaDetail}>
                {item.detail}
              </Text>
            </View>

            <View style={styles.dailyHealthAreaRight}>
              <Text style={styles.dailyHealthAreaScoreSmall}>
                {item.score !== null ? `${item.score}/100` : 'N/A'}
              </Text>

              <View
                style={[
                  styles.miniGradeBadge,
                  {
                    backgroundColor: getGradeColors(item.status).bg,
                    borderColor: getGradeColors(item.status).border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.miniGradeBadgeText,
                    { color: getGradeColors(item.status).text },
                  ]}
                >
                  {item.status}
                </Text>
              </View>
            </View>
          </View>
        ))}

        <Text style={styles.sectionTitle}>Quick Actions</Text>

        <View style={styles.rolloutActionGrid}>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Weekly Report</Text>
          </Pressable>


          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setMode('daily');
              setDailyTask('reach');
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Record Reach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setMode('daily');
              setDailyTask('sit_to_stand');
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Record Sit-to-Stand</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setMode('rehab');
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Record Rehab</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowDailyHealthOverview(false);
              setShowHistory(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>View History</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowDailyHealthOverview(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showBuilderTools) {
    const rollout = getRolloutDashboardStatus(savedSessions, testerNotes);
    const betaLaunch = getBetaLaunchReadiness(
      savedSessions,
      testerNotes,
      completedGuidedSteps
    );
    const testerAnalytics = getTesterAnalytics(testerNotes);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.builderToolsContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Builder Mode</Text>
        </View>

        <Text style={styles.title}>Builder Tools</Text>

        <Text style={styles.subtitle}>
          Internal tools for testing, feedback, rollout planning, and beta launch preparation.
        </Text>

        <View style={styles.builderSummaryCard}>
          <Text style={styles.sectionTitle}>Build Status</Text>

          <Text style={styles.metricLabelSmall}>Rollout Status</Text>
          <Text style={styles.builderStatusText}>
            {rollout.status}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Beta Readiness
          </Text>
          <Text style={styles.builderStatusText}>
            {betaLaunch.score}/100 — {betaLaunch.status}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Tester Analytics
          </Text>
          <Text style={styles.builderStatusText}>
            {testerAnalytics.status}
          </Text>
        </View>

        <View style={styles.builderSummaryCard}>
          <Text style={styles.sectionTitle}>Internal Progress</Text>

          <Text style={styles.metricLabelSmall}>Rollout Next Action</Text>
          <Text style={styles.builderStatusText}>
            {rollout.nextAction}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Guided Test Progress
          </Text>
          <Text style={styles.builderStatusText}>
            {getGuidedTestProgress(completedGuidedSteps).percent}% complete — {getGuidedTestProgress(completedGuidedSteps).nextStep}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Beta Launch Next Step
          </Text>
          <Text style={styles.builderStatusText}>
            {betaLaunch.nextStep}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Tester Analytics Fix Focus
          </Text>
          <Text style={styles.builderStatusText}>
            {testerAnalytics.fixFocus}
          </Text>
        </View>

        <View style={styles.builderStatsGrid}>
          <View style={styles.builderStatBox}>
            <Text style={styles.builderStatValue}>{savedSessions.length}</Text>
            <Text style={styles.builderStatLabel}>Movement Checks</Text>
          </View>

          <View style={styles.builderStatBox}>
            <Text style={styles.builderStatValue}>{testerNotes.length}</Text>
            <Text style={styles.builderStatLabel}>Tester Notes</Text>
          </View>

          <View style={styles.builderStatBox}>
            <Text style={styles.builderStatValue}>
              {getGuidedTestProgress(completedGuidedSteps).percent}%
            </Text>
            <Text style={styles.builderStatLabel}>Guided Test</Text>
          </View>

          <View style={styles.builderStatBox}>
            <Text style={styles.builderStatValue}>
              {getTestingReadiness(savedSessions).readinessScore}
            </Text>
            <Text style={styles.builderStatLabel}>Testing Readiness</Text>
          </View>
        </View>

        <Text style={styles.sectionTitle}>Testing Tools</Text>

        <View style={styles.builderToolGrid}>
          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowTestingGuide(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Testing Guide</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowGuidedTestWorkflow(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Guided Test</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowFeedbackNotes(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Feedback Notes</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowTesterAnalytics(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Tester Analytics</Text>
          </Pressable>
        </View>

        <Text style={styles.sectionTitle}>Launch Tools</Text>

        <View style={styles.builderToolGrid}>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowTrendEngine(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Trend Engine</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowYoloFramework(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>YOLO Framework</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Weekly Report</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowReportExport(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Movement Passport</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowRolloutDashboard(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Rollout Dashboard</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowBetaLaunchKit(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Beta Launch Kit</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowCameraSetupGuide(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>Camera Setup</Text>
          </Pressable>

          <Pressable
            style={styles.builderToolButton}
            onPress={() => {
              setShowBuilderTools(false);
              setShowHistory(true);
            }}
          >
            <Text style={styles.builderToolButtonText}>History</Text>
          </Pressable>
        </View>

        <View style={styles.builderWarningCard}>
          <Text style={styles.builderWarningTitle}>Hard Rule</Text>
          <Text style={styles.builderWarningText}>
            Normal users should not see most of these tools during testing. Use Builder Tools yourself, but keep the main home screen focused on movement checks.
          </Text>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowBuilderTools(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showReportExport) {
    const passport = getMovementPassport(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.reportExportContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Movement Passport</Text>
        </View>

        <Text style={styles.title}>Kinetra Passport</Text>

        <Text style={styles.subtitle}>
          A shareable snapshot of your movement profile, trends, baselines, watch areas, and recommended next action.
        </Text>

        <View style={styles.passportHeroCard}>
          <Text style={styles.sectionTitle}>Passport Summary</Text>

          <Text style={styles.passportHeadline}>
            {passport.passportHeadline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(passport.passportStatus).bg,
                borderColor: getGradeColors(passport.passportStatus).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(passport.passportStatus).text },
              ]}
            >
              {passport.passportStatus}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Movement Passport Score
          </Text>
          <Text style={styles.passportScoreText}>
            {passport.averageScore !== null ? `${passport.averageScore}/100` : 'N/A'}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Movement Fingerprint
          </Text>
          <Text style={styles.metricValueLarge}>
            {passport.movementFingerprint}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {passport.passportSummary}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Profile Snapshot</Text>

        <View style={styles.passportSnapshotGrid}>
          {passport.snapshotCards.map((card) => (
            <View key={card.label} style={styles.passportSnapshotCard}>
              <Text style={styles.passportSnapshotValue}>
                {card.value}
              </Text>
              <Text style={styles.passportSnapshotLabel}>
                {card.label}
              </Text>
              <Text style={styles.passportSnapshotDetail}>
                {card.detail}
              </Text>
            </View>
          ))}
        </View>

        <View style={styles.passportHighlightGrid}>
          <View style={styles.passportHighlightCard}>
            <Text style={styles.metricLabelSmall}>Strongest Area</Text>
            <Text style={styles.passportHighlightTitle}>
              {passport.strongestTask ? passport.strongestTask.label : 'Not available yet'}
            </Text>
            <Text style={styles.passportHighlightText}>
              {passport.strongestTask
                ? passport.strongestTask.summary
                : 'Save more checks to identify your strongest movement area.'}
            </Text>
          </View>

          <View style={styles.passportHighlightCard}>
            <Text style={styles.metricLabelSmall}>Main Watch Area</Text>
            <Text style={styles.passportHighlightTitle}>
              {passport.weakestTask ? passport.weakestTask.label : 'Not available yet'}
            </Text>
            <Text style={styles.passportHighlightText}>
              {passport.weakestTask
                ? passport.weakestTask.summary
                : 'Save more checks to identify your main watch area.'}
            </Text>
          </View>
        </View>

        <View style={styles.passportNextCard}>
          <Text style={styles.sectionTitle}>Recommended Next Action</Text>
          <Text style={styles.metricValueLarge}>
            {passport.nextAction}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Passport Readiness
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(passport.passportReadiness).bg,
                borderColor: getGradeColors(passport.passportReadiness).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(passport.passportReadiness).text },
              ]}
            >
              {passport.passportReadiness}
            </Text>
          </View>
        </View>

        <Text style={styles.sectionTitle}>Trend Summary</Text>

        <View style={styles.passportTrendGrid}>
          <View style={styles.passportTrendBox}>
            <Text style={styles.passportTrendValue}>
              {passport.trendSummary.improvingCount}
            </Text>
            <Text style={styles.passportTrendLabel}>Improving</Text>
          </View>

          <View style={styles.passportTrendBox}>
            <Text style={styles.passportTrendValue}>
              {passport.trendSummary.stableCount}
            </Text>
            <Text style={styles.passportTrendLabel}>Stable</Text>
          </View>

          <View style={styles.passportTrendBox}>
            <Text style={styles.passportTrendValue}>
              {passport.trendSummary.watchCount}
            </Text>
            <Text style={styles.passportTrendLabel}>Watch</Text>
          </View>

          <View style={styles.passportTrendBox}>
            <Text style={styles.passportTrendValue}>
              {passport.trendSummary.noDataCount}
            </Text>
            <Text style={styles.passportTrendLabel}>Missing</Text>
          </View>
        </View>

        <View style={styles.passportConnectedCard}>
          <Text style={styles.passportConnectedTitle}>Longitudinal Trend Engine</Text>
          <Text style={styles.passportConnectedText}>
            {passport.longitudinalTrends.engineStatus}
          </Text>
          <Text style={styles.passportConnectedText}>
            Confidence: {passport.longitudinalTrends.trendConfidence}
          </Text>
          <Text style={styles.passportConnectedText}>
            {passport.longitudinalTrends.mainFinding}
          </Text>
        </View>

        {passport.trendSummary.watchTasks.length > 0 ? (
          <View style={styles.passportWatchCard}>
            <Text style={styles.passportWatchTitle}>Recheck Recommended</Text>

            {passport.trendSummary.watchTasks.map((item) => (
              <Text key={item.task} style={styles.passportWatchText}>
                • {item.label}: {item.recommendation}
              </Text>
            ))}
          </View>
        ) : null}

        {passport.trendSummary.improvingTasks.length > 0 ? (
          <View style={styles.passportImprovingCard}>
            <Text style={styles.passportImprovingTitle}>Improving Areas</Text>

            {passport.trendSummary.improvingTasks.map((item) => (
              <Text key={item.task} style={styles.passportImprovingText}>
                • {item.label}: {item.summary}
              </Text>
            ))}
          </View>
        ) : null}

        <Text style={styles.sectionTitle}>Movement Area Passport</Text>

        {passport.taskTrends.map((item) => (
          <View key={item.task} style={styles.passportAreaCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.passportAreaTitle}>
                {item.label}
              </Text>

              <Text style={styles.passportAreaText}>
                {item.summary}
              </Text>

              <Text style={styles.passportAreaMeta}>
                {item.sessionCount} saved check{item.sessionCount === 1 ? '' : 's'} • {item.baselineScore !== null ? `Baseline ${item.baselineScore}/100` : 'Baseline building'}
              </Text>

              <Text style={styles.passportAreaMeta}>
                Trend: {item.trend}
              </Text>
            </View>

            <View style={styles.passportAreaRight}>
              <Text style={styles.passportAreaScore}>
                {item.latestScore !== null ? `${item.latestScore}/100` : 'N/A'}
              </Text>

              <View
                style={[
                  styles.miniGradeBadge,
                  {
                    backgroundColor: getGradeColors(item.status).bg,
                    borderColor: getGradeColors(item.status).border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.miniGradeBadgeText,
                    { color: getGradeColors(item.status).text },
                  ]}
                >
                  {item.status}
                </Text>
              </View>
            </View>
          </View>
        ))}

        <Text style={styles.sectionTitle}>Connected Reports</Text>

        <View style={styles.passportConnectedGrid}>
          <View style={styles.passportConnectedCard}>
            <Text style={styles.passportConnectedTitle}>Mobility Profile</Text>
            <Text style={styles.passportConnectedText}>
              {passport.mobilityProfile.profileLabel}
            </Text>
            <Text style={styles.passportConnectedText}>
              {passport.mobilityProfile.completionPercent}% complete
            </Text>
          </View>

          <View style={styles.passportConnectedCard}>
            <Text style={styles.passportConnectedTitle}>Weekly Report</Text>
            <Text style={styles.passportConnectedText}>
              {passport.weeklyReport.weeklyStatus}
            </Text>
            <Text style={styles.passportConnectedText}>
              {passport.weeklyReport.areasCheckedThisWeek} areas checked this week
            </Text>
          </View>

          <View style={styles.passportConnectedCard}>
            <Text style={styles.passportConnectedTitle}>Coach Focus</Text>
            <Text style={styles.passportConnectedText}>
              {passport.coach.whatNeedsAttention}
            </Text>
          </View>
        </View>

        <View style={styles.reportDisclaimerCard}>
          <Text style={styles.reportDisclaimerTitle}>Important</Text>
          <Text style={styles.reportDisclaimerText}>
            {APP_SAFETY_NOTE}
          </Text>
        </View>

        <Pressable
          style={styles.aiCoachInlineButton}
          onPress={() => {
            setShowReportExport(false);
            setShowAiCoach(true);
          }}
        >
          <Text style={styles.testingGuideButtonText}>Open AI Coach</Text>
        </Pressable>

        <Pressable
          style={styles.weeklyInlineButton}
          onPress={() => {
            setShowReportExport(false);
            setShowWeeklyReport(true);
          }}
        >
          <Text style={styles.testingGuideButtonText}>Open Weekly Report</Text>
        </Pressable>

        <Pressable
          style={styles.mobilityInlineButton}
          onPress={() => {
            setShowReportExport(false);
            setShowMobilityProfile(true);
          }}
        >
          <Text style={styles.testingGuideButtonText}>Open Mobility Profile</Text>
        </Pressable>

        <Pressable
          style={styles.trendInlineButton}
          onPress={() => {
            setShowReportExport(false);
            setShowTrendEngine(true);
          }}
        >
          <Text style={styles.testingGuideButtonText}>Open Trend Engine</Text>
        </Pressable>

        <Pressable
          style={styles.shareBetaButton}
          onPress={shareMovementReport}
        >
          <Text style={styles.buttonText}>Share Kinetra Passport</Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowReportExport(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showAiCoach) {
    const coach = getMovementCoachPlan(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.aiCoachContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>AI Coach</Text>
        </View>

        <Text style={styles.title}>Movement Coach</Text>

        <Text style={styles.subtitle}>
          Get a plain-English explanation of your movement profile, what to watch, and what to record next.
        </Text>

        <View style={styles.aiCoachHeroCard}>
          <Text style={styles.sectionTitle}>Coach Summary</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(coach.coachStatus).bg,
                borderColor: getGradeColors(coach.coachStatus).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(coach.coachStatus).text },
              ]}
            >
              {coach.coachStatus}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What This Means
          </Text>
          <Text style={styles.metricValueLarge}>
            {coach.coachSummary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What Looks Good
          </Text>
          <Text style={styles.metricValueLarge}>
            {coach.whatLooksGood}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What Needs Attention
          </Text>
          <Text style={styles.metricValueLarge}>
            {coach.whatNeedsAttention}
          </Text>
        </View>

        <View style={styles.aiCoachStatsGrid}>
          <View style={styles.aiCoachStatBox}>
            <Text style={styles.aiCoachStatValue}>{coach.profileComplete}%</Text>
            <Text style={styles.aiCoachStatLabel}>Profile Complete</Text>
          </View>

          <View style={styles.aiCoachStatBox}>
            <Text style={styles.aiCoachStatValue}>{coach.baselinesBuilt}%</Text>
            <Text style={styles.aiCoachStatLabel}>Baselines Built</Text>
          </View>

          <View style={styles.aiCoachStatBox}>
            <Text style={styles.aiCoachStatValue}>{coach.trustLevel}</Text>
            <Text style={styles.aiCoachStatLabel}>Trust Level</Text>
          </View>
        </View>

        <View style={styles.aiCoachCard}>
          <Text style={styles.sectionTitle}>Recommended Next Action</Text>
          <Text style={styles.metricValueLarge}>
            {coach.nextAction}
          </Text>
        </View>

        <View style={styles.aiCoachCard}>
          <Text style={styles.sectionTitle}>Simple Action Plan</Text>

          {coach.actionPlan.map((item, index) => (
            <View key={index} style={styles.aiCoachActionRow}>
              <View style={styles.aiCoachActionNumber}>
                <Text style={styles.aiCoachActionNumberText}>{index + 1}</Text>
              </View>

              <Text style={styles.aiCoachActionText}>
                {item}
              </Text>
            </View>
          ))}
        </View>

        <View style={styles.aiCoachCard}>
          <Text style={styles.sectionTitle}>Ask Claude for a Personal Note</Text>
          <Text style={styles.dailyTaskDescription}>
            Everything above is generated by this app&apos;s own rules, entirely on your
            phone. This is different: it sends only your latest score, grade, and recent
            trend for that task (never video, never raw pose data) to Claude, Anthropic&apos;s
            AI model, to write one short, personal note about it.
          </Text>

          <Pressable
            style={[styles.mainButton, { marginTop: 12 }, llmCoachLoading && styles.disabledButton]}
            onPress={requestLlmCoachNote}
            disabled={llmCoachLoading}
            accessibilityRole="button"
            accessibilityLabel="Ask Claude about my latest result"
            accessibilityState={{ disabled: llmCoachLoading, busy: llmCoachLoading }}
          >
            {llmCoachLoading ? (
              <View style={styles.analyzingRow}>
                <ActivityIndicator size="small" color="#ffffff" />
                <Text style={styles.buttonText}>Asking Claude...</Text>
              </View>
            ) : (
              <Text style={styles.buttonText}>
                {llmCoachNote ? 'Ask Again' : 'Ask Claude About My Latest Result'}
              </Text>
            )}
          </Pressable>

          {llmCoachNote ? (
            <View style={[styles.cameraSetupHeroCard, { marginTop: 12 }]}>
              <Text style={styles.metricLabelSmall}>Claude&apos;s Note</Text>
              <Text style={styles.metricValueLarge}>{llmCoachNote}</Text>
            </View>
          ) : null}

          {llmCoachUnavailableReason ? (
            <View style={[styles.cameraSetupCard, { marginTop: 12 }]}>
              <Text style={styles.sectionTitle}>Not Turned On Yet</Text>
              <Text style={styles.cameraMistakeText}>{llmCoachUnavailableReason}</Text>
            </View>
          ) : null}

          {llmCoachError ? (
            <View style={[styles.errorCard, { marginTop: 12 }]}>
              <Text style={styles.errorText}>{llmCoachError}</Text>
            </View>
          ) : null}
        </View>

        <View style={styles.aiCoachDisclaimerCard}>
          <Text style={styles.aiCoachDisclaimerTitle}>Important</Text>
          <Text style={styles.aiCoachDisclaimerText}>
            This coach is for movement awareness and tracking only. It does not diagnose, treat, or replace medical advice. Claude&apos;s note above is AI-generated commentary on your scores, not a clinical opinion.
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Coach Actions</Text>

        <View style={styles.rolloutActionGrid}>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowTrendEngine(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Trend Engine</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowGuidedOnboarding(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Start Here</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Weekly Report</Text>
          </Pressable>


          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowDailyHealthOverview(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Daily Overview</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowReportExport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Movement Passport</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowHistory(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>History</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowAiCoach(false);
              setShowCameraSetupGuide(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Camera Setup</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.shareBetaButton}
          onPress={shareCoachSummary}
        >
          <Text style={styles.buttonText}>Share Coach Summary</Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowAiCoach(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showWeeklyReport) {
    const weekly = getWeeklyHealthReport(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.weeklyReportContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Weekly Report</Text>
        </View>

        <Text style={styles.title}>Weekly Health Report</Text>

        <Text style={styles.subtitle}>
          Review your movement checks from the last 7 days and see what to focus on next.
        </Text>

        <View style={styles.weeklyReportHeroCard}>
          <Text style={styles.sectionTitle}>Weekly Summary</Text>

          <Text style={styles.weeklyReportHeadline}>
            {weekly.headline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(weekly.weeklyStatus).bg,
                borderColor: getGradeColors(weekly.weeklyStatus).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(weekly.weeklyStatus).text },
              ]}
            >
              {weekly.weeklyStatus}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {weekly.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Focus
          </Text>
          <Text style={styles.metricValueLarge}>
            {weekly.focus}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Recommended Next Action
          </Text>
          <Text style={styles.metricValueLarge}>
            {weekly.nextAction}
          </Text>
        </View>

        <View style={styles.weeklyStatsGrid}>
          <View style={styles.weeklyStatBox}>
            <Text style={styles.weeklyStatValue}>
              {weekly.totalChecksThisWeek}
            </Text>
            <Text style={styles.weeklyStatLabel}>Checks This Week</Text>
          </View>

          <View style={styles.weeklyStatBox}>
            <Text style={styles.weeklyStatValue}>
              {weekly.areasCheckedThisWeek}/6
            </Text>
            <Text style={styles.weeklyStatLabel}>Areas Checked</Text>
          </View>

          <View style={styles.weeklyStatBox}>
            <Text style={styles.weeklyStatValue}>
              {weekly.profileComplete}%
            </Text>
            <Text style={styles.weeklyStatLabel}>Profile Complete</Text>
          </View>

          <View style={styles.weeklyStatBox}>
            <Text style={styles.weeklyStatValue}>
              {weekly.baselinesBuilt}%
            </Text>
            <Text style={styles.weeklyStatLabel}>Baselines Built</Text>
          </View>
        </View>

        <View style={styles.weeklyReportCard}>
          <Text style={styles.sectionTitle}>Best Area This Week</Text>

          {weekly.bestArea ? (
            <>
              <Text style={styles.weeklyAreaTitle}>
                {weekly.bestArea.label}
              </Text>
              <Text style={styles.weeklyAreaScore}>
                {weekly.bestArea.score !== null ? `${weekly.bestArea.score}/100` : 'N/A'}
              </Text>
              <Text style={styles.metricValueLarge}>
                {weekly.bestArea.detail}
              </Text>
            </>
          ) : (
            <Text style={styles.metricValueLarge}>
              No best area yet. Record a movement check this week first.
            </Text>
          )}
        </View>

        <View style={styles.weeklyReportCard}>
          <Text style={styles.sectionTitle}>Area To Watch</Text>

          {weekly.areaToWatch ? (
            <>
              <Text style={styles.weeklyAreaTitle}>
                {weekly.areaToWatch.label}
              </Text>
              <Text style={styles.weeklyAreaScore}>
                {weekly.areaToWatch.score !== null ? `${weekly.areaToWatch.score}/100` : 'N/A'}
              </Text>
              <Text style={styles.metricValueLarge}>
                {weekly.areaToWatch.detail}
              </Text>
            </>
          ) : (
            <Text style={styles.metricValueLarge}>
              No area to watch yet. Keep recording weekly checks.
            </Text>
          )}
        </View>

        <Text style={styles.sectionTitle}>Weekly Movement Areas</Text>

        {weekly.weeklyItems.map((item) => (
          <View key={item.key} style={styles.weeklyAreaCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.weeklyAreaName}>
                {item.label}
              </Text>

              <Text style={styles.weeklyAreaDetail}>
                {item.weeklyCount} check{item.weeklyCount === 1 ? '' : 's'} this week
              </Text>

              <Text style={styles.weeklyAreaDetail}>
                {item.detail}
              </Text>
            </View>

            <View style={styles.weeklyAreaRight}>
              <Text style={styles.weeklyAreaScoreSmall}>
                {item.score !== null ? `${item.score}/100` : 'N/A'}
              </Text>

              <View
                style={[
                  styles.miniGradeBadge,
                  {
                    backgroundColor: getGradeColors(item.status).bg,
                    borderColor: getGradeColors(item.status).border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.miniGradeBadgeText,
                    { color: getGradeColors(item.status).text },
                  ]}
                >
                  {item.status}
                </Text>
              </View>
            </View>
          </View>
        ))}

        <Text style={styles.sectionTitle}>Weekly Actions</Text>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setShowTrendEngine(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Trend Engine</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setMode('daily');
              setDailyTask('reach');
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Record Reach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setMode('daily');
              setDailyTask('sit_to_stand');
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Record Sit-to-Stand</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowWeeklyReport(false);
              setShowHistory(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>History</Text>
          </Pressable>
        </View>

        <View style={styles.weeklyDisclaimerCard}>
          <Text style={styles.weeklyDisclaimerTitle}>Important</Text>
          <Text style={styles.weeklyDisclaimerText}>
            This weekly report is for movement awareness and tracking only. It does not diagnose, treat, or replace medical advice.
          </Text>
        </View>

        <Pressable
          style={styles.shareBetaButton}
          onPress={shareWeeklyReport}
        >
          <Text style={styles.buttonText}>Share Weekly Report</Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowWeeklyReport(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showMobilityProfile) {
    const profile = getFunctionalMobilityProfile(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.mobilityProfileContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Mobility Profile</Text>
        </View>

        <Text style={styles.title}>Kinetra Mobility Profile</Text>

        <Text style={styles.subtitle}>
          Combine your movement checks into one personal mobility profile across control, gait, balance, and functional mobility.
        </Text>

        <View style={styles.mobilityHeroCard}>
          <Text style={styles.sectionTitle}>Profile Summary</Text>

          <Text style={styles.mobilityHeadline}>
            {profile.headline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(profile.profileLabel).bg,
                borderColor: getGradeColors(profile.profileLabel).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(profile.profileLabel).text },
              ]}
            >
              {profile.profileLabel}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Overall Functional Mobility Score
          </Text>

          <Text style={styles.mobilityScoreText}>
            {profile.overallScore !== null ? `${profile.overallScore}/100` : 'N/A'}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Profile Type
          </Text>
          <Text style={styles.metricValueLarge}>
            {profile.profileType}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {profile.summary}
          </Text>
        </View>

        <View style={styles.mobilityStatsGrid}>
          <View style={styles.mobilityStatBox}>
            <Text style={styles.mobilityStatValue}>
              {profile.completionPercent}%
            </Text>
            <Text style={styles.mobilityStatLabel}>Profile Complete</Text>
          </View>

          <View style={styles.mobilityStatBox}>
            <Text style={styles.mobilityStatValue}>
              {profile.baselinePercent}%
            </Text>
            <Text style={styles.mobilityStatLabel}>Baselines Built</Text>
          </View>

          <View style={styles.mobilityStatBox}>
            <Text style={styles.mobilityStatValue}>
              {profile.domains.filter((domain) => domain.score !== null).length}/5
            </Text>
            <Text style={styles.mobilityStatLabel}>Domains Started</Text>
          </View>
        </View>

        <View style={styles.mobilityHighlightGrid}>
          <View style={styles.mobilityHighlightCard}>
            <Text style={styles.metricLabelSmall}>Strongest Domain</Text>
            <Text style={styles.mobilityHighlightTitle}>
              {profile.strongestDomain ? profile.strongestDomain.label : 'Not available yet'}
            </Text>
            <Text style={styles.mobilityHighlightText}>
              {profile.strongestDomain
                ? profile.strongestDomain.summary
                : 'Record more checks to identify your strongest movement domain.'}
            </Text>
          </View>

          <View style={styles.mobilityHighlightCard}>
            <Text style={styles.metricLabelSmall}>Main Watch Area</Text>
            <Text style={styles.mobilityHighlightTitle}>
              {profile.weakestDomain ? profile.weakestDomain.label : 'Not available yet'}
            </Text>
            <Text style={styles.mobilityHighlightText}>
              {profile.weakestDomain
                ? profile.weakestDomain.summary
                : 'Record more checks to identify your main watch area.'}
            </Text>
          </View>
        </View>

        <View style={styles.mobilityNextCard}>
          <Text style={styles.sectionTitle}>Recommended Next Action</Text>

          <Text style={styles.metricValueLarge}>
            {profile.recommendedNext}
          </Text>

          {profile.urgentRecheck ? (
            <>
              <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                Priority Domain
              </Text>
              <Text style={styles.mobilityHighlightTitle}>
                {profile.urgentRecheck.label}
              </Text>
              <Text style={styles.mobilityHighlightText}>
                {profile.urgentRecheck.summary}
              </Text>
            </>
          ) : null}
        </View>

        <Text style={styles.sectionTitle}>Mobility Domains</Text>

        {profile.domains.map((domain) => (
          <View key={domain.key} style={styles.mobilityDomainCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.mobilityDomainTitle}>
                {domain.label}
              </Text>

              <Text style={styles.mobilityDomainText}>
                {domain.description}
              </Text>

              <Text style={styles.mobilityDomainText}>
                {domain.summary}
              </Text>

              <Text style={styles.mobilityDomainMeta}>
                {domain.checkedCount}/{domain.totalTasks} checks started • {domain.baselineReadyCount} baseline-ready
              </Text>
            </View>

            <View style={styles.mobilityDomainRight}>
              <Text style={styles.mobilityDomainScore}>
                {domain.score !== null ? `${domain.score}/100` : 'N/A'}
              </Text>

              <View
                style={[
                  styles.miniGradeBadge,
                  {
                    backgroundColor: getGradeColors(domain.status).bg,
                    borderColor: getGradeColors(domain.status).border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.miniGradeBadgeText,
                    { color: getGradeColors(domain.status).text },
                  ]}
                >
                  {domain.status}
                </Text>
              </View>
            </View>
          </View>
        ))}

        <Text style={styles.sectionTitle}>Profile Actions</Text>

        <View style={styles.rolloutActionGrid}>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowTrendEngine(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Trend Engine</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowGuidedOnboarding(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Start Here</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowDailyHealthOverview(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Daily Overview</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Weekly Report</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowMobilityProfile(false);
              setShowReportExport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Movement Passport</Text>
          </Pressable>
        </View>

        <View style={styles.mobilityDisclaimerCard}>
          <Text style={styles.mobilityDisclaimerTitle}>Important</Text>
          <Text style={styles.mobilityDisclaimerText}>
            {APP_SAFETY_NOTE}
          </Text>
        </View>

        <Pressable
          style={styles.shareBetaButton}
          onPress={shareMobilityProfile}
        >
          <Text style={styles.buttonText}>Share Mobility Profile</Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowMobilityProfile(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showServerDiagnostics) {
    const serverBadge = getServerHealthBadgeLabel(serverHealth.status);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.serverDiagnosticsContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Backend</Text>
        </View>

        <Text style={styles.title}>Server Diagnostics</Text>

        <Text style={styles.subtitle}>
          Check whether the app can reach the backend before recording or sharing the app with someone else.
        </Text>

        <View style={styles.serverHeroCard}>
          <Text style={styles.sectionTitle}>Backend Status</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(serverBadge).bg,
                borderColor: getGradeColors(serverBadge).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(serverBadge).text },
              ]}
            >
              {serverBadge}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Current API URL
          </Text>

          <Text style={styles.serverUrlText}>
            {serverHealth.apiUrl}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Status Message
          </Text>

          <Text style={styles.metricValueLarge}>
            {serverHealth.message}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What This Means
          </Text>

          <Text style={styles.metricValueLarge}>
            {getServerHealthSummary(serverHealth.status)}
          </Text>

          {serverHealth.checkedAt ? (
            <>
              <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                Last Checked
              </Text>

              <Text style={styles.metricValueLarge}>
                {formatSessionDate(serverHealth.checkedAt)}
              </Text>
            </>
          ) : null}
        </View>

        <View style={styles.serverChecklistCard}>
          <Text style={styles.sectionTitle}>Local Testing Checklist</Text>

          <Text style={styles.serverChecklistText}>
            • Flask server is running.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Phone and computer are on the same Wi-Fi.
          </Text>
          <Text style={styles.serverChecklistText}>
            • API URL matches your computer IP address.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Firewall is not blocking port 5000.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Use short videos while testing.
          </Text>
        </View>

        <View style={styles.serverChecklistCard}>
          <Text style={styles.sectionTitle}>Rollout Checklist</Text>

          <Text style={styles.serverChecklistText}>
            • Deploy Flask backend to a public URL.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Set EXPO_PUBLIC_DEPLOYED_API_BASE_URL in .env to the deployed URL.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Add /health route to Flask.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Test analysis from a phone not on your home Wi-Fi.
          </Text>
          <Text style={styles.serverChecklistText}>
            • Keep uploads short enough that the backend does not time out.
          </Text>
        </View>

        <Pressable
          style={styles.shareBetaButton}
          onPress={checkAnalysisServer}
        >
          <Text style={styles.buttonText}>
            {serverHealth.status === 'checking' ? 'Checking...' : 'Check Server'}
          </Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowServerDiagnostics(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showTrendEngine) {
    const trendEngine = getLongitudinalTrendEngine(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.longitudinalContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Trend Engine</Text>
        </View>

        <Text style={styles.title}>Kinetra Trend Engine</Text>

        <Text style={styles.subtitle}>
          Track movement change over time across 7-day trends, 30-day trends, baselines, confidence, and recheck flags.
        </Text>

        <View style={styles.longitudinalHeroCard}>
          <Text style={styles.sectionTitle}>Trend Summary</Text>

          <Text style={styles.longitudinalHeadline}>
            {trendEngine.headline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(trendEngine.engineStatus).bg,
                borderColor: getGradeColors(trendEngine.engineStatus).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(trendEngine.engineStatus).text },
              ]}
            >
              {trendEngine.engineStatus}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Overall Trend Score
          </Text>

          <Text style={styles.longitudinalScoreText}>
            {trendEngine.overallTrendScore !== null ? `${trendEngine.overallTrendScore}/100` : 'N/A'}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Trend Confidence
          </Text>

          <Text style={styles.metricValueLarge}>
            {trendEngine.trendConfidence}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Main Finding
          </Text>

          <Text style={styles.metricValueLarge}>
            {trendEngine.mainFinding}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Recommended Next Action
          </Text>

          <Text style={styles.metricValueLarge}>
            {trendEngine.nextAction}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Trend Snapshot</Text>

        <View style={styles.longitudinalStatsGrid}>
          <View style={styles.longitudinalStatBox}>
            <Text style={styles.longitudinalStatValue}>
              {trendEngine.activeTrends.length}
            </Text>
            <Text style={styles.longitudinalStatLabel}>Active Areas</Text>
          </View>

          <View style={styles.longitudinalStatBox}>
            <Text style={styles.longitudinalStatValue}>
              {trendEngine.baselineReadyTrends.length}
            </Text>
            <Text style={styles.longitudinalStatLabel}>Baselines</Text>
          </View>

          <View style={styles.longitudinalStatBox}>
            <Text style={styles.longitudinalStatValue}>
              {trendEngine.improvingTrends.length}
            </Text>
            <Text style={styles.longitudinalStatLabel}>Improving</Text>
          </View>

          <View style={styles.longitudinalStatBox}>
            <Text style={styles.longitudinalStatValue}>
              {trendEngine.declineTrends.length}
            </Text>
            <Text style={styles.longitudinalStatLabel}>Watch</Text>
          </View>
        </View>

        {trendEngine.declineTrends.length > 0 ? (
          <View style={styles.longitudinalWatchCard}>
            <Text style={styles.longitudinalWatchTitle}>Recheck Flags</Text>

            {trendEngine.declineTrends.map((trend) => (
              <Text key={trend.task} style={styles.longitudinalWatchText}>
                • {trend.label}: {trend.recommendation}
              </Text>
            ))}
          </View>
        ) : null}

        {trendEngine.improvingTrends.length > 0 ? (
          <View style={styles.longitudinalImprovingCard}>
            <Text style={styles.longitudinalImprovingTitle}>Improving Trends</Text>

            {trendEngine.improvingTrends.map((trend) => (
              <Text key={trend.task} style={styles.longitudinalImprovingText}>
                • {trend.label}: {trend.summary}
              </Text>
            ))}
          </View>
        ) : null}

        <Text style={styles.sectionTitle}>Movement Area Trends</Text>

        {trendEngine.taskTrends.map((trend) => (
          <View key={trend.task} style={styles.longitudinalAreaCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.longitudinalAreaTitle}>
                {trend.label}
              </Text>

              <Text style={styles.longitudinalAreaText}>
                {trend.summary}
              </Text>

              <Text style={styles.longitudinalAreaMeta}>
                Latest: {trend.latestScore !== null ? `${trend.latestScore}/100` : 'N/A'} • Previous: {trend.previousScore !== null ? `${trend.previousScore}/100` : 'N/A'}
              </Text>

              <Text style={styles.longitudinalAreaMeta}>
                Baseline: {trend.baselineScore !== null ? `${trend.baselineScore}/100` : 'Building'} • Confidence: {trend.confidence}
              </Text>

              <Text style={styles.longitudinalAreaMeta}>
                7-day change: {trend.sevenDayStats.change !== null ? `${trend.sevenDayStats.change > 0 ? '+' : ''}${trend.sevenDayStats.change.toFixed(1)}` : 'N/A'} • 30-day change: {trend.thirtyDayStats.change !== null ? `${trend.thirtyDayStats.change > 0 ? '+' : ''}${trend.thirtyDayStats.change.toFixed(1)}` : 'N/A'}
              </Text>
            </View>

            <View style={styles.longitudinalAreaRight}>
              <Text style={styles.longitudinalAreaScore}>
                {trend.latestScore !== null ? `${trend.latestScore}/100` : 'N/A'}
              </Text>

              <View
                style={[
                  styles.miniGradeBadge,
                  {
                    backgroundColor: getGradeColors(trend.status).bg,
                    borderColor: getGradeColors(trend.status).border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.miniGradeBadgeText,
                    { color: getGradeColors(trend.status).text },
                  ]}
                >
                  {trend.status}
                </Text>
              </View>
            </View>
          </View>
        ))}

        <View style={styles.longitudinalInfoCard}>
          <Text style={styles.sectionTitle}>How To Read This</Text>

          <Text style={styles.longitudinalInfoText}>
            The 7-day trend shows short-term change. The 30-day trend shows longer-term direction. Baseline comparison shows whether the latest result is close to your own normal range.
          </Text>

          <Text style={styles.longitudinalInfoText}>
            Recheck flags do not mean injury or diagnosis. They mean the app saw a change worth repeating with the same camera setup.
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Trend Actions</Text>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTrendEngine(false);
              setShowReportExport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Movement Passport</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTrendEngine(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTrendEngine(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Weekly Report</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowTrendEngine(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.shareBetaButton}
          onPress={shareLongitudinalTrendReport}
        >
          <Text style={styles.buttonText}>Share Trend Report</Text>
        </Pressable>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowTrendEngine(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showGuidedOnboarding) {
    const onboarding = getOnboardingProgress(savedSessions);
    const suggestedNext = getSuggestedNextProfileTask(savedSessions);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.guidedOnboardingContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Start Here</Text>
        </View>

        <Text style={styles.title}>Build Your Kinetra Profile</Text>

        <Text style={styles.subtitle}>
          Start with guided movement checks so Kinetra can build your first mobility profile.
        </Text>

        <View style={styles.guidedHeroCard}>
          <Text style={styles.sectionTitle}>Setup Progress</Text>

          <Text style={styles.guidedHeadline}>
            {onboarding.headline}
          </Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(onboarding.status).bg,
                borderColor: getGradeColors(onboarding.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(onboarding.status).text },
              ]}
            >
              {onboarding.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Progress
          </Text>

          <Text style={styles.guidedProgressText}>
            {onboarding.completedCount}/{onboarding.totalCount} starter checks complete
          </Text>

          <View style={styles.guidedProgressTrack}>
            <View
              style={[
                styles.guidedProgressFill,
                { width: `${onboarding.completionPercent}%` },
              ]}
            />
          </View>

          <Text style={styles.metricValueLarge}>
            {onboarding.summary}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>How This Works</Text>

        <View style={styles.guidedInfoCard}>
          <Text style={styles.guidedOnboardingStepTitle}>1. Record</Text>
          <Text style={styles.guidedStepText}>
            Choose one guided movement check and record a short video with the correct camera setup.
          </Text>

          <Text style={styles.guidedOnboardingStepTitle}>2. Analyze</Text>
          <Text style={styles.guidedStepText}>
            The app reviews movement quality, confidence, score, and task-specific details.
          </Text>

          <Text style={styles.guidedOnboardingStepTitle}>3. Save</Text>
          <Text style={styles.guidedStepText}>
            Save the result so the app can build trends, baselines, weekly reports, and your Mobility Profile.
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Starter Checks</Text>

        {onboarding.steps.map((step) => (
          <View key={step.task} style={styles.guidedTaskCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.guidedTaskTitle}>
                {step.complete ? '✓ ' : ''}{step.label}
              </Text>

              <Text style={styles.guidedTaskText}>
                {step.reason}
              </Text>

              <Text style={styles.guidedTaskStatus}>
                {step.complete ? 'Completed' : 'Not completed yet'}
              </Text>
            </View>

            <Pressable
              style={[
                styles.guidedSmallButton,
                step.complete ? styles.guidedSmallButtonDone : null,
              ]}
              onPress={() => {
                setShowGuidedOnboarding(false);
                setMode('daily');
                setDailyTask(step.task);
                setShowCameraSetupGuide(true);
                setStarted(false);
              }}
            >
              <Text style={styles.guidedSmallButtonText}>
                {step.complete ? 'Repeat' : 'Start'}
              </Text>
            </Pressable>
          </View>
        ))}

        <View style={styles.guidedNextCard}>
          <Text style={styles.sectionTitle}>Recommended Next Check</Text>

          <Text style={styles.guidedNextTitle}>
            {suggestedNext.label}
          </Text>

          <Text style={styles.guidedTaskText}>
            {suggestedNext.reason}
          </Text>

          <Pressable
            style={styles.mainButton}
            onPress={() => {
              setShowGuidedOnboarding(false);
              setMode('daily');
              setDailyTask(suggestedNext.task);
              setShowCameraSetupGuide(true);
              setStarted(false);
            }}
          >
            <Text style={styles.buttonText}>
              Start {suggestedNext.label}
            </Text>
          </Pressable>
        </View>

        <View style={styles.guidedWhyCard}>
          <Text style={styles.sectionTitle}>Why Baselines Matter</Text>

          <Text style={styles.guidedStepText}>
            One check gives a snapshot. Repeated checks build your personal baseline. That lets the app compare you against your own normal range instead of guessing from random population averages.
          </Text>
        </View>

        <Text style={styles.sectionTitle}>After Setup</Text>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedOnboarding(false);
              setShowMobilityProfile(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Mobility Profile</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedOnboarding(false);
              setShowWeeklyReport(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Weekly Report</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedOnboarding(false);
              setShowAiCoach(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>AI Coach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowGuidedOnboarding(false);
              setShowDailyHealthOverview(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Daily Overview</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowGuidedOnboarding(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showYoloFramework) {
    const yoloFramework = getYoloFrameworkSummary();
    const activeModule = getActiveMovementModule(mode, dailyTask);

    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.yoloFrameworkContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>YOLO Framework</Text>
        </View>

        <Text style={styles.title}>YOLO Module Framework</Text>

        <Text style={styles.subtitle}>
          Organize current MediaPipe modules and future YOLO modules without breaking the existing app.
        </Text>

        <View style={styles.yoloHeroCard}>
          <Text style={styles.sectionTitle}>Framework Status</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(yoloFramework.status).bg,
                borderColor: getGradeColors(yoloFramework.status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(yoloFramework.status).text },
              ]}
            >
              {yoloFramework.status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {yoloFramework.summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Next Step
          </Text>
          <Text style={styles.metricValueLarge}>
            {yoloFramework.nextStep}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Hard Rule
          </Text>
          <Text style={styles.yoloWarningText}>
            {yoloFramework.warning}
          </Text>
        </View>

        <View style={styles.yoloStatsGrid}>
          <View style={styles.yoloStatBox}>
            <Text style={styles.yoloStatValue}>{yoloFramework.activeCount}</Text>
            <Text style={styles.yoloStatLabel}>Active Modules</Text>
          </View>

          <View style={styles.yoloStatBox}>
            <Text style={styles.yoloStatValue}>{yoloFramework.yoloCount}</Text>
            <Text style={styles.yoloStatLabel}>YOLO Modules</Text>
          </View>

          <View style={styles.yoloStatBox}>
            <Text style={styles.yoloStatValue}>{yoloFramework.foundationReadyCount}</Text>
            <Text style={styles.yoloStatLabel}>Ready Next</Text>
          </View>
        </View>

        <View style={styles.yoloCurrentModuleCard}>
          <Text style={styles.sectionTitle}>Current Selected Module</Text>

          <Text style={styles.yoloModuleTitle}>
            {activeModule.title}
          </Text>

          <Text style={styles.yoloModuleMeta}>
            Backend: {getBackendLabel(activeModule.backend)}
          </Text>

          <Text style={styles.yoloModuleText}>
            {activeModule.purpose}
          </Text>

          <Text style={styles.yoloModuleText}>
            {activeModule.currentState}
          </Text>
        </View>

        <Text style={styles.sectionTitle}>Module Registry</Text>

        {yoloFramework.modules.map((module) => (
          <View key={module.id} style={styles.yoloModuleCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.yoloModuleTitle}>
                {module.title}
              </Text>

              <Text style={styles.yoloModuleMeta}>
                {getBackendLabel(module.backend)} • {module.id}
              </Text>

              <Text style={styles.yoloModuleText}>
                {module.purpose}
              </Text>

              <Text style={styles.yoloModuleText}>
                {module.currentState}
              </Text>
            </View>

            <View
              style={[
                styles.miniGradeBadge,
                {
                  backgroundColor: getGradeColors(getModuleStatusLabel(module.status)).bg,
                  borderColor: getGradeColors(getModuleStatusLabel(module.status)).border,
                },
              ]}
            >
              <Text
                style={[
                  styles.miniGradeBadgeText,
                  { color: getGradeColors(getModuleStatusLabel(module.status)).text },
                ]}
              >
                {getModuleStatusLabel(module.status)}
              </Text>
            </View>
          </View>
        ))}

        <View style={styles.yoloRoadmapCard}>
          <Text style={styles.sectionTitle}>Next Build Order</Text>

          <Text style={styles.yoloRoadmapText}>1. Walking Analysis V1</Text>
          <Text style={styles.yoloRoadmapText}>2. Balance Analysis V1</Text>
          <Text style={styles.yoloRoadmapText}>3. Timed Up and Go V1</Text>
        </View>

        <Text style={styles.sectionTitle}>Framework Actions</Text>

        <View style={styles.rolloutActionGrid}>
          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowYoloFramework(false);
              setMode('daily');
              setDailyTask('reach');
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Select Reach</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowYoloFramework(false);
              setMode('daily');
              setDailyTask('sit_to_stand');
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Select Sit-to-Stand</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowYoloFramework(false);
              setMode('rehab');
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Select Rehab</Text>
          </Pressable>

          <Pressable
            style={styles.rolloutActionButton}
            onPress={() => {
              setShowYoloFramework(false);
              setShowBuilderTools(true);
            }}
          >
            <Text style={styles.rolloutActionButtonText}>Builder Tools</Text>
          </Pressable>
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowYoloFramework(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  if (showHistory) {
    return (
      <ScrollView
        style={styles.homeScroll}
        contentContainerStyle={styles.historyContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.heroBadge}>
          <Text style={styles.heroBadgeText}>Movement History</Text>
        </View>

        <Text style={styles.title}>Saved Checks</Text>

        <Text style={styles.subtitle}>
          Review Daily and Rehab checks, compare them with baseline, and watch movement quality change over time.
        </Text>

        <Pressable
          style={styles.dailyHealthInlineButton}
          onPress={() => {
            setShowHistory(false);
            setShowDailyHealthOverview(true);
          }}
        >
          <Text style={styles.testingGuideButtonText}>Open Daily Health Overview</Text>
        </Pressable>

        <View style={styles.movementOverviewCard}>
          <Text style={styles.sectionTitle}>Movement Change Overview</Text>

          <View
            style={[
              styles.gradeBadge,
              {
                backgroundColor: getGradeColors(getMovementOverview(savedSessions).status).bg,
                borderColor: getGradeColors(getMovementOverview(savedSessions).status).border,
              },
            ]}
          >
            <Text
              style={[
                styles.gradeBadgeText,
                { color: getGradeColors(getMovementOverview(savedSessions).status).text },
              ]}
            >
              {getMovementOverview(savedSessions).status}
            </Text>
          </View>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Summary
          </Text>
          <Text style={styles.metricValueLarge}>
            {getMovementOverview(savedSessions).summary}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            What To Watch
          </Text>
          <Text style={styles.metricValueLarge}>
            {getMovementOverview(savedSessions).watch}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Recommended Next Check
          </Text>
          <Text style={styles.metricValueLarge}>
            {getMovementOverview(savedSessions).nextStep}
          </Text>

          <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
            Movement Areas
          </Text>

          {getMovementOverview(savedSessions).items.map((item) => (
            <View key={item.key} style={styles.movementOverviewRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.movementOverviewLabel}>
                  {item.label}
                </Text>
                <Text style={styles.movementOverviewDetail}>
                  {item.detail}
                </Text>
              </View>

              <View style={styles.movementOverviewRight}>
                <Text style={styles.movementOverviewScore}>
                  {item.score !== null ? `${item.score}/100` : 'N/A'}
                </Text>

                <View
                  style={[
                    styles.miniGradeBadge,
                    {
                      backgroundColor: getGradeColors(item.status).bg,
                      borderColor: getGradeColors(item.status).border,
                    },
                  ]}
                >
                  <Text
                    style={[
                      styles.miniGradeBadgeText,
                      { color: getGradeColors(item.status).text },
                    ]}
                  >
                    {item.status}
                  </Text>
                </View>
              </View>
            </View>
          ))}
        </View>

        {(['reach', 'arm_raise', 'sit_to_stand', 'walking', 'balance', 'timed_up_and_go'] as const).map((task) => {
          const taskSessions = getSavedSessionsForTask(savedSessions, task);
          const latestSession = getLatestSessionForTask(savedSessions, task);
          const trendScores = getTaskTrendScores(savedSessions, task);
          const baseline = getBaselineForTask(savedSessions, task);

          return (
            <View key={task} style={styles.historyTaskCard}>
              <Text style={styles.historyTaskTitle}>
                {getDailyTaskLabel(task)}
              </Text>

              <Text style={styles.historyTaskSubtitle}>
                {taskSessions.length} saved check{taskSessions.length === 1 ? '' : 's'}
              </Text>

              {latestSession ? (
                <>
                  <Text style={styles.metricLabelSmall}>Latest Score</Text>

                  <Text style={styles.historyLatestScore}>
                    {latestSession.primary_score !== null
                      ? `${latestSession.primary_score}/100`
                      : 'N/A'}
                  </Text>

                  <View
                    style={[
                      styles.gradeBadge,
                      {
                        backgroundColor: getGradeColors(latestSession.primary_grade).bg,
                        borderColor: getGradeColors(latestSession.primary_grade).border,
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.gradeBadgeText,
                        { color: getGradeColors(latestSession.primary_grade).text },
                      ]}
                    >
                      {latestSession.primary_grade}
                    </Text>
                  </View>

                  <View style={styles.historyBaselineBox}>
                    <Text style={styles.metricLabelSmall}>Personal Baseline</Text>
                    <Text style={styles.historyBaselineText}>
                      {baseline.baselineScore !== null
                        ? `${baseline.baselineScore}/100 from ${baseline.sessionCount} checks`
                        : `Building baseline (${baseline.sessionCount}/3 checks)`}
                    </Text>
                  </View>

                  {trendScores.length >= 2 ? (
                    <View style={styles.historyGraphBlock}>
                      <Text style={styles.metricLabelSmall}>Trend</Text>
                      <ScoreGraph scores={trendScores} />
                    </View>
                  ) : (
                    <Text style={styles.historyEmptyText}>
                      Record this check at least twice to show a trend graph.
                    </Text>
                  )}

                  <View style={styles.historyListBlock}>
                    <Text style={styles.metricLabelSmall}>Recent Checks</Text>

                    {taskSessions.slice(0, 5).map((session) => (
                      <View key={session.id} style={styles.historySessionRow}>
                        <View>
                          <Text style={styles.historySessionDate}>
                            {formatSessionDate(session.timestamp)}
                          </Text>
                          <Text style={styles.historySessionGrade}>
                            {session.primary_grade}
                          </Text>
                        </View>

                        <Text style={styles.historySessionScore}>
                          {session.primary_score !== null
                            ? `${session.primary_score}/100`
                            : 'N/A'}
                        </Text>
                      </View>
                    ))}
                  </View>
                </>
              ) : (
                <Text style={styles.historyEmptyText}>
                  No saved {getDailyTaskLabel(task).toLowerCase()} checks yet. Record this movement from the Daily mode to start building a trend.
                </Text>
              )}
            </View>
          );
        })}

        <View style={styles.historyTaskCard}>
          <Text style={styles.historyTaskTitle}>Rehab Consistency</Text>

          <Text style={styles.historyTaskSubtitle}>
            {getSavedRehabSessions(savedSessions).length} saved rehab check{getSavedRehabSessions(savedSessions).length === 1 ? '' : 's'}
          </Text>

          {getLatestRehabSession(savedSessions) ? (
            <>
              <Text style={styles.metricLabelSmall}>Latest Consistency Score</Text>

              <Text style={styles.historyLatestScore}>
                {getLatestRehabSession(savedSessions)?.primary_score !== null &&
                  getLatestRehabSession(savedSessions)?.primary_score !== undefined
                  ? `${getLatestRehabSession(savedSessions)?.primary_score}/100`
                  : 'N/A'}
              </Text>

              <View
                style={[
                  styles.gradeBadge,
                  {
                    backgroundColor: getGradeColors(getLatestRehabSession(savedSessions)?.primary_grade || 'N/A').bg,
                    borderColor: getGradeColors(getLatestRehabSession(savedSessions)?.primary_grade || 'N/A').border,
                  },
                ]}
              >
                <Text
                  style={[
                    styles.gradeBadgeText,
                    {
                      color: getGradeColors(getLatestRehabSession(savedSessions)?.primary_grade || 'N/A').text,
                    },
                  ]}
                >
                  {getLatestRehabSession(savedSessions)?.primary_grade || 'N/A'}
                </Text>
              </View>

              <View style={styles.historyBaselineBox}>
                <Text style={styles.metricLabelSmall}>Rehab Baseline</Text>
                <Text style={styles.historyBaselineText}>
                  {getRehabBaseline(savedSessions).baselineScore !== null
                    ? `${getRehabBaseline(savedSessions).baselineScore}/100 from ${getRehabBaseline(savedSessions).sessionCount} checks`
                    : `Building baseline (${getRehabBaseline(savedSessions).sessionCount}/3 checks)`}
                </Text>
              </View>

              {getRehabTrendScores(savedSessions).length >= 2 ? (
                <View style={styles.historyGraphBlock}>
                  <Text style={styles.metricLabelSmall}>Rehab Consistency Trend</Text>
                  <ScoreGraph scores={getRehabTrendScores(savedSessions)} />
                </View>
              ) : (
                <Text style={styles.historyEmptyText}>
                  Record Rehab mode at least twice to show a consistency trend graph.
                </Text>
              )}

              <View style={styles.historyListBlock}>
                <Text style={styles.metricLabelSmall}>Recent Rehab Checks</Text>

                {getSavedRehabSessions(savedSessions).slice(0, 5).map((session) => (
                  <View key={session.id} style={styles.historySessionRow}>
                    <View>
                      <Text style={styles.historySessionDate}>
                        {formatSessionDate(session.timestamp)}
                      </Text>
                      <Text style={styles.historySessionGrade}>
                        {session.primary_grade}
                      </Text>
                    </View>

                    <Text style={styles.historySessionScore}>
                      {session.primary_score !== null
                        ? `${session.primary_score}/100`
                        : 'N/A'}
                    </Text>
                  </View>
                ))}
              </View>
            </>
          ) : (
            <Text style={styles.historyEmptyText}>
              No saved rehab consistency checks yet. Record from Rehab mode to start tracking consistency over time.
            </Text>
          )}
        </View>

        <Pressable
          style={styles.mainButton}
          onPress={() => setShowHistory(false)}
        >
          <Text style={styles.buttonText}>Back Home</Text>
        </Pressable>
      </ScrollView>
    );
  }

  return (
    <View style={styles.container}>
      {/* Mount the on-device pose worker once for the whole screen so the
          MediaPipe model stays warm between views. The WebView itself is
          1x1 transparent and pointer-events-none — it's invisible. */}
      <OnDevicePoseWorker />

      {!started ? (
        <ScrollView
          style={styles.homeScroll}
          contentContainerStyle={styles.homeScrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.heroBadge}>
            <Text style={styles.heroBadgeText}>{APP_BETA_LABEL}</Text>
          </View>

          <Text style={styles.title}>{APP_NAME}</Text>
          <View style={styles.publicHomeTopActions}>
            <View style={[styles.heroBadge, { marginBottom: 0 }]}>
              <Text style={styles.heroBadgeText}>{getModeLabel(mode)}</Text>
            </View>

            <Pressable
              style={styles.startHereTopButton}
              onPress={() => setShowGuidedOnboarding(true)}
            >
              <Text style={styles.howItWorksButtonText}>Start Here</Text>
            </Pressable>

            <Pressable
              style={styles.serverStatusTopButton}
              onPress={() => setShowServerDiagnostics(true)}
            >
              <Text style={styles.howItWorksButtonText}>Server</Text>
            </Pressable>

            <Pressable
              style={styles.howItWorksButton}
              onPress={() => setShowOnboarding(true)}
            >
              <Text style={styles.howItWorksButtonText}>How It Works</Text>
            </Pressable>

            <Pressable
              style={styles.howItWorksButton}
              onPress={() => setShowTransparency(true)}
            >
              <Text style={styles.howItWorksButtonText}>Transparency</Text>
            </Pressable>

            <Pressable
              style={styles.howItWorksButton}
              onPress={() => setShowTeamRoster(true)}
            >
              <Text style={styles.howItWorksButtonText}>Team Roster</Text>
            </Pressable>

            <Pressable
              style={styles.howItWorksButton}
              onPress={() => setShowSettings(true)}
              accessibilityRole="button"
              accessibilityLabel="Settings"
            >
              <Text style={styles.howItWorksButtonText}>Settings</Text>
            </Pressable>

            <Pressable
              style={[styles.howItWorksButton, { position: 'relative' }]}
              onPress={openWhatsNew}
              accessibilityRole="button"
              accessibilityLabel={
                lastSeenWhatsNewVersion !== APP_VERSION
                  ? "What's New (unread)"
                  : "What's New"
              }
            >
              <Text style={styles.howItWorksButtonText}>What&apos;s New</Text>
              {lastSeenWhatsNewVersion !== APP_VERSION ? (
                <View style={styles.unreadDot} />
              ) : null}
            </Pressable>

            <Pressable
              style={styles.dailyHealthTopButton}
              onPress={() => setShowDailyHealthOverview(true)}
            >
              <Text style={styles.howItWorksButtonText}>Daily Health Overview</Text>
            </Pressable>

            <Pressable
              style={styles.reportTopButton}
              onPress={() => setShowReportExport(true)}
            >
              <Text style={styles.howItWorksButtonText}>Movement Passport</Text>
            </Pressable>

            <Pressable
              style={styles.trendEngineTopButton}
              onPress={() => setShowTrendEngine(true)}
            >
              <Text style={styles.howItWorksButtonText}>Trend Engine</Text>
            </Pressable>

            <Pressable
              style={styles.aiCoachTopButton}
              onPress={() => setShowAiCoach(true)}
            >
              <Text style={styles.howItWorksButtonText}>AI Coach</Text>
            </Pressable>

            <Pressable
              style={styles.weeklyReportTopButton}
              onPress={() => setShowWeeklyReport(true)}
            >
              <Text style={styles.howItWorksButtonText}>Weekly Report</Text>
            </Pressable>

            <Pressable
              style={styles.mobilityProfileTopButton}
              onPress={() => setShowMobilityProfile(true)}
            >
              <Text style={styles.howItWorksButtonText}>Mobility Profile</Text>
            </Pressable>

            <Pressable
              style={styles.cameraSetupTopButton}
              onPress={() => setShowCameraSetupGuide(true)}
            >
              <Text style={styles.howItWorksButtonText}>Camera Setup</Text>
            </Pressable>

            <Pressable
              style={styles.builderToolsTopButton}
              onPress={() => setShowBuilderTools(true)}
            >
              <Text style={styles.howItWorksButtonText}>Builder Tools</Text>
            </Pressable>
          </View>

          <Text style={styles.subtitle}>
            {APP_TAGLINE} Record a short movement check and get a clear score, confidence level, trend, and next action.
          </Text>

          <View style={styles.startHereCard}>
            <Text style={styles.startHereCardTitle}>New here?</Text>
            <Text style={styles.startHereCardText}>
              Start with three guided checks so the app can build your first Mobility Profile.
            </Text>

            <Pressable
              style={styles.startHereCardButton}
              onPress={() => setShowGuidedOnboarding(true)}
            >
              <Text style={styles.buttonText}>Build My Profile</Text>
            </Pressable>
          </View>

          <View style={styles.serverStatusCard}>
            <Text style={styles.serverStatusCardTitle}>Analysis Server</Text>

            <Text style={styles.serverStatusCardText}>
              {getServerHealthBadgeLabel(serverHealth.status)} • {API_BASE_URL}
            </Text>

            <Text style={styles.serverStatusCardSubtext}>
              {serverHealth.message}
            </Text>

            <Pressable
              style={styles.serverStatusCardButton}
              onPress={checkAnalysisServer}
            >
              <Text style={styles.buttonText}>Check Server</Text>
            </Pressable>
          </View>

          {savedSessions.length > 0 ? (
            <View style={styles.savedSessionHomeBlock}>
              <Text style={styles.sessionCountText}>
                {savedSessions.length} saved movement check{savedSessions.length === 1 ? '' : 's'}
              </Text>

              {getSavedRehabSessions(savedSessions).length > 0 ? (
                <Text style={styles.sessionSubCountText}>
                  {getSavedRehabSessions(savedSessions).length} rehab consistency check{getSavedRehabSessions(savedSessions).length === 1 ? '' : 's'}
                </Text>
              ) : null}

              <View style={styles.homeOverviewPreview}>
                <Text style={styles.homeOverviewLabel}>Movement Overview</Text>
                <Text style={styles.homeOverviewStatus}>
                  {getMovementOverview(savedSessions).status}
                </Text>
                <Text style={styles.homeOverviewText}>
                  {getMovementOverview(savedSessions).summary}
                </Text>
              </View>

              <View style={styles.homeDailyHealthPreview}>
                <Text style={styles.homeOverviewLabel}>Daily Health Overview</Text>
                <Text style={styles.homeOverviewStatus}>
                  {getDailyHealthV2Overview(savedSessions).overallStatus}
                </Text>
                <Text style={styles.homeOverviewText}>
                  {getDailyHealthV2Overview(savedSessions).nextCheck}
                </Text>
              </View>

              <View style={styles.savedSessionButtonRow}>
                <Pressable
                  style={styles.viewHistoryButton}
                  onPress={() => setShowHistory(true)}
                >
                  <Text style={styles.viewHistoryButtonText}>View History</Text>
                </Pressable>

                <Pressable
                  style={styles.dailyHealthInlineButton}
                  onPress={() => setShowDailyHealthOverview(true)}
                >
                  <Text style={styles.testingGuideButtonText}>Daily Overview</Text>
                </Pressable>

                <Pressable
                  style={styles.builderSmallButton}
                  onPress={() => setShowBuilderTools(true)}
                >
                  <Text style={styles.testingGuideButtonText}>Builder</Text>
                </Pressable>

                <Pressable style={styles.clearHistoryButton} onPress={clearSavedSessions}>
                  <Text style={styles.clearHistoryButtonText}>Clear Saved Checks</Text>
                </Pressable>
              </View>
            </View>
          ) : null}

          {savedSessions.length === 0 ? (
            <View style={styles.emptyPublicHomeCard}>
              <Text style={styles.emptyPublicHomeTitle}>Start Your First Movement Check</Text>

              <Text style={styles.emptyPublicHomeText}>
                Record one simple movement to begin building your movement profile. Start with Reach if you want the easiest first check.
              </Text>

              <View style={styles.emptyPublicHomeButtonRow}>
                <Pressable
                  style={styles.emptyPublicPrimaryButton}
                  onPress={() => {
                    setMode('daily');
                    setDailyTask('reach');
                    setShowCameraSetupGuide(true);
                    setStarted(false);
                  }}
                >
                  <Text style={styles.buttonText}>Start Reach Check</Text>
                </Pressable>

                <Pressable
                  style={styles.emptyPublicSecondaryButton}
                  onPress={() => setShowDailyHealthOverview(true)}
                >
                  <Text style={styles.secondaryButtonText}>View Overview</Text>
                </Pressable>
              </View>
            </View>
          ) : null}

          <View style={styles.activeModulePreviewCard}>
            <Text style={styles.homeOverviewLabel}>Active Analysis Module</Text>

            <Text style={styles.homeOverviewStatus}>
              {getActiveMovementModule(mode, dailyTask).title}
            </Text>

            <Text style={styles.homeOverviewText}>
              {getBackendLabel(getActiveMovementModule(mode, dailyTask).backend)} backend • {getActiveMovementModule(mode, dailyTask).purpose}
            </Text>
          </View>

          <View style={styles.modeSelectorRow}>
            {(['rep', 'rehab', 'daily', 'lab'] as const).map((m) => (
              <Pressable
                key={m}
                style={[
                  styles.modeChip,
                  mode === m && styles.modeChipActive,
                ]}
                onPress={() => setMode(m)}
              >
                <Text
                  style={[
                    styles.modeChipText,
                    mode === m && styles.modeChipTextActive,
                  ]}
                >
                  {m.toUpperCase()}
                </Text>
              </Pressable>
            ))}
          </View>

          <View style={styles.dailyTaskBlock}>
            <Text style={styles.dailyTaskTitle}>Which Arm Are You Testing?</Text>

            <View style={styles.dailyTaskRow}>
              {(['right', 'left'] as const).map((s) => (
                <Pressable
                  key={s}
                  style={[
                    styles.dailyTaskChip,
                    selectedSide === s && styles.dailyTaskChipActive,
                  ]}
                  onPress={() => setSelectedSide(s)}
                >
                  <Text
                    style={[
                      styles.dailyTaskChipText,
                      selectedSide === s && styles.dailyTaskChipTextActive,
                    ]}
                  >
                    {s === 'right' ? 'Right Arm' : 'Left Arm'}
                  </Text>
                </Pressable>
              ))}
            </View>

            <Text style={styles.dailyTaskDescription}>
              Kinetra tracks whichever arm you select here -- make sure that arm is the
              one clearly visible to the camera during your recording.
            </Text>

            <View style={{ marginTop: 12 }}>
              {calibratedThresholds ? (
                <>
                  <Text style={styles.dailyTaskDescription}>
                    Personalized rep range active: flexed below {calibratedThresholds.flex}°,
                    extended above {calibratedThresholds.extend}°.
                  </Text>
                  <Pressable
                    style={[styles.dailyTaskChip, { marginTop: 8, flex: 0, alignSelf: 'flex-start', paddingHorizontal: 14 }]}
                    onPress={() => {
                      persistCalibrationForSide(selectedSide, null);
                      setPendingCalibrationSuggestion(null);
                      setCalibrationMessage(null);
                    }}
                  >
                    <Text style={styles.dailyTaskChipText}>Reset to Default Range</Text>
                  </Pressable>
                </>
              ) : relevantPendingCalibrationSuggestion ? (
                <View style={styles.quickStartBox}>
                  <Text style={styles.quickStartTitle}>Calibration Recording Complete</Text>
                  <Text style={styles.quickStartText}>
                    We measured your range as {relevantPendingCalibrationSuggestion.observedMin}° to{' '}
                    {relevantPendingCalibrationSuggestion.observedMax}°. Suggested personalized
                    thresholds: flexed below {relevantPendingCalibrationSuggestion.flex}°, extended
                    above {relevantPendingCalibrationSuggestion.extend}°.
                  </Text>
                  <View style={[styles.dailyTaskRow, { marginTop: 10, marginBottom: 0 }]}>
                    <Pressable
                      style={styles.dailyTaskChip}
                      onPress={() => {
                        persistCalibrationForSide(relevantPendingCalibrationSuggestion.side, {
                          flex: relevantPendingCalibrationSuggestion.flex,
                          extend: relevantPendingCalibrationSuggestion.extend,
                        });
                        setPendingCalibrationSuggestion(null);
                      }}
                    >
                      <Text style={styles.dailyTaskChipText}>Use These</Text>
                    </Pressable>
                    <Pressable
                      style={styles.dailyTaskChip}
                      onPress={() => setPendingCalibrationSuggestion(null)}
                    >
                      <Text style={styles.dailyTaskChipText}>Keep Default</Text>
                    </Pressable>
                  </View>
                </View>
              ) : (
                <>
                  <Pressable
                    style={[styles.dailyTaskChip, { flex: 0, alignSelf: 'flex-start', paddingHorizontal: 14 }]}
                    onPress={() => {
                      setIsCalibrating(true);
                      setCalibrationMessage(
                        'Calibration mode is on. Record one more take, moving through your full comfortable range of motion.'
                      );
                    }}
                  >
                    <Text style={styles.dailyTaskChipText}>
                      {isCalibrating ? 'Calibrating: Record Your Next Take' : 'Calibrate My Range'}
                    </Text>
                  </Pressable>
                  {calibrationMessage ? (
                    <Text style={[styles.dailyTaskDescription, { marginTop: 8 }]}>
                      {calibrationMessage}
                    </Text>
                  ) : null}
                </>
              )}
            </View>
          </View>

          <View style={styles.dailyTaskBlock}>
            <Text style={styles.dailyTaskTitle}>Consistency Streak</Text>

            {(() => {
              const streaks = getConsistencyStreaks(savedSessions);
              const hasAnyCalibration =
                !!calibratedThresholdsBySide.left || !!calibratedThresholdsBySide.right;
              const badges = getEarnedBadges(savedSessions, hasAnyCalibration);
              const earnedBadges = badges.filter((b) => b.earned);

              return (
                <>
                  <Text style={styles.dailyTaskDescription}>
                    {streaks.currentStreak > 0
                      ? `${streaks.currentStreak} day${streaks.currentStreak === 1 ? '' : 's'} in a row${streaks.activeToday ? ' -- checked in today' : ' -- check in today to keep it going'}.`
                      : savedSessions.length > 0
                        ? "Your streak reset. Check in today to start a new one."
                        : 'Save your first check to start a streak.'}
                    {streaks.longestStreak > streaks.currentStreak
                      ? ` Longest streak so far: ${streaks.longestStreak} days.`
                      : ''}
                  </Text>

                  <View style={styles.setupChipWrap}>
                    {badges.map((badge) => (
                      <View
                        key={badge.id}
                        style={[
                          styles.setupChip,
                          !badge.earned && {
                            backgroundColor: 'rgba(100, 116, 139, 0.12)',
                            borderColor: 'rgba(100, 116, 139, 0.28)',
                          },
                        ]}
                      >
                        <Text
                          style={[
                            styles.setupChipText,
                            // #94a3b8 (not the dimmer #64748b) -- checked
                            // against WCAG AA contrast requirements, see
                            // ACCESSIBILITY_NOTES.md. Also reuses a color
                            // already used elsewhere in the app instead of
                            // introducing a new one.
                            !badge.earned && { color: '#94a3b8' },
                          ]}
                        >
                          {badge.earned ? '✓ ' : ''}{badge.label}
                        </Text>
                      </View>
                    ))}
                  </View>

                  <Text style={[styles.historyEmptyText, { marginTop: 8 }]}>
                    {earnedBadges.length}/{badges.length} badges earned -- all based on
                    showing up and using the app fully, never on getting a high score.
                  </Text>
                </>
              );
            })()}
          </View>

          <View style={styles.dailyTaskBlock}>
            <Text style={styles.dailyTaskTitle}>Team Screening</Text>

            <Text style={styles.dailyTaskDescription}>
              For a coach or PE teacher screening several athletes with one phone: turn this
              on, type the athlete&apos;s name before each recording, and every result gets
              tagged so you can review the whole roster afterward.
            </Text>

            <Pressable
              style={[
                styles.dailyTaskChip,
                { marginTop: 10, flex: 0, alignSelf: 'flex-start', paddingHorizontal: 14 },
                teamModeEnabled && styles.dailyTaskChipActive,
              ]}
              onPress={() => setTeamModeEnabled(!teamModeEnabled)}
              accessibilityRole="switch"
              accessibilityState={{ checked: teamModeEnabled }}
              accessibilityLabel="Team Screening"
              accessibilityHint="Tags each saved recording with an athlete name for later review"
            >
              <Text
                style={[
                  styles.dailyTaskChipText,
                  teamModeEnabled && styles.dailyTaskChipTextActive,
                ]}
              >
                {teamModeEnabled ? 'Team Screening: On' : 'Team Screening: Off'}
              </Text>
            </Pressable>

            {teamModeEnabled ? (
              <View style={{ marginTop: 12 }}>
                <Text style={styles.inputLabel}>Athlete Name</Text>
                <TextInput
                  style={styles.feedbackInput}
                  value={athleteNameInput}
                  onChangeText={setAthleteNameInput}
                  accessibilityLabel="Athlete name for this recording"
                  placeholder="Example: Jordan, #14, Alex R."
                  placeholderTextColor="#64748b"
                />
                {!athleteNameInput.trim() ? (
                  <Text style={[styles.dailyTaskDescription, { marginTop: 8 }]}>
                    Enter a name before recording, or this result will save without one.
                  </Text>
                ) : null}
              </View>
            ) : null}
          </View>

          {mode === 'daily' ? (
            <View style={styles.dailyTaskBlock}>
              <Text style={styles.dailyTaskTitle}>Choose Daily Movement Check</Text>

              <View style={styles.dailyTaskRow}>
                {(['reach', 'arm_raise', 'sit_to_stand', 'walking', 'balance', 'timed_up_and_go'] as const).map((task) => (
                  <Pressable
                    key={task}
                    style={[
                      styles.dailyTaskChip,
                      dailyTask === task && styles.dailyTaskChipActive,
                    ]}
                    onPress={() => setDailyTask(task)}
                  >
                    <Text
                      style={[
                        styles.dailyTaskChipText,
                        dailyTask === task && styles.dailyTaskChipTextActive,
                      ]}
                    >
                      {getDailyTaskLabel(task)}
                    </Text>
                  </Pressable>
                ))}
              </View>

              <Text style={styles.dailyTaskDescription}>
                {getDailyTaskDescription(dailyTask)}
              </Text>

              <View style={styles.dailyInstructionList}>
                {getDailyTaskInstructions(dailyTask).map((instruction, index) => (
                  <Text key={index} style={styles.dailyInstructionText}>
                    • {instruction}
                  </Text>
                ))}
              </View>

              <View style={styles.quickStartBox}>
                <Text style={styles.quickStartTitle}>Best setup for this check</Text>

                <Text style={styles.quickStartText}>
                  {dailyTask === 'reach'
                    ? 'Camera: upper body visible. Focus: shoulder, elbow, wrist, and hand.'
                    : dailyTask === 'arm_raise'
                      ? 'Camera: side or front angle. Focus: shoulder, elbow, wrist, and full arm path.'
                      : dailyTask === 'walking'
                        ? 'Camera: side view. Focus: hips, knees, ankles, feet, and several visible steps.'
                        : dailyTask === 'balance'
                          ? 'Camera: full body visible. Focus: shoulders, hips, torso, ankles, feet, and visible sway.'
                          : dailyTask === 'timed_up_and_go'
                            ? 'Camera: wide view. Focus: chair, full body, walking path, turn or return, and complete sequence.'
                            : 'Camera: side view. Focus: full body, hips, knees, ankles, and feet.'}
                </Text>
              </View>
            </View>
          ) : null}

          {mode === 'rehab' ? (
            <View style={styles.rehabHomeBlock}>
              <Text style={styles.dailyTaskTitle}>Rehab Consistency Check</Text>

              <Text style={styles.dailyTaskDescription}>
                Record a slow, controlled repeated movement. The app will review how consistent and repeatable the movement was.
              </Text>

              <View style={styles.dailyInstructionList}>
                <Text style={styles.dailyInstructionText}>
                  • Move slowly and stay in control.
                </Text>
                <Text style={styles.dailyInstructionText}>
                  • Repeat the same movement several times.
                </Text>
                <Text style={styles.dailyInstructionText}>
                  • Keep the moving joint visible the whole time.
                </Text>
                <Text style={styles.dailyInstructionText}>
                  • Use the same camera angle each time so comparisons are fair.
                </Text>
              </View>

              <View style={styles.quickStartBox}>
                <Text style={styles.quickStartTitle}>Best setup for Rehab mode</Text>
                <Text style={styles.quickStartText}>
                  Use a short, clear video of one repeated controlled movement. Avoid rushing or changing the movement halfway through.
                </Text>
              </View>
            </View>
          ) : null}

          <View style={styles.startFeatureGrid}>
            <View style={styles.startFeatureCard}>
              <Text style={styles.startFeatureTitle}>
                {mode === 'daily'
                  ? 'Movement Health Check'
                  : mode === 'rehab'
                    ? 'Consistency Check'
                    : 'Per-Rep Grading'}
              </Text>
              <Text style={styles.startFeatureText}>
                {mode === 'daily'
                  ? 'Stability, control, efficiency, and daily movement quality.'
                  : mode === 'rehab'
                    ? 'Repeatability, control stability, and consistency.'
                    : 'Smoothness, symmetry, and control.'}
              </Text>
            </View>

            <View style={styles.startFeatureCard}>
              <Text style={styles.startFeatureTitle}>
                {mode === 'daily'
                  ? 'Everyday Motion Insight'
                  : mode === 'rehab'
                    ? 'Progress Over Time'
                    : 'Explainable Metrics'}
              </Text>
              <Text style={styles.startFeatureText}>
                {mode === 'daily'
                  ? `Current check: ${getDailyTaskLabel(dailyTask)}. Designed for everyday movement analysis.`
                  : mode === 'rehab'
                    ? 'Compare controlled movement quality across sessions.'
                    : 'Not just counts. Real movement quality.'}
              </Text>
            </View>
          </View>

          <Pressable
            style={styles.mainButton}
            onPress={() => {
              setShowHistory(false);
              setShowTestingGuide(false);
              setShowFeedbackNotes(false);
              setShowRolloutDashboard(false);
              setShowGuidedTestWorkflow(false);
              setShowBetaLaunchKit(false);
              setShowTesterAnalytics(false);
              setShowDailyHealthOverview(false);
              setShowBuilderTools(false);
              setShowReportExport(false);
              setShowAiCoach(false);
              setShowWeeklyReport(false);
              setShowYoloFramework(false);
              setShowMobilityProfile(false);
              setShowTrendEngine(false);
              setShowGuidedOnboarding(false);
              setShowServerDiagnostics(false);
              setVideoUri(null);
              setRecording(false);
              setCameraReady(false);
              setAnalysisResult(null);
              setAnalysisError(null);
              setAnalysisStatusMessage('');
              setComparisonSession(null);
              setShowDetails(false);

              if (mode === 'daily' || mode === 'rehab') {
                setShowCameraSetupGuide(true);
                setStarted(false);
              } else {
                setShowCameraSetupGuide(false);
                setStarted(true);
              }
            }}
          >
            <Text style={styles.buttonText}>Start</Text>
          </Pressable>
        </ScrollView>
      ) : videoUri ? (
        <View style={styles.fullScreen}>
          <VideoView
            player={player}
            style={styles.camera}
            nativeControls
            contentFit="contain"
          />

          {showResultsPanel ? (

            <ScrollView
              style={styles.resultsPanel}
              contentContainerStyle={styles.resultsContent}
              showsVerticalScrollIndicator={false}
            >
              <View style={styles.panelHeader}>
                <View>
                  <Text style={styles.panelEyebrow}>
                    {mode === 'daily'
                      ? `${getModeLabel(mode)} • ${analysisResult?.daily_task_label || getDailyTaskLabel(dailyTask)}`
                      : getModeLabel(mode)}
                  </Text>
                  <Text style={styles.panelTitle}>Movement Analysis</Text>
                </View>
              </View>

              <View style={styles.panelActionRow}>
                <Pressable
                  style={styles.panelReplayButton}
                  onPress={replayVideo}
                >
                  <Text style={styles.buttonText}>Replay</Text>
                </Pressable>

                <Pressable
                  style={[styles.analyzeButton, isAnalyzing && styles.disabledButton]}
                  onPress={() => {
                    if (videoUri && !isAnalyzing) {
                      uploadVideo(videoUri);
                    }
                  }}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <View style={styles.analyzingRow}>
                      <ActivityIndicator size="small" color="#ffffff" />
                      <Text style={styles.buttonText}>Analyzing...</Text>
                    </View>
                  ) : (
                    <Text style={styles.buttonText}>Analyze Video</Text>
                  )}
                </Pressable>
              </View>

              {isAnalyzing ? (
                <View style={styles.analysisProgressCard}>
                  <ActivityIndicator size="small" color="#93c5fd" />

                  <View style={{ flex: 1 }}>
                    <Text style={styles.analysisProgressTitle}>Analyzing Movement</Text>
                    <Text style={styles.analysisProgressText}>
                      {analysisStatusMessage || 'Processing your video...'}
                    </Text>
                    <Text style={styles.analysisProgressHint}>
                      Daily checks work best with short, clear recordings.
                    </Text>
                  </View>
                </View>
              ) : null}

              {analysisError ? (
                <View style={styles.errorCard}>
                  <Text style={styles.errorTitle}>Analysis Could Not Finish</Text>
                  <Text style={styles.errorText}>{analysisError}</Text>

                  <View style={styles.errorActionRow}>
                    <Pressable
                      style={styles.errorRetryButton}
                      onPress={() => {
                        if (videoUri && !isAnalyzing) {
                          uploadVideo(videoUri);
                        }
                      }}
                    >
                      <Text style={styles.errorActionText}>Try Again</Text>
                    </Pressable>

                    <Pressable
                      style={styles.errorRetakeButton}
                      onPress={() => {
                        setVideoUri(null);
                        setAnalysisResult(null);
                        setAnalysisError(null);
                        setAnalysisStatusMessage('');
                        setComparisonSession(null);
                        setShowDetails(false);
                      }}
                    >
                      <Text style={styles.errorActionText}>Retake</Text>
                    </Pressable>
                  </View>
                </View>
              ) : null}

              {analysisResult ? (
                <>
                  <View style={styles.summaryHero}>
                    <View style={styles.summaryHeroTop}>
                      <View>
                        <Text style={styles.summaryEyebrow}>
                          {mode === 'daily'
                            ? analysisResult.interpretation.summary
                            : analysisResult.interpretation.summary}
                        </Text>
                        <Text style={styles.summaryMainTitle}>
                          {mode === 'daily' ? 'Daily Movement Check' : getOverallLabel(analysisResult)}
                        </Text>
                      </View>
                      <View style={styles.repsBubble}>
                        <Text style={styles.repsBubbleValue}>{analysisResult.reps}</Text>
                        <Text style={styles.repsBubbleLabel}>
                          {mode === 'daily'
                            ? analysisResult.daily_task === 'sit_to_stand'
                              ? 'Moves'
                              : 'Checks'
                            : 'Reps'}
                        </Text>
                      </View>
                    </View>

                    <View style={styles.summaryPillRow}>
                      <SummaryPill
                        label="Extension Speed"
                        value={
                          analysisResult.max_extension_speed !== null
                            ? analysisResult.max_extension_speed.toFixed(2)
                            : 'N/A'
                        }
                      />
                      <SummaryPill
                        label="Flexion Speed"
                        value={
                          analysisResult.max_flexion_speed !== null
                            ? analysisResult.max_flexion_speed.toFixed(2)
                            : 'N/A'
                        }
                      />
                    </View>

                    {mode === 'daily' ? (
                      <View style={styles.sectionBlock}>
                        <Text style={styles.sectionTitle}>Daily Check</Text>

                        <View style={styles.metricCard}>
                          <Text style={styles.metricTitle}>
                            {analysisResult.daily_task_label || getDailyTaskLabel(dailyTask)}
                          </Text>

                          <Text style={styles.metricValueLarge}>
                            {getDailyTaskResultDescription(analysisResult.daily_task || dailyTask)}
                          </Text>
                        </View>
                      </View>
                    ) : null}



                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Movement Health Score</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricTitle}>
                          {analysisResult.movement_health_score !== null && analysisResult.movement_health_score !== undefined
                            ? `${analysisResult.movement_health_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.movement_health_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.movement_health_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.movement_health_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.movement_health_grade || 'N/A'}
                          </Text>
                        </View>

                        <Text style={styles.metricValueLarge}>
                          {mode === 'daily'
                            ? 'This broader score gives a general movement-quality snapshot. Use the task-specific analysis below for the most relevant result.'
                            : 'This score summarizes movement smoothness, symmetry, control, efficiency, and performance trend.'}
                        </Text>
                      </View>
                    </View>
                  </View>

                  <View style={styles.aiCoachCurrentCard}>
                    <Text style={styles.sectionTitle}>AI Coach</Text>

                    <View
                      style={[
                        styles.gradeBadge,
                        {
                          backgroundColor: getGradeColors(
                            getCurrentResultCoachAdvice(
                              analysisResult,
                              comparisonSession,
                              savedSessions
                            ).status
                          ).bg,
                          borderColor: getGradeColors(
                            getCurrentResultCoachAdvice(
                              analysisResult,
                              comparisonSession,
                              savedSessions
                            ).status
                          ).border,
                        },
                      ]}
                    >
                      <Text
                        style={[
                          styles.gradeBadgeText,
                          {
                            color: getGradeColors(
                              getCurrentResultCoachAdvice(
                                analysisResult,
                                comparisonSession,
                                savedSessions
                              ).status
                            ).text,
                          },
                        ]}
                      >
                        {
                          getCurrentResultCoachAdvice(
                            analysisResult,
                            comparisonSession,
                            savedSessions
                          ).status
                        }
                      </Text>
                    </View>

                    <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                      Summary
                    </Text>
                    <Text style={styles.metricValueLarge}>
                      {
                        getCurrentResultCoachAdvice(
                          analysisResult,
                          comparisonSession,
                          savedSessions
                        ).summary
                      }
                    </Text>

                    <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                      Trust
                    </Text>
                    <Text style={styles.metricValueLarge}>
                      {
                        getCurrentResultCoachAdvice(
                          analysisResult,
                          comparisonSession,
                          savedSessions
                        ).trust
                      }
                    </Text>

                    <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                      Focus
                    </Text>
                    <Text style={styles.metricValueLarge}>
                      {
                        getCurrentResultCoachAdvice(
                          analysisResult,
                          comparisonSession,
                          savedSessions
                        ).focus
                      }
                    </Text>

                    <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                      Next Action
                    </Text>
                    <Text style={styles.metricValueLarge}>
                      {
                        getCurrentResultCoachAdvice(
                          analysisResult,
                          comparisonSession,
                          savedSessions
                        ).nextAction
                      }
                    </Text>
                  </View>

                  <View style={styles.sectionBlock}>
                    <Text style={styles.sectionTitle}>Signal Quality</Text>

                    <View style={styles.metricCard}>
                      <View
                        style={[
                          styles.gradeBadge,
                          {
                            backgroundColor: getGradeColors(analysisResult.signal_quality?.grade || 'N/A').bg,
                            borderColor: getGradeColors(analysisResult.signal_quality?.grade || 'N/A').border,
                          },
                        ]}
                      >
                        <Text
                          style={[
                            styles.gradeBadgeText,
                            { color: getGradeColors(analysisResult.signal_quality?.grade || 'N/A').text },
                          ]}
                        >
                          {analysisResult.signal_quality?.grade || 'N/A'}
                        </Text>
                      </View>

                      <Text style={styles.metricValueLarge}>
                        {analysisResult.signal_quality?.message || 'No signal quality data available.'}
                      </Text>

                      <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                        Valid Frames
                      </Text>

                      <Text style={styles.metricValueLarge}>
                        {analysisResult.signal_quality
                          ? `${analysisResult.signal_quality.valid_frames}/${analysisResult.signal_quality.total_frames}`
                          : 'N/A'}
                      </Text>
                    </View>
                  </View>

                  {mode === 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Rehab Session Summary</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Overall Rehab Summary</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).headline
                              ).bg,
                              borderColor: getGradeColors(
                                getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).headline
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).headline
                                ).text,
                              },
                            ]}
                          >
                            {getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).headline}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          What Happened
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).mainMessage}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Main Focus Area
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).focusArea}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Check
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabSessionSummary(analysisResult, comparisonSession, savedSessions).nextCheck}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Rehab Consistency</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Controlled Movement Score</Text>

                        <Text style={styles.metricTitle}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).score !== null
                            ? `${getRehabConsistencyAnalysis(analysisResult, comparisonSession).score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getRehabConsistencyAnalysis(analysisResult, comparisonSession).status
                              ).bg,
                              borderColor: getGradeColors(
                                getRehabConsistencyAnalysis(analysisResult, comparisonSession).status
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getRehabConsistencyAnalysis(analysisResult, comparisonSession).status
                                ).text,
                              },
                            ]}
                          >
                            {getRehabConsistencyAnalysis(analysisResult, comparisonSession).status}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Summary
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).summary}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Repeatability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).repeatability}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Control Stability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).controlStability}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Biggest Limitation
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).limitation}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Compared With Previous Rehab Check
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).previousComparison}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Recommended Next Check
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).nextStep}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'rehab' && comparisonSession ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Previous Rehab Check</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Previous Consistency Score</Text>
                        <Text style={styles.metricValueLarge}>
                          {comparisonSession.primary_score !== null
                            ? `${comparisonSession.primary_score}/100`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Current Consistency Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabConsistencyAnalysis(analysisResult, comparisonSession).score !== null
                            ? `${getRehabConsistencyAnalysis(analysisResult, comparisonSession).score}/100`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Change
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getScoreChangeText(
                            getRehabConsistencyAnalysis(analysisResult, comparisonSession).score,
                            comparisonSession.primary_score
                          )}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Previous Status
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {comparisonSession.primary_grade}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Rehab Baseline</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Baseline Status</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getRehabBaselineInterpretation(analysisResult, savedSessions).status
                              ).bg,
                              borderColor: getGradeColors(
                                getRehabBaselineInterpretation(analysisResult, savedSessions).status
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getRehabBaselineInterpretation(analysisResult, savedSessions).status
                                ).text,
                              },
                            ]}
                          >
                            {getRehabBaselineInterpretation(analysisResult, savedSessions).status}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Your Rehab Baseline
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).baselineScore !== null
                            ? `${getRehabBaselineInterpretation(analysisResult, savedSessions).baselineScore}/100`
                            : 'Not built yet'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Today vs Baseline
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).difference !== null
                            ? `${getRehabBaselineInterpretation(analysisResult, savedSessions).difference! > 0 ? '+' : ''}${getRehabBaselineInterpretation(analysisResult, savedSessions).difference} points`
                            : 'Not enough data yet'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          What It Means
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).summary}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Detail
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).detail}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Step
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).nextStep}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Rehab Checks Used
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabBaselineInterpretation(analysisResult, savedSessions).sessionCount} saved check{getRehabBaselineInterpretation(analysisResult, savedSessions).sessionCount === 1 ? '' : 's'}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Rehab Trend Interpretation</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Trend Status</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getRehabTrendInterpretation(analysisResult, comparisonSession).status
                              ).bg,
                              borderColor: getGradeColors(
                                getRehabTrendInterpretation(analysisResult, comparisonSession).status
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getRehabTrendInterpretation(analysisResult, comparisonSession).status
                                ).text,
                              },
                            ]}
                          >
                            {getRehabTrendInterpretation(analysisResult, comparisonSession).status}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          What Changed
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabTrendInterpretation(analysisResult, comparisonSession).summary}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Why It Matters
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabTrendInterpretation(analysisResult, comparisonSession).detail}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Step
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getRehabTrendInterpretation(analysisResult, comparisonSession).nextStep}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Confidence & Next Action</Text>

                      <View style={styles.metricCard}>
                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(getDailyConfidence(analysisResult).grade).bg,
                              borderColor: getGradeColors(getDailyConfidence(analysisResult).grade).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(getDailyConfidence(analysisResult).grade).text },
                            ]}
                          >
                            {getDailyConfidence(analysisResult).grade}
                          </Text>
                        </View>

                        <Text style={styles.metricValueLarge}>
                          {getDailyConfidence(analysisResult).message}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Action
                        </Text>

                        <Text style={styles.metricValueLarge}>
                          {getDailyConfidence(analysisResult).nextAction}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' && comparisonSession ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Last Check Data</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Current Metric</Text>
                        <Text style={styles.metricValueLarge}>
                          {getPrimaryTaskScore(analysisResult).label}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Previous Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {comparisonSession.primary_score !== null
                            ? `${comparisonSession.primary_score}/100`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Current Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getPrimaryTaskScore(analysisResult).score !== null
                            ? `${getPrimaryTaskScore(analysisResult).score}/100`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Change
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getScoreChangeText(
                            getPrimaryTaskScore(analysisResult).score,
                            comparisonSession.primary_score
                          )}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Previous Grade
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {comparisonSession.primary_grade}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Personal Baseline</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Baseline Status</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getBaselineInterpretation(analysisResult, savedSessions).status
                              ).bg,
                              borderColor: getGradeColors(
                                getBaselineInterpretation(analysisResult, savedSessions).status
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getBaselineInterpretation(analysisResult, savedSessions).status
                                ).text,
                              },
                            ]}
                          >
                            {getBaselineInterpretation(analysisResult, savedSessions).status}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Your Baseline
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).baselineScore !== null
                            ? `${getBaselineInterpretation(analysisResult, savedSessions).baselineScore}/100`
                            : 'Not built yet'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Today vs Baseline
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).difference !== null
                            ? `${getBaselineInterpretation(analysisResult, savedSessions).difference! > 0 ? '+' : ''}${getBaselineInterpretation(analysisResult, savedSessions).difference} points`
                            : 'Not enough data yet'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          What It Means
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).summary}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Detail
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).detail}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Step
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).nextStep}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Baseline Checks Used
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getBaselineInterpretation(analysisResult, savedSessions).sessionCount} saved check{getBaselineInterpretation(analysisResult, savedSessions).sessionCount === 1 ? '' : 's'}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Daily Assessment</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Overall Assessment</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getDailyInsightV2(analysisResult, comparisonSession).assessment
                              ).bg,
                              borderColor: getGradeColors(
                                getDailyInsightV2(analysisResult, comparisonSession).assessment
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getDailyInsightV2(analysisResult, comparisonSession).assessment
                                ).text,
                              },
                            ]}
                          >
                            {getDailyInsightV2(analysisResult, comparisonSession).assessment}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Biggest Strength
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getDailyInsightV2(analysisResult, comparisonSession).strength}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Biggest Limitation
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getDailyInsightV2(analysisResult, comparisonSession).limitation}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Recommended Next Check
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getDailyInsightV2(analysisResult, comparisonSession).recommendedNextCheck}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Trend Interpretation</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Trend Status</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(
                                getTrendInterpretation(analysisResult, comparisonSession).status
                              ).bg,
                              borderColor: getGradeColors(
                                getTrendInterpretation(analysisResult, comparisonSession).status
                              ).border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              {
                                color: getGradeColors(
                                  getTrendInterpretation(analysisResult, comparisonSession).status
                                ).text,
                              },
                            ]}
                          >
                            {getTrendInterpretation(analysisResult, comparisonSession).status}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          What Changed
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getTrendInterpretation(analysisResult, comparisonSession).summary}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Why It Matters
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getTrendInterpretation(analysisResult, comparisonSession).detail}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Next Step
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {getTrendInterpretation(analysisResult, comparisonSession).nextStep}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' ? (
                    <View style={styles.sectionBlock}>
                      <Pressable
                        style={styles.detailsToggleButton}
                        onPress={() => setShowDetails(!showDetails)}
                      >
                        <Text style={styles.detailsToggleButtonText}>
                          {showDetails ? 'Hide Detailed Metrics' : 'Show Detailed Metrics'}
                        </Text>
                      </Pressable>

                      <Text style={styles.detailsToggleHint}>
                        {showDetails
                          ? 'Detailed metrics are useful for deeper review, but the main result is already summarized above.'
                          : 'Keep this hidden for a simpler result. Open it only if you want the raw task metrics.'}
                      </Text>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'sit_to_stand' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Sit-to-Stand Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Lower Body Signal</Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.lower_body_signal_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.lower_body_signal_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.lower_body_signal_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.lower_body_signal_grade || 'N/A'}
                          </Text>
                        </View>

                        <Text style={styles.metricLabelSmall}>Summary</Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.summary || 'Sit-to-stand analysis generated.'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Transition Detected
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.transition_detected ? 'Yes' : 'No'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Key Findings
                        </Text>

                        <Text style={styles.metricValueLarge}>
                          Rise Stability: {analysisResult.task_analysis.rise_stability_grade || 'N/A'}
                        </Text>

                        <Text style={styles.metricValueLarge}>
                          Lower Body Signal: {analysisResult.task_analysis.lower_body_signal_grade || 'N/A'}
                        </Text>

                        <Text style={styles.metricValueLarge}>
                          Transition Duration:{' '}
                          {analysisResult.task_analysis.transition_duration_sec !== null &&
                            analysisResult.task_analysis.transition_duration_sec !== undefined
                            ? `${formatTaskMetric(
                              analysisResult.task_analysis.transition_duration_sec
                            )} sec`
                            : 'N/A'}
                        </Text>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Sit-to-Stand Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.note}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Rise Stability Score
                        </Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.rise_stability_score !== null &&
                            analysisResult.task_analysis.rise_stability_score !== undefined
                            ? `${analysisResult.task_analysis.rise_stability_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.rise_stability_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.rise_stability_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.rise_stability_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.rise_stability_grade || 'N/A'}
                          </Text>
                        </View>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Transition Duration
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.transition_duration_sec !== null &&
                            analysisResult.task_analysis.transition_duration_sec !== undefined
                            ? `${formatTaskMetric(analysisResult.task_analysis.transition_duration_sec)} sec`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Knee Angle Mean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.right_knee_angle_mean)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Knee Extension Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.knee_extension_range)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Angle Mean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.right_hip_angle_mean)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Extension Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.hip_extension_range)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Mean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.torso_lean_mean)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Max Torso Lean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.max_torso_lean)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Control
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.torso_control)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Lower Body Valid Frames
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.lower_body_valid_frames} frames
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Lower Body Signal Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.lower_body_signal_score)}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'reach' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Reach Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Reach Stability Score</Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.reach_stability_score !== null &&
                            analysisResult.task_analysis.reach_stability_score !== undefined
                            ? `${analysisResult.task_analysis.reach_stability_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.reach_stability_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.reach_stability_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.reach_stability_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.reach_stability_grade || 'N/A'}
                          </Text>
                        </View>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Reach Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Max Reach Distance
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.reach_distance_max)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Min Reach Distance
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.reach_distance_min)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Reach Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.reach_range)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Reach Smoothness
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.reach_smoothness)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Endpoint Steadiness
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.endpoint_steadiness)}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'arm_raise' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Arm Raise Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Arm Raise Stability Score</Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.arm_raise_stability_score !== null &&
                            analysisResult.task_analysis.arm_raise_stability_score !== undefined
                            ? `${analysisResult.task_analysis.arm_raise_stability_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.arm_raise_stability_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.arm_raise_stability_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.arm_raise_stability_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.arm_raise_stability_grade || 'N/A'}
                          </Text>
                        </View>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Arm Raise Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Arm Raise Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.arm_raise_range)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Arm Raise Smoothness
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.arm_raise_smoothness)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Top Steadiness
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.top_steadiness)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Mean Arm Distance
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.arm_distance_mean)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Mean Elbow Angle
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.elbow_angle_mean)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Arm Raise Signal Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.arm_raise_signal_score)}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'walking' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Walking Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Walking Stability Score</Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.walking_stability_score !== null &&
                            analysisResult.task_analysis.walking_stability_score !== undefined
                            ? `${analysisResult.task_analysis.walking_stability_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.walking_stability_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.walking_stability_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.walking_stability_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.walking_stability_grade || 'N/A'}
                          </Text>
                        </View>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Walking Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Quality
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.walking_signal_grade || 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.walking_signal_score !== null &&
                            analysisResult.task_analysis.walking_signal_score !== undefined
                            ? `${Math.round(analysisResult.task_analysis.walking_signal_score * 100)}%`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Estimated Step Cycles
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.estimated_step_cycles !== null &&
                            analysisResult.task_analysis.estimated_step_cycles !== undefined
                            ? `${analysisResult.task_analysis.estimated_step_cycles}`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Estimated Cadence
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.cadence_estimate !== null &&
                            analysisResult.task_analysis.cadence_estimate !== undefined
                            ? `${analysisResult.task_analysis.cadence_estimate.toFixed(1)} steps/min`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Path Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.hip_path_range, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Vertical Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.hip_vertical_variability, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Step Rhythm Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.step_rhythm_variability, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Left Knee Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.left_knee_range, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Right Knee Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.right_knee_range, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Knee Range Difference
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.knee_range_difference, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Note
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.note || 'Walking V1 analysis generated.'}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'balance' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Balance Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Balance Stability Score</Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.balance_stability_score !== null &&
                            analysisResult.task_analysis.balance_stability_score !== undefined
                            ? `${analysisResult.task_analysis.balance_stability_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.balance_stability_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.balance_stability_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.balance_stability_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.balance_stability_grade || 'N/A'}
                          </Text>
                        </View>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Balance Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Quality
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.balance_signal_grade || 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.balance_signal_score !== null &&
                            analysisResult.task_analysis.balance_signal_score !== undefined
                            ? `${Math.round(analysisResult.task_analysis.balance_signal_score * 100)}%`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Sway Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.hip_sway_range, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Shoulder Sway Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.shoulder_sway_range, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Vertical Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.hip_vertical_variability, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Shoulder Vertical Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.shoulder_vertical_variability, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Mean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.torso_lean_mean, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Max
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.torso_lean_max, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.torso_lean_variability, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Ankle Center Drift
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.ankle_center_drift, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Support Width Mean
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.support_width_mean, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Note
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.note || 'Balance V1 analysis generated.'}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {mode === 'daily' &&
                    showDetails &&
                    analysisResult.daily_task === 'timed_up_and_go' &&
                    analysisResult.task_analysis ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Timed Up and Go Analysis</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Functional Mobility Score</Text>

                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_mobility_score !== null &&
                            analysisResult.task_analysis.tug_mobility_score !== undefined
                            ? `${analysisResult.task_analysis.tug_mobility_score}/100`
                            : 'N/A'}
                        </Text>

                        <View
                          style={[
                            styles.gradeBadge,
                            {
                              backgroundColor: getGradeColors(analysisResult.task_analysis.tug_mobility_grade || 'N/A').bg,
                              borderColor: getGradeColors(analysisResult.task_analysis.tug_mobility_grade || 'N/A').border,
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.gradeBadgeText,
                              { color: getGradeColors(analysisResult.task_analysis.tug_mobility_grade || 'N/A').text },
                            ]}
                          >
                            {analysisResult.task_analysis.tug_mobility_grade || 'N/A'}
                          </Text>
                        </View>

                        {analysisResult.task_analysis.task_insights &&
                          analysisResult.task_analysis.task_insights.length > 0 ? (
                          <>
                            <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                              Timed Up and Go Insights
                            </Text>

                            {analysisResult.task_analysis.task_insights.map((insight, index) => (
                              <Text
                                key={index}
                                style={[
                                  styles.metricValueLarge,
                                  index > 0 ? { marginTop: 8 } : { marginTop: 0 },
                                ]}
                              >
                                • {insight}
                              </Text>
                            ))}
                          </>
                        ) : null}

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Quality
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_signal_grade || 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Signal Score
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_signal_score !== null &&
                            analysisResult.task_analysis.tug_signal_score !== undefined
                            ? `${Math.round(analysisResult.task_analysis.tug_signal_score * 100)}%`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Visible Sequence Duration
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_duration_sec !== null &&
                            analysisResult.task_analysis.tug_duration_sec !== undefined
                            ? `${analysisResult.task_analysis.tug_duration_sec.toFixed(1)} sec`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Body Path Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_path_range, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Return Pattern Detected
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_return_pattern_detected ? 'Yes' : 'No'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Direction Changes
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_direction_changes !== null &&
                            analysisResult.task_analysis.tug_direction_changes !== undefined
                            ? `${analysisResult.task_analysis.tug_direction_changes}`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Estimated Step Cycles
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_estimated_step_cycles !== null &&
                            analysisResult.task_analysis.tug_estimated_step_cycles !== undefined
                            ? `${analysisResult.task_analysis.tug_estimated_step_cycles}`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Estimated Cadence
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.tug_cadence_estimate !== null &&
                            analysisResult.task_analysis.tug_cadence_estimate !== undefined
                            ? `${analysisResult.task_analysis.tug_cadence_estimate.toFixed(1)} steps/min`
                            : 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Knee Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_knee_range, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Range
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_hip_range, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Max
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_torso_lean_max, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Torso Lean Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_torso_lean_variability, 2)}°
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Hip Vertical Variability
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {formatTaskMetric(analysisResult.task_analysis.tug_hip_vertical_variability, 4)}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Note
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.task_analysis.note || 'Timed Up and Go V1 analysis generated.'}
                        </Text>
                      </View>
                    </View>
                  ) : null}


                  {(mode !== 'daily' || showDetails) ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Movement Signature</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricTitle}>
                          {analysisResult.movement_signature?.label || 'No signature available'}
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.movement_signature?.description || 'Analyze more movement data to generate a movement profile.'}
                        </Text>
                      </View>
                    </View>
                  ) : null}

                  {(mode !== 'daily' || showDetails) ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Key Insights</Text>

                      <View style={styles.metricCard}>
                        {analysisResult.key_insights && analysisResult.key_insights.length > 0 ? (
                          analysisResult.key_insights.map((insight, index) => (
                            <Text
                              key={index}
                              style={[
                                styles.metricValueLarge,
                                index > 0 ? { marginTop: 10 } : null
                              ]}
                            >
                              • {insight}
                            </Text>
                          ))
                        ) : (
                          <Text style={styles.metricValueLarge}>No insights available</Text>
                        )}
                      </View>
                    </View>
                  ) : null}

                  {analysisResult.score_trend &&
                    analysisResult.score_trend.length >= 2 &&
                    mode !== 'daily' &&
                    mode !== 'rehab' ? (

                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Performance Trend</Text>

                      <View style={styles.metricCard}>
                        <Text style={styles.metricLabelSmall}>Summary</Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.performance_summary || 'N/A'}
                        </Text>

                        <Text style={[styles.metricLabelSmall, styles.metricLabelSpacing]}>
                          Score Trend
                        </Text>
                        <Text style={styles.metricValueLarge}>
                          {analysisResult.score_trend && analysisResult.score_trend.length > 0
                            ? analysisResult.score_trend.join(' → ')
                            : 'N/A'}
                        </Text>

                        {analysisResult.score_trend && analysisResult.score_trend.length > 1 ? (
                          <View style={{ marginTop: 10, alignItems: 'center' }}>
                            <ScoreGraph
                              scores={analysisResult.score_trend}
                              bestRep={analysisResult.best_rep?.rep}
                              worstRep={analysisResult.worst_rep?.rep}
                            />
                          </View>
                        ) : null}
                      </View>
                    </View>
                  ) : null}

                  {analysisResult.best_rep &&
                    analysisResult.worst_rep &&
                    analysisResult.best_rep.rep !== analysisResult.worst_rep.rep &&
                    mode !== 'daily' &&
                    mode !== 'rehab' ? (

                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Best vs Worst Rep</Text>

                      {analysisResult.best_rep && analysisResult.worst_rep ? (
                        <>
                          <View style={styles.metricCard}>
                            <Text style={styles.metricLabelSmall}>Best Rep</Text>
                            <Text style={styles.metricValueLarge}>
                              Rep {analysisResult.best_rep.rep} — {analysisResult.best_rep.score !== null ? `${analysisResult.best_rep.score}/100` : 'N/A'}
                            </Text>
                            <Text style={styles.metricValueLarge}>
                              {analysisResult.best_rep.score_grade}
                            </Text>
                          </View>

                          <View style={[styles.metricCard, { marginTop: 10 }]}>
                            <Text style={styles.metricLabelSmall}>Worst Rep</Text>
                            <Text style={styles.metricValueLarge}>
                              Rep {analysisResult.worst_rep.rep} — {analysisResult.worst_rep.score !== null ? `${analysisResult.worst_rep.score}/100` : 'N/A'}
                            </Text>
                            <Text style={styles.metricValueLarge}>
                              {analysisResult.worst_rep.score_grade}
                            </Text>
                          </View>
                        </>
                      ) : (
                        <View style={styles.metricCard}>
                          <Text style={styles.metricValueLarge}>
                            Not enough valid reps to compare
                          </Text>
                        </View>
                      )}
                    </View>
                  ) : null}

                  {mode !== 'daily' && mode !== 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Global Metrics</Text>

                      <View style={styles.metricsGrid}>
                        <MetricCard
                          title="Smoothness"
                          grade={analysisResult.global_metrics.smoothness_grade}
                          value={formatMetric(analysisResult.global_metrics.smoothness)}
                        />
                        <MetricCard
                          title="Symmetry"
                          grade={analysisResult.global_metrics.symmetry_grade}
                          value={formatMetric(analysisResult.global_metrics.symmetry)}
                        />
                        <MetricCard
                          title="Control"
                          grade={analysisResult.global_metrics.control_grade}
                          value={formatMetric(analysisResult.global_metrics.control)}
                        />

                        <MetricCard
                          title="Efficiency"
                          grade={analysisResult.global_metrics.efficiency_grade}
                          value={formatMetric(analysisResult.global_metrics.efficiency)}
                        />
                      </View>
                    </View>
                  ) : null}
                  {mode !== 'daily' && mode !== 'rehab' ? (
                    <View style={styles.sectionBlock}>
                      <Text style={styles.sectionTitle}>Per-Rep Analysis</Text>

                      {analysisResult.rep_analysis.length === 0 ? (
                        <View style={styles.emptyRepCard}>
                          <Text style={styles.emptyRepTitle}>No reps detected</Text>
                          <Text style={styles.emptyRepText}>
                            Try a clearer camera angle or larger range of motion.
                          </Text>
                        </View>
                      ) : (
                        analysisResult.rep_analysis.map((rep) => (
                          <View key={rep.rep} style={styles.repCard}>
                            <View style={styles.repCardHeader}>
                              <View>
                                <Text style={styles.repTitle}>
                                  Rep {rep.rep}
                                </Text>
                                <Text style={styles.repSubtitle}>
                                  Frames {rep.start} → {rep.end}
                                </Text>
                              </View>

                              <View style={styles.repIndexBubble}>
                                <Text style={styles.repIndexText}>{rep.rep}</Text>
                              </View>
                            </View>

                            <View style={{ marginBottom: 10 }}>
                              <Text style={{ color: '#94a3b8', fontSize: 12 }}>
                                Rep Score
                              </Text>
                              <Text style={{ color: '#ffffff', fontSize: 22, fontWeight: '800' }}>
                                {rep.score !== null ? `${rep.score}/100` : 'N/A'}
                              </Text>
                              <Text style={{ color: '#93c5fd', fontWeight: '700' }}>
                                {rep.score_grade}
                              </Text>
                            </View>

                            <View style={styles.repMetricRow}>
                              <Text style={styles.repMetricLabel}>Smoothness</Text>
                              <View
                                style={[
                                  styles.inlineGradeBadge,
                                  {
                                    backgroundColor: getGradeColors(rep.smoothness_grade).bg,
                                    borderColor: getGradeColors(rep.smoothness_grade).border,
                                  },
                                ]}
                              >
                                <Text
                                  style={[
                                    styles.inlineGradeText,
                                    { color: getGradeColors(rep.smoothness_grade).text },
                                  ]}
                                >
                                  {rep.smoothness_grade}
                                </Text>
                              </View>
                            </View>
                            <Text style={styles.repMetricValue}>
                              Value: {rep.smoothness !== null ? rep.smoothness.toFixed(3) : 'N/A'}
                            </Text>

                            <View style={styles.repMetricRow}>
                              <Text style={styles.repMetricLabel}>Symmetry</Text>
                              <View
                                style={[
                                  styles.inlineGradeBadge,
                                  {
                                    backgroundColor: getGradeColors(rep.symmetry_grade).bg,
                                    borderColor: getGradeColors(rep.symmetry_grade).border,
                                  },
                                ]}
                              >
                                <Text
                                  style={[
                                    styles.inlineGradeText,
                                    { color: getGradeColors(rep.symmetry_grade).text },
                                  ]}
                                >
                                  {rep.symmetry_grade}
                                </Text>
                              </View>
                            </View>
                            <Text style={styles.repMetricValue}>
                              Value: {rep.symmetry !== null ? rep.symmetry.toFixed(3) : 'N/A'}
                            </Text>

                            <View style={styles.repMetricRow}>
                              <Text style={styles.repMetricLabel}>Control</Text>
                              <View
                                style={[
                                  styles.inlineGradeBadge,
                                  {
                                    backgroundColor: getGradeColors(rep.control_grade).bg,
                                    borderColor: getGradeColors(rep.control_grade).border,
                                  },
                                ]}
                              >
                                <Text
                                  style={[
                                    styles.inlineGradeText,
                                    { color: getGradeColors(rep.control_grade).text },
                                  ]}
                                >
                                  {rep.control_grade}
                                </Text>
                              </View>
                            </View>
                            <Text style={styles.repMetricValue}>
                              Value: {rep.control !== null ? rep.control.toFixed(3) : 'N/A'}
                            </Text>
                          </View>
                        ))
                      )}
                    </View>
                  ) : null}
                </>
              ) : (
                <View style={styles.preAnalysisCard}>
                  <Text style={styles.preAnalysisTitle}>Ready to Analyze</Text>
                  <Text style={styles.preAnalysisText}>
                    {mode === 'daily'
                      ? `Analyze your ${getDailyTaskLabel(dailyTask).toLowerCase()} check. Short, clear videos with the needed body parts visible work best.`
                      : 'Analyze movement quality, stability, symmetry, control, efficiency, and performance trends using biomechanical AI analysis.'}
                  </Text>
                </View>
              )}
            </ScrollView>
          ) : (
            <Pressable
              style={styles.showPanelButton}
              onPress={() => setShowResultsPanel(true)}
            >
              <Text style={styles.buttonText}>Show Analysis</Text>
            </Pressable>
          )}

          <View style={styles.bottomControls}>
            <Pressable
              style={styles.secondaryButton}
              onPress={() => {
                setShowHistory(false);
                setShowTestingGuide(false);
                setStarted(true);
                setVideoUri(null);
                setRecording(false);
                setCameraReady(false);
                setAnalysisResult(null);
                setAnalysisError(null);
                setAnalysisStatusMessage('');
                setComparisonSession(null);
                setShowDetails(false);
              }}
            >
              <Text style={styles.buttonText}>Record Again</Text>
            </Pressable>

            <Pressable
              style={styles.dangerButton}
              onPress={() => {
                setShowHistory(false);
                setShowTestingGuide(false);
                setShowFeedbackNotes(false);
                setShowCameraSetupGuide(false);
                setShowRolloutDashboard(false);
                setShowGuidedTestWorkflow(false);
                setShowBetaLaunchKit(false);
                setShowTesterAnalytics(false);
                setShowDailyHealthOverview(false);
                setShowBuilderTools(false);
                setShowReportExport(false);
                setShowAiCoach(false);
                setShowWeeklyReport(false);
                setShowYoloFramework(false);
                setShowMobilityProfile(false);
                setShowTrendEngine(false);
                setShowGuidedOnboarding(false);
                setShowServerDiagnostics(false);
                setStarted(false);
                setVideoUri(null);
                setRecording(false);
                setCameraReady(false);
                setAnalysisResult(null);
                setAnalysisError(null);
                setAnalysisStatusMessage('');
                setComparisonSession(null);
                setShowDetails(false);
              }}
            >
              <Text style={styles.buttonText}>Home</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <View style={styles.fullScreen}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="back"
            mode="video"
            mute
            onCameraReady={() => setCameraReady(true)}
          />

          <View style={styles.topStatus}>
            <Text style={styles.statusText}>
              {cameraReady ? 'Camera ready' : 'Loading camera...'}
            </Text>
          </View>

          <View style={styles.captureHintCard}>
            <Text style={styles.captureHintEyebrow}>{getModeLabel(mode)}</Text>

            <Text style={styles.captureHintTitle}>
              {mode === 'daily'
                ? `Record ${getDailyTaskLabel(dailyTask)}`
                : 'Capture a clear movement sequence'}
            </Text>

            <Text style={styles.captureHintText}>
              {mode === 'daily'
                ? 'Before recording, check these four things:'
                : mode === 'rehab'
                  ? 'Before recording, check these four things:'
                  : 'Keep the full body region visible and use a stable camera angle for more accurate movement analysis.'}
            </Text>

            {mode === 'daily' ? (
              <View style={styles.recordingChecklistBox}>
                {getRecordingQualityChecklist(dailyTask).map((item, index) => (
                  <Text key={index} style={styles.recordingChecklistText}>
                    ✓ {item}
                  </Text>
                ))}
              </View>
            ) : null}

            {mode === 'rehab' ? (
              <View style={styles.recordingChecklistBox}>
                <Text style={styles.recordingChecklistText}>
                  ✓ Moving joint stays visible.
                </Text>
                <Text style={styles.recordingChecklistText}>
                  ✓ Movement is slow and controlled.
                </Text>
                <Text style={styles.recordingChecklistText}>
                  ✓ Same movement repeats several times.
                </Text>
                <Text style={styles.recordingChecklistText}>
                  ✓ Camera stays stable.
                </Text>
              </View>
            ) : null}

            {/* On-device pose indicator (Step 1). Hidden until the model is
                either loaded or has errored, so we don't flash UI before we
                know whether the brain is alive. */}
            {onDevicePose.error ? (
              <Text style={styles.recordingChecklistText}>
                On-device pose: unavailable ({onDevicePose.error})
              </Text>
            ) : onDevicePose.ready ? (
              <Text style={styles.recordingChecklistText}>
                On-device joints: {onDeviceJointCount ?? 0}/33
                {recording ? ' (updating every second)' : ''}
              </Text>
            ) : (
              <Text style={styles.recordingChecklistText}>
                On-device pose: warming up…
              </Text>
            )}
          </View>

          {mode === 'daily' || mode === 'rehab' ? (
            <Pressable
              style={styles.captureSetupHelpButton}
              onPress={() => {
                setStarted(false);
                setShowCameraSetupGuide(true);
              }}
            >
              <Text style={styles.captureSetupHelpText}>Review Camera Setup</Text>
            </Pressable>
          ) : null}

          <View style={styles.bottomControls}>
            {!recording ? (
              <Pressable
                style={[
                  styles.recordButton,
                  !cameraReady && styles.disabledButton,
                ]}
                onPress={startRecording}
                disabled={!cameraReady}
              >
                <Text style={styles.buttonText}>Record</Text>
              </Pressable>
            ) : (
              <Pressable style={styles.stopButton} onPress={stopRecording}>
                <Text style={styles.buttonText}>Stop</Text>
              </Pressable>
            )}

            <Pressable
              style={styles.secondaryButton}
              onPress={() => {
                setStarted(false);
                setRecording(false);
                setCameraReady(false);
                setVideoUri(null);
                setShowDetails(false);
                setShowResultsPanel(false);
                setAnalysisResult(null);
                setAnalysisError(null);
                setAnalysisStatusMessage('');
                setComparisonSession(null);
                setOnDeviceJointCount(null);
              }}
            >
              <Text style={styles.buttonText}>Back</Text>
            </Pressable>
          </View>
        </View>
      )}
    </View>
  );
}

function ScoreGraph({
  scores,
  bestRep,
  worstRep
}: {
  scores: number[];
  bestRep?: number;
  worstRep?: number;
}) {
  if (!scores || scores.length === 0) return null;

  const width = 300;
  const height = 120;
  const padding = 20;

  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);

  const range = maxScore - minScore || 1;

  const getX = (index: number) =>
    padding + (index / (scores.length - 1 || 1)) * (width - 2 * padding);

  const getY = (score: number) =>
    height - padding - ((score - minScore) / range) * (height - 2 * padding);

  return (
    <Svg width={width} height={height}>
      <Line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke="#555"
        strokeWidth="1"
      />
      <Line
        x1={padding}
        y1={padding}
        x2={padding}
        y2={height - padding}
        stroke="#555"
        strokeWidth="1"
      />
      {scores.map((score, i) => {
        if (i === 0) return null;

        return (
          <Line
            key={`line-${i}`}
            x1={getX(i - 1)}
            y1={getY(scores[i - 1])}
            x2={getX(i)}
            y2={getY(score)}
            stroke="#93c5fd"
            strokeWidth="2"
          />
        );
      })}

      {scores.map((score, i) => {
        let color = "#3b82f6";
        let radius = 3;

        // score-based color
        if (score >= 75) color = "#22c55e";
        else if (score >= 55) color = "#eab308";
        else color = "#ef4444";

        // best rep highlight
        if (bestRep === i + 1) {
          color = "#16a34a";
          radius = 5;
        }

        // worst rep highlight
        if (worstRep === i + 1) {
          color = "#dc2626";
          radius = 5;
        }

        return (
          <Circle
            key={`point-${i}`}
            cx={getX(i)}
            cy={getY(score)}
            r={radius}
            fill={color}
          />
        );
      })}
    </Svg>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
    // Pin absolutely-positioned children (the hidden pose WebView) inside
    // this view so they don't get laid out by the surrounding flow or
    // pushed against the screen edge.
    position: 'relative',
  },
  centeredContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: '#0f172a',
  },
  homeScroll: {
    flex: 1,
    backgroundColor: '#0f172a',
  },

  homeScrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    // No justifyContent here: when the home content is taller than the
    // screen, vertical centering would push the top card off-screen and
    // leave an empty gap at the bottom. Let it stack from the top.
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },
  fullScreen: {
    flex: 1,
    backgroundColor: '#000000',
  },
  camera: {
    flex: 1,
  },
  heroBadge: {
    backgroundColor: 'rgba(37, 99, 235, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.35)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 18,
  },
  heroBadgeText: {
    color: '#93c5fd',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  title: {
    fontSize: 32,
    fontWeight: '800',
    color: '#ffffff',
    marginBottom: 12,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 17,
    color: '#94a3b8',
    marginBottom: 28,
    textAlign: 'center',
    paddingHorizontal: 16,
    lineHeight: 25,
    maxWidth: 420,
  },
  savedSessionHomeBlock: {
    alignItems: 'center',
    marginBottom: 18,
  },

  sessionCountText: {
    color: '#93c5fd',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  sessionSubCountText: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },

  clearHistoryButton: {
    backgroundColor: 'rgba(51, 65, 85, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.18)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  clearHistoryButtonText: {
    color: '#cbd5e1',
    fontSize: 12,
    fontWeight: '700',
  },
  savedSessionButtonRow: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },

  viewHistoryButton: {
    backgroundColor: '#2563eb',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  viewHistoryButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },

  historyContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  historyTaskCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 20,
    padding: 16,
    marginBottom: 16,
  },

  historyTaskTitle: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 4,
  },

  historyTaskSubtitle: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 14,
  },

  historyLatestScore: {
    color: '#ffffff',
    fontSize: 30,
    fontWeight: '900',
    marginTop: 4,
    marginBottom: 8,
  },

  historyGraphBlock: {
    marginTop: 16,
    alignItems: 'center',
  },

  historyListBlock: {
    marginTop: 16,
  },

  historySessionRow: {
    marginTop: 10,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.12)',
    borderRadius: 14,
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  historySessionDate: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '700',
  },

  historySessionGrade: {
    color: '#94a3b8',
    fontSize: 12,
    marginTop: 3,
  },

  historySessionScore: {
    color: '#93c5fd',
    fontSize: 16,
    fontWeight: '800',
  },

  historyEmptyText: {
    color: '#94a3b8',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 8,
  },
  historyBaselineBox: {
    marginTop: 14,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 14,
    padding: 12,
  },

  historyBaselineText: {
    color: '#cbd5e1',
    fontSize: 14,
    fontWeight: '700',
    marginTop: 4,
  },
  movementOverviewCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 18,
  },

  movementOverviewRow: {
    marginTop: 10,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  movementOverviewLabel: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
  },

  movementOverviewDetail: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },

  movementOverviewRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  movementOverviewScore: {
    color: '#93c5fd',
    fontSize: 14,
    fontWeight: '900',
  },

  miniGradeBadge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingVertical: 4,
    paddingHorizontal: 8,
  },

  miniGradeBadgeText: {
    fontSize: 10,
    fontWeight: '800',
  },

  homeOverviewPreview: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },

  homeOverviewLabel: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 4,
    textAlign: 'center',
  },

  homeOverviewStatus: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    textAlign: 'center',
    marginBottom: 4,
  },

  homeOverviewText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
    textAlign: 'center',
  },
  testingGuideContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  testingReadinessCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  testingReadinessScore: {
    color: '#ffffff',
    fontSize: 36,
    fontWeight: '900',
    marginBottom: 8,
  },

  testingStatsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 14,
  },

  testingStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  testingStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  testingStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
  },

  testingNoteCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 14,
    marginBottom: 16,
  },

  testingNoteTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  testingNoteText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  testingTaskCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 14,
  },

  testingTaskNumber: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '900',
    marginBottom: 4,
  },

  testingTaskTitle: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '900',
    marginBottom: 6,
  },

  testingTaskGoal: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 10,
  },

  testingTaskInstruction: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
    marginTop: 4,
  },

  testingGuideButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  testingGuideButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '800',
  },

  testingGuideStandaloneButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },
  feedbackContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  feedbackSummaryCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  feedbackFormCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  inputLabel: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 12,
    marginBottom: 6,
  },

  feedbackInput: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.22)',
    borderRadius: 12,
    padding: 12,
    color: '#ffffff',
    fontSize: 14,
  },

  feedbackTextArea: {
    width: '100%',
    minHeight: 82,
    backgroundColor: 'rgba(15, 23, 42, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.22)',
    borderRadius: 12,
    padding: 12,
    color: '#ffffff',
    fontSize: 14,
    textAlignVertical: 'top',
  },

  feedbackNoteCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 14,
  },

  feedbackNoteTitle: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '900',
    marginBottom: 4,
  },

  feedbackNoteDate: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 8,
  },

  feedbackNoteText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },

  feedbackNotesButton: {
    backgroundColor: 'rgba(79, 70, 229, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.45)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  feedbackNotesStandaloneButton: {
    backgroundColor: 'rgba(79, 70, 229, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.45)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },
  cameraSetupContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  cameraSetupHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  cameraSetupCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  cameraMistakeCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  cameraMistakeText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 20,
    marginTop: 6,
  },

  setupChipWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },

  setupChip: {
    backgroundColor: 'rgba(37, 99, 235, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.32)',
    borderRadius: 999,
    paddingVertical: 6,
    paddingHorizontal: 10,
  },

  setupChipText: {
    color: '#bfdbfe',
    fontSize: 12,
    fontWeight: '800',
  },

  setupStepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginTop: 12,
  },

  setupStepNumber: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  setupStepNumberText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '900',
  },

  setupStepText: {
    flex: 1,
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
  },

  cameraSetupTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(37, 99, 235, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 8,
    marginBottom: 10,
  },

  captureSetupHelpButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 12,
  },

  captureSetupHelpText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '800',
  },

  testingGuideInlineButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginTop: 12,
  },
  rolloutDashboardContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  rolloutHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  rolloutSectionCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  rolloutWarningText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },

  rolloutStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  rolloutStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  rolloutStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  rolloutStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  rolloutActionGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  rolloutActionButton: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: 'center',
  },

  rolloutActionButtonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '900',
  },

  rolloutChecklistCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  rolloutChecklistText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 21,
    marginTop: 4,
  },

  rolloutTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.32)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 10,
  },

  rolloutSmallButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.35)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  homeRolloutPreview: {
    width: '100%',
    backgroundColor: 'rgba(20, 83, 45, 0.28)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.24)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },

  rolloutStandaloneButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },
  guidedWorkflowContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  guidedProgressCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  guidedProgressPercent: {
    color: '#ffffff',
    fontSize: 38,
    fontWeight: '900',
    marginBottom: 8,
  },

  guidedStepCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 14,
  },

  guidedStepHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    marginBottom: 8,
  },

  guidedStepNumber: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '900',
    marginBottom: 3,
  },

  guidedStepTitle: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '900',
  },

  guidedStepGoal: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 10,
  },

  guidedInstructionText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
    marginTop: 4,
  },

  guidedCheckButton: {
    backgroundColor: 'rgba(51, 65, 85, 0.95)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.25)',
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 999,
  },

  guidedCheckButtonDone: {
    backgroundColor: 'rgba(21, 128, 61, 0.32)',
    borderColor: 'rgba(34, 197, 94, 0.5)',
  },

  guidedCheckButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '900',
  },

  guidedLaunchButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginTop: 12,
  },

  guidedLaunchButtonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '900',
  },

  guidedReminderCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  guidedReminderText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  guidedWorkflowTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(79, 70, 229, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 10,
  },

  guidedWorkflowStandaloneButton: {
    backgroundColor: 'rgba(79, 70, 229, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.34)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },

  homeGuidedPreview: {
    width: '100%',
    backgroundColor: 'rgba(49, 46, 129, 0.34)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.24)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },
  betaLaunchContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  betaLaunchHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  betaLaunchScore: {
    color: '#ffffff',
    fontSize: 38,
    fontWeight: '900',
    marginBottom: 8,
  },

  betaLaunchWarningText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },

  betaLaunchStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  betaLaunchStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  betaLaunchStatValue: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    textAlign: 'center',
  },

  betaLaunchStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  betaLaunchCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  betaInviteText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 14,
  },

  shareBetaButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.45)',
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
  },

  betaTesterGroup: {
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    marginTop: 10,
  },

  betaTesterGroupTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 4,
  },

  betaTesterGroupText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },

  betaLaunchChecklistCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  betaChecklistText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 21,
    marginTop: 4,
  },

  betaLaunchTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 10,
  },

  betaLaunchStandaloneButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },

  homeBetaLaunchPreview: {
    width: '100%',
    backgroundColor: 'rgba(20, 83, 45, 0.28)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.24)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },
  testerAnalyticsContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  testerAnalyticsHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  testerAnalyticsStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  testerAnalyticsStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  testerAnalyticsStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  testerAnalyticsStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  testerAnalyticsCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  analyticsIssueLabel: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '900',
    marginTop: 10,
    marginBottom: 4,
  },

  analyticsIssueValue: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
  },

  analyticsEmptyText: {
    color: '#94a3b8',
    fontSize: 13,
    lineHeight: 19,
  },

  analyticsCategoryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 12,
    padding: 12,
    marginTop: 8,
  },

  analyticsCategoryName: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '800',
    flex: 1,
  },

  analyticsCategoryCount: {
    color: '#93c5fd',
    fontSize: 16,
    fontWeight: '900',
    marginLeft: 10,
  },

  testerAnalyticsTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(37, 99, 235, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 10,
  },

  testerAnalyticsStandaloneButton: {
    backgroundColor: 'rgba(37, 99, 235, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.34)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },

  homeTesterAnalyticsPreview: {
    width: '100%',
    backgroundColor: 'rgba(30, 64, 175, 0.24)',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.24)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },
  dailyHealthOverviewContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  dailyHealthHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  dailyHealthHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  dailyHealthStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  dailyHealthStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  dailyHealthStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  dailyHealthStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  dailyHealthCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  dailyHealthAreaTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    marginBottom: 6,
  },

  dailyHealthAreaScore: {
    color: '#93c5fd',
    fontSize: 24,
    fontWeight: '900',
    marginBottom: 8,
  },

  dailyHealthAreaCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  dailyHealthAreaName: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  dailyHealthAreaDetail: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 17,
  },

  dailyHealthAreaRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  dailyHealthAreaScoreSmall: {
    color: '#93c5fd',
    fontSize: 14,
    fontWeight: '900',
  },

  dailyHealthTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 10,
  },

  dailyHealthStandaloneButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 16,
  },

  dailyHealthInlineButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 16,
  },

  homeDailyHealthPreview: {
    width: '100%',
    backgroundColor: 'rgba(20, 83, 45, 0.28)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.24)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },
  publicHomeTopActions: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },

  builderToolsTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(51, 65, 85, 0.52)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.32)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  builderToolsContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  builderSummaryCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.22)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  builderStatusText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
    lineHeight: 21,
  },

  builderStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  builderStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  builderStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  builderStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  builderToolGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  builderToolButton: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(51, 65, 85, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.28)',
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 12,
    alignItems: 'center',
  },

  builderToolButtonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '900',
    textAlign: 'center',
  },

  builderWarningCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  builderWarningTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  builderWarningText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  builderSmallButton: {
    backgroundColor: 'rgba(51, 65, 85, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.28)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  emptyPublicHomeCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.22)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
    alignItems: 'center',
  },

  emptyPublicHomeTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    textAlign: 'center',
    marginBottom: 8,
  },

  emptyPublicHomeText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
    marginBottom: 14,
  },

  emptyPublicHomeButtonRow: {
    width: '100%',
    gap: 10,
  },

  emptyPublicPrimaryButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.45)',
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
  },

  emptyPublicSecondaryButton: {
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.28)',
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
  },
  reportTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  reportExportContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  reportHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  reportHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  reportStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  reportStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  reportStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  reportStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  reportAreaCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  reportAreaTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  reportAreaText: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 17,
  },

  reportAreaRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  reportAreaScore: {
    color: '#93c5fd',
    fontSize: 14,
    fontWeight: '900',
  },

  reportDisclaimerCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  reportDisclaimerTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  reportDisclaimerText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },
  passportHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  passportHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  passportScoreText: {
    color: '#7dd3fc',
    fontSize: 38,
    fontWeight: '900',
    marginTop: 4,
  },

  passportSnapshotGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  passportSnapshotCard: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.18)',
    borderRadius: 16,
    padding: 14,
  },

  passportSnapshotValue: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: '900',
    marginBottom: 4,
  },

  passportSnapshotLabel: {
    color: '#7dd3fc',
    fontSize: 13,
    fontWeight: '900',
    marginBottom: 4,
  },

  passportSnapshotDetail: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
  },

  passportHighlightGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  passportHighlightCard: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.18)',
    borderRadius: 16,
    padding: 14,
  },

  passportHighlightTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900',
    marginTop: 6,
    marginBottom: 6,
  },

  passportHighlightText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
  },

  passportNextCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.18)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  passportTrendGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  passportTrendBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '22%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  passportTrendValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  passportTrendLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  passportWatchCard: {
    width: '100%',
    backgroundColor: 'rgba(161, 98, 7, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.32)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  passportWatchTitle: {
    color: '#fde68a',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 8,
  },

  passportWatchText: {
    color: '#fef3c7',
    fontSize: 13,
    lineHeight: 20,
  },

  passportImprovingCard: {
    width: '100%',
    backgroundColor: 'rgba(21, 128, 61, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.32)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  passportImprovingTitle: {
    color: '#86efac',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 8,
  },

  passportImprovingText: {
    color: '#bbf7d0',
    fontSize: 13,
    lineHeight: 20,
  },

  passportAreaCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  passportAreaTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  passportAreaText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },

  passportAreaMeta: {
    color: '#7dd3fc',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 5,
  },

  passportAreaRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  passportAreaScore: {
    color: '#7dd3fc',
    fontSize: 14,
    fontWeight: '900',
  },

  passportConnectedGrid: {
    width: '100%',
    gap: 10,
    marginBottom: 16,
  },

  passportConnectedCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.16)',
    borderRadius: 16,
    padding: 14,
  },

  passportConnectedTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  passportConnectedText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
  },
  trendEngineTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(14, 165, 233, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  longitudinalContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  longitudinalHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  longitudinalHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  longitudinalScoreText: {
    color: '#7dd3fc',
    fontSize: 38,
    fontWeight: '900',
    marginTop: 4,
  },

  longitudinalStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  longitudinalStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '22%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.18)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  longitudinalStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  longitudinalStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  longitudinalWatchCard: {
    width: '100%',
    backgroundColor: 'rgba(161, 98, 7, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.32)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  longitudinalWatchTitle: {
    color: '#fde68a',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 8,
  },

  longitudinalWatchText: {
    color: '#fef3c7',
    fontSize: 13,
    lineHeight: 20,
  },

  longitudinalImprovingCard: {
    width: '100%',
    backgroundColor: 'rgba(21, 128, 61, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.32)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  longitudinalImprovingTitle: {
    color: '#86efac',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 8,
  },

  longitudinalImprovingText: {
    color: '#bbf7d0',
    fontSize: 13,
    lineHeight: 20,
  },

  longitudinalAreaCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  longitudinalAreaTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  longitudinalAreaText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },

  longitudinalAreaMeta: {
    color: '#7dd3fc',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 5,
  },

  longitudinalAreaRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  longitudinalAreaScore: {
    color: '#7dd3fc',
    fontSize: 14,
    fontWeight: '900',
  },

  longitudinalInfoCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  longitudinalInfoText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 8,
  },

  trendInlineButton: {
    backgroundColor: 'rgba(14, 165, 233, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 12,
  },
  aiCoachTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(79, 70, 229, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  aiCoachContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  aiCoachHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  aiCoachStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  aiCoachStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '30%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  aiCoachStatValue: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    textAlign: 'center',
  },

  aiCoachStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  aiCoachCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  aiCoachActionRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginTop: 12,
  },

  aiCoachActionNumber: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(79, 70, 229, 0.92)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  aiCoachActionNumberText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '900',
  },

  aiCoachActionText: {
    flex: 1,
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
  },

  aiCoachDisclaimerCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  aiCoachDisclaimerTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  aiCoachDisclaimerText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  aiCoachInlineButton: {
    backgroundColor: 'rgba(79, 70, 229, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 12,
  },

  aiCoachCurrentCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(165, 180, 252, 0.24)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },
  weeklyReportTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(14, 165, 233, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  weeklyReportContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  weeklyReportHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  weeklyReportHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  weeklyStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  weeklyStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  weeklyStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    textAlign: 'center',
  },

  weeklyStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  weeklyReportCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  weeklyAreaTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    marginBottom: 6,
  },

  weeklyAreaScore: {
    color: '#7dd3fc',
    fontSize: 24,
    fontWeight: '900',
    marginBottom: 8,
  },

  weeklyAreaCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  weeklyAreaName: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  weeklyAreaDetail: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 17,
  },

  weeklyAreaRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  weeklyAreaScoreSmall: {
    color: '#7dd3fc',
    fontSize: 14,
    fontWeight: '900',
  },

  weeklyDisclaimerCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  weeklyDisclaimerTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  weeklyDisclaimerText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  weeklyInlineButton: {
    backgroundColor: 'rgba(14, 165, 233, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 12,
  },
  mobilityProfileTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(168, 85, 247, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(216, 180, 254, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  mobilityProfileContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  mobilityHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(216, 180, 254, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  mobilityHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  mobilityScoreText: {
    color: '#d8b4fe',
    fontSize: 36,
    fontWeight: '900',
    marginTop: 4,
  },

  mobilityStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  mobilityStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '30%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  mobilityStatValue: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '900',
    textAlign: 'center',
  },

  mobilityStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  mobilityHighlightGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  mobilityHighlightCard: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '45%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(216, 180, 254, 0.18)',
    borderRadius: 16,
    padding: 14,
  },

  mobilityHighlightTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900',
    marginTop: 6,
    marginBottom: 6,
  },

  mobilityHighlightText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
  },

  mobilityNextCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(216, 180, 254, 0.18)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  mobilityDomainCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  mobilityDomainTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  mobilityDomainText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },

  mobilityDomainMeta: {
    color: '#a78bfa',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 6,
  },

  mobilityDomainRight: {
    alignItems: 'flex-end',
    gap: 6,
  },

  mobilityDomainScore: {
    color: '#d8b4fe',
    fontSize: 14,
    fontWeight: '900',
  },

  mobilityDisclaimerCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  mobilityDisclaimerTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 6,
  },

  mobilityDisclaimerText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  mobilityInlineButton: {
    backgroundColor: 'rgba(168, 85, 247, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(216, 180, 254, 0.35)',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
    marginBottom: 12,
  },
  startHereTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  startHereCard: {
    width: '100%',
    backgroundColor: 'rgba(21, 128, 61, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.26)',
    borderRadius: 20,
    padding: 16,
    marginBottom: 16,
  },

  startHereCardTitle: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '900',
    marginBottom: 6,
  },

  startHereCardText: {
    color: '#bbf7d0',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 12,
  },

  startHereCardButton: {
    backgroundColor: '#16a34a',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 999,
    alignItems: 'center',
  },

  guidedOnboardingContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  guidedHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  guidedHeadline: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 10,
  },

  guidedProgressText: {
    color: '#86efac',
    fontSize: 24,
    fontWeight: '900',
    marginBottom: 10,
  },

  guidedProgressTrack: {
    width: '100%',
    height: 10,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    borderRadius: 999,
    overflow: 'hidden',
    marginBottom: 12,
  },

  guidedProgressFill: {
    height: '100%',
    backgroundColor: '#22c55e',
    borderRadius: 999,
  },

  guidedInfoCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  guidedOnboardingStepTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginTop: 8,
    marginBottom: 4,
  },

  guidedStepText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },

  guidedTaskCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  guidedTaskTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900',
    marginBottom: 4,
  },

  guidedTaskText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
  },

  guidedTaskStatus: {
    color: '#86efac',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 6,
  },

  guidedSmallButton: {
    backgroundColor: 'rgba(34, 197, 94, 0.18)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.35)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 999,
  },

  guidedSmallButtonDone: {
    backgroundColor: 'rgba(30, 64, 175, 0.18)',
    borderColor: 'rgba(96, 165, 250, 0.35)',
  },

  guidedSmallButtonText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '900',
  },

  guidedNextCard: {
    width: '100%',
    backgroundColor: 'rgba(21, 128, 61, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(134, 239, 172, 0.24)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  guidedNextTitle: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: '900',
    marginBottom: 8,
  },

  guidedWhyCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },
  yoloFrameworkContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  yoloHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  yoloWarningText: {
    color: '#fde68a',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },

  yoloStatsGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },

  yoloStatBox: {
    flexGrow: 1,
    flexShrink: 1,
    minWidth: '30%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    alignItems: 'center',
  },

  yoloStatValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '900',
  },

  yoloStatLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 3,
    textAlign: 'center',
  },

  yoloCurrentModuleCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.22)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  yoloModuleCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },

  yoloModuleTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    marginBottom: 4,
  },

  yoloModuleMeta: {
    color: '#facc15',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 6,
  },

  yoloModuleText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 3,
  },

  yoloRoadmapCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.18)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  yoloRoadmapText: {
    color: '#cbd5e1',
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 22,
  },

  activeModulePreviewCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.18)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 12,
  },
  feedbackActionPlanCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.28)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  priorityFeedbackItem: {
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 14,
    padding: 12,
    marginTop: 10,
  },

  priorityFeedbackType: {
    color: '#fde68a',
    fontSize: 12,
    fontWeight: '900',
    marginBottom: 4,
  },

  priorityFeedbackText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },

  homeFeedbackPreview: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.84)',
    borderWidth: 1,
    borderColor: 'rgba(250, 204, 21, 0.22)',
    borderRadius: 16,
    padding: 12,
    marginBottom: 10,
  },
  rehabHomeBlock: {
    gap: 12,
    marginBottom: 12,
  },
  onboardingContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  onboardingCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.16)',
    borderRadius: 20,
    padding: 16,
    marginBottom: 12,
    flexDirection: 'row',
    gap: 12,
  },

  onboardingStepNumber: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#2563eb',
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '900',
    textAlign: 'center',
    lineHeight: 34,
  },

  onboardingStepTextBlock: {
    flex: 1,
  },

  onboardingStepTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 5,
  },

  onboardingStepText: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
  },

  onboardingNoteCard: {
    width: '100%',
    backgroundColor: 'rgba(127, 29, 29, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.28)',
    borderRadius: 18,
    padding: 14,
    marginTop: 4,
    marginBottom: 18,
  },

  onboardingNoteTitle: {
    color: '#fca5a5',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 6,
  },

  onboardingNoteText: {
    color: '#fecaca',
    fontSize: 13,
    lineHeight: 19,
  },

  howItWorksButton: {
    backgroundColor: 'rgba(51, 65, 85, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.18)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginBottom: 14,
  },

  howItWorksButtonText: {
    color: '#cbd5e1',
    fontSize: 12,
    fontWeight: '800',
  },

  unreadDot: {
    position: 'absolute',
    top: -3,
    right: -3,
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: '#38bdf8',
    borderWidth: 1.5,
    borderColor: '#0f172a',
  },

  quickStartBox: {
    marginTop: 12,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 14,
    padding: 12,
  },

  quickStartTitle: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 5,
  },

  quickStartText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },
  recordingChecklistBox: {
    marginTop: 10,
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    borderRadius: 14,
    padding: 12,
    gap: 6,
  },

  recordingChecklistText: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
  },

  detailsToggleButton: {
    backgroundColor: '#2563eb',
    borderWidth: 1,
    borderColor: 'rgba(147, 197, 253, 0.45)',
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 14,
    alignItems: 'center',
  },

  detailsToggleButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
  },

  detailsToggleHint: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 8,
    textAlign: 'center',
  },
  analysisProgressCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.94)',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.22)',
    borderRadius: 18,
    padding: 14,
    marginBottom: 14,
    flexDirection: 'row',
    gap: 12,
    alignItems: 'center',
  },

  analysisProgressTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
    marginBottom: 4,
  },

  analysisProgressText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 18,
  },

  analysisProgressHint: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 17,
    marginTop: 4,
  },

  errorActionRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 12,
  },

  errorRetryButton: {
    flex: 1,
    backgroundColor: '#2563eb',
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
  },

  errorRetakeButton: {
    flex: 1,
    backgroundColor: 'rgba(51, 65, 85, 0.95)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.22)',
    paddingVertical: 10,
    borderRadius: 12,
    alignItems: 'center',
  },

  errorActionText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '800',
  },
  modeSelectorRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 22,
  },
  dailyTaskBlock: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 18,
    padding: 14,
    marginBottom: 18,
  },

  dailyTaskTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
    marginBottom: 10,
  },

  dailyTaskRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 10,
  },

  dailyTaskChip: {
    flex: 1,
    backgroundColor: '#334155',
    paddingVertical: 9,
    paddingHorizontal: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    alignItems: 'center',
  },

  dailyTaskChipActive: {
    backgroundColor: '#2563eb',
    borderColor: 'rgba(147, 197, 253, 0.45)',
  },

  dailyTaskChipText: {
    color: '#cbd5e1',
    fontSize: 12,
    fontWeight: '700',
  },

  dailyTaskChipTextActive: {
    color: '#ffffff',
  },

  dailyTaskDescription: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 19,
  },
  dailyInstructionList: {
    marginTop: 10,
    gap: 6,
  },

  dailyInstructionText: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 18,
  },
  modeChip: {
    backgroundColor: '#334155',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
  },
  modeChipActive: {
    backgroundColor: '#2563eb',
    borderColor: 'rgba(147, 197, 253, 0.45)',
  },
  modeChipText: {
    color: '#cbd5e1',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  modeChipTextActive: {
    color: '#ffffff',
  },
  startFeatureGrid: {
    width: '100%',
    gap: 12,
    marginBottom: 24,
  },
  startFeatureCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.9)',
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
    borderRadius: 18,
    padding: 16,
  },
  startFeatureTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 6,
  },
  startFeatureText: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
  },
  mainButton: {
    backgroundColor: '#2563eb',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 14,
    shadowColor: '#2563eb',
    shadowOpacity: 0.28,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 8 },
    elevation: 6,
  },
  analyzeButton: {
    flex: 1,
    backgroundColor: '#2563eb',
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: 'center',
    shadowColor: '#2563eb',
    shadowOpacity: 0.25,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 5,
  },
  secondaryButton: {
    backgroundColor: '#334155',
    paddingVertical: 14,
    paddingHorizontal: 22,
    borderRadius: 14,
  },
  secondaryButtonText: {
    color: '#cbd5e1',
    fontSize: 14,
    fontWeight: '700',
  },
  dangerButton: {
    backgroundColor: '#b91c1c',
    paddingVertical: 14,
    paddingHorizontal: 22,
    borderRadius: 14,
  },
  recordButton: {
    backgroundColor: '#16a34a',
    paddingVertical: 16,
    paddingHorizontal: 28,
    borderRadius: 999,
    shadowColor: '#16a34a',
    shadowOpacity: 0.22,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 5,
  },
  stopButton: {
    backgroundColor: '#dc2626',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 999,
    shadowColor: '#dc2626',
    shadowOpacity: 0.22,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 5,
  },
  disabledButton: {
    opacity: 0.55,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  topStatus: {
    position: 'absolute',
    top: 58,
    alignSelf: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.82)',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
  },
  statusText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  captureHintCard: {
    position: 'absolute',
    top: 110,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(15, 23, 42, 0.86)',
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
  },
  captureHintEyebrow: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 6,
    letterSpacing: 0.5,
  },
  captureHintTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 6,
  },
  captureHintText: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
  },
  showPanelButton: {
    position: 'absolute',
    top: 88,
    alignSelf: 'center',
    backgroundColor: '#2563eb',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 999,
    shadowColor: '#2563eb',
    shadowOpacity: 0.25,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 5,
  },
  bottomControls: {
    position: 'absolute',
    bottom: 40,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  resultsPanel: {
    position: 'absolute',
    top: 88,
    left: 12,
    right: 12,
    maxHeight: 430,
    backgroundColor: 'rgba(15, 23, 42, 0.86)',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.14)',
  },
  resultsContent: {
    padding: 16,
    paddingBottom: 20,
  },
  panelHeader: {
    marginBottom: 10,
  },
  panelActionRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 14,
  },
  panelReplayButton: {
    flex: 1,
    backgroundColor: '#334155',
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: 'center',
  },
  panelEyebrow: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 4,
    letterSpacing: 0.5,
  },
  panelTitle: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '800',
  },
  analyzingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  summaryHero: {
    backgroundColor: 'rgba(30, 41, 59, 0.98)',
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.18)',
    marginBottom: 16,
  },
  summaryHeroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  summaryEyebrow: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 4,
    letterSpacing: 0.5,
  },
  summaryMainTitle: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: '800',
  },
  repsBubble: {
    backgroundColor: 'rgba(37, 99, 235, 0.18)',
    borderColor: 'rgba(96, 165, 250, 0.3)',
    borderWidth: 1,
    borderRadius: 18,
    paddingVertical: 10,
    paddingHorizontal: 14,
    alignItems: 'center',
    minWidth: 92,
  },
  repsBubbleValue: {
    color: '#ffffff',
    fontSize: 22,
    fontWeight: '800',
  },
  repsBubbleLabel: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 2,
  },
  summaryPillRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 16,
  },
  summaryPill: {
    flex: 1,
    backgroundColor: 'rgba(51, 65, 85, 0.92)',
    borderRadius: 16,
    padding: 12,
  },
  summaryPillLabel: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 6,
    textTransform: 'uppercase',
  },
  summaryPillValue: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
  },
  sectionBlock: {
    marginTop: 4,
    marginBottom: 14,
  },
  sectionTitle: {
    color: '#93c5fd',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 12,
  },
  metricsGrid: {
    gap: 10,
  },
  metricCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.12)',
  },
  metricTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 10,
  },
  metricLabelSmall: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  metricLabelSpacing: {
    marginTop: 12,
  },
  gradeBadge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingVertical: 7,
    paddingHorizontal: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  gradeBadgeText: {
    fontSize: 13,
    fontWeight: '700',
  },
  metricValue: {
    color: '#e2e8f0',
    fontSize: 20,
    fontWeight: '800',
  },
  metricValueLarge: {
    color: '#e2e8f0',
    fontSize: 16,
    fontWeight: '700',
    lineHeight: 24,
  },
  repCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    borderRadius: 18,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.12)',
  },
  repCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
    alignItems: 'center',
  },
  repTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 4,
  },
  repSubtitle: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
  },
  repIndexBubble: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: 'rgba(37, 99, 235, 0.18)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(96, 165, 250, 0.3)',
  },
  repIndexText: {
    color: '#bfdbfe',
    fontSize: 14,
    fontWeight: '800',
  },
  repMetricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
    gap: 8,
  },
  repMetricLabel: {
    color: '#e2e8f0',
    fontSize: 14,
    fontWeight: '700',
  },
  repMetricValue: {
    color: '#94a3b8',
    fontSize: 13,
    lineHeight: 18,
    marginTop: 6,
    marginBottom: 2,
  },
  inlineGradeBadge: {
    borderRadius: 999,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderWidth: 1,
    maxWidth: '64%',
  },
  inlineGradeText: {
    fontSize: 12,
    fontWeight: '700',
  },
  preAnalysisCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.95)',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.12)',
  },
  preAnalysisTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 8,
  },
  preAnalysisText: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 21,
  },
  emptyRepCard: {
    backgroundColor: 'rgba(51, 65, 85, 0.9)',
    borderRadius: 16,
    padding: 14,
  },
  emptyRepTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 6,
  },
  emptyRepText: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
  },
  errorCard: {
    backgroundColor: 'rgba(127, 29, 29, 0.95)',
    borderRadius: 18,
    padding: 14,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: 'rgba(248, 113, 113, 0.2)',
  },
  errorTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
    marginBottom: 8,
  },
  errorText: {
    color: '#fecaca',
    fontSize: 14,
    lineHeight: 20,
  },
  serverStatusTopButton: {
    alignSelf: 'center',
    backgroundColor: 'rgba(14, 165, 233, 0.16)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.34)',
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    marginTop: 0,
    marginBottom: 4,
  },

  serverStatusCard: {
    width: '100%',
    backgroundColor: 'rgba(14, 165, 233, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.24)',
    borderRadius: 20,
    padding: 16,
    marginBottom: 16,
  },

  serverStatusCardTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '900',
    marginBottom: 6,
  },

  serverStatusCardText: {
    color: '#7dd3fc',
    fontSize: 13,
    fontWeight: '900',
    marginBottom: 6,
  },

  serverStatusCardSubtext: {
    color: '#cbd5e1',
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 12,
  },

  serverStatusCardButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 11,
    paddingHorizontal: 14,
    borderRadius: 999,
    alignItems: 'center',
  },

  serverDiagnosticsContent: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
    paddingBottom: 48,
    backgroundColor: '#0f172a',
  },

  serverHeroCard: {
    width: '100%',
    backgroundColor: 'rgba(30, 41, 59, 0.96)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.24)',
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },

  serverUrlText: {
    color: '#7dd3fc',
    fontSize: 13,
    fontWeight: '800',
    lineHeight: 19,
  },

  serverChecklistCard: {
    width: '100%',
    backgroundColor: 'rgba(15, 23, 42, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.16)',
    borderRadius: 18,
    padding: 16,
    marginBottom: 16,
  },

  serverChecklistText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 20,
    marginTop: 4,
  },
});