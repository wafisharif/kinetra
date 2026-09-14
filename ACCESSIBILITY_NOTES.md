# Accessibility pass — 2026-09-13

## Color contrast (WCAG AA, 4.5:1 for normal text)

Audited the entire grade-badge palette used throughout the app (`getGradeColors`)
by computing each text color's contrast ratio against its badge background
composited onto the app's page background (`#0f172a`), and against the page
background directly. All five badge variants (green/blue/yellow/red/gray) pass
comfortably -- the tightest is the red variant at 8.97:1, well above the 4.5:1
AA minimum for normal text and above the 7:1 AAA threshold too.

Also checked the app's body/secondary text colors (`subtitle`, `dailyTaskDescription`,
`historyEmptyText`, `errorText`, `cameraMistakeText`, button text on the primary
blue button): all pass AA, several by a wide margin.

One real issue found and fixed: the new "unearned badge" dimmed state (Consistency
Streak card) originally used `#64748b`, which only reaches 3.33:1 against its
badge background -- below the 4.5:1 AA minimum for normal-size text (the badge
text is 12px, which is "normal" not "large" under WCAG's size thresholds).
Changed to `#94a3b8` (an existing color already used elsewhere in the app for
secondary text), which reaches 6.18:1. Also note: earned vs. unearned badges
are distinguished by both color AND a "✓ " text prefix, so the information
isn't conveyed by color alone even before this fix.

## Screen reader labels

Added `accessibilityRole` / `accessibilityLabel` / `accessibilityState` to the
interactive elements added this session: the Team Screening toggle and athlete
name input, the Daily Reminder toggle and time-of-day options, the "Ask Claude"
button (with a `busy` state while loading), and the new Settings nav chip.
Toggles use `accessibilityRole="switch"` with `checked` state; the reminder
time options use `accessibilityRole="radio"`.

## Scope and what's NOT covered

This file (`index.tsx`) is roughly 17,000 lines with hundreds of pre-existing
`Pressable`/`Text` elements built up over Phases 0-3. Retrofitting
accessibility labels across literally all of them in one pass would mean
touching a very large number of unrelated call sites in a single session --
real risk of an unrelated typo or mis-paste breaking something elsewhere,
directly against the "don't break anything" priority for this session. This
pass covers (a) auditing color contrast app-wide, since that's a single
reusable function and a small fixed palette, and (b) adding proper
accessibility labeling to everything built this session, so nothing new
ships less accessible than it should. A full accessibility labeling sweep of
the pre-existing screens is real, valuable, well-scoped follow-up work -- just
not something to rush through inside an already-large session.

## Dynamic font scaling

Not independently verified against the OS's actual accessibility font-scale
setting (that requires a real device/simulator with the setting changed, not
something a headless browser export can simulate). New text this session
follows the app's existing pattern of unconstrained `Text` components with no
`numberOfLines` truncation in the new cards, so it should scale the same way
the rest of the app already does -- neither better nor worse than existing
screens.
