// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * End-to-end regression suite against the exported production web build.
 * This is the same technique used throughout development to catch real
 * bugs (a horizontal text-overflow bug in the Mobility Profile and
 * Movement Passport screens was found and fixed this way) -- it runs
 * against actual rendered DOM in a real browser, not just unit-level
 * logic, so it catches whole classes of bugs (bad navigation, missing
 * data, broken conditional rendering) that type-checking cannot.
 */

function daysAgo(n, hour = 12) {
  const d = new Date();
  d.setHours(hour, 0, 0, 0);
  d.setDate(d.getDate() - n);
  return d.toISOString();
}

const SESSIONS = [
  { id: '1', timestamp: daysAgo(0), mode: 'daily', daily_task: 'reach', daily_task_label: 'Overhead Reach', primary_score: 88, primary_grade: 'Good', confidence_grade: 'High', side: 'right', thresholds_calibrated: true },
  { id: '2', timestamp: daysAgo(1), mode: 'daily', daily_task: 'arm_raise', daily_task_label: 'Arm Raise', primary_score: 75, primary_grade: 'Fair', confidence_grade: 'High', side: 'left', thresholds_calibrated: false, athlete_name: 'Jordan' },
  { id: '3', timestamp: daysAgo(2), mode: 'rehab', primary_score: 40, primary_grade: 'Poor - Recheck Needed', confidence_grade: 'Moderate', side: 'right', thresholds_calibrated: false, athlete_name: 'Alex R.' },
];

async function clickText(page, text, opts = {}) {
  const loc = page.getByText(text, { exact: opts.exact ?? false }).first();
  await loc.scrollIntoViewIfNeeded();
  await loc.click({ force: true, timeout: 5000 });
}

async function seedAndReload(page) {
  await page.goto('/');
  await page.evaluate((sessions) => {
    window.localStorage.setItem('has_seen_onboarding_v1', 'true');
    // Regression test infra fix: the Home screen gates its whole render
    // behind a one-time "What Brings You Here?" goal picker
    // (kinetra_user_goal_v1) that this suite never seeded, so every test
    // below was silently hitting the goal picker instead of Home -- not a
    // bug in the app, a gap in this suite's seed data. 'curious' is the
    // same default the app itself falls back to when no goal is set.
    window.localStorage.setItem('kinetra_user_goal_v1', 'curious');
    window.localStorage.setItem('movement_sessions_v1', JSON.stringify(sessions));
    window.localStorage.setItem('calibrated_thresholds_v1', JSON.stringify({ left: null, right: { flex: 90, extend: 120 } }));
  }, SESSIONS);
  await page.reload();
  await page.waitForTimeout(1200);
}

// Sessions + Daily Check-Ins crafted so the Insight Engine's rules are each
// deterministically satisfied exactly once, with no ambiguity about which
// check-in a session should be compared against:
//   - Streak: 3 consecutive check-in days (today, yesterday, 2 days ago).
//   - Soreness<->score correlation (Reach task): 4 Reach sessions spaced 2
//     days apart (so each is >36h from every other check-in but its own),
//     alternating a "sore" day (Shoulders/Neck reported) with a "clear" day.
//     Sore-day average (56.5) is far enough below clear-day average (89)
//     to clear the engine's 8-point threshold.
//   - Left/right asymmetry (Arm Raise task): 2 right-side sessions well
//     above 2 left-side sessions (30-point gap, past the 15-point threshold).
const INSIGHT_SCENARIO_SESSIONS = [
  { id: 'r1', timestamp: daysAgo(10, 12), mode: 'daily', daily_task: 'reach', daily_task_label: 'Reach', primary_score: 55, primary_grade: 'Fair', side: 'right' },
  { id: 'r2', timestamp: daysAgo(12, 12), mode: 'daily', daily_task: 'reach', daily_task_label: 'Reach', primary_score: 90, primary_grade: 'Good', side: 'right' },
  { id: 'r3', timestamp: daysAgo(14, 12), mode: 'daily', daily_task: 'reach', daily_task_label: 'Reach', primary_score: 58, primary_grade: 'Fair', side: 'right' },
  { id: 'r4', timestamp: daysAgo(16, 12), mode: 'daily', daily_task: 'reach', daily_task_label: 'Reach', primary_score: 88, primary_grade: 'Good', side: 'right' },
  { id: 'a1', timestamp: daysAgo(20, 12), mode: 'daily', daily_task: 'arm_raise', daily_task_label: 'Arm Raise', primary_score: 90, primary_grade: 'Good', side: 'right' },
  { id: 'a2', timestamp: daysAgo(21, 12), mode: 'daily', daily_task: 'arm_raise', daily_task_label: 'Arm Raise', primary_score: 85, primary_grade: 'Good', side: 'right' },
  { id: 'a3', timestamp: daysAgo(22, 12), mode: 'daily', daily_task: 'arm_raise', daily_task_label: 'Arm Raise', primary_score: 60, primary_grade: 'Fair', side: 'left' },
  { id: 'a4', timestamp: daysAgo(23, 12), mode: 'daily', daily_task: 'arm_raise', daily_task_label: 'Arm Raise', primary_score: 55, primary_grade: 'Fair', side: 'left' },
];

const INSIGHT_SCENARIO_CHECKINS = [
  { id: 'c0', timestamp: daysAgo(0, 12), feeling: 3, soreness: [] },
  { id: 'c1', timestamp: daysAgo(1, 12), feeling: 3, soreness: [] },
  { id: 'c2', timestamp: daysAgo(2, 12), feeling: 3, soreness: [] },
  { id: 'c10', timestamp: daysAgo(10, 12), feeling: 3, soreness: ['Shoulders'] },
  { id: 'c12', timestamp: daysAgo(12, 12), feeling: 4, soreness: [] },
  { id: 'c14', timestamp: daysAgo(14, 12), feeling: 2, soreness: ['Shoulders', 'Neck'] },
  { id: 'c16', timestamp: daysAgo(16, 12), feeling: 5, soreness: [] },
];

async function seedInsightScenarioAndReload(page) {
  await page.goto('/');
  await page.evaluate(({ sessions, checkIns }) => {
    window.localStorage.setItem('has_seen_onboarding_v1', 'true');
    window.localStorage.setItem('movement_sessions_v1', JSON.stringify(sessions));
    window.localStorage.setItem('daily_checkins_v1', JSON.stringify(checkIns));
  }, { sessions: INSIGHT_SCENARIO_SESSIONS, checkIns: INSIGHT_SCENARIO_CHECKINS });
  await page.reload();
  await page.waitForTimeout(1200);
}

test.describe('Kinetra regression suite', () => {
  test.beforeEach(async ({ page }) => {
    await seedAndReload(page);
  });

  test('home renders with saved sessions and nav chips', async ({ page }) => {
    const text = await page.locator('body').innerText();
    expect(text).toContain('Kinetra');
    expect(text).toContain('Settings');
    expect(text).toContain('Explore');
    // Regression check: confirms the feature-preview text for the current
    // default mode ('daily', since the "declutter navigation, broaden
    // appeal" redesign made Daily Movement Health the default experience
    // instead of Rep Quality Coach) renders on the home screen. The
    // mode-selector chips themselves (and their getModeDescription() text)
    // moved behind "Show Advanced Options" in that same redesign -- see the
    // "team roster" test above for the same collapsed-by-default pattern.
    expect(text).toContain('Stability, control, efficiency, and daily movement quality.');

    // "Your Summary" (along with How It Works, Camera Setup, Weekly
    // Report, Mobility Profile, and Privacy & Transparency) is collapsed
    // behind the "More" chip as of the tile-card Home redesign -- same
    // collapsed-by-default pattern as "Show Advanced Options" above.
    // Shouldn't be in the initial render, but should appear once expanded.
    expect(text).not.toContain('Your Summary');
    expect(text).toContain('More');

    await clickText(page, 'More');
    await page.waitForTimeout(300);
    const expandedText = await page.locator('body').innerText();
    expect(expandedText).toContain('Your Summary');
  });

  test('team roster shows athletes and flags the poor grade', async ({ page }) => {
    // Team Roster is a coach-only tool, collapsed by default: Show Advanced
    // Options -> enable Team Screening -> View Team Roster.
    await clickText(page, 'Show Advanced Options');
    await page.waitForTimeout(300);
    await clickText(page, 'Team Screening: Off');
    await page.waitForTimeout(300);
    await clickText(page, 'Team Roster');
    await page.waitForTimeout(600);
    const text = await page.locator('body').innerText();
    expect(text).toContain('Jordan');
    expect(text).toContain('Alex R.');
    expect(text).toContain('Flagged for follow-up');
  });

  test('settings shows per-arm calibration and reset clears it', async ({ page }) => {
    await clickText(page, 'Settings');
    await page.waitForTimeout(600);
    let text = await page.locator('body').innerText();
    expect(text).toContain('Right arm: personalized');
    expect(text).toContain('Left arm: using default range');

    await clickText(page, 'Reset Both to Default Range');
    await page.waitForTimeout(400);
    text = await page.locator('body').innerText();
    expect(text).toContain('Right arm: using default range');
    expect(text).toContain('Left arm: using default range');
  });

  test('transparency screen renders', async ({ page }) => {
    // "Privacy & Transparency" lives behind the "More" chip now.
    await clickText(page, 'More');
    await page.waitForTimeout(300);
    await clickText(page, 'Transparency');
    await page.waitForTimeout(600);
    const text = await page.locator('body').innerText();
    expect(text).toContain('How Kinetra Actually Measures Movement');
  });

  test('AI Coach screen has both the rule-based coach and Ask AI section', async ({ page }) => {
    await clickText(page, 'AI Coach');
    await page.waitForTimeout(600);
    const text = await page.locator('body').innerText();
    expect(text).toContain('Coach Summary');
    expect(text).toContain('Ask AI for a Personal Note');
  });

  test('Mobility Profile highlight cards wrap their text instead of clipping it', async ({ page }) => {
    // Regression test for the flexGrow-without-flexShrink bug found during
    // the layout QA pass: the "Strongest Domain" / "Main Watch Area" cards
    // used to clip their summary text off the right edge of the viewport
    // instead of wrapping it. Assert the text elements stay within bounds
    // at a narrow (375px) viewport, where the bug was originally caught.
    await page.setViewportSize({ width: 375, height: 800 });
    // "Mobility Profile" lives behind the "More" chip now.
    await clickText(page, 'More');
    await page.waitForTimeout(300);
    await clickText(page, 'Mobility Profile');
    await page.waitForTimeout(600);

    const loc = page.getByText('Strongest Domain', { exact: true }).first();
    await loc.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);

    const overflowing = await page.evaluate((viewportWidth) => {
      const all = Array.from(document.querySelectorAll('div, span, p'));
      return all
        .filter((el) => el.children.length === 0 && el.textContent && el.textContent.trim().length > 1)
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width > 1 && (r.right > viewportWidth + 3 || r.left < -3)).length;
    }, 375);
    expect(overflowing).toBe(0);
  });

  test('Movement Passport highlight cards wrap their text instead of clipping it', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    // "Your Summary" lives behind the "More" chip now.
    await clickText(page, 'More');
    await page.waitForTimeout(300);
    await clickText(page, 'Your Summary');
    await page.waitForTimeout(600);

    const loc = page.getByText('Strongest Area', { exact: true }).first();
    await loc.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);

    const overflowing = await page.evaluate((viewportWidth) => {
      const all = Array.from(document.querySelectorAll('div, span, p'));
      return all
        .filter((el) => el.children.length === 0 && el.textContent && el.textContent.trim().length > 1)
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width > 1 && (r.right > viewportWidth + 3 || r.left < -3)).length;
    }, 375);
    expect(overflowing).toBe(0);
  });

  test('home shows the Daily Check-In card and the check-in flow saves data', async ({ page }) => {
    // Regression + feature test for the low-effort, no-camera Daily
    // Check-In: every other tracking surface in the app requires recording
    // a movement first. This confirms the check-in entry point is visible
    // on first load, and that completing it round-trips through
    // AsyncStorage (localStorage on web) and updates the home screen.
    let text = await page.locator('body').innerText();
    expect(text).toContain('Quick Daily Check-In');
    expect(text).toContain('Check In Now');

    await clickText(page, 'Check In Now');
    await page.waitForTimeout(400);
    text = await page.locator('body').innerText();
    expect(text).toContain('How Are You Feeling Today?');

    await clickText(page, 'Good', { exact: true });
    await clickText(page, 'Knees', { exact: true });
    await clickText(page, 'Save Check-In');
    await page.waitForTimeout(400);

    text = await page.locator('body').innerText();
    expect(text).toContain("Saved today's check-in.");

    await clickText(page, 'Back Home');
    await page.waitForTimeout(400);
    text = await page.locator('body').innerText();
    expect(text).toContain("Today's Check-In Done");

    const stored = await page.evaluate(() => window.localStorage.getItem('daily_checkins_v1'));
    const parsed = JSON.parse(stored);
    expect(parsed.length).toBe(1);
    expect(parsed[0].feeling).toBe(4);
    expect(parsed[0].soreness).toContain('Knees');
  });

  test('Insights screen surfaces streak, correlation, and asymmetry patterns', async ({ page }) => {
    // Regression test for the Insight Engine -- the app's cross-referencing
    // feature that combines subjective Daily Check-Ins with objective
    // recorded scores. Uses a hand-crafted, deterministic scenario (see
    // INSIGHT_SCENARIO_* above) so this asserts on the engine's actual
    // math, not just that the screen renders something.
    await seedInsightScenarioAndReload(page);

    await clickText(page, 'Insights', { exact: true });
    await page.waitForTimeout(600);
    const text = await page.locator('body').innerText();

    expect(text).toContain('3-day check-in streak');
    expect(text).toContain('Reach scores are lower on Shoulders');
    expect(text).toContain('Arm Raise: right side outperforming left by 30 points');
  });

  test('no unexpected console or page errors across the flows above', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await seedAndReload(page);
    await clickText(page, 'Show Advanced Options');
    await page.waitForTimeout(300);
    await clickText(page, 'Team Screening: Off');
    await page.waitForTimeout(300);
    await clickText(page, 'Team Roster');
    await page.waitForTimeout(400);
    await clickText(page, 'Back Home');
    await page.waitForTimeout(300);
    await clickText(page, 'AI Coach');
    await page.waitForTimeout(400);

    // The jsQR worker script is fetched from a CDN that this sandbox's
    // network policy blocks -- a known environment artifact, not an app
    // bug (confirmed harmless: QR scanning simply isn't exercised in these
    // flows). Excluded here for the same reason it was excluded throughout
    // manual verification.
    const filtered = errors.filter((e) => !/jsqr/i.test(e));
    expect(filtered).toEqual([]);
  });

  // ----- Quick Tests: Finger Tapping, 4-Meter Walk, Dual-Task Walk --------
  // These three use page.clock (Playwright's fake-timer API) instead of
  // real waits, so a 10-second tap test or a multi-second walk timer
  // finishes instantly in the test run while still exercising the exact
  // same Date.now()-driven code path the real app runs. Every assertion
  // below is a value hand-computed from the published formulas (CAPSIT-PD
  // tap rate/rhythm/asymmetry, the 4-Meter Walk Test's distance/time, and
  // the CCNA Dual-Task Gait Cost formula), not just "some number appears" --
  // this is the suite's real check that the math is exactly right.

  test('Finger Tapping records taps, rhythm, decrement, and left/right asymmetry correctly', async ({ page }) => {
    await page.clock.install();
    await page.clock.pauseAt(new Date());
    await clickText(page, 'Finger Tapping');
    await page.waitForTimeout(300);

    let text = await page.locator('body').innerText();
    expect(text).toContain('Right Hand');
    expect(text).toContain('CAPSIT-PD');

    await clickText(page, 'Start 10-Second Test');
    await page.waitForTimeout(100);

    // 6 taps, exactly 1500ms apart (t = 0, 1500, 3000, 4500, 6000, 7500),
    // then let the clock run out past the 10s window.
    for (let i = 0; i < 6; i++) {
      await clickText(page, 'TAP', { exact: true });
      await page.clock.runFor(1500);
    }
    await page.clock.runFor(2000);
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('Right Hand Done');
    // tapRateHz = 6 taps / 10s = 0.6
    expect(text).toContain('0.6 taps/sec');
    // All 5 intervals are exactly 1500ms apart -> zero rhythm variability.
    expect(text).toContain('0% CV');
    // First half (0-5s): taps at 0,1500,3000,4500 = 4 taps -> 0.8/s.
    // Second half (5-10s): taps at 6000,7500 = 2 taps -> 0.4/s.
    // decrement = (0.8-0.4)/0.8 * 100 = 50% slower.
    expect(text).toContain('50% slower');

    // Left hand, zero taps -- the edge case that must not divide by zero
    // or render NaN anywhere (tap rate, rhythm CV, decrement, asymmetry).
    await clickText(page, 'Test Left Hand Too');
    await page.waitForTimeout(300);
    text = await page.locator('body').innerText();
    expect(text).toContain('Left Hand');

    await clickText(page, 'Start 10-Second Test');
    await page.clock.runFor(11000);
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).not.toContain('NaN');
    expect(text).toContain('0.0 taps/sec');

    await clickText(page, 'See Full Results');
    await page.waitForTimeout(300);
    text = await page.locator('body').innerText();
    expect(text).toContain('Results Saved');
    expect(text).toContain('Left/Right Asymmetry');
    // better=0.6, worse=0 -> 100 * (0.6-0)/(0.6+0) = 100%.
    expect(text).toContain('100%');
    expect(text).not.toContain('NaN');

    await clickText(page, 'Back Home');
  });

  test('4-Meter Walk computes gait speed, bands, and the vs-last-walk comparison correctly', async ({ page }) => {
    await page.clock.install();
    await page.clock.pauseAt(new Date());
    await clickText(page, '4-Meter Walk');
    await page.waitForTimeout(300);

    let text = await page.locator('body').innerText();
    expect(text).toContain('Gait Speed Check');

    // Walk 1: 2 seconds for 4 meters -> 2.00 m/s (typical).
    await clickText(page, 'Start', { exact: true });
    await page.clock.runFor(2000);
    await clickText(page, 'Stop', { exact: true });
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('2.0s');
    expect(text).toContain('2.00 m/s');
    expect(text).toContain('Typical range');

    // Walk 2: 5 seconds for 4 meters -> 0.80 m/s, exactly on the
    // borderline cut-point (>=0.8 is borderline, not slow).
    await clickText(page, 'Walk Again');
    await page.waitForTimeout(200);
    await clickText(page, 'Start', { exact: true });
    await page.clock.runFor(5000);
    await clickText(page, 'Stop', { exact: true });
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('5.0s');
    expect(text).toContain('0.80 m/s');
    expect(text).toContain('Borderline');
    // vs. the first walk: (0.80-2.00)/2.00 * 100 = -60% -> "60% slower".
    expect(text).toContain('60% slower than your last walk');

    await clickText(page, 'Back Home');
  });

  test('Dual-Task Walk computes plain/dual gait speed and Dual-Task Cost correctly', async ({ page }) => {
    await page.clock.install();
    await page.clock.pauseAt(new Date());
    await clickText(page, 'Dual-Task Walk');
    await page.waitForTimeout(300);

    let text = await page.locator('body').innerText();
    expect(text).toContain('Dual-Task Walk');

    await clickText(page, 'Naming Animals');
    await clickText(page, 'Begin');
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('Walk 1 of 2: Plain Walk');

    // Plain walk: 4 seconds for 4 meters -> 1.00 m/s.
    await clickText(page, 'Start', { exact: true });
    await page.clock.runFor(4000);
    await clickText(page, 'Stop', { exact: true });
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('4.0s');
    expect(text).toContain('1.00 m/s');

    await clickText(page, 'Continue');
    await page.waitForTimeout(200);
    text = await page.locator('body').innerText();
    expect(text).toContain('Walk 2 of 2');
    expect(text).toContain('Say a different animal');

    // Dual-task walk: 8 seconds for 4 meters -> 0.50 m/s.
    await clickText(page, 'Start', { exact: true });
    await page.clock.runFor(8000);
    await clickText(page, 'Stop', { exact: true });
    await page.waitForTimeout(200);

    text = await page.locator('body').innerText();
    expect(text).toContain('1.00 m/s');
    expect(text).toContain('0.50 m/s');
    // DTGC = (1.00-0.50)/1.00 * 100 = 50%.
    expect(text).toContain('50%');
    // band for 0.50 m/s is "slow" (<0.8).
    expect(text).toContain('Below 0.8 m/s while dual-tasking');
    expect(text).toContain('Your walk slowed by 50% while counting');
    expect(text).not.toContain('NaN');

    await clickText(page, 'Back Home');
  });
});
