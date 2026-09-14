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
    window.localStorage.setItem('movement_sessions_v1', JSON.stringify(sessions));
    window.localStorage.setItem('calibrated_thresholds_v1', JSON.stringify({ left: null, right: { flex: 90, extend: 120 } }));
  }, SESSIONS);
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
    expect(text).toContain('Team Roster');
    expect(text).toContain('Settings');
  });

  test('team roster shows athletes and flags the poor grade', async ({ page }) => {
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
    await clickText(page, 'Movement Passport');
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

  test('no unexpected console or page errors across the flows above', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await seedAndReload(page);
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
});
