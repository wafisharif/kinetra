// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Runs the Playwright regression suite against the static web export
 * (`npx expo export --platform web`, served from ./dist). CI builds `dist`
 * itself before this config's webServer starts it -- see
 * .github/workflows/ci.yml.
 */
module.exports = defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: 'http://localhost:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    permissions: ['camera', 'microphone'],
    launchOptions: {
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
      // In CI (a fresh checkout), `npx playwright install --with-deps
      // chromium` downloads the browser matching the pinned
      // @playwright/test version below, so no explicit executablePath is
      // needed there. PLAYWRIGHT_LOCAL_CHROMIUM lets a sandboxed dev
      // environment that already has a browser pre-baked at a fixed path
      // point at it instead of re-downloading.
      executablePath: process.env.PLAYWRIGHT_LOCAL_CHROMIUM || undefined,
    },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'npx serve -l 4173 -s dist',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 60000,
  },
});
