import { defineConfig } from '@playwright/test';
const port = 5174;
export default defineConfig({
  testDir: './tests', fullyParallel: false, workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium', trace: 'retain-on-failure' },
  webServer: { command: `npm run dev -- --port ${port}`, url: `http://127.0.0.1:${port}`, reuseExistingServer: false },
});
