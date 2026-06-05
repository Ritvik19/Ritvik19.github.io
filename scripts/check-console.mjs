import { chromium } from 'playwright';

const url = process.argv[2] || 'http://127.0.0.1:8000/pyradox/';
const logs = { errors: [], warnings: [], other: [], failed: [] };

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on('requestfailed', (req) => {
  logs.failed.push(`${req.failure()?.errorText || 'failed'} ${req.url()}`);
});

page.on('console', (msg) => {
  const entry = `[${msg.type()}] ${msg.text()}`;
  if (msg.type() === 'error') logs.errors.push(entry);
  else if (msg.type() === 'warning') logs.warnings.push(entry);
  else logs.other.push(entry);
});
page.on('pageerror', (err) => logs.errors.push(`[pageerror] ${err.message}`));

const resp = await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
const title = await page.locator('#project-title').textContent().catch(() => '');
const sections = await page.locator('#project-contents section').count();

console.log('URL:', url);
console.log('HTTP:', resp?.status());
console.log('Title:', title?.trim() || '(empty)');
console.log('Sections:', sections);
console.log('\n=== Console errors ===');
if (logs.errors.length) logs.errors.forEach((e) => console.log(e));
else console.log('(none)');
console.log('\n=== Console warnings ===');
if (logs.warnings.length) logs.warnings.forEach((w) => console.log(w));
else console.log('(none)');
console.log('\n=== Failed requests ===');
if (logs.failed.length) logs.failed.forEach((f) => console.log(f));
else console.log('(none)');

await browser.close();
