// Self-contained Playwright no-backend smoke test for web/wizard.html.
// Run: node web/tests/wizard_smoke.mjs
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = resolve(__dirname, '..');         // web/
const PORT = 8090;
const ORIGIN = `http://localhost:${PORT}`;

// ── stub responses keyed by path suffix ──
function stubFor(pathname) {
  if (pathname === '/api/packagings') return [];
  if (pathname.endsWith('/samples')) return { samples: [] };
  if (/\/api\/packagings\/[^/]+$/.test(pathname)) {
    if (pathname.endsWith('/cfgdraft')) {
      return { key: 'cfgdraft', display_name: 'Cfg', status: 'configured',
               pipeline: 'detector_ocr', image_count: 60, sub_regions: ['lot'],
               config: { lot_patterns: ['(?i)XX\\d+'], fields_extracted: ['lot','exp','product'],
                         sheet_checks: ['lot'], message_template_key: 'lot_exp',
                         product_aliases: [{ canonical: 'Houjicha', keywords: ['houjicha'] }] } };
    }
    return { key: 'demo', display_name: 'Demo', status: 'draft',
             pipeline: 'detector_ocr', image_count: 0, sub_regions: ['lot'] };
  }
  return {};
}

// ── checks registry — tasks push [name, async (page)=>{}] ──
export const checks = [];

checks.push(['showConfirm resolves true on OK', async (page) => {
  const ok = await page.evaluate(async () => {
    const p = showConfirm({ title: 't', body: 'b' });
    document.querySelector('#confirm-ok').click();
    return await p;
  });
  if (ok !== true) throw new Error(`expected true, got ${ok}`);
}]);

checks.push(['showConfirm resolves false on Cancel', async (page) => {
  const v = await page.evaluate(async () => {
    const p = showConfirm({ title: 't', body: 'b' });
    document.querySelector('#confirm-cancel').click();
    return await p;
  });
  if (v !== false) throw new Error(`expected false, got ${v}`);
}]);

checks.push(['no native confirm() left in source', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  const bare = (src.match(/[^w]confirm\(/g) || []);   // showConfirm( allowed
  if (bare.length) throw new Error(`found ${bare.length} native confirm( call(s)`);
}]);

checks.push(['saveAnnotation failure toasts and does not mark labeled', async (page) => {
  await page.evaluate(() => {
    curDraftKey = 'demo';
    annot.images = [{ name: 'a.jpg', labeled: false, bbox_count: 0 }];
    annot.curIdx = 0;
    annot.bboxes = [{ x1: 1, y1: 1, x2: 9, y2: 9, label: 'lot' }];
    annot.saveTimer = null;
  });
  await page.route('**/annotations/**', route => route.fulfill({ status: 500, body: '{"detail":"x"}' }));
  const r = await page.evaluate(async () => {
    saveAnnotation();
    await new Promise(res => setTimeout(res, 500));   // wait out 250ms debounce + fetch
    const toast = document.getElementById('toast');
    return { labeled: annot.images[0].labeled, toastShown: toast && toast.classList.contains('show') };
  });
  await page.unroute('**/annotations/**');
  if (r.labeled !== false) throw new Error('image marked labeled despite save failure');
  if (!r.toastShown) throw new Error('no toast on save failure');
}]);

checks.push(['no async-flow alert() left (validation alerts allowed)', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  const banned = [
    'alert(`Clone', 'alert(`Archive', 'alert(`Unarchive',
    'alert(`ลบไม่สำเร็จ', 'alert(`Prelabel', 'alert(`สร้าง packaging',
    'alert(`อัพโหลดล้มเหลว', 'alert(`บันทึก config', 'alert(`Start full training',
    'alert(`Deploy failed', 'alert(msg)',
  ];
  const hit = banned.filter(b => src.includes(b));
  if (hit.length) throw new Error(`async alert() still present: ${hit.join(', ')}`);
}]);

// ── runner ──
async function main() {
  const server = spawn('python', ['-m', 'http.server', String(PORT), '--directory', WEB_DIR],
    { stdio: 'ignore' });
  await new Promise(r => setTimeout(r, 1000));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.addInitScript((origin) => { window.API_BASE_OVERRIDE = origin; }, ORIGIN);
  await page.route('**/api/**', (route) => {
    const u = new URL(route.request().url());
    route.fulfill({ status: 200, contentType: 'application/json',
                    body: JSON.stringify(stubFor(u.pathname)) });
  });
  await page.goto(`${ORIGIN}/wizard.html`, { waitUntil: 'networkidle' });

  let failed = 0;
  for (const [name, fn] of checks) {
    try { await fn(page); console.log(`PASS  ${name}`); }
    catch (e) { failed++; console.log(`FAIL  ${name}\n      ${e.message}`); }
  }
  await browser.close();
  server.kill();
  console.log(`\n${checks.length - failed}/${checks.length} passed`);
  process.exit(failed ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
