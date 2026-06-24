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

checks.push(['dead views + promoProd removed', async (page) => {
  const r = await page.evaluate(() => ({
    staging: !!document.getElementById('view-staging'),
    success: !!document.getElementById('view-success'),
    promoProd: typeof promoProd,
  }));
  if (r.staging) throw new Error('#view-staging still present');
  if (r.success) throw new Error('#view-success still present');
  if (r.promoProd !== 'undefined') throw new Error('promoProd still defined/referenced');
}]);

checks.push(['no demo hardcode in step1/step4 inputs', async (page) => {
  const r = await page.evaluate(() => ({
    name: document.getElementById('inp-display-name').value,
    key: document.getElementById('inp-key').value,
    desc: document.getElementById('inp-desc').value,
    lotRows: document.querySelectorAll('#lot-rows .lot-row').length,
    firstLot: document.querySelector('#lot-rows input')?.value || '',
  }));
  if (r.name || r.key || r.desc) throw new Error('step1 inputs still prefilled');
  if (r.lotRows !== 1 || r.firstLot) throw new Error('step4 lot examples still hardcoded');
}]);

checks.push(['prefillStep4FromDraft restores saved config', async (page) => {
  const r = await page.evaluate(async () => {
    curDraftKey = 'cfgdraft';
    cropMode = 'single';
    await prefillStep4FromDraft();
    const on = (sel) => !!document.querySelector(sel);
    return {
      pattern: document.getElementById('rx-display').textContent,
      lotRows: document.querySelectorAll('#lot-rows .lot-row').length,
      product: on('#sp4 [data-group="fields"] .cbitem.on[data-field="product"]'),
      tpl: document.querySelector('#sp4 .tpl-opt.on')?.dataset.template,
      aliasRows: document.querySelectorAll('#pa-rows .pa-row').length,
    };
  });
  if (r.pattern !== '(?i)XX\\d+') throw new Error(`pattern not restored: ${r.pattern}`);
  if (r.lotRows !== 1) throw new Error('example rows not reset to 1');
  if (!r.product) throw new Error('product field not toggled on');
  if (r.tpl !== 'lot_exp') throw new Error(`template not selected: ${r.tpl}`);
  if (r.aliasRows !== 1) throw new Error(`alias rows: ${r.aliasRows}`);
}]);

checks.push(['drawer step-map covers trained/training_full', async (page) => {
  const r = await page.evaluate(() => {
    const mk = (status) => renderDrawerBody(
      { key: 'd', display_name: 'D', status, pipeline: 'detector_ocr', image_count: 60,
        conf_threshold: null, accuracy: null },
      { samples: [] });
    return { trained: mk('trained'), training: mk('training_full') };
  });
  if (!r.trained.includes('Step 5 / 5')) throw new Error('trained not step 5');
  if (!r.training.includes('Step 5 / 5')) throw new Error('training_full not step 5');
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
