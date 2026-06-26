// Self-contained Playwright no-backend smoke test for web/wizard.html.
// Run: node web/tests/wizard_smoke.mjs
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, extname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = resolve(__dirname, '..');         // web/
const PORT = 8090;
const ORIGIN = `http://localhost:${PORT}`;
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};

async function startServer() {
  const server = createServer(async (req, res) => {
    try {
      const u = new URL(req.url || '/', ORIGIN);
      const rel = u.pathname === '/' ? '/wizard.html' : decodeURIComponent(u.pathname);
      const file = resolve(WEB_DIR, '.' + rel);
      if (!file.startsWith(WEB_DIR)) {
        res.writeHead(403).end('Forbidden');
        return;
      }
      const data = await readFile(file);
      res.writeHead(200, { 'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream' });
      res.end(data);
    } catch (_) {
      res.writeHead(404).end('Not found');
    }
  });
  await new Promise((resolveListen) => server.listen(PORT, 'localhost', resolveListen));
  return server;
}

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

// Open the wizard in a fresh isolated context (own localStorage).
// seed = a session object to pre-write under 'wizardAuth', or null.
async function openWizard(page, { seed = null, waitUntil = 'domcontentloaded' } = {}) {
  const ctx = await page.context().browser().newContext();
  const p = await ctx.newPage();
  await p.addInitScript((o) => { window.API_BASE_OVERRIDE = o; }, ORIGIN);
  if (seed) await p.addInitScript((s) => {
    if (sessionStorage.getItem('__wizardSeeded')) return;
    localStorage.setItem('wizardAuth', s);
    sessionStorage.setItem('__wizardSeeded', '1');
  }, JSON.stringify(seed));
  await p.route('https://accounts.google.com/**', (r) => r.abort());
  await p.route('**/api/**', (route) => {
    const u = new URL(route.request().url());
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(stubFor(u.pathname)) });
  });
  await p.goto(`${ORIGIN}/wizard.html`, { waitUntil });
  return { ctx, p };
}

// ── checks registry — tasks push [name, async (page)=>{}] ──
export const checks = [];

checks.push(['login gate blocks without a session', async (page) => {
  const { ctx, p } = await openWizard(page, { seed: null });
  const visible = await p.evaluate(() => {
    const g = document.getElementById('login-gate');
    return !!g && !g.classList.contains('hidden');
  });
  await ctx.close();
  if (!visible) throw new Error('gate not visible without a session');
}]);

checks.push(['login gate reveals dashboard with a valid session', async (page) => {
  const seed = { email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000 };
  const { ctx, p } = await openWizard(page, { seed });
  let hidden = false;
  try {
    await p.waitForFunction(() => {
      const g = document.getElementById('login-gate');
      return g && g.classList.contains('hidden');
    }, { timeout: 5000 });
    hidden = true;
  } catch (_) {}
  await ctx.close();
  if (!hidden) throw new Error('gate still visible with a valid session');
}]);

checks.push(['auth claim validation accepts @tdfb.co, rejects others/expired', async (page) => {
  const res = await page.evaluate(() => {
    const mk = (o) => 'x.' + btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_') + '.y';
    const future = Math.floor(Date.now() / 1000) + 3600;
    const good = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: true, exp: future })));
    const badDomain = isAllowedClaims(decodeJwtPayload(mk({ hd: 'gmail.com', email: 'a@gmail.com', email_verified: true, exp: future })));
    const unverified = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: false, exp: future })));
    const expired = isAllowedClaims(decodeJwtPayload(mk({ hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: true, exp: 1 })));
    return { good, badDomain, unverified, expired };
  });
  if (res.good !== true) throw new Error('valid @tdfb.co claims rejected');
  if (res.badDomain !== false) throw new Error('non-tdfb.co domain accepted');
  if (res.unverified !== false) throw new Error('unverified email accepted');
  if (res.expired !== false) throw new Error('expired token accepted');
}]);

checks.push(['decodeJwtPayload handles unpadded base64url (real Google tokens)', async (page) => {
  const r = await page.evaluate(() => {
    // Real Google JWTs strip '=' padding (the original synthetic tests kept it,
    // hiding this path). Vary payload length over the achievable residues
    // (base64 length is never %4==1) to guard the unpadding+decode logic.
    const mkRaw = (o) => 'x.' + btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '') + '.y';
    for (let i = 0; i < 4; i++) {
      const payload = { hd: 'tdfb.co', email: 'a@tdfb.co', email_verified: true, exp: 9999999999, pad: 'x'.repeat(i) };
      let decoded;
      try { decoded = decodeJwtPayload(mkRaw(payload)); }
      catch (e) { return 'threw at i=' + i + ': ' + e.message; }
      if (decoded.email !== 'a@tdfb.co' || decoded.hd !== 'tdfb.co') return 'bad decode at i=' + i;
    }
    return 'ok';
  });
  if (r !== 'ok') throw new Error(r);
}]);

checks.push(['onGoogleCredential gates out non-tdfb.co and admits @tdfb.co', async (page) => {
  const { ctx, p } = await openWizard(page, { seed: null });
  const r = await p.evaluate(() => {
    const mk = (o) => 'x.' + btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_') + '.y';
    const future = Math.floor(Date.now() / 1000) + 3600;
    onGoogleCredential({ credential: mk({ hd: 'gmail.com', email: 'a@gmail.com', email_verified: true, exp: future }) });
    const blockedAfterBad = !document.getElementById('login-gate').classList.contains('hidden');
    const storedAfterBad = !!localStorage.getItem('wizardAuth');
    onGoogleCredential({ credential: mk({ hd: 'tdfb.co', email: 'ok@tdfb.co', email_verified: true, exp: future }) });
    const openedAfterGood = document.getElementById('login-gate').classList.contains('hidden');
    const storedAfterGood = !!localStorage.getItem('wizardAuth');
    return { blockedAfterBad, storedAfterBad, openedAfterGood, storedAfterGood };
  });
  await ctx.close();
  if (!r.blockedAfterBad) throw new Error('gate opened for non-tdfb.co account');
  if (r.storedAfterBad) throw new Error('session stored for non-tdfb.co account');
  if (!r.openedAfterGood) throw new Error('gate did not open for @tdfb.co account');
  if (!r.storedAfterGood) throw new Error('session not stored for @tdfb.co account');
}]);

checks.push(['GIS client script is present', async (page) => {
  const has = await page.evaluate(() =>
    !!document.querySelector('script[src="https://accounts.google.com/gsi/client"]'));
  if (!has) throw new Error('GIS client script tag missing');
}]);

checks.push(['topbar shows signed-in email; sign-out returns to gate', async (page) => {
  const seed = { email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000 };
  const { ctx, p } = await openWizard(page, { seed });
  await p.waitForFunction(() => {
    const g = document.getElementById('login-gate');
    return g && g.classList.contains('hidden');
  }, { timeout: 5000 });
  const email = await p.evaluate(() => document.getElementById('tu-email')?.textContent || '');
  if (email !== 'tester@tdfb.co') { await ctx.close(); throw new Error(`chip email wrong: "${email}"`); }
  await p.click('#tu-signout');
  let backToGate = false;
  try {
    await p.waitForFunction(() => {
      const g = document.getElementById('login-gate');
      return g && !g.classList.contains('hidden');
    }, { timeout: 5000 });
    backToGate = true;
  } catch (_) {}
  const cleared = await p.evaluate(() => !localStorage.getItem('wizardAuth'));
  await ctx.close();
  if (!backToGate) throw new Error('gate did not return after sign-out');
  if (!cleared) throw new Error('session not cleared after sign-out');
}]);

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

checks.push(['imageGate is 50 fresh / 30 edit', async (page) => {
  const r = await page.evaluate(() => {
    curDraftKey = 'freshkey';
    const fresh = imageGate();
    curDraftKey = 'freshkey__edit';
    const edit = imageGate();
    return { fresh, edit };
  });
  if (r.fresh !== 50) throw new Error(`fresh gate ${r.fresh}`);
  if (r.edit !== 30) throw new Error(`edit gate ${r.edit}`);
}]);

checks.push(['legacy crop-mode advanced toggle removed', async (page) => {
  const src = await page.evaluate(() => document.documentElement.outerHTML);
  if (src.includes('srToggleAdvanced')) throw new Error('srToggleAdvanced still present');
  const tabs = await page.evaluate(() => document.querySelectorAll('.sr-mode-tab').length);
  if (tabs !== 3) throw new Error(`expected 3 mode tabs, got ${tabs}`);
}]);

checks.push(['dashboard subtitle count is dynamic', async (page) => {
  await page.route('**/api/packagings', route => route.fulfill({ status: 200,
    contentType: 'application/json', body: JSON.stringify([
      { key: 'a', display_name: 'A', status: 'active', pipeline: 'detector_ocr', image_count: 60, accuracy: null, conf_threshold: 0.6 },
      { key: 'b', display_name: 'B', status: 'active', pipeline: 'detector_ocr', image_count: 60, accuracy: null, conf_threshold: 0.6 },
    ]) }), { times: 1 });
  const txt = await page.evaluate(async () => {
    await loadDashboard();
    return document.getElementById('dash-sub').textContent;
  });
  if (!txt.includes('2')) throw new Error(`subtitle did not reflect count: "${txt}"`);
}]);

checks.push(['HEIC removed from upload label', async (page) => {
  const t = await page.evaluate(() => document.querySelector('.upload-desc')?.textContent || '');
  if (/HEIC/i.test(t)) throw new Error('HEIC still in upload label');
  if (!/JPG/i.test(t)) throw new Error('upload label lost JPG');
}]);

// ── runner ──
async function main() {
  const server = await startServer();
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.addInitScript((origin) => { window.API_BASE_OVERRIDE = origin; }, ORIGIN);
  await page.addInitScript(() => {
    localStorage.setItem('wizardAuth', JSON.stringify({
      email: 'tester@tdfb.co', name: 'Tester', picture: '', exp: Date.now() + 3600000,
    }));
  });
  await page.route('https://accounts.google.com/**', (r) => r.abort());
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
  server.close();
  console.log(`\n${checks.length - failed}/${checks.length} passed`);
  process.exit(failed ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
