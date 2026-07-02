"""Capture annotated screenshots of the Packaging Wizard for the user manual.

Drives the TEST_MODE wizard (served at :8091, backend :8081) with Playwright,
navigates each screen via the wizard's own JS functions, injects highlight
overlays (gold ring + numbered badge) over the buttons the manual points at,
and saves PNGs to docs/manual/img/.

Run the harness first:  .\\scripts\\run_test_wizard.ps1   (separate terminal)
Then:                    python scripts/capture_manual_shots.py
"""
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

WIZARD_URL = "http://localhost:8091/wizard.html"
API_BASE = "http://localhost:8081"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "manual" / "img"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#E8A020"

# Active packaging used as the running example in the manual. print_sticker_back is
# the first shipped multi_field class AND reads product names — so it exercises both
# the new step-1 multi_field crop mode and the drawer product-names editor.
# Must exist in the TEST_MODE config (run_test_wizard.ps1 seeds it from prod config).
EXAMPLE_KEY = "print_sticker_back"
EXAMPLE_NAME = "สติกเกอร์หลังซอง Matcha"

# Throwaway draft built from real matcha-sticker photos so the step-3 "label" shot shows
# the new multi_field class with a gold lot box + the 2-chip label picker. Built after the
# dashboard shot and deleted at the end so it never pollutes dashboard.png.
DEMO_DRAFT_KEY = "print_sticker_demo"
DEMO_SRC_IMAGES = REPO / "images" / "print_sticker_back"
DEMO_DRAFT_DIR = REPO / "data" / "test" / "drafts" / DEMO_DRAFT_KEY
DEMO_LEAD_IMAGE = "1000016046.jpg"   # clean Matcha M2 sticker — select this in step 3


def build_demo_draft():
    """Create the print_sticker_demo draft from matcha photos + detector bboxes.

    Pulls bboxes from the live samples endpoint (detector run server-side), copies the
    source photos into the draft, and writes one annotation per image. Idempotent.
    """
    if not DEMO_SRC_IMAGES.exists():
        print(f"build_demo_draft SKIP — no source images at {DEMO_SRC_IMAGES}")
        return False
    try:
        url = f"{API_BASE}/api/packagings/{EXAMPLE_KEY}/samples?count=14"
        with urllib.request.urlopen(url, timeout=120) as r:
            samples = json.loads(r.read().decode("utf-8")).get("samples", [])
    except Exception as e:
        print("build_demo_draft ERR fetching samples:", e)
        return False

    (DEMO_DRAFT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DEMO_DRAFT_DIR / "annotations").mkdir(parents=True, exist_ok=True)
    samples = [s for s in samples if any(rg.get("bbox") for rg in s.get("regions", []))]
    samples.sort(key=lambda s: 0 if s["name"] == DEMO_LEAD_IMAGE else 1)

    count = 0
    for s in samples:
        name = s["name"]
        if not (DEMO_SRC_IMAGES / name).exists():
            continue
        bboxes = [{"x1": rg["bbox"][0], "y1": rg["bbox"][1],
                   "x2": rg["bbox"][2], "y2": rg["bbox"][3],
                   "label": rg.get("label") or "lot_exp"}
                  for rg in s["regions"] if rg.get("bbox")]
        if not bboxes:
            continue
        shutil.copy2(DEMO_SRC_IMAGES / name, DEMO_DRAFT_DIR / "images" / name)
        (DEMO_DRAFT_DIR / "annotations" / f"{name}.json").write_text(
            json.dumps({"bboxes": bboxes, "updated_at": "2026-06-24T00:00:00+00:00"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    meta = {
        "key": DEMO_DRAFT_KEY, "display_name": EXAMPLE_NAME,
        "description": "สติกเกอร์หลังซอง Matcha — LOT/วันหมดอายุอยู่ล่างขวา, ชื่อสินค้าอยู่บน",
        "pipeline": "detector_ocr", "sub_regions": ["lot_exp", "product_size"],
        "detection_mode": "multi_field", "status": "configured",
        "created_at": "2026-06-24T00:00:00+00:00", "updated_at": "2026-06-24T00:00:00+00:00",
        "config": {"lot_patterns": [r"(?i)[A-Z]{1}\d{17,}[A-Z0-9]*"],
                   "fields_extracted": ["lot", "exp", "product", "size"],
                   "sheet_checks": ["lot", "exp"],
                   "message_template_key": "default_full", "product_aliases": []},
        "images": [],
    }
    (DEMO_DRAFT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"build_demo_draft OK — {count} labeled matcha images")
    return count > 0


def cleanup_demo_draft():
    if DEMO_DRAFT_DIR.exists():
        shutil.rmtree(DEMO_DRAFT_DIR, ignore_errors=True)
        print("cleaned up demo draft")

OVERLAY_JS = r"""
() => {
  window.__clearOv = () => document.querySelectorAll('.__ov').forEach(e => e.remove());
  window.__ov = (sel, num, color) => {
    color = color || '#E8A020';
    const el = (typeof sel === 'string') ? document.querySelector(sel) : sel;
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const sx = window.scrollX, sy = window.scrollY;
    const ring = document.createElement('div');
    ring.className = '__ov';
    Object.assign(ring.style, {
      position: 'absolute', left: (r.left + sx - 6) + 'px', top: (r.top + sy - 6) + 'px',
      width: (r.width + 12) + 'px', height: (r.height + 12) + 'px',
      border: '3px solid ' + color, borderRadius: '10px',
      zIndex: 99999, pointerEvents: 'none',
      boxShadow: '0 0 0 2px rgba(255,255,255,.5), 0 6px 22px rgba(232,160,32,.35)'
    });
    document.body.appendChild(ring);
    if (num) {
      const b = document.createElement('div');
      b.className = '__ov';
      b.textContent = num;
      Object.assign(b.style, {
        position: 'absolute', left: (r.left + sx - 17) + 'px', top: (r.top + sy - 17) + 'px',
        width: '30px', height: '30px', background: color, color: '#1a1206',
        borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontWeight: '800', fontSize: '16px', fontFamily: 'system-ui, sans-serif',
        zIndex: 100000, pointerEvents: 'none', boxShadow: '0 3px 10px rgba(0,0,0,.45)'
      });
      document.body.appendChild(b);
    }
    return true;
  };
}
"""


def main():
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 920},
                                  device_scale_factor=2)
        page = ctx.new_page()

        def goto_dash():
            page.goto(WIZARD_URL, wait_until="networkidle")
            page.wait_for_selector(".pkg-card", timeout=15000)
            page.wait_for_timeout(900)

        def ov_init():
            page.evaluate(OVERLAY_JS)

        def clear():
            page.evaluate("() => window.__clearOv && window.__clearOv()")

        def ov(sel, num=None, color=ACCENT):
            return page.evaluate("(a) => window.__ov(a.sel, a.num, a.color)",
                                 {"sel": sel, "num": num, "color": color})

        def shot(name, full=False):
            page.wait_for_timeout(350)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), full_page=full)
            ok = path.exists() and path.stat().st_size > 0
            results.append((name, ok))
            print(f"[{'OK' if ok else 'FAIL'}] {name}.png")

        # ── 1. Dashboard ───────────────────────────────
        try:
            goto_dash()
            # hide the leftover 'new_tea_bag_box' test-draft card so it doesn't show in the
            # manual dashboard (the draft itself is kept — steps 2/4/5 still drive it)
            page.evaluate("""() => {
                const strip = document.querySelector('.pkg-photo-strip[data-key="new_tea_bag_box"]');
                const card = strip && strip.closest('.pkg-card');
                if (card) card.remove();
            }""")
            page.wait_for_timeout(200)
            ov_init()
            ov(".pkg-card-add", "1")
            ov(".pkg-card", "2")
            shot("dashboard", full=True)  # full page so the bottom card row isn't cut off
            clear()
        except Exception as e:
            print("dashboard ERR", e)

        # ── 2 & 13. Drawer (active pkg: edit/clone/archive + conf) ──
        try:
            page.evaluate("(a) => openDrawer(a.key, {display_name: a.name})",
                          {"key": EXAMPLE_KEY, "name": EXAMPLE_NAME})
            # samples endpoint re-runs the detector live on full-res images (~2s each),
            # so a product class with large photos can take ~20-40s before the drawer body renders
            page.wait_for_selector("#drawer-actions", state="visible", timeout=60000)
            page.wait_for_timeout(1100)
            ov_init()
            ov("#conf-input", "1") or ov(".drawer-body", "1")
            ov("[onclick^='cloneActive']", "2")
            ov("[onclick^='archivePackaging']", "3")
            shot("drawer")
            clear()
            # archive-focused variant
            ov_init()
            ov("[onclick^='archivePackaging']", "1")
            shot("archive")
            clear()
            # product-names editor (runtime alias edit) — print_sticker_back reads product
            scrolled = page.evaluate(
                "() => { const r=document.getElementById('drawer-pa-rows');"
                " if(r){ r.scrollIntoView({block:'center'}); return true } return false }")
            if scrolled:
                page.wait_for_timeout(450)
                ov_init()
                ov("#drawer-pa-rows", "1")
                ov("[onclick^='saveDrawerAliases']", "2")
                ov("[onclick^='revertDrawerAliases']", "3") or ov("[onclick^='addProdAliasTo']", "3")
                shot("drawer-product")
                clear()
            else:
                print("drawer-product SKIP — no #drawer-pa-rows (is EXAMPLE_KEY a product-reading class?)")
            page.evaluate("() => closeDrawer()")
            page.wait_for_timeout(400)
        except Exception as e:
            print("drawer ERR", e)

        # ── 3. Step 1 — name + pipeline + crop mode ─────
        try:
            page.evaluate("() => startWizard()")
            page.wait_for_selector("#sp1", state="visible", timeout=8000)
            page.wait_for_timeout(500)
            ov_init()
            ov("#inp-display-name", "1")
            ov("#inp-pipeline", "2")
            ov(".sr-mode-tabs", "3")
            shot("step1")
            clear()
        except Exception as e:
            print("step1 ERR", e)

        # ── 3b. Step 1 — multi_field crop mode (field ticks + box picker) ──
        try:
            page.evaluate("() => srSetMode('multi_field')")
            page.wait_for_timeout(500)
            page.evaluate("() => { const e=document.getElementById('sr-fields');"
                          " if(e) e.scrollIntoView({block:'center'}); }")
            page.wait_for_timeout(300)
            ov_init()
            ov(".sr-mode-tab[data-mode='multi_field']", "1")
            ov("#sr-fields", "2")
            shot("step1-multi")
            clear()
            page.evaluate("() => srSetMode('single')")  # reset for downstream steps
        except Exception as e:
            print("step1-multi ERR", e)

        # Load the real draft into wizard state for steps 2-5
        page.evaluate("() => { curDraftKey = 'new_tea_bag_box'; }")

        # ── 4. Step 2 — upload zone ─────────────────────
        try:
            page.evaluate("() => goStep(2)")
            page.wait_for_selector("#sp2", state="visible", timeout=8000)
            page.wait_for_timeout(700)
            ov_init()
            ov("#uz", "1")
            shot("step2")
            clear()
        except Exception as e:
            print("step2 ERR", e)

        # ── 5. Step 3 — annotation (real bbox, NEW class print_sticker_back) ──
        # Use a demo draft built from real matcha-sticker photos + detector-derived
        # bboxes (data/test/drafts/print_sticker_demo) so the label step shows the
        # new multi_field class with a gold lot box + the 2-chip label picker.
        # Built here (after dashboard.png) and removed at the end so it never shows
        # as an extra card on the dashboard shot.
        build_demo_draft()
        try:
            page.evaluate("() => { curDraftKey = 'print_sticker_demo'; }")
            page.evaluate("() => goStep(3)")
            page.wait_for_selector("#sp3", state="visible", timeout=8000)
            # match the breadcrumb to the matcha class (it reads the step-1 name input)
            page.evaluate("""(name) => {
                const inp = document.getElementById('inp-display-name'); if (inp) inp.value = name;
                const tt = document.getElementById('topbar-title');
                if (tt) tt.innerHTML = `<span style="font-size:13px;color:var(--t3)">Packaging</span>`
                  + `<span style="color:var(--t3);margin:0 6px">›</span>`
                  + `<span style="font-size:14px;font-weight:500">${name}</span>`;
            }""", EXAMPLE_NAME)
            page.wait_for_timeout(1800)
            # prefer the clean Matcha M2 sticker; fall back to first labeled image
            page.evaluate("""() => {
                const imgs = annot.images || [];
                let i = imgs.findIndex(im => im.name === '1000016046.jpg');
                if (i < 0) i = imgs.findIndex(im => im.labeled);
                if (i >= 0) annotSelect(i);
            }""")
            page.wait_for_timeout(2100)  # wait for full-res image load + canvas bbox draw
            ov_init()
            ov("#annot-canvas-wrap", "1")
            ov("#annot-thumbs", "2")
            ov(".annot-progress", "3")
            shot("step3")
            clear()
            # reset draft + name for step 4 (keeps step4/5 on the generic example)
            page.evaluate("""() => {
                curDraftKey = 'new_tea_bag_box';
                const inp = document.getElementById('inp-display-name'); if (inp) inp.value = 'New Tea Bag Box';
            }""")
        except Exception as e:
            print("step3 ERR", e)

        # ── 6. Step 4 — config (lot pattern / fields / template) ──
        try:
            page.evaluate("() => goStep(4)")
            page.wait_for_selector("#sp4", state="visible", timeout=8000)
            page.wait_for_timeout(700)
            ov_init()
            ov("#lot-rows", "1")
            ov(".regex-preview", "2")
            ov("[data-group='fields']", "3")
            shot("step4")  # Message Template is shown in step4-product.png
            clear()
        except Exception as e:
            print("step4 ERR", e)

        # ── 7. Step 4 — product aliases card ────────────
        try:
            page.evaluate("""() => {
                const el = document.querySelector("[data-group='fields'] [data-field='product']");
                if (el && !el.classList.contains('on')) { toggleCb(el); }
                syncProductAliasVisibility();
                if (typeof addProdAlias === 'function' && !document.querySelector('#pa-rows .pa-row')) addProdAlias();
            }""")
            page.wait_for_timeout(600)
            page.evaluate("() => { const c=document.getElementById('pa-card'); if(c) c.scrollIntoView(); }")
            page.wait_for_timeout(500)
            ov_init()
            ov("#pa-card", "1")
            shot("step4-product", full=True)
            clear()
        except Exception as e:
            print("step4-product ERR", e)

        # ── 8. Step 5a — pre-training (publish + open notebook) ──
        try:
            page.set_viewport_size({"width": 1280, "height": 600})  # tighten frame for short step-5 cards
            page.evaluate("() => goStep(5)")
            page.wait_for_selector("#sp5", state="visible", timeout=8000)
            page.wait_for_timeout(900)
            page.evaluate("""() => renderStep5_preTraining(
                {key:'new_tea_bag_box', display_name:'New Tea Bag Box', status:'configured'})""")
            page.wait_for_timeout(400)
            ov_init()
            ov("#btn-full-start", "1")
            shot("step5a")
            clear()
        except Exception as e:
            print("step5a ERR", e)

        # ── 9. Step 5b — waiting for Colab (sync model) ──
        try:
            page.evaluate("""() => renderStep5_training(
                {key:'new_tea_bag_box', display_name:'New Tea Bag Box', status:'training_full'})""")
            page.wait_for_timeout(400)
            ov_init()
            ov("#btn-sync-model", "1")
            shot("step5b")
            clear()
        except Exception as e:
            print("step5b ERR", e)

        # ── 10. Step 5c — eval + deploy (passing floor) ──
        try:
            page.set_viewport_size({"width": 1280, "height": 920})  # eval card is taller
            mock = {
                "eval": {"detector_mAP_50": 0.912, "precision": 0.934, "recall": 0.881,
                          "epochs": 100, "imgsz": 640, "train_count": 42, "val_count": 11},
                "hard_floor": {"passed": True,
                                "hard_floor": {"detector_mAP_50": 0.65, "precision": 0.70}},
            }
            page.evaluate("""(ev) => renderStep5_eval(
                {key:'new_tea_bag_box', display_name:'New Tea Bag Box', status:'trained'}, ev)""", mock)
            page.wait_for_timeout(500)
            ov_init()
            ov(".eval-grid", "1")
            ov(".floor-pass", "2")
            ov("#btn-deploy", "3") or ov(".deploy-sec", "3")
            shot("step5c", full=True)
            clear()
        except Exception as e:
            print("step5c ERR", e)

        # ── 11. Success ─────────────────────────────────
        try:
            page.set_viewport_size({"width": 1180, "height": 720})  # center the celebration card
            page.evaluate("() => showView('success')")
            page.wait_for_timeout(700)
            ov_init()
            ov("[onclick=\"showView('dashboard')\"]", "1")
            shot("success")
            clear()
            page.set_viewport_size({"width": 1440, "height": 920})
        except Exception as e:
            print("success ERR", e)

        # ── 12. Edit-mode banner + prelabel (clone an active pkg) ──
        try:
            edit_key = page.evaluate("""async (k) => {
                try { await api('DELETE', '/api/packagings/' + k + '__edit'); } catch (e) {}
                const d = await api('POST', '/api/packagings/' + k + '/clone');
                return d.key;
            }""", EXAMPLE_KEY)
            print("clone ->", edit_key)
            page.evaluate("(k) => { curDraftKey = k; showView('wizard'); goStep(2); }", edit_key)
            page.wait_for_selector("#edit-mode-banner", state="visible", timeout=8000)
            page.wait_for_timeout(800)
            ov_init()
            ov("#edit-mode-banner", "1")
            shot("edit-banner")
            clear()
            # prelabel bar on step 3
            page.evaluate("() => goStep(3)")
            page.wait_for_timeout(1500)
            page.evaluate("() => { const b=document.getElementById('prelabel-bar'); if(b) b.style.display='block'; }")
            page.wait_for_timeout(400)
            ov_init()
            ov("#btn-prelabel", "1")
            shot("edit-prelabel")
            clear()
            # cleanup the clone
            page.evaluate("(k) => api('DELETE', '/api/packagings/'+encodeURIComponent(k)).catch(()=>{})", edit_key)
        except Exception as e:
            print("edit-banner ERR", e)

        browser.close()

    cleanup_demo_draft()
    ok = sum(1 for _, o in results if o)
    print(f"\n{ok}/{len(results)} screenshots captured -> {OUT}")
    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
