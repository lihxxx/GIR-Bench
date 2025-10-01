// ----- Theme -----
const root = document.documentElement;
const themeBtn = document.getElementById('theme-btn');
const savedTheme = localStorage.getItem('theme');
if (savedTheme) root.setAttribute('data-theme', savedTheme);
themeBtn?.addEventListener('click', () => {
    const cur = root.getAttribute('data-theme');
    const next = cur === 'light' ? null : 'light';
    if (next) root.setAttribute('data-theme', next); else root.removeAttribute('data-theme');
    localStorage.setItem('theme', next || '');
});

// ----- Tabs (supports multiple groups) -----
// For every .tabs group, switch the nearest following .tabpanes
document.querySelectorAll('.tabs').forEach(group => {
    const buttons = group.querySelectorAll('.tab');
    const panesWrap = group.nextElementSibling && group.nextElementSibling.classList.contains('tabpanes')
        ? group.nextElementSibling
        : null;
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            // buttons active state (current group only)
            buttons.forEach(x => x.classList.remove('active'));
            btn.classList.add('active');

            if (!panesWrap) return;
            // panel switching (prefer data-target; fallback to #pane-{data-tab})
            const target = btn.getAttribute('data-target') || ('#pane-' + (btn.getAttribute('data-tab') || '').trim());
            const targetEl = panesWrap.querySelector(target);
            // Use direct children to avoid :scope compatibility issues
            Array.from(panesWrap.children).forEach(p => p.classList.remove('active'));
            if (targetEl) targetEl.classList.add('active');
        });
    });
});

// ----- Compare sliders -----
const bindCompare = (rangeId, topId) => {
    const range = document.getElementById(rangeId);
    const top = document.getElementById(topId);
    range?.addEventListener('input', e => { top.style.width = e.target.value + '%'; });
};
bindCompare('cmpRange', 'cmpTop');         // Qualitative section
bindCompare('cmpRange-uni', 'cmpTop-uni'); // Examples / Uni tab

// ----- Sortable table -----
const table = document.getElementById('lb');
const getRows = () => Array.from(table?.querySelectorAll('tbody tr') || []);
const parseNum = v => (v === '—' || v === '–' || v === '-' ? NaN : parseFloat(v));
let sortState = { key: 'ucg_g', asc: false };
function sortBy(key) {
    if (!table) return;
    // Column indices in the table:
    // 0: Model, 1: Type, 2: UCG-U, 3: UCG-G, 4: T2I, 5: Edit
    const idxMap = { model: 0, type: 1, ucg_u: 2, ucg_g: 3, t2i: 4, edit: 5 };
    const idx = idxMap[key];
    if (idx === undefined) return; // unknown key
    const rows = getRows();
    rows.sort((a, b) => {
        if (key === 'model' || key === 'type') {
            const A = (a.children[idx]?.textContent || '').trim().toLowerCase();
            const B = (b.children[idx]?.textContent || '').trim().toLowerCase();
            return sortState.asc ? A.localeCompare(B) : B.localeCompare(A);
        } else {
            const A = parseNum((a.children[idx]?.textContent || '').trim());
            const B = parseNum((b.children[idx]?.textContent || '').trim());
            if (isNaN(A) && isNaN(B)) return 0;
            if (isNaN(A)) return 1;
            if (isNaN(B)) return -1;
            return sortState.asc ? A - B : B - A;
        }
    });
    const tbody = table.querySelector('tbody');
    rows.forEach(r => tbody.appendChild(r));
}
document.querySelectorAll('.sort').forEach(el => {
    el.addEventListener('click', () => {
        const key = el.dataset.key;
        if (sortState.key === key) { sortState.asc = !sortState.asc; } else { sortState.key = key; sortState.asc = false; }
        sortBy(key);
        applyTypeFilter();
    });
});
// Type filter
const typeFilter = document.getElementById('typeFilter');
function applyTypeFilter() {
    const val = typeFilter?.value || 'all';
    getRows().forEach(tr => {
        // Type is the 2nd column (index 1)
        const typeCell = tr.children[1];
        const type = (typeCell && typeCell.textContent ? typeCell.textContent : '').trim();
        tr.style.display = (val === 'all' || type === val) ? '' : 'none';
    });
}
typeFilter?.addEventListener('change', applyTypeFilter);

// Initial draw
sortBy('ucg_g');
applyTypeFilter();

// ----- Mini bars (demo data) -----
const barsData = [
    { label: 'Numerical Reasoning', value: 0.36, hint: 'Upper-bound (proprietary) demo' },
    { label: 'Spatial Layout', value: 0.76, hint: 'Upper-bound (proprietary) demo' },
    { label: 'Text Rendering (implicit)', value: 0.81, hint: 'Upper-bound (proprietary) demo' },
    { label: 'Visual Puzzle (norm. FID)', value: 0.40, hint: 'Demo' },
    { label: 'Visual Logic (Sudoku)', value: 0.25, hint: 'Demo' },
    { label: 'Reasoning Perception (IoU)', value: 0.44, hint: 'Demo' },
];
const barsHost = document.getElementById('bars');
if (barsHost) {
    barsData.forEach(d => {
        const wrap = document.createElement('div'); wrap.className = 'bar';
        const label = document.createElement('div'); label.innerHTML = `<b>${d.label}</b><div class="caption">${d.hint}</div>`;
        const track = document.createElement('div'); track.className = 'track';
        const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = (d.value * 100) + '%';
        track.appendChild(fill); wrap.appendChild(label); wrap.appendChild(track); barsHost.appendChild(wrap);
    });
}

// ----- Copy BibTeX -----
const copyBtn = document.getElementById('copyBib');
const copyState = document.getElementById('copyState');
copyBtn?.addEventListener('click', async () => {
    const txt = document.getElementById('bib').innerText.trim();
    try {
        await navigator.clipboard.writeText(txt);
        copyState.textContent = 'Copied ✓';
    } catch {
        copyState.textContent = 'Copy failed';
    } finally {
        setTimeout(() => copyState.textContent = 'Ready', 1500);
    }
});

// ----- Back to top -----
document.getElementById('scrollTop')?.addEventListener('click', () =>
    window.scrollTo({ top: 0, behavior: 'smooth' })
);

// ----- Email: assemble mailto + copy -----
document.querySelectorAll('.mailto').forEach(link => {
    const user = link.getAttribute('data-user');
    const domain = link.getAttribute('data-domain');
    if (user && domain) {
        const addr = `${user}@${domain}`;
        link.setAttribute('href', `mailto:${addr}`);
        link.textContent = addr;
    }
});

document.querySelectorAll('.copy-mail').forEach(btn => {
    btn.addEventListener('click', async () => {
        const user = btn.getAttribute('data-user');
        const domain = btn.getAttribute('data-domain');
        if (!user || !domain) return;
        const addr = `${user}@${domain}`;
        try {
            await navigator.clipboard.writeText(addr);
            btn.textContent = 'Copied ✓';
        } catch {
            btn.textContent = 'Copy failed';
        } finally {
            setTimeout(() => btn.textContent = 'Copy', 1500);
        }
    });
});

// ----- Lightbox for overview + example images -----
(() => {
    const triggers = Array.from(document.querySelectorAll('#overview-img, .tabpanes .example'));
    if (!triggers.length) return;
    // Create backdrop once
    const backdrop = document.createElement('div');
    backdrop.className = 'lightbox-backdrop';
    backdrop.setAttribute('role', 'dialog');
    backdrop.setAttribute('aria-modal', 'true');
    backdrop.setAttribute('aria-label', 'Image preview');
    const full = document.createElement('img');
    full.className = 'lightbox-img';
    backdrop.appendChild(full);
    document.body.appendChild(backdrop);

    const open = (src, alt) => {
        full.src = src;
        full.alt = alt || 'Preview';
        backdrop.classList.add('active');
        document.body.style.overflow = 'hidden';
    };
    const close = () => {
        backdrop.classList.remove('active');
        document.body.style.overflow = '';
    };

    triggers.forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', () => open(img.src, img.alt));
    });
    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) close();
    });
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') close();
    });
})();
