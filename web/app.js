
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const statusEl = document.getElementById('status');
const statusMessageEl = document.getElementById('statusMessage');
const imageChipEl = document.getElementById('imageChip');
const imageChipTextEl = document.getElementById('imageChipText');
const classTagsEl = document.getElementById('classTags');
const epsEl = document.getElementById('eps');
const epsText = document.getElementById('epsText');
const gotoEl = document.getElementById('gotoIdx');
const btnPrevEl = document.getElementById('btnPrev');
const btnNextEl = document.getElementById('btnNext');
const btnUndoPointEl = document.getElementById('btnUndoPoint');
const btnUndoInstEl = document.getElementById('btnUndoInst');
const btnConfirmEl = document.getElementById('btnConfirm');
const btnSaveEl = document.getElementById('btnSave');
const btnLoadSourceEl = document.getElementById('btnLoadSource');
const sourceEl = document.getElementById('sourcePath');
const classInputEl = document.getElementById('classInput');
const instanceListEl = document.getElementById('instanceList');
const instanceStatsEl = document.getElementById('instanceStats');
const modePointBtn = document.getElementById('modePointBtn');
const modeBoxBtn = document.getElementById('modeBoxBtn');
const modeBrushBtn = document.getElementById('modeBrushBtn');
const brushTypeBtn = document.getElementById('brushTypeBtn');
const brushRadiusEl = document.getElementById('brushRadius');
const pVisitedEl = document.getElementById('pVisited');
const pLabeledEl = document.getElementById('pLabeled');
const pFlaggedEl = document.getElementById('pFlagged');
const pInstancesEl = document.getElementById('pInstances');
const pSummaryEl = document.getElementById('pSummary');
const pVisitFillEl = document.getElementById('pVisitFill');
const pFlagListEl = document.getElementById('pFlagList');
const overviewOverlayEl = document.getElementById('overviewOverlay');
const overviewGridEl = document.getElementById('overviewGrid');
const settingsOverlayEl = document.getElementById('settingsOverlay');
const shortcutTableEl = document.getElementById('shortcutTable');
const settingsHintEl = document.getElementById('settingsHint');
const viewBrightnessEl = document.getElementById('viewBrightness');
const viewContrastEl = document.getElementById('viewContrast');
const viewSaturateEl = document.getElementById('viewSaturate');
const viewBrightnessValEl = document.getElementById('viewBrightnessVal');
const viewContrastValEl = document.getElementById('viewContrastVal');
const viewSaturateValEl = document.getElementById('viewSaturateVal');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const themeSelectEl = document.getElementById('themeSelect');
const localeSelectEl = document.getElementById('localeSelect');
let state = null;
let progress = null;
let overviewData = [];
let overviewFilter = 'all';
let frameImg = new Image();
let hasFrame = false;
let currentImageName = "";
let viewScale = 1.0;
let viewTx = 0.0;
let viewTy = 0.0;
let dragPan = false;
let panLastX = 0.0;
let panLastY = 0.0;
let boxMode = false;
let dragBox = false;
let boxStartX = 0.0;
let boxStartY = 0.0;
let boxEndX = 0.0;
let boxEndY = 0.0;
let editMode = false;
let brushErase = false;
let dragBrush = false;
let brushX = null;
let brushY = null;
let brushBusy = false;
let queuedBrush = null;
let lastBrushSentAt = 0;
let embeddingPollTimer = null;
let modelPollTimer = null;
let brushLineStart = null;
let recordingShortcutId = null;

const SHORTCUT_STORAGE_KEY = 'snapseg.shortcuts.v1';
const VIEW_ADJUST_STORAGE_KEY = 'snapseg.view_adjust.v1';
const THEME_STORAGE_KEY = 'snapseg.theme.v1';
const DEFAULT_THEME = 'dark';
const SUPPORTED_THEMES = ['dark', 'Graphite', 'Fog', 'Soft'];
const LOCALE_STORAGE_KEY = 'snapseg.locale.v1';
const DEFAULT_LOCALE = 'en';
const SUPPORTED_LOCALES = ['en', 'zh-TW'];
let currentLocale = DEFAULT_LOCALE;
let messages = {};
const THEME_MARKERS = {
  dark: "🔵",
  Graphite: "⚫",
  Fog: "⚪",
  Soft: "🟤",
};
const shortcutSpecs = [
  { id: 'save', label: 'Save', default: 'S', handler: () => act('save') },
  { id: 'confirm', label: 'Confirm', default: 'Enter', handler: () => act('confirm') },
  { id: 'undo', label: 'Undo Point', default: 'U', handler: () => act('undo') },
  { id: 'undo_instance', label: 'Undo Instance', default: 'Backspace', handler: () => act('undo_instance') },
  { id: 'reset', label: 'Reset Prompt', default: 'R', handler: () => act('reset') },
  { id: 'next', label: 'Next Image (Space)', default: 'Space', handler: () => act('next') },
  { id: 'next_arrow', label: 'Next Image (Right)', default: 'ArrowRight', handler: () => act('next') },
  { id: 'prev', label: 'Prev Image', default: 'ArrowLeft', handler: () => act('prev') },
  { id: 'class_next', label: 'Class Next', default: 'N', handler: () => act('class_next') },
  { id: 'class_prev', label: 'Class Prev', default: 'P', handler: () => act('class_prev') },
  { id: 'toggle_flag', label: 'Toggle Flag', default: 'F', handler: () => act('toggle_flag') },
  { id: 'toggle_edit', label: 'Toggle Brush Mode', default: 'E', handler: () => toggleEditMode() },
  { id: 'toggle_box', label: 'Toggle Box Mode', default: 'B', handler: () => toggleBoxMode() },
  { id: 'revert_mask', label: 'Revert Brush Edit', default: 'T', handler: () => act('revert_mask') },
  { id: 'zoom_in', label: 'Zoom In', default: '+', handler: () => zoomIn() },
  { id: 'zoom_out', label: 'Zoom Out', default: '-', handler: () => zoomOut() },
  { id: 'zoom_reset', label: 'Zoom Reset', default: '0', handler: () => zoomReset() },
  { id: 'undo_ctrl', label: 'Undo (Ctrl+Z)', default: 'Ctrl+Z', handler: () => act('undo') },
];
const defaultShortcuts = Object.fromEntries(shortcutSpecs.map((s) => [s.id, s.default]));
const shortcutHandlers = Object.fromEntries(shortcutSpecs.map((s) => [s.id, s.handler]));
let userShortcuts = { ...defaultShortcuts };
let viewAdjust = { brightness: 100, contrast: 100, saturate: 100 };

function esc(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function themeColor(token, fallback = '') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(token);
  const out = (v || '').trim();
  return out || fallback;
}

function applyThemeTokens(tokens) {
  const root = document.documentElement;
  for (const [k, v] of Object.entries(tokens || {})) {
    if (!k.startsWith('--')) continue;
    root.style.setProperty(k, String(v));
  }
}

async function loadTheme(themeName, opts = { persist: true, markReady: true }) {
  const themeByLower = Object.fromEntries(SUPPORTED_THEMES.map((n) => [String(n).toLowerCase(), n]));
  const normalized = themeByLower[String(themeName || '').toLowerCase()] || DEFAULT_THEME;
  const res = await fetch(`/themes/${encodeURIComponent(normalized)}.json?ts=${Date.now()}`);
  if (!res.ok) throw new Error(`theme ${normalized} not found`);
  const tokens = await res.json();
  applyThemeTokens(tokens);
  document.documentElement.setAttribute('data-theme', normalized);
  if (themeSelectEl) themeSelectEl.value = normalized;
  if (opts.persist !== false) localStorage.setItem(THEME_STORAGE_KEY, normalized);
  if (opts.markReady !== false) document.documentElement.setAttribute('data-theme-ready', '1');
  renderCanvas();
}

async function initTheme() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY) || DEFAULT_THEME;
  try {
    await loadTheme(saved, { persist: false, markReady: true });
  } catch {
    document.documentElement.setAttribute('data-theme', DEFAULT_THEME);
    document.documentElement.setAttribute('data-theme-ready', '1');
  }
}

function t(key, vars = {}) {
  let template = messages[key];
  if (template === undefined || template === null || template === '') template = key;
  return String(template).replace(/\{(\w+)\}/g, (_, k) => (vars[k] !== undefined ? String(vars[k]) : `{${k}}`));
}

async function loadLocaleMessages(locale) {
  const normalized = SUPPORTED_LOCALES.includes(locale) ? locale : DEFAULT_LOCALE;
  let next = {};
  try {
    const res = await fetch(`/locales/${encodeURIComponent(normalized)}.json?ts=${Date.now()}`);
    if (!res.ok) throw new Error(`locale ${normalized} not found`);
    next = await res.json();
  } catch {
    if (normalized !== DEFAULT_LOCALE) return loadLocaleMessages(DEFAULT_LOCALE);
    next = {};
  }
  currentLocale = normalized;
  messages = next || {};
  localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
  applyI18n();
  await drawFrame();
}

function setText(id, key, vars = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = t(key, vars);
}

function setThemeOptionLabel(id, themeKey, nameKey) {
  const el = document.getElementById(id);
  if (!el) return;
  const marker = THEME_MARKERS[themeKey] || "●";
  el.textContent = `${marker} ${t(nameKey)}`;
}

function applyI18n() {
  document.documentElement.lang = currentLocale === 'zh-TW' ? 'zh-Hant' : 'en';
  document.title = t('app.doc_title');
  if (localeSelectEl) localeSelectEl.value = currentLocale;

  setText('settingsTitle', 'settings.title');
  setText('settingsCloseBtn', 'common.close');
  setText('settingsThemeTitle', 'settings.theme');
  setThemeOptionLabel('themeDarkOption', 'dark', 'theme.dark');
  setThemeOptionLabel('themeGraphiteOption', 'Graphite', 'theme.graphite');
  setThemeOptionLabel('themeFogOption', 'Fog', 'theme.fog');
  setThemeOptionLabel('themeSoftOption', 'Soft', 'theme.soft');
  setText('settingsLanguageTitle', 'settings.language');
  setText('localeEnOption', 'language.en');
  setText('localeZhOption', 'language.zh_tw');
  setText('resetDefaultsBtn', 'settings.reset_defaults');
  setText('overviewTitle', 'overview.title');
  setText('overviewFilterAll', 'overview.filter_all');
  setText('overviewFilterFlagged', 'overview.filter_flagged');
  setText('overviewFilterLabeled', 'overview.filter_labeled');
  setText('overviewFilterUnseen', 'overview.filter_unseen');
  setText('overviewCloseBtn', 'common.close');
  setText('imageChipLabel', 'top.image');
  setText('gotoBtn', 'top.go');
  setText('confirmLabel', 'top.confirm');
  setText('saveLabel', 'top.save');
  setText('appTitle', 'app.title');
  setText('sourceSummary', 'left.source');
  setText('sourceDesc', 'left.source_desc');
  setText('pickFolderBtn', 'left.pick_folder');
  setText('pickImageBtn', 'left.pick_image');
  setText('btnLoadSource', 'left.load_source');
  setText('classesSummary', 'left.classes');
  setText('classListLabel', 'left.class_list');
  setText('activeClassLabel', 'left.active_class');
  setText('classPrevLabel', 'left.class_prev_short');
  setText('classNextLabel', 'left.class_next_short');
  setText('settingsSummary', 'left.settings');
  setText('viewAdjustmentTitle', 'left.view_adjust');
  setText('brightnessLabel', 'left.brightness');
  setText('contrastLabel', 'left.contrast');
  setText('saturationLabel', 'left.saturation');
  setText('resetViewBtn', 'left.reset_view');
  setText('brushRadiusLabel', 'left.brush_radius');
  setText('epsilonLabel', 'left.epsilon');
  setText('applyEpsilonBtn', 'left.apply_epsilon');
  setText('progressSummary', 'right.progress');
  setText('metricVisited', 'right.visited');
  setText('metricLabeled', 'right.labeled');
  setText('metricFlagged', 'right.flagged');
  setText('metricInstances', 'right.instances');
  setText('confirmedSummary', 'right.confirmed');
  setText('instStatsTitle', 'right.annotation_stats');
  setText('sessionSummary', 'right.session');
  setText('shortcutsSummary', 'right.shortcuts');
  setText('scConfirm', 'shortcuts.confirm');
  setText('scSave', 'shortcuts.save');
  setText('scNext', 'shortcuts.next');
  setText('scPrev', 'shortcuts.prev');
  setText('scBox', 'shortcuts.box');
  setText('scBrush', 'shortcuts.brush');
  setText('scFlag', 'shortcuts.flag');
  setText('scUndoInst', 'shortcuts.undo_instance');
  setText('scUndoReset', 'shortcuts.undo_reset');
  setText('scRevert', 'shortcuts.revert');
  setText('scBrushRadius', 'shortcuts.brush_radius');
  setText('scZoomPan', 'shortcuts.zoom_pan');

  sourceEl.placeholder = t('left.pick_source_placeholder');
  classInputEl.placeholder = t('left.class_input_placeholder');
  gotoEl.placeholder = t('top.goto_index');
  modePointBtn.textContent = t('mode.point');
  modeBoxBtn.textContent = t('mode.box');
  modeBrushBtn.textContent = t('mode.brush');

  btnPrevEl.title = t('tooltip.prev');
  btnNextEl.title = t('tooltip.next');
  btnUndoPointEl.title = t('tooltip.undo_point');
  btnUndoInstEl.title = t('tooltip.undo_instance');
  document.getElementById('btnResetPrompt').title = t('tooltip.reset_prompt');
  document.getElementById('btnToggleFlag').title = t('tooltip.flag');
  btnConfirmEl.title = t('tooltip.confirm');
  btnSaveEl.title = t('tooltip.save');
  document.getElementById('btnSettings').title = t('tooltip.settings');
  document.querySelector('[onclick="zoomIn()"]').title = t('tooltip.zoom_in');
  document.querySelector('[onclick="zoomOut()"]').title = t('tooltip.zoom_out');
  document.querySelector('[onclick="zoomReset()"]').title = t('tooltip.zoom_reset');
  document.querySelector('[onclick="openOverview()"]').title = t('tooltip.overview');
  document.querySelector('[onclick="gotoImage()"]').title = t('tooltip.goto');
  document.querySelector('[onclick="act(\'class_prev\')"]').title = t('tooltip.class_prev');
  document.querySelector('[onclick="act(\'class_next\')"]').title = t('tooltip.class_next');
  modePointBtn.title = t('tooltip.mode_point');
  modeBoxBtn.title = t('tooltip.mode_box');
  modeBrushBtn.title = t('tooltip.mode_brush');
  if (themeSelectEl) themeSelectEl.value = document.documentElement.getAttribute('data-theme') || DEFAULT_THEME;
  renderShortcutTable();
  if (!recordingShortcutId && settingsHintEl) settingsHintEl.textContent = t('settings.hint_customize');
  if (epsText && state) {
    epsText.textContent = t('epsilon.current', { value: Number(state.polygon_epsilon_ratio || 0.005).toFixed(4) });
  }
  if (pSummaryEl && progress) {
    const visitRate = Number(progress.visit_rate || 0);
    pSummaryEl.textContent = t('progress.visit_rate', { rate: (visitRate * 100).toFixed(1) });
  }
}

function changeLocale(locale) {
  loadLocaleMessages(locale);
}

async function changeTheme(theme) {
  try {
    await loadTheme(theme, { persist: true, markReady: true });
  } catch {
    setStatusMessage(t('error.theme_load_failed'));
  }
}

function setStatusMessage(msg) {
  statusMessageEl.textContent = msg || '';
}

function normalizeKeyName(key) {
  if (!key) return '';
  if (key === ' ') return 'Space';
  if (key === 'Esc') return 'Escape';
  if (key.length === 1) return key.toUpperCase();
  return key;
}

function eventToShortcut(e) {
  const key = normalizeKeyName(e.key);
  if (!key || key === 'Control' || key === 'Shift' || key === 'Alt' || key === 'Meta') return '';
  const mods = [];
  if (e.ctrlKey || e.metaKey) mods.push('Ctrl');
  if (e.altKey) mods.push('Alt');
  if (e.shiftKey) mods.push('Shift');
  return `${mods.length ? `${mods.join('+')}+` : ''}${key}`;
}

function saveShortcuts() {
  localStorage.setItem(SHORTCUT_STORAGE_KEY, JSON.stringify(userShortcuts));
}

function clampNum(v, lo, hi) {
  return Math.max(lo, Math.min(hi, Number(v)));
}

function saveViewAdjust() {
  localStorage.setItem(VIEW_ADJUST_STORAGE_KEY, JSON.stringify(viewAdjust));
}

function loadViewAdjust() {
  let parsed = {};
  try {
    parsed = JSON.parse(localStorage.getItem(VIEW_ADJUST_STORAGE_KEY) || '{}') || {};
  } catch {
    parsed = {};
  }
  viewAdjust.brightness = clampNum(parsed.brightness ?? 100, 50, 170);
  viewAdjust.contrast = clampNum(parsed.contrast ?? 100, 50, 220);
  viewAdjust.saturate = clampNum(parsed.saturate ?? 100, 0, 220);
}

function syncViewAdjustUI() {
  if (viewBrightnessEl) viewBrightnessEl.value = String(Math.round(viewAdjust.brightness));
  if (viewContrastEl) viewContrastEl.value = String(Math.round(viewAdjust.contrast));
  if (viewSaturateEl) viewSaturateEl.value = String(Math.round(viewAdjust.saturate));
  if (viewBrightnessValEl) viewBrightnessValEl.textContent = `${Math.round(viewAdjust.brightness)}%`;
  if (viewContrastValEl) viewContrastValEl.textContent = `${Math.round(viewAdjust.contrast)}%`;
  if (viewSaturateValEl) viewSaturateValEl.textContent = `${Math.round(viewAdjust.saturate)}%`;
}

function applyViewAdjustFromControls() {
  viewAdjust.brightness = clampNum(viewBrightnessEl?.value ?? 100, 50, 170);
  viewAdjust.contrast = clampNum(viewContrastEl?.value ?? 100, 50, 220);
  viewAdjust.saturate = clampNum(viewSaturateEl?.value ?? 100, 0, 220);
  syncViewAdjustUI();
  saveViewAdjust();
  renderCanvas();
}

function resetViewAdjust() {
  viewAdjust = { brightness: 100, contrast: 100, saturate: 100 };
  syncViewAdjustUI();
  saveViewAdjust();
  renderCanvas();
}

function currentCanvasFilter() {
  return `brightness(${viewAdjust.brightness}%) contrast(${viewAdjust.contrast}%) saturate(${viewAdjust.saturate}%)`;
}

function loadShortcuts() {
  let parsed = {};
  try {
    parsed = JSON.parse(localStorage.getItem(SHORTCUT_STORAGE_KEY) || '{}') || {};
  } catch {
    parsed = {};
  }
  userShortcuts = { ...defaultShortcuts };
  for (const spec of shortcutSpecs) {
    const raw = parsed[spec.id];
    if (typeof raw === 'string' && raw.trim()) userShortcuts[spec.id] = raw.trim();
  }
}

function getShortcutToActionMap() {
  const m = {};
  for (const spec of shortcutSpecs) {
    const k = userShortcuts[spec.id];
    if (k) m[k] = spec.id;
  }
  return m;
}

function renderShortcutTable() {
  if (!shortcutTableEl) return;
  const rows = [
    `<div class="shortcutHead">${esc(t('settings.action'))}</div><div class="shortcutHead">${esc(t('settings.key'))}</div><div class="shortcutHead">${esc(t('settings.bind'))}</div>`
  ];
  for (const spec of shortcutSpecs) {
    const label = t(`shortcut_label.${spec.id}`) === `shortcut_label.${spec.id}` ? spec.label : t(`shortcut_label.${spec.id}`);
    const key = userShortcuts[spec.id] || '-';
    const rec = recordingShortcutId === spec.id;
    rows.push(`<div class="shortcutCell">${esc(label)}</div>`);
    rows.push(`<div class="shortcutCell shortcutKey">${esc(key)}</div>`);
    rows.push(
      `<button class="shortcutActionBtn${rec ? ' recording' : ''}" onclick="startShortcutRecord('${spec.id}')">${rec ? esc(t('settings.press_key')) : esc(t('settings.set'))}</button>`
    );
  }
  shortcutTableEl.innerHTML = rows.join('');
}

function openSettings() {
  recordingShortcutId = null;
  if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_customize');
  syncViewAdjustUI();
  renderShortcutTable();
  settingsOverlayEl.style.display = 'flex';
}

function closeSettings() {
  recordingShortcutId = null;
  settingsOverlayEl.style.display = 'none';
}

function startShortcutRecord(actionId) {
  recordingShortcutId = actionId;
  if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_press_key');
  renderShortcutTable();
}

function resetShortcuts() {
  userShortcuts = { ...defaultShortcuts };
  saveShortcuts();
  recordingShortcutId = null;
  if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_reset_done');
  renderShortcutTable();
}

function setStatusFieldRows(rows) {
  statusEl.innerHTML = rows.map((r) => {
    const warnClass = r.warn ? ' warn' : '';
    return `<div class="k">${esc(r.k)}</div><div class="v${warnClass}">${esc(r.v)}</div>`;
  }).join('');
}

function updateImageChip() {
  if (!imageChipEl || !imageChipTextEl || !state) return;
  if (!state.ready) {
    imageChipTextEl.textContent = '-';
    imageChipEl.title = '-';
    return;
  }
  const text = `${state.image_index}/${state.image_total} (${state.image_name || '-'})`;
  imageChipTextEl.textContent = text;
  imageChipEl.title = text;
}

function updateTopActionAvailability() {
  if (!state || !state.ready) {
    if (btnPrevEl) btnPrevEl.disabled = true;
    if (btnNextEl) btnNextEl.disabled = true;
    if (btnUndoPointEl) btnUndoPointEl.disabled = true;
    if (btnUndoInstEl) btnUndoInstEl.disabled = true;
    if (btnConfirmEl) btnConfirmEl.disabled = true;
    if (btnSaveEl) btnSaveEl.disabled = true;
    return;
  }
  const idx = Number(state.image_index || 0);
  const total = Number(state.image_total || 0);
  const points = Number(state.points || 0);
  const hasBox = Boolean(state.has_box_prompt);
  const instances = Number(state.instances || 0);
  const hasMask = Boolean(state.has_mask);

  if (btnPrevEl) btnPrevEl.disabled = !(idx > 1);
  if (btnNextEl) btnNextEl.disabled = !(idx >= 1 && idx < total);
  if (btnUndoPointEl) btnUndoPointEl.disabled = !((points > 0) || hasBox || hasMask);
  if (btnUndoInstEl) btnUndoInstEl.disabled = !(instances > 0);
  if (btnConfirmEl) btnConfirmEl.disabled = !hasMask;
  // Save only after at least one confirmed instance exists.
  if (btnSaveEl) btnSaveEl.disabled = !(instances > 0);
}

function updateModelUI() {
  if (!btnLoadSourceEl || !state) return;
  const status = String(state.model_status || 'idle');
  const elapsedMs = Number(state.model_loading_elapsed_ms || 0);
  if (status === 'ready') {
    btnLoadSourceEl.disabled = false;
    btnLoadSourceEl.textContent = t('left.load_source');
    return;
  }
  if (status === 'error') {
    btnLoadSourceEl.disabled = true;
    btnLoadSourceEl.textContent = t('state.model_error');
    if (state.model_error) setStatusMessage(t('error.model_failed', { err: state.model_error }));
    return;
  }
  const sec = elapsedMs > 0 ? ` (${Math.round(elapsedMs / 1000)}s)` : '';
  if (elapsedMs > 60000) {
    btnLoadSourceEl.textContent = t('state.model_stalled', { sec });
    setStatusMessage(t('error.model_slow'));
  } else {
    btnLoadSourceEl.textContent = t('state.model_loading', { sec });
  }
  btnLoadSourceEl.disabled = true;
}

function statusPills(item) {
  const pills = [];
  if (item.visited) pills.push(`<span class="statusPill visited">${esc(t('right.visited'))}</span>`);
  if (item.labeled) pills.push(`<span class="statusPill labeled">${esc(t('right.labeled'))}</span>`);
  if (item.flagged) pills.push(`<span class="statusPill flagged">${esc(t('right.flagged'))}</span>`);
  if (pills.length === 0) pills.push(`<span class="statusPill">${esc(t('overview.filter_unseen'))}</span>`);
  return pills.join('');
}

function filteredOverviewItems() {
  if (overviewFilter === 'flagged') return overviewData.filter((x) => x.flagged);
  if (overviewFilter === 'labeled') return overviewData.filter((x) => x.labeled);
  if (overviewFilter === 'unseen') return overviewData.filter((x) => !x.visited);
  return overviewData;
}

function renderOverviewGrid() {
  const rows = filteredOverviewItems().map((it) => {
    const idx = Number(it.index || 0);
    const name = esc(it.name || '-');
    const currentClass = it.is_current ? ' current' : '';
    return (
      `<div class="thumbCard${currentClass}">` +
      `<div class="thumbBox"><img loading="lazy" src="/api/thumb?index=${idx}&size=260" alt="${name}"></div>` +
      `<div class="thumbName" title="${name}">#${idx} ${name}</div>` +
      `<div class="thumbMeta">${statusPills(it)} ${t('overview.inst')}: ${Number(it.instances || 0)}</div>` +
      `<button class="btn flagJump" onclick="jumpToImage(${idx}); closeOverview();">${t('overview.jump')}</button>` +
      `</div>`
    );
  }).join('');
  overviewGridEl.innerHTML = rows || `<div class="status">${esc(t('overview.no_images'))}</div>`;
}

async function loadOverview() {
  const res = await fetch('/api/overview');
  const data = await res.json();
  overviewData = Array.isArray(data.items) ? data.items : [];
  renderOverviewGrid();
}

function setOverviewFilter(name) {
  overviewFilter = name;
  document.querySelectorAll('.tagBtn[data-filter]').forEach((el) => {
    if (el.getAttribute('data-filter') === name) el.classList.add('active');
    else el.classList.remove('active');
  });
  renderOverviewGrid();
}

async function openOverview() {
  if (!state || !state.ready) return;
  showLoading(t('loading.overview'));
  await loadOverview();
  hideLoading();
  overviewOverlayEl.style.display = 'block';
}

function closeOverview() {
  overviewOverlayEl.style.display = 'none';
}

function showLoading(msg) {
  loadingText.textContent = msg || t('loading.default');
  loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
}

function zoomReset() {
  viewScale = 1.0;
  viewTx = 0.0;
  viewTy = 0.0;
  clampViewTransform();
  renderCanvas();
}

function clampViewTransform() {
  if (!state || !state.ready || cv.width <= 0 || cv.height <= 0) return;
  const scaledW = state.width * viewScale;
  const scaledH = state.height * viewScale;
  if (scaledW <= cv.width) {
    viewTx = (cv.width - scaledW) * 0.5;
  } else {
    const minTx = cv.width - scaledW;
    viewTx = Math.max(minTx, Math.min(0, viewTx));
  }
  if (scaledH <= cv.height) {
    viewTy = (cv.height - scaledH) * 0.5;
  } else {
    const minTy = cv.height - scaledH;
    viewTy = Math.max(minTy, Math.min(0, viewTy));
  }
}

function zoomAt(cx, cy, scaleFactor) {
  const oldScale = viewScale;
  const newScale = Math.max(0.5, Math.min(18.0, oldScale * scaleFactor));
  if (Math.abs(newScale - oldScale) < 1e-6) return;
  viewTx = cx - (cx - viewTx) * (newScale / oldScale);
  viewTy = cy - (cy - viewTy) * (newScale / oldScale);
  viewScale = newScale;
  clampViewTransform();
  renderCanvas();
}

function zoomIn() {
  zoomAt(cv.width / 2, cv.height / 2, 1.2);
}

function zoomOut() {
  zoomAt(cv.width / 2, cv.height / 2, 1.0 / 1.2);
}

function updateModeButtons() {
  const mode = boxMode ? 'box' : (editMode ? 'brush' : 'point');
  modePointBtn.classList.toggle('active', mode === 'point');
  modeBoxBtn.classList.toggle('active', mode === 'box');
  modeBrushBtn.classList.toggle('active', mode === 'brush');
  modePointBtn.classList.toggle('pb', mode === 'point');
  modeBoxBtn.classList.toggle('pb', mode === 'box');
  modeBrushBtn.classList.toggle('br', mode === 'brush');
}

function updateBrushButtons() {
  if (!brushTypeBtn) return;
  brushTypeBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21h18"/><path d="M7 14l5-9 5 9"/></svg><span class="lbl">${t('brush.label')}: ${brushErase ? t('brush.erase') : t('brush.add')}</span>`;
  brushTypeBtn.style.borderColor = brushErase ? themeColor('--ui-warn-btn-line') : themeColor('--ui-primary-btn-line');
}

function clearButtonFocus() {
  const ae = document.activeElement;
  if (ae && ae.tagName && ae.tagName.toUpperCase() === 'BUTTON') {
    ae.blur();
  }
}

function setMode(mode) {
  if (mode === 'box') {
    boxMode = true;
    editMode = false;
  } else if (mode === 'brush') {
    boxMode = false;
    editMode = true;
  } else {
    boxMode = false;
    editMode = false;
  }
  dragBox = false;
  dragBrush = false;
  updateModeButtons();
  updateBrushButtons();
  renderCanvas();
}

function toggleBoxMode() {
  if (boxMode) setMode('point');
  else setMode('box');
}

function toggleEditMode() {
  if (editMode) setMode('point');
  else setMode('brush');
}

function toggleBrushType() {
  brushErase = !brushErase;
  updateBrushButtons();
}

function canvasToImage(e) {
  const rect = cv.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cv.width / rect.width);
  const cy = (e.clientY - rect.top) * (cv.height / rect.height);
  const imgX = (cx - viewTx) / viewScale;
  const imgY = (cy - viewTy) / viewScale;
  const x = Math.max(0, Math.min(state.width - 1, imgX));
  const y = Math.max(0, Math.min(state.height - 1, imgY));
  return { x, y };
}

function renderCanvas() {
  if (!hasFrame || !state) return;
  if (!state.ready) {
    cv.width = 1280;
    cv.height = 720;
    const g = ctx.createLinearGradient(0, 0, cv.width, cv.height);
    g.addColorStop(0, themeColor('--canvas-welcome-grad-1'));
    g.addColorStop(1, themeColor('--canvas-welcome-grad-2'));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = themeColor('--canvas-welcome-title');
    ctx.font = '700 34px "Segoe UI","Helvetica Neue",Arial,sans-serif';
    ctx.fillText(t('canvas.welcome_title'), 72, 180);
    ctx.font = '500 20px "Segoe UI","Helvetica Neue",Arial,sans-serif';
    ctx.fillStyle = themeColor('--canvas-welcome-text');
    ctx.fillText(t('canvas.welcome_step1'), 72, 240);
    ctx.fillText(t('canvas.welcome_step2'), 72, 280);
    ctx.fillText(t('canvas.welcome_step3'), 72, 320);
    ctx.strokeStyle = themeColor('--canvas-welcome-stroke');
    ctx.lineWidth = 2;
    ctx.strokeRect(60, 120, 760, 260);
    return;
  }
  if (!frameImg || !frameImg.complete || frameImg.naturalWidth === 0) {
    return;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.setTransform(viewScale, 0, 0, viewScale, viewTx, viewTy);
  ctx.filter = currentCanvasFilter();
  ctx.drawImage(frameImg, 0, 0, state.width, state.height);
  ctx.filter = 'none';
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (dragBox) {
    const x = Math.min(boxStartX, boxEndX);
    const y = Math.min(boxStartY, boxEndY);
    const w = Math.abs(boxEndX - boxStartX);
    const h = Math.abs(boxEndY - boxStartY);
    ctx.setTransform(viewScale, 0, 0, viewScale, viewTx, viewTy);
    ctx.lineWidth = 2.0 / Math.max(0.01, viewScale);
    ctx.strokeStyle = themeColor('--canvas-box-stroke');
    ctx.strokeRect(x, y, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  if (editMode && state && state.ready && brushX !== null && brushY !== null) {
    const r = Math.max(1, Number(brushRadiusEl.value || 12));
    ctx.setTransform(viewScale, 0, 0, viewScale, viewTx, viewTy);
    ctx.lineWidth = 1.6 / Math.max(0.01, viewScale);
    ctx.strokeStyle = brushErase ? themeColor('--canvas-brush-erase') : themeColor('--canvas-brush-add');
    ctx.beginPath();
    ctx.arc(brushX, brushY, r, 0, Math.PI * 2);
    ctx.stroke();
    if (brushLineStart) {
      ctx.setLineDash([6 / Math.max(0.01, viewScale), 5 / Math.max(0.01, viewScale)]);
      ctx.strokeStyle = themeColor('--canvas-box-stroke');
      ctx.lineWidth = 1.4 / Math.max(0.01, viewScale);
      ctx.beginPath();
      ctx.moveTo(brushLineStart.x, brushLineStart.y);
      ctx.lineTo(brushX, brushY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
}

function renderInstanceList() {
  if (!state || !state.ready || !state.instances_detail || state.instances_detail.length === 0) {
    instanceListEl.innerHTML = t('instance.none');
    if (instanceStatsEl) instanceStatsEl.innerHTML = `<div class="status">${esc(t('instance.no_stats'))}</div>`;
    return;
  }
  const rows = state.instances_detail.map((it) => {
    const idx = Number(it.index) + 1;
    const label = String(it.label || 'object');
    const score = Number(it.score || 0).toFixed(4);
    let dot = themeColor('--canvas-instance-dot');
    if (Array.isArray(it.color_bgr) && it.color_bgr.length === 3) {
      const b = Number(it.color_bgr[0] || 0);
      const g = Number(it.color_bgr[1] || 0);
      const r = Number(it.color_bgr[2] || 0);
      dot = `rgb(${r},${g},${b})`;
    }
    return (
      `<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin:4px 0;">` +
      `<span style="display:flex;align-items:center;gap:8px;"><span style="width:10px;height:10px;border-radius:50%;background:${dot};display:inline-block;flex:none;"></span>#${idx} ${label} (${t('instance.score')} ${score})</span>` +
      `<button class="btn warn" style="width:84px;margin:0;padding:6px 8px;" onclick="deleteInstance(${Number(it.index)})">${t('instance.delete')}</button>` +
      `</div>`
    );
  });
  instanceListEl.innerHTML = rows.join('');
  renderInstanceStats();
}

function fmtInt(n) {
  return Number(n || 0).toLocaleString();
}

function fmtFloat(n, d = 2) {
  return Number(n || 0).toFixed(d);
}

function renderInstanceStats() {
  if (!instanceStatsEl) return;
  if (!state || !state.ready || !Array.isArray(state.instances_detail) || state.instances_detail.length === 0) {
    instanceStatsEl.innerHTML = `<div class="status">${esc(t('instance.no_stats'))}</div>`;
    return;
  }
  const byClass = {};
  for (const it of state.instances_detail) {
    const k = String(it.label || 'object');
    if (!byClass[k]) byClass[k] = [];
    byClass[k].push({
      score: Number(it.score || 0),
      area: Number(it.area_px || 0),
    });
  }
  const rows = Object.entries(byClass).map(([cls, arr]) => {
    const count = arr.length;
    const scoreAvg = arr.reduce((a, b) => a + b.score, 0) / Math.max(1, count);
    const areaAvg = arr.reduce((a, b) => a + b.area, 0) / Math.max(1, count);
    const areaMin = Math.min(...arr.map((x) => x.area));
    const areaMax = Math.max(...arr.map((x) => x.area));
    const scoreWarn = scoreAvg < 0.75;
    const rangeWarn = areaMin > 0 && (areaMax / areaMin) > 10.0;
    return {
      cls,
      count,
      scoreAvg,
      areaAvg,
      areaMin,
      areaMax,
      scoreWarn,
      rangeWarn,
    };
  });
  const head = `<table class="instStatsTable"><thead><tr><th>${esc(t('stats.class'))}</th><th>${esc(t('stats.count'))}</th><th>${esc(t('stats.avg_score'))}</th><th>${esc(t('stats.avg_area'))}</th><th>${esc(t('stats.area_range'))}</th></tr></thead><tbody>`;
  const body = rows.map((r) => {
    const scoreCls = r.scoreWarn ? 'instStatsWarn' : '';
    const rangeCls = r.rangeWarn ? 'instStatsWarn' : '';
    return (
      `<tr>` +
      `<td>${esc(r.cls)}</td>` +
      `<td>${fmtInt(r.count)}</td>` +
      `<td class="${scoreCls}">${fmtFloat(r.scoreAvg, 2)}</td>` +
      `<td>${fmtInt(Math.round(r.areaAvg))} px簡</td>` +
      `<td class="${rangeCls}">${fmtInt(r.areaMin)} ~ ${fmtInt(r.areaMax)}</td>` +
      `</tr>`
    );
  }).join('');
  instanceStatsEl.innerHTML = `${head}${body}</tbody></table>`;
}

function renderClassTags() {
  if (!state || !Array.isArray(state.class_list)) {
    classTagsEl.innerHTML = '';
    return;
  }
  const rows = state.class_list.map((name, i) => {
    const active = Number(i) === Number(state.class_idx) ? ' active' : '';
    return `<button class="classTag${active}" onclick="setClass(${i})" title="${esc(t('class.set_class', { index: i + 1, name }))}">${esc(name)}</button>`;
  });
  classTagsEl.innerHTML = rows.join('');
}

async function getProgress() {
  const res = await fetch('/api/progress');
  progress = await res.json();
  const visited = Number(progress.visited_count || 0);
  const labeled = Number(progress.labeled_count || 0);
  const flagged = Number(progress.flagged_count || 0);
  const total = Number(progress.total_images || 0);
  const totalInstances = Number(progress.total_instances || 0);
  const visitRate = Number(progress.visit_rate || 0);

  pVisitedEl.textContent = `${visited}/${total}`;
  pLabeledEl.textContent = `${labeled}`;
  pFlaggedEl.textContent = `${flagged}`;
  pInstancesEl.textContent = `${totalInstances}`;
  pSummaryEl.textContent = t('progress.visit_rate', { rate: (visitRate * 100).toFixed(1) });
  pVisitFillEl.style.width = `${Math.max(0, Math.min(100, visitRate * 100))}%`;

  const items = Array.isArray(progress.flagged_items) ? progress.flagged_items : [];
  if (items.length === 0) {
    pFlagListEl.innerHTML = t('progress.flagged_none');
  } else {
    const rows = items.map((it) => {
      const idx = Number(it.index || 0);
      const name = esc(it.name || '-');
      const current = idx === Number(progress.current_index || 0);
      const tag = current ? ` ${t('progress.current_tag')}` : '';
      return (
        `<div class="flagRow">` +
        `<span class="flagName" title="#${idx} ${name}${tag}">#${idx} ${name}${tag}</span>` +
        `<button class="btn flagJump" onclick="jumpToImage(${idx})">${t('overview.jump')}</button>` +
        `</div>`
      );
    }).join('');
    pFlagListEl.innerHTML = `${t('progress.flagged_list')}:${rows}`;
  }
}

async function getState() {
  const res = await fetch('/api/state');
  state = await res.json();
  updateImageChip();
  updateTopActionAvailability();
  renderClassTags();
  const activeClass = (state.class_list && state.class_list.length > 0) ? state.class_list[state.class_idx] : '-';
  const promptMode = boxMode ? t('mode.box') : (editMode ? t('mode.brush') : t('mode.point'));
  const modelSource = state.model_source || t('state.unknown');
  const modelStatus = state.model_status || t('state.idle');
  const rows = [
    { k: t('state.image'), v: `${state.image_index}/${state.image_total} (${state.image_name || '-'})` },
    { k: t('state.class'), v: `${activeClass}` },
    { k: t('state.instances'), v: `${state.instances}` },
    { k: t('state.flagged'), v: `${state.flagged}` },
    { k: t('state.prompt_mode'), v: `${promptMode} (${t('state.box_prompt')}: ${state.has_box_prompt})` },
    { k: t('state.points'), v: `${state.points}` },
    { k: t('state.score'), v: `${state.score}` },
    { k: t('state.latency'), v: `${state.latency_ms} ms` },
    { k: t('state.backend'), v: `${state.backend_requested} -> ${state.backend}` },
    { k: t('state.model'), v: `${state.model_id}` },
    { k: t('state.model_source'), v: `${modelSource}` },
    { k: t('state.model_status'), v: `${modelStatus}` },
    { k: t('state.save_queue'), v: `${state.save_queue}` },
    { k: t('state.autosave_queue'), v: `${state.autosave_queue}` },
    { k: t('state.vram_free'), v: `${state.prefetch_free_gb} GB` },
    { k: t('state.prefetch_guard'), v: `${state.prefetch_paused_low_vram}` },
    { k: t('state.autosave_file'), v: `${state.autosave}` },
    { k: t('state.zoom'), v: `${viewScale.toFixed(2)}x` },
  ];
  setStatusFieldRows(rows);
  setStatusMessage(state.embedding_error || state.backend_warning || '');
  epsEl.value = Number(state.polygon_epsilon_ratio || 0.005);
  epsText.textContent = t('epsilon.current', { value: Number(state.polygon_epsilon_ratio || 0.005).toFixed(4) });
  gotoEl.max = String(state.image_total);
  gotoEl.placeholder = t('top.goto_range', { total: state.image_total });
  if (state.source_path) sourceEl.value = state.source_path;
  const editingClassInput = (document.activeElement === classInputEl);
  if (!editingClassInput && state.class_list && state.class_list.length > 0) {
    classInputEl.value = state.class_list.join(',');
  }
  renderInstanceList();
  updateModelUI();
}

async function drawFrame() {
  await Promise.all([getState(), getProgress()]);
  if (state && state.ready && !state.embedding_ready) {
    if (!embeddingPollTimer) {
      embeddingPollTimer = setTimeout(async () => {
        embeddingPollTimer = null;
        if (state && state.ready && !state.embedding_ready) {
          await drawFrame();
        }
      }, 300);
    }
  } else if (embeddingPollTimer) {
    clearTimeout(embeddingPollTimer);
    embeddingPollTimer = null;
  }
  if (state && state.model_status === 'loading') {
    if (!modelPollTimer) {
      modelPollTimer = setTimeout(async () => {
        modelPollTimer = null;
        if (state && state.model_status === 'loading') {
          await drawFrame();
        }
      }, 500);
    }
  } else if (modelPollTimer) {
    clearTimeout(modelPollTimer);
    modelPollTimer = null;
  }
  if (!state.ready) {
    hasFrame = true;
    renderCanvas();
    return;
  }
  const imageChanged = currentImageName !== state.image_name;
  if (imageChanged) currentImageName = state.image_name;
  const nextImg = new Image();
  nextImg.onload = () => {
    cv.width = state.width;
    cv.height = state.height;
    frameImg = nextImg;
    hasFrame = true;
    // Reset after canvas size is updated; prevents stale pan/zoom offsets.
    if (imageChanged) {
      viewScale = 1.0;
      viewTx = 0.0;
      viewTy = 0.0;
    }
    clampViewTransform();
    renderCanvas();
  };
  nextImg.src = `/api/frame?fmt=png&ts=${Date.now()}`;
}

async function act(action) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action})});
  await drawFrame();
}

async function setClass(idx) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action:'set_class', class_idx: Number(idx)})});
  await drawFrame();
}

async function deleteInstance(idx) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'delete_instance', index: Number(idx)})
  });
  await drawFrame();
}

async function submitBox(x1, y1, x2, y2) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  await fetch('/api/box', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x1, y1, x2, y2})
  });
  await drawFrame();
}

async function submitBrushLine(x1, y1, x2, y2, erase) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  await fetch('/api/brush-line', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      x1, y1, x2, y2,
      radius: Number(brushRadiusEl.value || 12),
      erase: !!erase,
    })
  });
  await drawFrame();
}

async function pumpBrush() {
  if (brushBusy || !queuedBrush) return;
  const now = Date.now();
  if (now - lastBrushSentAt < 35) {
    setTimeout(pumpBrush, 20);
    return;
  }
  const q = queuedBrush;
  queuedBrush = null;
  brushBusy = true;
  try {
    await fetch('/api/brush', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        x: q.x,
        y: q.y,
        radius: Number(brushRadiusEl.value || 12),
        erase: !!q.erase,
      }),
    });
    await drawFrame();
  } finally {
    lastBrushSentAt = Date.now();
    brushBusy = false;
    if (queuedBrush) setTimeout(pumpBrush, 0);
  }
}

function enqueueBrush(x, y, erase) {
  queuedBrush = { x, y, erase };
  pumpBrush();
}

async function endBrushStroke() {
  if (!state || !state.ready) return;
  while (brushBusy || queuedBrush) {
    await new Promise((r) => setTimeout(r, 20));
  }
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'brush_end'})
  });
}

async function applyEpsilon() {
  if (!state || !state.ready) return;
  clearButtonFocus();
  const v = Number(epsEl.value);
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'set_epsilon', epsilon: v})
  });
  await drawFrame();
}

async function gotoImage() {
  if (!state || !state.ready) return;
  clearButtonFocus();
  const idx = Number(gotoEl.value);
  if (!Number.isFinite(idx)) return;
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'goto', index: Math.round(idx)})
  });
  await drawFrame();
}

async function jumpToImage(idx) {
  if (!state || !state.ready) return;
  clearButtonFocus();
  const n = Number(idx);
  if (!Number.isFinite(n)) return;
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'goto', index: Math.round(n)})
  });
  await drawFrame();
}

async function applyConfig() {
  const source_path = sourceEl.value.trim();
  const classes = classInputEl.value.trim();
  if (!source_path) {
    setStatusMessage(t('error.pick_source_first'));
    return;
  }
  if (state && state.model_status !== 'ready' && state.model_status !== 'error') {
    const sec = Number(state.model_loading_elapsed_ms || 0);
    const tip = sec > 0 ? ` (${Math.round(sec / 1000)}s)` : '';
    setStatusMessage(t('error.model_loading', { tip }));
    return;
  }
  if (state && state.model_status === 'error') {
    setStatusMessage(t('error.model_failed', { err: state.model_error || t('state.unknown') }));
    return;
  }
  showLoading(t('loading.source'));
  const res = await fetch('/api/config', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source_path, classes})
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({detail:t('state.unknown')}));
    setStatusMessage(t('error.generic', { err: err.detail || t('state.unknown') }));
    hideLoading();
    return;
  }
  // Force viewport reset after loading a new source to avoid stale pan/zoom state.
  currentImageName = "";
  viewScale = 1.0;
  viewTx = 0.0;
  viewTy = 0.0;
  await drawFrame();
  hideLoading();
}

async function pickFolder() {
  const res = await fetch('/api/pick-folder', { method: 'POST' });
  if (!res.ok) return;
  const data = await res.json();
  if (data.path) sourceEl.value = data.path;
}

async function pickImage() {
  const res = await fetch('/api/pick-image', { method: 'POST' });
  if (!res.ok) return;
  const data = await res.json();
  if (data.path) sourceEl.value = data.path;
}

cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('auxclick', (e) => {
  if (e.button === 1) e.preventDefault();
});
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!state) return;
  const rect = cv.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cv.width / rect.width);
  const cy = (e.clientY - rect.top) * (cv.height / rect.height);
  let deltaY = e.deltaY;
  if (e.deltaMode === 1) deltaY *= 16;
  if (e.deltaMode === 2) deltaY *= window.innerHeight;
  const factor = Math.exp(-deltaY * 0.0015);
  zoomAt(cx, cy, factor);
}, { passive: false });

cv.addEventListener('mousedown', async (e) => {
  if (!state || !state.ready) return;
  if ((e.button === 0 && e.shiftKey) || e.button === 1) {
    e.preventDefault();
    dragPan = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
    cv.style.cursor = 'grabbing';
    return;
  }
  if (editMode && (e.button === 0 || e.button === 2)) {
    const p = canvasToImage(e);
    brushX = p.x;
    brushY = p.y;
    if (e.ctrlKey) {
      const erase = (e.button === 2) ? true : brushErase;
      if (!brushLineStart) {
        brushLineStart = { x: p.x, y: p.y, erase };
      } else {
        const start = brushLineStart;
        brushLineStart = null;
        await submitBrushLine(start.x, start.y, p.x, p.y, erase);
      }
      renderCanvas();
      return;
    }
    brushLineStart = null;
    dragBrush = true;
    const erase = (e.button === 2) ? true : brushErase;
    enqueueBrush(p.x, p.y, erase);
    renderCanvas();
    return;
  }
  if (boxMode && e.button === 0) {
    const p = canvasToImage(e);
    dragBox = true;
    boxStartX = p.x;
    boxStartY = p.y;
    boxEndX = p.x;
    boxEndY = p.y;
    renderCanvas();
    return;
  }
  const p = canvasToImage(e);
  const x = p.x;
  const y = p.y;
  const label = (e.button === 2) ? 0 : 1;
  await fetch('/api/click', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({x, y, label})});
  await drawFrame();
});

cv.addEventListener('mousemove', async (e) => {
  if (state && state.ready) {
    const p = canvasToImage(e);
    brushX = p.x;
    brushY = p.y;
  }
  if (dragPan) {
    const dx = (e.clientX - panLastX) * (cv.width / cv.getBoundingClientRect().width);
    const dy = (e.clientY - panLastY) * (cv.height / cv.getBoundingClientRect().height);
    panLastX = e.clientX;
    panLastY = e.clientY;
    viewTx += dx;
    viewTy += dy;
    clampViewTransform();
    renderCanvas();
    return;
  }
  if (dragBrush) {
    const erase = (e.buttons & 2) ? true : brushErase;
    enqueueBrush(brushX, brushY, erase);
    renderCanvas();
    return;
  }
  if (dragBox) {
    const p = canvasToImage(e);
    boxEndX = p.x;
    boxEndY = p.y;
    renderCanvas();
    return;
  }
  if (editMode) {
    renderCanvas();
  }
});

window.addEventListener('mouseup', async () => {
  if (dragPan) {
    dragPan = false;
    cv.style.cursor = 'crosshair';
    return;
  }
  if (dragBox) {
    dragBox = false;
    const x1 = Math.min(boxStartX, boxEndX);
    const y1 = Math.min(boxStartY, boxEndY);
    const x2 = Math.max(boxStartX, boxEndX);
    const y2 = Math.max(boxStartY, boxEndY);
    if ((x2 - x1) >= 2 && (y2 - y1) >= 2) {
      await submitBox(x1, y1, x2, y2);
    } else {
      renderCanvas();
    }
  }
  if (dragBrush) {
    dragBrush = false;
    await endBrushStroke();
  }
});

window.addEventListener('keydown', async (e) => {
  if (recordingShortcutId) {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      recordingShortcutId = null;
      if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_record_cancel');
      renderShortcutTable();
      return;
    }
    const shortcut = eventToShortcut(e);
    if (!shortcut) return;
    const owner = shortcutSpecs.find((s) => userShortcuts[s.id] === shortcut && s.id !== recordingShortcutId);
    if (owner) {
      const ownerLabel = t(`shortcut_label.${owner.id}`) === `shortcut_label.${owner.id}` ? owner.label : t(`shortcut_label.${owner.id}`);
      if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_conflict', { shortcut, owner: ownerLabel });
      return;
    }
    userShortcuts[recordingShortcutId] = shortcut;
    saveShortcuts();
    const savedSpec = shortcutSpecs.find((s) => s.id === recordingShortcutId);
    const savedLabel = savedSpec ? (t(`shortcut_label.${savedSpec.id}`) === `shortcut_label.${savedSpec.id}` ? savedSpec.label : t(`shortcut_label.${savedSpec.id}`)) : t('settings.shortcut');
    if (settingsHintEl) settingsHintEl.textContent = t('settings.hint_saved', {
      label: savedLabel,
      shortcut,
    });
    recordingShortcutId = null;
    renderShortcutTable();
    return;
  }
  if (e.key === 'Escape' && settingsOverlayEl.style.display === 'flex') {
    closeSettings();
    return;
  }
  if (e.key === 'Escape' && overviewOverlayEl.style.display === 'block') {
    closeOverview();
    return;
  }
  if (e.key === 'Escape' && brushLineStart) {
    brushLineStart = null;
    renderCanvas();
    return;
  }
  const tag = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toUpperCase() : "";
  const isTypingContext = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  if (isTypingContext) return;
  if (tag === 'BUTTON' && (e.key === ' ' || e.key === 'Enter')) {
    e.preventDefault();
  }
  const shortcut = eventToShortcut(e);
  const actionId = getShortcutToActionMap()[shortcut];
  if (actionId && shortcutHandlers[actionId]) {
    e.preventDefault();
    return shortcutHandlers[actionId]();
  }
  if (normalizeKeyName(e.key) === '[') {
    e.preventDefault();
    brushRadiusEl.value = String(Math.max(1, Number(brushRadiusEl.value || 12) - 1));
    return renderCanvas();
  }
  if (normalizeKeyName(e.key) === ']') {
    e.preventDefault();
    brushRadiusEl.value = String(Math.min(64, Number(brushRadiusEl.value || 12) + 1));
    return renderCanvas();
  }
  if (/^[1-9]$/.test(e.key)) return setClass(Number(e.key) - 1);
});

window.addEventListener('keyup', (e) => {
  const tag = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toUpperCase() : "";
  const isTypingContext = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  if (isTypingContext) return;
  if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
  }
});

if (viewBrightnessEl) viewBrightnessEl.addEventListener('input', applyViewAdjustFromControls);
if (viewContrastEl) viewContrastEl.addEventListener('input', applyViewAdjustFromControls);
if (viewSaturateEl) viewSaturateEl.addEventListener('input', applyViewAdjustFromControls);

updateModeButtons();
updateBrushButtons();
loadViewAdjust();
syncViewAdjustUI();
loadShortcuts();

(async () => {
  await initTheme();
  const savedLocale = localStorage.getItem(LOCALE_STORAGE_KEY) || DEFAULT_LOCALE;
  await loadLocaleMessages(savedLocale);
})();

// Stage 2A: expose inline-handler targets explicitly for safety.
Object.assign(window, {
  closeSettings,
  changeTheme,
  changeLocale,
  resetShortcuts,
  closeOverview,
  setOverviewFilter,
  openOverview,
  gotoImage,
  jumpToImage,
  setMode,
  openSettings,
  act,
  zoomIn,
  zoomOut,
  zoomReset,
  pickFolder,
  pickImage,
  applyConfig,
  resetViewAdjust,
  applyEpsilon,
  deleteInstance,
  setClass,
  startShortcutRecord,
});

