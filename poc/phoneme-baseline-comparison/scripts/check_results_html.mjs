/** Verify report data and rendering functions without launching a browser. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const base = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const html = readFileSync(resolve(base, 'evaluation-results.html'), 'utf8');
const payloadText = html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payloadText);
assert.equal(data.conditions.rows.length, 1080);
assert.equal(data.differences.rows.length, 792);
assert.equal(data.strata.rows.length, 9720);
for (const [file, expected] of Object.entries(data.sources)) {
  assert.equal(createHash('sha256').update(readFileSync(resolve(base, file))).digest('hex'), expected);
}

// These objects collect text and event handlers only. They do not implement a
// browser, open a page, or interact with the user's computer.
class TextSink {
  constructor(id, dataset = {}) {
    this.id = id;
    this.dataset = dataset;
    this.value = '';
    this.innerHTML = '';
    this.textContent = '';
    this.listeners = {};
    this.attributes = {};
    this.parentElement = { setAttribute() {} };
  }
  addEventListener(event, listener) { this.listeners[event] = listener; }
  setAttribute(name, value) { this.attributes[name] = value; }
  click() { if (this.download) downloads.push(this.download); this.listeners.click?.({ target: this }); }
  remove() {}
}
const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new TextSink(m[1])]));
for (const match of html.matchAll(/<select[^>]+id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  elements.get(match[1]).value = match[2].match(/<option value="([^"]*)"/)[1];
}
elements.get('report-data').textContent = payloadText;
const roleButtons = ['verification', 'cross_text_verification'].map(role => new TextSink(null, { primaryRole: role }));
const sortButtons = [...html.matchAll(/data-sort="([^"]+)"/g)].map(m => new TextSink(null, { sort: m[1] }));
const queries = {
  '[data-primary-role]': roleButtons,
  '[data-sort]': sortButtons,
  '#condition-filters select': [...elements.values()].filter(e => e.id.startsWith('filter-') && e.id !== 'filter-reset'),
  '#paired-filters select': [...elements.values()].filter(e => e.id.startsWith('pair-')),
};
const downloads = [];
let exportedBlob;
const document = {
  getElementById(id) { assert.ok(elements.has(id), `Unknown element ${id}`); return elements.get(id); },
  querySelectorAll(selector) { assert.ok(selector in queries, `Unknown selector ${selector}`); return queries[selector]; },
  createElement() { return new TextSink(null); },
  body: { appendChild() {} },
};
const context = vm.createContext({ document, Blob, URL: { createObjectURL(blob) { exportedBlob = blob; return 'blob:unit-test'; }, revokeObjectURL() {} }, setTimeout(fn) { fn(); } });
new vm.Script(script, { filename: 'phase7-report.js' }).runInContext(context);
const evaluate = expression => vm.runInContext(expression, context);
const text = id => elements.get(id).innerHTML;
const change = (id, value) => { elements.get(id).value = value; elements.get(id).listeners.change(); };
const rowCount = id => (text(id).match(/<tr>/g) ?? []).length;

assert.equal(rowCount('primary-body'), 3);
assert.match(text('primary-body'), /34 \/ 750/);
assert.match(text('primary-body'), /65 \/ 10,500/);
assert.match(text('primary-meaning'), /拒否が0件/);
roleButtons[1].click();
assert.match(text('primary-body'), /1 \/ 450/);
assert.match(text('primary-meaning'), /誤受入は23 \/ 6,300件/);
assert.equal(roleButtons[1].attributes['aria-pressed'], 'true');
assert.equal(rowCount('paired-primary'), 4);
assert.equal(rowCount('length-body'), 6);
assert.equal(evaluate('selected.length'), 15);
change('filter-cap', 'max_1s');
assert.equal(evaluate('selected.length'), 3);
assert.match(text('conditions-body'), /95.200%/);
assert.equal(evaluate('selected.filter(r => r.conditional_status === "not_evaluable").length'), 2);
change('filter-support', 'common');
assert.equal(evaluate('selected.length'), 3);
assert.equal(evaluate('selected.filter(r => r.conditional_status === "not_evaluable").length'), 3);
assert.match(text('conditions-body'), /4.933%/);
assert.match(text('conditions-body'), /100.000%/);
const detailIndex = evaluate('selected[0].index');
elements.get('conditions-body').listeners.click({ target: { closest() { return { dataset: { detail: String(detailIndex) } }; } } });
assert.equal(elements.get('condition-detail').hidden, false);
assert.match(text('condition-detail'), /評価不能/);
change('pair-kind', 'cap_minus_full');
change('pair-support', 'common');
assert.equal(rowCount('paired-body'), 12);
assert.match(text('paired-body'), /capごとに集合が変化/);
elements.get('filter-reset').click();
assert.equal(evaluate('selected.length'), 15);
for (const key of evaluate('filterKeys.join(",")').split(',')) change('filter-' + key, '');
assert.equal(evaluate('selected.length'), 1080);
assert.equal(rowCount('conditions-body'), 30);
assert.equal(elements.get('page-label').textContent, '1 / 36');
elements.get('page-next').click();
assert.equal(elements.get('page-label').textContent, '2 / 36');
sortButtons.find(b => b.dataset.sort === 'far').click();
assert.equal(elements.get('page-label').textContent, '1 / 36');
assert.equal(evaluate('selected.filter(r => r.far !== null).every((r,i,rows) => i === 0 || rows[i-1].far <= r.far)'), true);
assert.equal(evaluate('selected[selected.length-1].far'), null);
assert.equal(rowCount('strata-body'), 300);
assert.match(elements.get('strata-count').textContent, /4,320行/);
change('stratum-dimension', 'actual_input_seconds');
assert.match(elements.get('strata-count').textContent, /5,400行/);
elements.get('filter-reset').click();
elements.get('export-filtered').click();
assert.deepEqual(downloads, ['phase7-filtered-results.csv']);
assert.equal((await exportedBlob.text()).trim().split('\r\n').length, 16);
console.log('Passed: embedded source hashes, primary counts, zero/NE distinction, cross-text switch, filters, shared-support coverage, details, paired cohorts, sorting, pagination, strata and CSV export.');
