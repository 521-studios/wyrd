// Manifest-driven SPA. Adding a new generator on the server requires no client
// changes — input forms are built from each generator's input_schema.

const API_BASE = window.WYRD_API_BASE || "/api";

const $ = (id) => document.getElementById(id);

let manifest = null;
let currentGenerator = null;

async function loadManifest() {
    const resp = await fetch(`${API_BASE}/manifest`);
    manifest = await resp.json();
    const select = $("generator-select");
    select.replaceChildren();
    for (const g of manifest.generators) {
        const opt = document.createElement("option");
        opt.value = g.name;
        opt.textContent = g.display_name;
        select.appendChild(opt);
    }
    select.onchange = () => selectGenerator(select.value);

    // Honor ?generator= and ?seed= from the URL on first load.
    const url = new URL(window.location.href);
    const initial = url.searchParams.get("generator") || manifest.generators[0]?.name;
    if (initial) {
        select.value = initial;
        selectGenerator(initial);
    }
}

function selectGenerator(name) {
    currentGenerator = manifest.generators.find((g) => g.name === name);
    if (!currentGenerator) return;
    $("generator-description").textContent = currentGenerator.description;
    renderForm(currentGenerator.input_schema);
    _renderAbout(currentGenerator.details);
    // Switching generators invalidates any prior results — different generators
    // produce different output shapes, and a stale legend would mislead.
    $("results").replaceChildren();
    $("legend").hidden = true;
    $("seed-line").hidden = true;
    _updateOutputVisibility();
}

// #output is the panel containing #about, #results, #legend, and #seed-line.
// It should be visible whenever any of those have content and hidden otherwise,
// so the empty styled panel never shows on first load or on a generator with
// no details.
function _updateOutputVisibility() {
    const hasAbout = !$("about").hidden;
    const hasResults = $("results").children.length > 0;
    $("output").hidden = !hasAbout && !hasResults;
}

function _renderAbout(details) {
    const div = $("about");
    div.replaceChildren();
    if (!details) {
        div.hidden = true;
        return;
    }
    const heading = document.createElement("h3");
    heading.className = "panel-title";
    heading.textContent = "What is this?";
    div.appendChild(heading);
    // `details` is a static string defined in the generator's Python source —
    // it never contains user input or anything fetched at request time. Same-
    // origin doesn't prevent XSS; the safety here is that the content cannot
    // be influenced by anything the user does. If a future generator builds
    // `details` from external data, this branch must use a sanitizer.
    const body = document.createElement("div");
    body.innerHTML = details;
    div.appendChild(body);
    div.hidden = false;
}

function _renderLegend(legend) {
    const el = $("legend");
    el.replaceChildren();
    if (!legend || legend.length === 0) {
        el.hidden = true;
        return;
    }
    el.appendChild(document.createTextNode("Sources: "));
    legend.forEach((entry, i) => {
        if (i > 0) el.appendChild(document.createTextNode(" · "));
        const code = document.createElement("strong");
        code.textContent = entry.code;
        el.appendChild(code);
        el.appendChild(document.createTextNode(" " + entry.name));
    });
    el.hidden = false;
}

function _buildSelectField(key, prop, urlVal) {
    const input = document.createElement("select");
    input.id = `field-${key}`;
    input.name = key;
    for (const v of prop.enum) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        input.appendChild(opt);
    }
    input.value = urlVal || prop.default || prop.enum[0];
    return input;
}

// wyrd-awo: dependent-select rendering for fields keyed off the 'culture'
// field. The schema carries `x-options-by-culture` mapping each culture
// to its allowed option list (e.g. era cells per era family). On render
// the field shows options for the currently-selected culture, and the
// culture-select's onchange rebuilds the option list while preserving
// the current selection if still valid. Falls back to the first
// culture's options when no culture select exists yet (defensive — the
// kenning generator always exposes culture, but the helper stays
// schema-agnostic).
function _buildDependentSelectField(key, prop, urlVal) {
    const input = document.createElement("select");
    input.id = `field-${key}`;
    input.name = key;
    const initialCulture = _currentCultureValue() || _firstCultureFromMap(prop["x-options-by-culture"]);
    // Pass urlVal-or-default explicitly so a fresh select (whose .value
    // is "") doesn't accidentally beat prop.default in the fallback
    // chain. Also load-bears for the order-agnostic-render claim — if
    // the culture field hasn't been built yet, this initial pass picks
    // a best-guess culture's options, and the post-render sync in
    // _wireDependentSelects re-applies the same urlVal/default once
    // the real culture is known.
    const preferred = urlVal !== null ? urlVal : prop.default;
    _populateDependentOptions(input, prop, initialCulture, preferred);
    return input;
}

function _currentCultureValue() {
    const cultureField = document.getElementById("field-culture");
    return cultureField ? cultureField.value : null;
}

function _firstCultureFromMap(optionsByCulture) {
    const keys = Object.keys(optionsByCulture || {});
    return keys.length > 0 ? keys[0] : null;
}

function _populateDependentOptions(select, prop, culture, preferredValue) {
    const map = prop["x-options-by-culture"] || {};
    const options = map[culture] || [];
    const previous = preferredValue !== null && preferredValue !== undefined
        ? preferredValue
        : select.value;
    select.replaceChildren();
    for (const v of options) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v === "" ? "(no filter)" : v;
        select.appendChild(opt);
    }
    // Preserve a previously-chosen value when the new option set still
    // contains it (e.g. switching English ↔ Scottish keeps 'modern'
    // selected); otherwise fall back to the schema default. The bare
    // includes() check (no truthy guard) is intentional: "" is the
    // 'no filter' option and a valid selection to preserve across
    // culture changes — wrapping with `previous &&` would silently
    // drop the empty selection on every culture change.
    if (options.includes(previous)) {
        select.value = previous;
    } else if (prop.default !== undefined && options.includes(prop.default)) {
        select.value = prop.default;
    } else if (options.length > 0) {
        select.value = options[0];
    }
}

// Wire all `x-options-by-culture` fields to repopulate when the culture
// select changes. Called once after renderForm finishes building all
// fields, since the dependee may be rendered before its dependent.
//
// `url` is forwarded so the initial sync can re-read URL params per
// dependent key — without it, the urlVal a dependent built with would
// be lost when sync rebuilds the option list against the actual
// culture (load-bearing for the order-agnostic-render claim: even if
// the dependent renders before the dependee, the URL value survives).
function _wireDependentSelects(schema, url) {
    const cultureField = document.getElementById("field-culture");
    if (!cultureField) return;
    const dependentKeys = [];
    for (const [key, prop] of Object.entries(schema.properties || {})) {
        if (prop["x-options-by-culture"]) dependentKeys.push([key, prop]);
    }
    if (dependentKeys.length === 0) return;
    const sync = (isInitial) => {
        for (const [key, prop] of dependentKeys) {
            const select = document.getElementById(`field-${key}`);
            if (!select) continue;
            // On the initial sync, prefer the URL param (if any) and
            // fall back to the schema default. On a culture-change
            // event we want the previously-chosen value to survive
            // when valid for the new culture, so pass null and let
            // _populateDependentOptions read select.value.
            let preferred = null;
            if (isInitial) {
                const urlVal = url.searchParams.get(key);
                preferred = urlVal !== null ? urlVal : prop.default;
            }
            _populateDependentOptions(select, prop, cultureField.value, preferred);
        }
    };
    cultureField.addEventListener("change", () => sync(false));
    // One initial sync after wire-up so the dependent's options match
    // the culture field's actual value, even when schema iteration
    // built the dependent before the dependee. _buildDependentSelectField
    // takes a best-guess at render time, but the post-render sync is
    // the load-bearing source of truth.
    sync(true);
}

function _buildTagGrid(key, prop, url) {
    const grid = document.createElement("div");
    grid.className = "tag-grid";
    grid.dataset.field = key;
    const urlVals = url.searchParams.getAll(key);
    const selected = new Set(urlVals.length ? urlVals : prop.default || []);
    for (const v of prop.items.enum) {
        const tagLabel = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.name = key;
        cb.value = v;
        cb.checked = selected.has(v);
        tagLabel.appendChild(cb);
        tagLabel.appendChild(document.createTextNode(v));
        grid.appendChild(tagLabel);
    }
    return grid;
}

function _buildNumberField(key, prop, urlVal) {
    const input = document.createElement("input");
    input.type = "number";
    input.id = `field-${key}`;
    input.name = key;
    if (prop.minimum !== undefined) input.min = prop.minimum;
    if (prop.maximum !== undefined) input.max = prop.maximum;
    if (prop.type === "integer") input.step = 1;
    if (urlVal !== null) input.value = urlVal;
    else if (prop.default !== undefined) input.value = prop.default;
    return input;
}

function _buildTextField(key, prop, urlVal) {
    const input = document.createElement("input");
    input.type = "text";
    input.id = `field-${key}`;
    input.name = key;
    if (urlVal !== null) input.value = urlVal;
    else if (prop.default !== undefined) input.value = prop.default;
    return input;
}

function _buildBooleanField(key, prop, urlVal) {
    // Render boolean params as a checkbox so the form value type matches
    // the schema. Without this the field falls through to a text input
    // and a stringified default ("false") gets shipped to the server,
    // where bool("false") is True — silent gate inversion (wyrd-yan).
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `field-${key}`;
    input.name = key;
    if (urlVal !== null) input.checked = urlVal === "true" || urlVal === "1";
    else input.checked = prop.default === true;
    return input;
}

function _buildField(key, prop, url) {
    const urlVal = url.searchParams.get(key);
    if (prop["x-options-by-culture"]) {
        return _buildDependentSelectField(key, prop, urlVal);
    }
    if (prop.type === "string" && Array.isArray(prop.enum)) {
        return _buildSelectField(key, prop, urlVal);
    }
    if (prop.type === "array" && prop.items?.enum) {
        return _buildTagGrid(key, prop, url);
    }
    if (prop.type === "integer" || prop.type === "number") {
        return _buildNumberField(key, prop, urlVal);
    }
    if (prop.type === "boolean") {
        return _buildBooleanField(key, prop, urlVal);
    }
    return _buildTextField(key, prop, urlVal);
}

function renderForm(schema) {
    const form = $("params-form");
    form.replaceChildren();
    if (!schema || !schema.properties) return;

    const heading = document.createElement("h3");
    heading.className = "panel-title";
    heading.textContent = "Options";
    form.appendChild(heading);

    const url = new URL(window.location.href);

    for (const [key, prop] of Object.entries(schema.properties)) {
        if (key === "seed") continue; // exposed via the share link, not the form
        const wrap = document.createElement("div");
        wrap.className = "field";

        const label = document.createElement("label");
        label.textContent = prop.description ? `${key} — ${prop.description}` : key;
        label.htmlFor = `field-${key}`;
        wrap.appendChild(label);

        wrap.appendChild(_buildField(key, prop, url));
        form.appendChild(wrap);
    }
    // After all fields exist, wire any dependent-select fields to their
    // dependee (currently always the culture field). Done in a second
    // pass since the dependee may be rendered before its dependents in
    // the schema iteration order. `url` is forwarded so the initial
    // sync can re-read URL params per dependent key.
    _wireDependentSelects(schema, url);
}

function readForm() {
    const params = {};
    if (!currentGenerator) return params;

    for (const [key, prop] of Object.entries(currentGenerator.input_schema.properties || {})) {
        if (key === "seed") continue;
        if (prop.type === "array" && prop.items?.enum) {
            const grid = document.querySelector(`[data-field="${key}"]`);
            const values = [...grid.querySelectorAll("input:checked")].map((cb) => cb.value);
            if (values.length) params[key] = values;
        } else if (prop.type === "boolean") {
            // Read the checkbox's checked state, NOT its value (which is
            // always the literal string "on"). Sends a real JSON boolean
            // so the server doesn't have to guess from string tokens.
            const el = document.getElementById(`field-${key}`);
            if (el) params[key] = el.checked;
        } else {
            const el = document.getElementById(`field-${key}`);
            if (el && el.value !== "") {
                params[key] = prop.type === "integer" ? parseInt(el.value, 10) : el.value;
            }
        }
    }
    return params;
}

async function sha256Hex(text) {
    // CloudFront OAC + Lambda Function URL requires POST/PUT clients to send
    // x-amz-content-sha256 — CloudFront doesn't compute the body hash itself.
    const buf = new TextEncoder().encode(text);
    const hashBuf = await crypto.subtle.digest("SHA-256", buf);
    return [...new Uint8Array(hashBuf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function _renderResultItem(r) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = r.result;
    const explanation = document.createElement("p");
    explanation.className = "explanation";
    explanation.textContent = r.explanation || "";
    li.appendChild(name);
    li.appendChild(explanation);

    // wyrd-9kh.6: structured per-element citation list. The explainer
    // text already shows "cited by ..." inline (truncated to 3 + count),
    // but the full list per element lives in components and deserves its
    // own disclosure so a curious GM can see who attests each morpheme.
    const citations = _renderCitationsDetail(r.components || []);
    if (citations) li.appendChild(citations);

    // wyrd-qhs0 Phase 2d: etymological-provenance panel — surfaces the
    // four wyrd-ha9q renderings (original_script + transliteration +
    // english_shaped + IPA) for non-Latin source-lang morphemes. Skipped
    // when none of the components carries any rendering data, which is
    // the common case for English / Celtic / Romance generation.
    const renderings = _renderProvenancePanel(r.components || []);
    if (renderings) li.appendChild(renderings);

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Components (raw)";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(r.components, null, 2);
    details.appendChild(summary);
    details.appendChild(pre);
    li.appendChild(details);
    return li;
}

function _renderProvenancePanel(components) {
    // wyrd-qhs0 Phase 2d. components[i].renderings is a sparse map:
    //   { <lang_field>: { <canonical_form>: { original_script, transliteration, english_shaped, ipa, dialect } } }
    // Returns null when no component has ANY rendering data, so the
    // disclosure doesn't render for English-only outputs.
    const anyRenderings = components.some(
        (c) => c.renderings && Object.keys(c.renderings).length > 0,
    );
    if (!anyRenderings) return null;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Etymological provenance";
    details.appendChild(summary);
    const list = document.createElement("ul");
    list.className = "provenance";
    for (const c of components) {
        if (!c.renderings || Object.keys(c.renderings).length === 0) continue;
        const li = document.createElement("li");
        const usage = document.createElement("strong");
        usage.textContent = c.usage;
        li.appendChild(usage);
        for (const [langField, formMap] of Object.entries(c.renderings)) {
            for (const [canonicalForm, slots] of Object.entries(formMap)) {
                const row = _renderProvenanceRow(langField, canonicalForm, slots);
                if (row) li.appendChild(row);
            }
        }
        list.appendChild(li);
    }
    details.appendChild(list);
    return details;
}

function _renderProvenanceRow(langField, canonicalForm, slots) {
    // One <div class="provenance-row"> per (lang_field, canonical_form)
    // surfacing whichever of the four renderings the lexicon has.
    // Each rendering is its own labeled span so the SPA can style /
    // hide individual columns. Rendering keys with null values are
    // skipped — the panel only shows what's actually populated.
    const row = document.createElement("div");
    row.className = "provenance-row";
    const lang = document.createElement("span");
    lang.className = "provenance-lang";
    lang.textContent = langField;
    row.appendChild(lang);
    const renderings = [
        { key: "original_script", label: "native" },
        { key: "transliteration", label: "translit" },
        { key: "english_shaped", label: "english" },
        { key: "ipa", label: "ipa" },
    ];
    let any = false;
    for (const { key, label } of renderings) {
        const value = slots[key];
        if (!value) continue;
        any = true;
        const span = document.createElement("span");
        span.className = `provenance-${key}`;
        const labelSpan = document.createElement("span");
        labelSpan.className = "provenance-label";
        labelSpan.textContent = `${label}:`;
        const valueSpan = document.createElement("span");
        valueSpan.className = "provenance-value";
        valueSpan.textContent = value;
        if (key === "ipa" && slots.dialect) {
            valueSpan.textContent = `${value} (${slots.dialect})`;
        }
        span.appendChild(labelSpan);
        span.appendChild(valueSpan);
        row.appendChild(span);
    }
    if (!any) return null;
    // Surface the canonical_form too — useful when it differs from
    // original_script (e.g. unvocalized vs vocalized Hebrew) and the
    // lang form array shows it as the lookup key. Inserted AFTER the
    // any-check so the canonical-form node isn't built for skipped
    // rows (and so we don't depend on insertBefore's `null`-on-missing
    // child-index coercion).
    const canonical = document.createElement("span");
    canonical.className = "provenance-canonical";
    canonical.textContent = canonicalForm;
    row.insertBefore(canonical, row.children[1]);
    return row;
}

function _renderCitationsDetail(components) {
    // Returns null if no component carries any scholarly citations —
    // skip the disclosure rather than render an empty 'no citations'
    // affordance that adds noise to a name with all-rando-port morphemes.
    const anyCitations = components.some((c) => Array.isArray(c.citations) && c.citations.length);
    if (!anyCitations) return null;
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Citations";
    details.appendChild(summary);
    const list = document.createElement("ul");
    list.className = "citations";
    for (const c of components) {
        const li = document.createElement("li");
        const usage = document.createElement("strong");
        usage.textContent = c.usage;
        li.appendChild(usage);
        li.appendChild(document.createTextNode(" — "));
        const cites = Array.isArray(c.citations) ? c.citations : [];
        if (cites.length) {
            const wrap = document.createElement("span");
            wrap.className = "citation-list";
            for (let i = 0; i < cites.length; i++) {
                if (i > 0) wrap.appendChild(document.createTextNode(", "));
                const span = document.createElement("span");
                span.className = "citation";
                span.textContent = cites[i];
                wrap.appendChild(span);
            }
            li.appendChild(wrap);
        } else {
            const muted = document.createElement("span");
            muted.className = "muted";
            muted.textContent = "no scholarly citations yet";
            li.appendChild(muted);
        }
        list.appendChild(li);
    }
    details.appendChild(list);
    return details;
}

function _renderResults(body) {
    const list = $("results");
    list.replaceChildren();
    for (const r of body.results) {
        list.appendChild(_renderResultItem(r));
    }
    $("seed").textContent = body.seed;
    $("about").hidden = true;
    _renderLegend(currentGenerator?.legend);
    $("seed-line").hidden = false;
    _updateOutputVisibility();
}

async function _renderError(resp) {
    const list = $("results");
    list.replaceChildren();
    const err = await resp.json().catch(() => ({}));
    const li = document.createElement("li");
    li.textContent = `error: ${err.error || resp.status}${err.detail ? " — " + err.detail : ""}`;
    list.appendChild(li);
    $("seed").textContent = "";
    $("about").hidden = true;
    $("legend").hidden = true;
    $("seed-line").hidden = true;
    _updateOutputVisibility();
}

function _wireShareLink(body, requestParams) {
    // Merge server-echoed parameters with the original request params so values
    // the dispatcher pops before echoing (e.g. count) still make it into the
    // share URL.
    const merged = { ...requestParams, ...(body.parameters || {}) };
    delete merged.seed; // seed is set explicitly below from the resolved value

    $("copy-link").onclick = () => {
        const share = new URL(window.location.href);
        share.searchParams.set("generator", currentGenerator.name);
        share.searchParams.set("seed", body.seed);
        for (const [k, v] of Object.entries(merged)) {
            if (Array.isArray(v)) {
                share.searchParams.delete(k);
                v.forEach((x) => share.searchParams.append(k, x));
            } else {
                share.searchParams.set(k, v);
            }
        }
        navigator.clipboard.writeText(share.toString()).catch((err) => {
            console.error("clipboard write failed:", err);
        });
    };
}

async function roll() {
    const params = readForm();
    const url = new URL(window.location.href);
    const seedFromUrl = url.searchParams.get("seed");
    if (seedFromUrl) params.seed = parseInt(seedFromUrl, 10);

    const requestBody = JSON.stringify(params);
    const bodyHash = await sha256Hex(requestBody);
    const resp = await fetch(`${API_BASE}/${currentGenerator.name}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "x-amz-content-sha256": bodyHash,
        },
        body: requestBody,
    });

    if (!resp.ok) {
        await _renderError(resp);
        // Clear the share-link handler so it can't copy a stale link from a
        // previous successful roll.
        $("copy-link").onclick = null;
        return;
    }
    const body = await resp.json();
    _renderResults(body);

    // Once we've rolled, drop the URL seed so the next roll is fresh unless the
    // user explicitly clicks the share link.
    if (url.searchParams.has("seed")) {
        url.searchParams.delete("seed");
        window.history.replaceState({}, "", url.toString());
    }
    _wireShareLink(body, params);
}

document.addEventListener("DOMContentLoaded", () => {
    loadManifest();
    $("roll").onclick = roll;
});
