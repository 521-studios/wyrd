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
    select.innerHTML = "";
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

function _buildField(key, prop, url) {
    const urlVal = url.searchParams.get(key);
    if (prop.type === "string" && Array.isArray(prop.enum)) {
        return _buildSelectField(key, prop, urlVal);
    }
    if (prop.type === "array" && prop.items?.enum) {
        return _buildTagGrid(key, prop, url);
    }
    if (prop.type === "integer" || prop.type === "number") {
        return _buildNumberField(key, prop, urlVal);
    }
    return _buildTextField(key, prop, urlVal);
}

function renderForm(schema) {
    const form = $("params-form");
    form.innerHTML = "";
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
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Components";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(r.components, null, 2);
    details.appendChild(summary);
    details.appendChild(pre);
    li.appendChild(name);
    li.appendChild(explanation);
    li.appendChild(details);
    return li;
}

function _renderResults(body) {
    const list = $("results");
    list.innerHTML = "";
    for (const r of body.results) {
        list.appendChild(_renderResultItem(r));
    }
    $("seed").textContent = body.seed;
    $("output").hidden = false;
}

async function _renderError(resp) {
    const list = $("results");
    list.innerHTML = "";
    const err = await resp.json().catch(() => ({}));
    const li = document.createElement("li");
    li.textContent = `error: ${err.error || resp.status}${err.detail ? " — " + err.detail : ""}`;
    list.appendChild(li);
    $("seed").textContent = "";
    $("output").hidden = false;
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
