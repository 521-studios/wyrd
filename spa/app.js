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

function renderForm(schema) {
    const form = $("params-form");
    form.innerHTML = "";
    if (!schema || !schema.properties) return;

    const url = new URL(window.location.href);

    for (const [key, prop] of Object.entries(schema.properties)) {
        if (key === "seed") continue; // exposed via the share link, not the form
        const wrap = document.createElement("div");
        wrap.className = "field";

        const label = document.createElement("label");
        label.textContent = prop.description ? `${key} — ${prop.description}` : key;
        label.htmlFor = `field-${key}`;
        wrap.appendChild(label);

        let input;
        const urlVal = url.searchParams.get(key);
        if (prop.type === "string" && Array.isArray(prop.enum)) {
            input = document.createElement("select");
            input.id = `field-${key}`;
            input.name = key;
            for (const v of prop.enum) {
                const opt = document.createElement("option");
                opt.value = v;
                opt.textContent = v;
                input.appendChild(opt);
            }
            input.value = urlVal || prop.default || prop.enum[0];
            wrap.appendChild(input);
        } else if (prop.type === "array" && prop.items?.enum) {
            input = document.createElement("div");
            input.className = "tag-grid";
            input.dataset.field = key;
            const selected = new Set(
                urlVal ? url.searchParams.getAll(key) : prop.default || []
            );
            for (const v of prop.items.enum) {
                const tagLabel = document.createElement("label");
                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.name = key;
                cb.value = v;
                cb.checked = selected.has(v);
                tagLabel.appendChild(cb);
                tagLabel.appendChild(document.createTextNode(v));
                input.appendChild(tagLabel);
            }
            wrap.appendChild(input);
        } else if (prop.type === "integer" || prop.type === "number") {
            input = document.createElement("input");
            input.type = "number";
            input.id = `field-${key}`;
            input.name = key;
            if (urlVal !== null) input.value = urlVal;
            else if (prop.default !== undefined) input.value = prop.default;
            wrap.appendChild(input);
        } else {
            input = document.createElement("input");
            input.type = "text";
            input.id = `field-${key}`;
            input.name = key;
            if (urlVal !== null) input.value = urlVal;
            else if (prop.default !== undefined) input.value = prop.default;
            wrap.appendChild(input);
        }

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
        const err = await resp.json().catch(() => ({}));
        $("result").textContent = `error: ${err.error || resp.status}`;
        $("explanation").textContent = err.detail || "";
        $("output").hidden = false;
        return;
    }
    const body = await resp.json();
    $("result").textContent = body.result;
    $("explanation").textContent = body.explanation || "";
    $("components").textContent = JSON.stringify(body.components, null, 2);
    $("seed").textContent = body.seed;
    $("output").hidden = false;

    // Once we've rolled, drop the URL seed so the next roll is fresh unless the
    // user explicitly clicks the share link.
    if (url.searchParams.has("seed")) {
        url.searchParams.delete("seed");
        window.history.replaceState({}, "", url.toString());
    }
    $("copy-link").onclick = () => {
        const share = new URL(window.location.href);
        share.searchParams.set("generator", currentGenerator.name);
        share.searchParams.set("seed", body.seed);
        for (const [k, v] of Object.entries(body.parameters || {})) {
            if (Array.isArray(v)) {
                share.searchParams.delete(k);
                v.forEach((x) => share.searchParams.append(k, x));
            } else {
                share.searchParams.set(k, v);
            }
        }
        navigator.clipboard.writeText(share.toString());
    };
}

document.addEventListener("DOMContentLoaded", () => {
    loadManifest();
    $("roll").onclick = roll;
});
