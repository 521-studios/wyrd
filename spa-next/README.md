# wyrd SPA (next)

Svelte 5 + Vite rewrite of the wyrd generator UI. Part of the [wyrd-ga8h](../) SPA epic — built alongside the existing `spa/` so the old SPA stays deployable until the wyrd-20pz cutover.

## Layout

Three columns: **Configure** (params + roll), **Output** (results + saved library), **Inspect & Transform** (morpheme detail + transform pipeline).

## Dev

```bash
# In one terminal: Flask API
cd ..
. .venv/bin/activate
WYRD_SPA_DIR=spa flask --app wyrd.app run

# In another: Vite dev server (proxies /api/* to Flask)
cd spa-next
npm install
npm run dev
# → http://localhost:5173
```

## Build

```bash
npm run build  # → dist/
```

The wyrd-20pz cutover PR will switch `deploy.yml` to upload from `spa-next/dist/` instead of `spa/`.

## PR sequence (see beads)

1. wyrd-z3lp — scaffold + 3-column layout (this PR)
2. wyrd-hcmc — col 1 form + col 2 result list
3. wyrd-yxf6 — col 3 inspector (morpheme detail)
4. wyrd-kppy — col 3 transform pipeline + Rewind
5. wyrd-hpjg — col 3 direct manipulation
6. wyrd-34tn — save / load via localStorage
7. wyrd-jh75 — mobile layout
8. wyrd-tz35 — share-link continuity
9. wyrd-20pz — cutover + delete `spa/`
