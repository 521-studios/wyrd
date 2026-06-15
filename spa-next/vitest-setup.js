// wyrd-200v: vitest setup — adds the @testing-library/jest-dom matchers
// (toBeInTheDocument, toHaveClass, toHaveAttribute, …) to expect for the
// component tests. Harmless for the pure-logic unit tests (they just don't use
// the DOM matchers). Auto-cleanup between component tests is handled by the
// svelteTesting() plugin in vite.config.js.
import '@testing-library/jest-dom/vitest';
