import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

// wyrd-z3lp: Svelte 5 entry. mount() is the Svelte 5 API (replaces
// `new App({ target })` from earlier versions).
const app = mount(App, {
  target: document.getElementById('app'),
});

export default app;
