import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

import { shellHtml } from "./src/ui/sections";

const APP_ROOT = '<div id="app"></div>';

/**
 * Writes the script-free part of the page -- masthead, headline, lede -- into
 * index.html, so the served HTML carries the page's h1 heading before any
 * script runs. The markup comes from the same `shellHtml()` the app mounts,
 * so the two cannot drift apart.
 */
function staticShell(): Plugin {
  return {
    name: "carmine-static-shell",
    transformIndexHtml(html) {
      if (!html.includes(APP_ROOT)) {
        throw new Error(`index.html has no empty ${APP_ROOT} to render the static shell into`);
      }
      return html.replace(APP_ROOT, `<div id="app">${shellHtml()}</div>`);
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [staticShell()],
  build: {
    outDir: "../docs",
    emptyOutDir: true,
  },
  test: {
    passWithNoTests: true,
  },
});
