/**
 * Page assembly.
 *
 * One place owns the look; the mirror renders it, the rail edits it, and
 * the explainer's last frame shows it. Each is handed a copy on every
 * change, so none can mutate another's state behind its back.
 *
 * The page's accent follows the lipstick on the face: the mirror's glow,
 * the wordmark's dot and the active controls all take that shade, so the
 * page answers each choice as visibly as the mirror does.
 */

import { PRESETS, type LookConfig } from "../engine/look";
import { createExplainer } from "./explainer";
import { createMirror } from "./mirror";
import { createRail, type Change } from "./rail";
import { shellHtml } from "./sections";
import { initTheme } from "./theme";

/** The look the page opens on. Everyday is the one that reads as makeup
 * rather than as a demo of makeup. */
const OPENING_LOOK = "everyday";

/** The accent when no lipstick is on: carmine itself. */
const CARMINE = "#960018";

function startingLook(): LookConfig {
  const preset = PRESETS[OPENING_LOOK] ?? Object.values(PRESETS)[0];
  return JSON.parse(JSON.stringify(preset)) as LookConfig;
}

/**
 * Whether the sample runs without being asked. It costs the face model and
 * its runtime, about 15 MB, so it waits for a click when the reader has
 * asked the browser to save data or the connection is 2G-slow -- and never
 * during the selftest, whose timing check must not share the page with a
 * second render loop. A "3g" estimate is not enough on its own: Chrome
 * derives it mostly from round-trip time, so a fast link with high latency
 * is labelled 3g too.
 */
function shouldAutostart(): boolean {
  if (new URLSearchParams(window.location.search).get("selftest") === "1") {
    return false;
  }
  const connection = (navigator as Navigator & {
    connection?: { saveData?: boolean; effectiveType?: string };
  }).connection;
  if (connection?.saveData) {
    return false;
  }
  return !/^(slow-2g|2g)$/.test(connection?.effectiveType ?? "");
}

function setAccent(look: LookConfig): void {
  const shade = look.lipstick.intensity > 0 ? look.lipstick.color : CARMINE;
  document.documentElement.style.setProperty("--shade", shade);
}

export function mountApp(root: HTMLElement): void {
  let look = startingLook();

  root.innerHTML = shellHtml(
    '<div class="studio-slot"></div>',
    '<div class="rail-slot"></div>',
  );

  const howSection = root.querySelector<HTMLElement>("#how");
  const explainer = howSection ? createExplainer(howSection, look) : null;

  const mirror = createMirror({
    look,
    autostart: shouldAutostart(),
    onSampleReady(scene) {
      explainer?.setScene(scene);
    },
    onSourceChange(source) {
      root.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((node) => {
        const pressed =
          (node.dataset.action === "camera" && source === "camera") ||
          (node.dataset.action === "photo" && source === "photo");
        node.dataset.active = String(pressed);
      });
    },
  });

  const rail = createRail({
    look,
    onChange(next: LookConfig, change: Change) {
      look = next;
      mirror.setLook(next, change !== "intensity");
      explainer?.setLook(next);
      setAccent(next);
    },
  });

  root.querySelector(".studio-slot")?.replaceWith(mirror.element);
  root.querySelector(".rail-slot")?.replaceWith(rail.element);

  root.querySelector<HTMLButtonElement>('[data-action="camera"]')?.addEventListener("click", () => {
    mirror.useCamera();
  });
  root.querySelector<HTMLButtonElement>('[data-action="photo"]')?.addEventListener("click", () => {
    mirror.choosePhoto();
  });

  setAccent(look);

  const toggle = root.querySelector<HTMLButtonElement>("#theme-toggle");
  if (toggle) {
    initTheme(toggle);
  }
}
