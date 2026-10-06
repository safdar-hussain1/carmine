/**
 * Page assembly.
 *
 * One place owns the look; the mirror renders it, the controls edit it, and
 * the story's last chapter shows it. Each is handed a copy on every change,
 * so none can mutate another's state behind its back.
 *
 * The page's accent follows the lipstick on the face: the giant masthead,
 * the glow around the picture, the wordmark's dot and the active controls
 * all take that shade, and a highlight sweeps across the masthead each time
 * it changes, so the page answers each choice as visibly as the mirror does.
 */

import { PRESETS, type LookConfig, type ProductName } from "../engine/look";
import { createControls, type Change } from "./controls";
import { createMirror } from "./mirror";
import { shellHtml } from "./sections";
import { createStory } from "./story";

/** The look the page opens on. Everyday is the one that reads as makeup
 * rather than as a demo of makeup. */
const OPENING_LOOK = "everyday";

/** The accent when no lipstick is on: carmine itself. */
const CARMINE = "#c4123a";

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

export function mountApp(root: HTMLElement): void {
  let look = startingLook();

  // The build writes the whole page into index.html, so it is usually on
  // screen already, entrance animations running. It is kept, not drawn
  // again: only the live parts are slotted into it, so nothing blinks or
  // replays when the script arrives late on a slow connection.
  if (!root.querySelector(".hero")) {
    root.innerHTML = shellHtml();
  }
  const placeholderStage = root.querySelector<HTMLElement>(".hero__stage .stage");
  const poster = placeholderStage?.querySelector<HTMLImageElement>(".stage__poster") ?? null;

  const storySection = root.querySelector<HTMLElement>("#how");
  const story = storySection ? createStory(storySection, look) : null;

  const mirror = createMirror({
    look,
    poster,
    autostart: shouldAutostart(),
    onSampleReady(scene) {
      story?.setScene(scene);
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

  const masthead = root.querySelector<HTMLElement>(".masthead");
  let accent = "";
  const setAccent = (next: LookConfig) => {
    const shade = next.lipstick.intensity > 0 ? next.lipstick.color : CARMINE;
    if (shade === accent) {
      return;
    }
    accent = shade;
    document.documentElement.style.setProperty("--shade", shade);
    masthead?.classList.toggle("is-swept");
  };

  const controls = createControls({
    look,
    onChange(next: LookConfig, change: Change) {
      look = next;
      mirror.setLook(next, change !== "intensity");
      story?.setLook(next);
      setAccent(next);
    },
  });

  placeholderStage?.replaceWith(mirror.stage);
  root.querySelector(".dock--pending")?.replaceWith(controls.dock);
  root.append(controls.drawer);
  controls.toolsSlot.append(mirror.tools);

  // On a stacked layout the drawer rises over the lower half of the screen.
  // Scroll so the part of the face being edited -- lips, eyes, cheeks --
  // sits in the strip still visible above it.
  const stacked = window.matchMedia?.("(max-width: 74.99rem)");
  const header = root.querySelector<HTMLElement>(".site-header");
  controls.onDrawerFocus((product: ProductName) => {
    if (!stacked?.matches) {
      return;
    }
    requestAnimationFrame(() => {
      const sheet = root.querySelector<HTMLElement>(".studio");
      const featureY = mirror.featureY(product);
      if (!sheet || featureY === null) {
        root.querySelector(".hero")?.scrollIntoView({ block: "start", behavior: "smooth" });
        return;
      }
      const top = header?.getBoundingClientRect().height ?? 0;
      const bottom = window.innerHeight - sheet.getBoundingClientRect().height;
      const stageTop = mirror.stage.getBoundingClientRect().top + window.scrollY;
      const target = stageTop + featureY - (top + (bottom - top) * 0.55);
      window.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    });
  });

  root.querySelector<HTMLButtonElement>('[data-action="camera"]')?.addEventListener("click", () => {
    mirror.useCamera();
  });
  root.querySelector<HTMLButtonElement>('[data-action="photo"]')?.addEventListener("click", () => {
    mirror.choosePhoto();
  });

  // The masthead floats over the hero; once the page scrolls it settles
  // onto a solid bar so the links stay readable over the sections below.
  const onScroll = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 24);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  setAccent(look);
}
