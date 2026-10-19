/**
 * Page assembly.
 *
 * One place owns the look; the mirror renders it and the shade bar edits it.
 * Each is handed a copy on every change, so neither can mutate the other's
 * state behind its back. The lookbook and the story paint the sample on
 * their own, from the same scene the mirror built.
 *
 * The page's accent follows the lipstick on the face: the light around the
 * studio screen, the wordmark's dot and the active controls all take that
 * shade, so the page answers each choice as visibly as the mirror does.
 */

import { animate, stagger } from "motion";
import { PRESETS, type LookConfig } from "../engine/look";
import { createControls, type Change } from "./controls";
import { createLookbook } from "./lookbook";
import { createMirror } from "./mirror";
import { shellHtml } from "./sections";
import { createStory } from "./story";
import { initTheme } from "./theme";

/** The look the page opens on. Everyday is the one that reads as makeup
 * rather than as a demo of makeup. */
const OPENING_LOOK = "everyday";

/** The accent when no lipstick is on: carmine itself. */
const CARMINE = "#c4123a";

function startingLook(): LookConfig {
  const preset = PRESETS[OPENING_LOOK] ?? Object.values(PRESETS)[0];
  return JSON.parse(JSON.stringify(preset)) as LookConfig;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/**
 * Whether the sample plays without being asked. It costs nothing to
 * download beyond the page itself -- its landmarks ship with the page -- so
 * it always does, except during the selftest, whose timing check must not
 * share the page with an animation.
 */
function shouldAutostart(): boolean {
  return new URLSearchParams(window.location.search).get("selftest") !== "1";
}

export function mountApp(root: HTMLElement): void {
  let look = startingLook();

  // The build writes the whole page into index.html, so it is usually on
  // screen already. It is kept, not drawn again: only the live parts are
  // slotted into it, so nothing blinks when the script arrives.
  if (!root.querySelector(".hero")) {
    root.innerHTML = shellHtml();
  }
  const placeholderStage = root.querySelector<HTMLElement>(".screen__media .stage");
  const poster = placeholderStage?.querySelector<HTMLImageElement>(".stage__poster") ?? null;

  const storySection = root.querySelector<HTMLElement>("#how");
  const story = storySection ? createStory(storySection) : null;

  let controls: ReturnType<typeof createControls> | null = null;
  const lookbookSection = root.querySelector<HTMLElement>("#looks");
  const lookbook = lookbookSection
    ? createLookbook(lookbookSection, (name) => {
        const preset = PRESETS[name];
        if (!preset) {
          return;
        }
        look = JSON.parse(JSON.stringify(preset)) as LookConfig;
        controls?.refresh(look);
        setAccent(look);
        root.querySelector(".hero")?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth" });
        // The sweep plays once the mirror is back on screen.
        window.setTimeout(() => mirror.setLook(look, true), prefersReducedMotion() ? 0 : 450);
      })
    : null;

  // On a phone the buttons sit below the screen, so a camera or a photo
  // could start out of sight; bring the screen back unless most of it shows.
  const revealScreen = () => {
    const screen = root.querySelector<HTMLElement>(".screen");
    if (!screen) {
      return;
    }
    const rect = screen.getBoundingClientRect();
    const showing = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
    if (showing < rect.height * 0.6) {
      screen.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    }
  };

  const mirror = createMirror({
    look,
    poster,
    autostart: shouldAutostart(),
    onSampleReady(scene) {
      story?.setScene(scene);
      // The four looks are painted once and shared by the lookbook and the
      // thumbnails in the shade bar.
      const thumbs = lookbook?.render(scene);
      if (thumbs) {
        controls?.setThumbnails(thumbs);
      }
    },
    onSourceChange(source) {
      root.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((node) => {
        const pressed =
          (node.dataset.action === "camera" && source === "camera") ||
          (node.dataset.action === "photo" && source === "photo");
        node.dataset.active = String(pressed);
      });
      if (source !== "sample") {
        revealScreen();
      }
    },
  });

  let accent = "";
  const setAccent = (next: LookConfig) => {
    const shade = next.lipstick.intensity > 0 ? next.lipstick.color : CARMINE;
    if (shade !== accent) {
      accent = shade;
      document.documentElement.style.setProperty("--shade", shade);
    }
  };

  controls = createControls({
    look,
    onChange(next: LookConfig, change: Change) {
      look = next;
      mirror.setLook(next, change !== "intensity");
      setAccent(next);
    },
  });

  placeholderStage?.replaceWith(mirror.stage);
  root.querySelector(".tools--pending")?.replaceWith(mirror.tools);
  root.querySelector(".bar--pending")?.replaceWith(controls.element);

  const cameraButton = root.querySelector<HTMLButtonElement>('[data-action="camera"]');
  const photoButton = root.querySelector<HTMLButtonElement>('[data-action="photo"]');
  cameraButton?.addEventListener("click", () => mirror.useCamera());
  photoButton?.addEventListener("click", () => mirror.choosePhoto());
  // Reaching for either button starts the face model downloading, so the
  // camera or the photo is ready sooner when the click lands.
  for (const button of [cameraButton, photoButton]) {
    button?.addEventListener("pointerenter", () => mirror.warmUp(), { once: true });
    button?.addEventListener("focus", () => mirror.warmUp(), { once: true });
  }

  // The masthead floats over the page; once it scrolls it settles onto a
  // solid bar so the links stay readable over the sections below.
  const header = root.querySelector<HTMLElement>(".site-header");
  const onScroll = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 16);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const themeToggle = root.querySelector<HTMLButtonElement>("#theme-toggle");
  if (themeToggle) {
    initTheme(themeToggle);
  }

  setAccent(look);

  // One entrance for the screen's furniture, while the face assembles
  // behind it: the headline and buttons rise in, then the shade bar.
  if (!prefersReducedMotion()) {
    const copy = root.querySelectorAll(".hero__copy > *");
    animate(copy, { opacity: [0, 1], y: [18, 0] }, { duration: 0.9, delay: stagger(0.08, { startDelay: 0.25 }), ease: [0.22, 1, 0.36, 1] });
    animate(controls.element, { opacity: [0, 1], y: [24, 0] }, { duration: 0.9, delay: 0.7, ease: [0.22, 1, 0.36, 1] });
    animate(mirror.tools, { opacity: [0, 1] }, { duration: 0.8, delay: 0.9 });
  }
}
