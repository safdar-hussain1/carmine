/**
 * The shade bar: one strip of controls along the bottom of the mirror.
 *
 * It works like a camera app's filter strip. A row of tabs -- Looks, then
 * the six products -- picks what the strip below it shows: the four looks
 * as small portraits of the face wearing them, or one product's range drawn
 * as the thing you would buy (lipstick bullets, pressed-powder pans,
 * pencils), with a None choice first, and beside it the shade's name, its
 * intensity and, for lipstick, its finish.
 *
 * Every control edits one field of the `LookConfig` the mirror renders, and
 * nothing here trusts a copy of its own: it is handed the current look on
 * every refresh and redraws its selected states from it. That is what keeps
 * a look, a tab's colour dot and a swatch from ever disagreeing about which
 * shade is on the face.
 *
 * Choosing None sets a product's intensity to zero (which is what the
 * engine reads), but the bar remembers the intensity it had, so picking a
 * shade again returns you to your setting rather than to a default.
 */

import { PRESETS, type Finish, type LookConfig, type ProductName } from "../engine/look";
import { FINISHES, PRODUCTS, type ProductMeta } from "./shades";

/** What changed, so the page can decide whether to animate. Sliders cut: a
 * drag already moves continuously, and a transition would trail the thumb. */
export type Change = "look" | "shade" | "toggle" | "finish" | "intensity";

export interface Controls {
  element: HTMLElement;
  refresh(look: LookConfig): void;
  /** Paints each look's portrait once the page has rendered them. */
  setThumbnails(thumbnails: Record<string, CanvasImageSource>): void;
}

interface ControlsOptions {
  look: LookConfig;
  onChange(look: LookConfig, change: Change): void;
}

export interface LookMeta {
  name: string;
  label: string;
  note: string;
}

/** Lightest to boldest, each with the line that tells them apart. */
export const LOOKS: LookMeta[] = [
  { name: "bare", label: "Bare", note: "A tint on lips and brows" },
  { name: "everyday", label: "Everyday", note: "Soft colour all over" },
  { name: "velvet", label: "Velvet", note: "Deep matte lip and liner" },
  { name: "glass", label: "Glass", note: "Gloss and glow" },
];

/** How each product's shades are drawn: as the thing you would buy. */
const SWATCH_KIND: Record<ProductName, "bullet" | "pan" | "pencil"> = {
  lipstick: "bullet",
  eyeshadow: "pan",
  eyeliner: "pencil",
  brows: "pencil",
  blush: "pan",
  highlighter: "pan",
};

/** The product's name as it reads in "No …". */
const NONE_LABEL: Record<ProductName, string> = {
  lipstick: "No lipstick",
  eyeshadow: "No eyeshadow",
  eyeliner: "No liner",
  brows: "No brow tint",
  blush: "No blush",
  highlighter: "No highlighter",
};

const THUMB_SIZE = 112;

/**
 * How strongly a shade goes on when it is picked from the range, if the
 * product is fainter than that. A range is for seeing shades side by side,
 * and at a preset's light touch neighbouring reds look alike on the face.
 * It never lowers a stronger setting, and never overrides a level the
 * visitor chose with the slider.
 */
const PICK_FLOOR: Record<ProductName, number> = {
  lipstick: 0.75,
  eyeshadow: 0.45,
  eyeliner: 0.7,
  brows: 0.4,
  blush: 0.35,
  highlighter: 0.5,
};

function clone(look: LookConfig): LookConfig {
  return {
    lipstick: { ...look.lipstick },
    eyeshadow: { ...look.eyeshadow },
    eyeliner: { ...look.eyeliner },
    brows: { ...look.brows },
    blush: { ...look.blush },
    highlighter: { ...look.highlighter },
    smoothing: look.smoothing,
  };
}

export function sameLook(a: LookConfig, b: LookConfig): boolean {
  return PRODUCTS.every((product) => {
    const left = a[product.name];
    const right = b[product.name];
    return (
      left.color.toLowerCase() === right.color.toLowerCase() &&
      Math.abs(left.intensity - right.intensity) < 1e-6 &&
      left.finish === right.finish
    );
  });
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

/** One shade, drawn as a bullet, a pressed pan or a pencil tip. */
function swatch(kind: "bullet" | "pan" | "pencil", hex: string, name: string): HTMLButtonElement {
  const node = el("button", `swatch swatch--${kind}`);
  node.type = "button";
  node.style.setProperty("--swatch", hex);
  node.title = name;
  node.setAttribute("aria-label", name);
  node.setAttribute("aria-pressed", "false");
  node.dataset.hex = hex;
  const art = el("span", "swatch__art");
  art.setAttribute("aria-hidden", "true");
  node.append(art);
  return node;
}

/**
 * Arrow keys move between the buttons of a row, and only one of them is in
 * the tab order at a time: sixteen lipsticks are one stop for the keyboard,
 * not sixteen.
 */
function rovingRow(row: HTMLElement, items: HTMLElement[]): (index: number) => void {
  const setCurrent = (index: number) => {
    items.forEach((item, i) => {
      item.tabIndex = i === index ? 0 : -1;
    });
  };
  row.addEventListener("keydown", (event) => {
    const current = items.indexOf(document.activeElement as HTMLElement);
    if (current < 0) {
      return;
    }
    let next = current;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      next = Math.min(items.length - 1, current + 1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      next = Math.max(0, current - 1);
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = items.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    setCurrent(next);
    items[next].focus();
    items[next].scrollIntoView({ block: "nearest", inline: "nearest" });
  });
  setCurrent(0);
  return setCurrent;
}

export function createControls(options: ControlsOptions): Controls {
  let look = clone(options.look);
  /** Intensity to restore when a product is switched back on. */
  const remembered = new Map<ProductName, number>();
  /** Products whose intensity the visitor set with the slider. */
  const chosenByHand = new Set<ProductName>();

  const emit = (change: Change) => {
    options.onChange(clone(look), change);
    refresh(look);
  };

  const root = el("div", "bar");
  root.setAttribute("role", "region");
  root.setAttribute("aria-label", "Makeup");

  const tabList = el("div", "bar__tabs");
  tabList.setAttribute("role", "tablist");
  tabList.setAttribute("aria-label", "Makeup");
  const panels = el("div", "bar__panels");
  root.append(tabList, panels);

  interface Tab {
    key: string;
    tab: HTMLButtonElement;
    panel: HTMLElement;
  }
  const tabs: Tab[] = [];
  let selected = "looks";

  function addTab(key: string, label: string, panel: HTMLElement): HTMLButtonElement {
    const tab = el("button", "tab");
    tab.type = "button";
    tab.id = `tab-${key}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `panel-${key}`);
    panel.id = `panel-${key}`;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tab.id);
    panel.tabIndex = -1;
    const dot = el("span", "tab__dot");
    dot.setAttribute("aria-hidden", "true");
    tab.append(dot, el("span", "tab__label", label));
    tab.addEventListener("click", () => select(key));
    tabList.append(tab);
    panels.append(panel);
    tabs.push({ key, tab, panel });
    return tab;
  }

  // ---- looks ------------------------------------------------------------

  const looksPanel = el("div", "looks");
  const looksRow = el("div", "looks__row");
  looksRow.setAttribute("role", "group");
  looksRow.setAttribute("aria-label", "Looks");
  looksPanel.append(looksRow);
  const presetButtons: Array<{ name: string; node: HTMLButtonElement; thumb: HTMLCanvasElement }> = [];
  for (const meta of LOOKS) {
    const preset = PRESETS[meta.name];
    if (!preset) {
      continue;
    }
    const node = el("button", "preset");
    node.type = "button";
    node.title = meta.note;
    node.setAttribute("aria-pressed", "false");
    const frame = el("span", "preset__frame");
    frame.setAttribute("aria-hidden", "true");
    const thumb = el("canvas", "preset__thumb");
    thumb.width = THUMB_SIZE;
    thumb.height = THUMB_SIZE;
    // Until the portraits are painted, the look's own shades stand in.
    const pans = el("span", "preset__pans");
    for (const product of PRODUCTS) {
      if (preset[product.name].intensity > 0) {
        const pan = el("span");
        pan.style.setProperty("--pan", preset[product.name].color);
        pans.append(pan);
      }
    }
    frame.append(pans, thumb);
    const words = el("span", "preset__words");
    words.append(el("span", "preset__name", meta.label), el("span", "preset__note", meta.note));
    node.append(frame, words);
    node.addEventListener("click", () => {
      look = clone(preset);
      remembered.clear();
      chosenByHand.clear();
      emit("look");
    });
    looksRow.append(node);
    presetButtons.push({ name: meta.name, node, thumb });
  }
  rovingRow(looksRow, presetButtons.map((entry) => entry.node));
  const looksTab = addTab("looks", "Looks", looksPanel);
  looksTab.classList.add("tab--looks");

  // ---- products -----------------------------------------------------------

  interface ProductControls {
    meta: ProductMeta;
    tab: HTMLButtonElement;
    panel: HTMLElement;
    none: HTMLButtonElement;
    shadeName: HTMLElement;
    swatches: HTMLButtonElement[];
    focusShade: (index: number) => void;
    slider: HTMLInputElement;
    sliderValue: HTMLOutputElement;
    finishButtons: Array<{ value: Finish; node: HTMLButtonElement }>;
  }
  const controls: ProductControls[] = [];

  for (const meta of PRODUCTS) {
    const panel = el("div", "product");
    panel.dataset.product = meta.name;

    const row = el("div", `product__shades product__shades--${SWATCH_KIND[meta.name]}`);
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", `${meta.label} shades`);

    const none = el("button", "none");
    none.type = "button";
    none.title = NONE_LABEL[meta.name];
    none.setAttribute("aria-label", NONE_LABEL[meta.name]);
    none.setAttribute("aria-pressed", "false");
    none.addEventListener("click", () => {
      if (look[meta.name].intensity > 0) {
        remembered.set(meta.name, look[meta.name].intensity);
        look[meta.name].intensity = 0;
        emit("toggle");
      }
    });
    row.append(none);

    const side = el("div", "product__side");
    const shadeName = el("p", "product__shade");
    shadeName.setAttribute("aria-live", "polite");

    const swatches: HTMLButtonElement[] = [];
    for (const shade of meta.shades) {
      const node = swatch(SWATCH_KIND[meta.name], shade.hex, shade.name);
      node.addEventListener("click", () => {
        const product = look[meta.name];
        product.color = shade.hex;
        if (product.intensity <= 0) {
          product.intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
        }
        if (!chosenByHand.has(meta.name)) {
          product.intensity = Math.max(product.intensity, PICK_FLOOR[meta.name]);
        }
        emit("shade");
      });
      // Pointing at a shade previews its name where the worn shade's name
      // sits, so the range can be read without committing to anything.
      const preview = () => {
        shadeName.dataset.preview = "true";
        shadeName.textContent = shade.name;
      };
      node.addEventListener("pointerenter", preview);
      node.addEventListener("focus", preview);
      node.addEventListener("pointerleave", () => refresh(look));
      node.addEventListener("blur", () => refresh(look));
      row.append(node);
      swatches.push(node);
    }
    const focusShade = rovingRow(row, [none, ...swatches]);

    const sliderRow = el("div", "slider");
    const sliderId = `intensity-${meta.name}`;
    const sliderLabel = el("label", "slider__label", "Intensity");
    sliderLabel.htmlFor = sliderId;
    const slider = el("input", "slider__input");
    slider.id = sliderId;
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.step = "1";
    const sliderValue = el("output", "slider__value", "0%");
    sliderValue.htmlFor.add(sliderId);
    slider.addEventListener("input", () => {
      const value = Number(slider.value) / 100;
      chosenByHand.add(meta.name);
      look[meta.name].intensity = value;
      if (value > 0) {
        remembered.set(meta.name, value);
      }
      emit("intensity");
    });
    sliderRow.append(sliderLabel, slider, sliderValue);

    side.append(shadeName);
    const finishButtons: Array<{ value: Finish; node: HTMLButtonElement }> = [];
    if (meta.hasFinish) {
      const segmented = el("div", "segmented");
      segmented.setAttribute("role", "group");
      segmented.setAttribute("aria-label", "Lipstick finish");
      for (const finish of FINISHES) {
        const node = el("button", undefined, finish.label);
        node.type = "button";
        node.title = finish.note;
        node.setAttribute("aria-pressed", "false");
        node.addEventListener("click", () => {
          look[meta.name].finish = finish.value;
          emit("finish");
        });
        segmented.append(node);
        finishButtons.push({ value: finish.value, node });
      }
      side.append(segmented);
    }
    side.append(sliderRow);
    panel.append(row, side);

    const tab = addTab(meta.name, meta.label, panel);
    controls.push({
      meta,
      tab,
      panel,
      none,
      shadeName,
      swatches,
      focusShade,
      slider,
      sliderValue,
      finishButtons,
    });
  }

  function select(key: string, focusTab = false): void {
    selected = key;
    for (const entry of tabs) {
      const active = entry.key === key;
      entry.tab.setAttribute("aria-selected", String(active));
      entry.tab.tabIndex = active ? 0 : -1;
      entry.panel.hidden = !active;
      if (active && focusTab) {
        entry.tab.focus();
      }
    }
    root.dataset.tab = key;
  }

  // Arrow keys move between tabs, as the tab pattern promises; selection
  // follows focus, since showing a panel costs nothing.
  tabList.addEventListener("keydown", (event) => {
    const index = tabs.findIndex((entry) => entry.key === selected);
    let next = index;
    if (event.key === "ArrowRight") {
      next = (index + 1) % tabs.length;
    } else if (event.key === "ArrowLeft") {
      next = (index - 1 + tabs.length) % tabs.length;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = tabs.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    select(tabs[next].key, true);
  });

  // ---- keeping every control honest --------------------------------------

  function refresh(next: LookConfig): void {
    look = clone(next);

    for (const { name, node } of presetButtons) {
      node.setAttribute("aria-pressed", String(sameLook(look, PRESETS[name])));
    }

    for (const control of controls) {
      const product = look[control.meta.name];
      const on = product.intensity > 0;
      const shadeIndex = control.meta.shades.findIndex(
        (s) => s.hex.toLowerCase() === product.color.toLowerCase(),
      );
      const name = shadeIndex >= 0 ? control.meta.shades[shadeIndex].name : product.color;

      control.panel.dataset.on = String(on);
      control.panel.style.setProperty("--product", product.color);
      control.tab.dataset.on = String(on);
      control.tab.style.setProperty("--dot", product.color);
      control.tab.setAttribute("aria-label", `${control.meta.label}: ${on ? name : "none"}`);
      control.none.setAttribute("aria-pressed", String(!on));
      delete control.shadeName.dataset.preview;
      control.shadeName.textContent = on ? name : NONE_LABEL[control.meta.name];
      control.swatches.forEach((node, i) => {
        node.setAttribute("aria-pressed", String(on && i === shadeIndex));
      });
      // The keyboard lands on what is worn.
      if (!control.panel.contains(document.activeElement)) {
        control.focusShade(on && shadeIndex >= 0 ? shadeIndex + 1 : 0);
      }
      const percent = Math.round(product.intensity * 100);
      control.slider.value = String(percent);
      control.slider.style.setProperty("--fill", `${percent}%`);
      control.sliderValue.textContent = `${percent}%`;
      for (const finish of control.finishButtons) {
        finish.node.setAttribute("aria-pressed", String(finish.value === product.finish));
      }
    }
  }

  select(selected);
  refresh(options.look);

  return {
    element: root,
    refresh,
    setThumbnails(thumbnails) {
      for (const { name, node, thumb } of presetButtons) {
        const source = thumbnails[name];
        const ctx = thumb.getContext("2d");
        if (!source || !ctx) {
          continue;
        }
        ctx.drawImage(source, 0, 0, thumb.width, thumb.height);
        node.classList.add("preset--painted");
      }
    },
  };
}
