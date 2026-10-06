/**
 * The controls: a dock beside the mirror, and a drawer with every product.
 *
 * The dock holds what most people reach for -- the four looks and the
 * lipstick range, drawn as lipstick bullets -- plus the mirror's own tools.
 * "All products" opens a drawer with the other five products, one at a
 * time behind tabs. On a wide screen the drawer slides in from the left,
 * over the text, so the face on the right stays in view while you work; on
 * a phone it rises from the bottom under the face.
 *
 * Every control edits one field of the `LookConfig` the mirror renders, and
 * nothing here trusts a copy of its own: it is handed the current look on
 * every refresh and redraws its selected states from it. That is what keeps
 * a look, a bullet, a tab's colour dot and a swatch from ever disagreeing
 * about which shade is on the face.
 *
 * Switching a product off sets its intensity to zero (which is what the
 * engine reads), but the controls remember the intensity it had, so
 * switching back on returns you to your setting rather than to a default.
 */

import { PRESETS, type Finish, type LookConfig, type ProductName } from "../engine/look";
import { ICONS } from "./icons";
import { FINISHES, PRODUCTS, type ProductMeta } from "./shades";

/** What changed, so the page can decide whether to blend or cut. Sliders
 * cut: a drag already moves continuously, and blending would make the face
 * lag behind the thumb. */
export type Change = "look" | "shade" | "toggle" | "finish" | "intensity";

export interface Controls {
  dock: HTMLElement;
  drawer: HTMLElement;
  /** Where the mirror's tool buttons go, at the end of the dock. */
  toolsSlot: HTMLElement;
  refresh(look: LookConfig): void;
  /** Called when the drawer opens and whenever it switches product, so the
   * page can bring that part of the face into view. */
  onDrawerFocus(callback: (product: ProductName) => void): void;
}

interface ControlsOptions {
  look: LookConfig;
  onChange(look: LookConfig, change: Change): void;
}

interface LookMeta {
  name: string;
  label: string;
  note: string;
}

/** Lightest to boldest, each with the line that tells them apart. */
const LOOKS: LookMeta[] = [
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

function sameLook(a: LookConfig, b: LookConfig): boolean {
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

function iconButton(className: string, label: string, icon: string): HTMLButtonElement {
  const node = el("button", className);
  node.type = "button";
  node.innerHTML = `${icon}<span>${label}</span>`;
  return node;
}

/** One shade, drawn as a bullet, a pressed pan or a pencil tip. */
function swatch(kind: "bullet" | "pan" | "pencil", hex: string, name: string, className: string): HTMLButtonElement {
  const node = el("button", `${className} ${className}--${kind}`);
  node.type = "button";
  node.style.setProperty("--swatch", hex);
  node.title = name;
  node.setAttribute("aria-label", name);
  node.setAttribute("aria-pressed", "false");
  node.dataset.hex = hex;
  const art = el("span", `${className}__art`);
  art.setAttribute("aria-hidden", "true");
  node.append(art);
  return node;
}

export function createControls(options: ControlsOptions): Controls {
  let look = clone(options.look);
  /** Intensity to restore when a product is switched back on. */
  const remembered = new Map<ProductName, number>();
  const drawerListeners: Array<(product: ProductName) => void> = [];

  const emit = (change: Change) => {
    options.onChange(clone(look), change);
    refresh(look);
  };

  function pickShade(meta: ProductMeta, hex: string): void {
    look[meta.name].color = hex;
    if (look[meta.name].intensity <= 0) {
      look[meta.name].intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
    }
    emit("shade");
  }

  // ---- the dock ---------------------------------------------------------

  const dock = el("div", "dock");

  const looksGroup = el("div", "dock__group");
  const looksLabel = el("p", "dock__label", "Looks");
  looksLabel.id = "looks-label";
  const presetRow = el("div", "presets");
  presetRow.setAttribute("role", "group");
  presetRow.setAttribute("aria-labelledby", looksLabel.id);
  const presetButtons: Array<{ name: string; node: HTMLButtonElement }> = [];
  for (const meta of LOOKS) {
    const preset = PRESETS[meta.name];
    if (!preset) {
      continue;
    }
    const node = el("button", "preset");
    node.type = "button";
    node.title = meta.note;
    node.setAttribute("aria-pressed", "false");
    const pans = el("span", "preset__pans");
    pans.setAttribute("aria-hidden", "true");
    for (const product of PRODUCTS) {
      if (preset[product.name].intensity <= 0) {
        continue;
      }
      const pan = el("span");
      pan.style.setProperty("--pan", preset[product.name].color);
      pans.append(pan);
    }
    node.append(pans, el("span", "preset__name", meta.label));
    node.addEventListener("click", () => {
      look = clone(preset);
      remembered.clear();
      emit("look");
    });
    presetRow.append(node);
    presetButtons.push({ name: meta.name, node });
  }
  looksGroup.append(looksLabel, presetRow);

  const lipstick = PRODUCTS.find((product) => product.name === "lipstick") as ProductMeta;
  const lipsGroup = el("div", "dock__group");
  const lipsLabel = el("p", "dock__label");
  lipsLabel.id = "lips-label";
  const lipsShade = el("span", "dock__shade");
  lipsLabel.append("Lipstick", lipsShade);
  const bulletRow = el("div", "bullets");
  bulletRow.setAttribute("role", "group");
  bulletRow.setAttribute("aria-labelledby", lipsLabel.id);
  const bullets: HTMLButtonElement[] = [];
  for (const shade of lipstick.shades) {
    const node = swatch("bullet", shade.hex, shade.name, "bullet");
    node.classList.remove("bullet--bullet");
    node.addEventListener("click", () => pickShade(lipstick, shade.hex));
    // Pointing at a bullet previews its name, so the range can be read
    // without committing to anything.
    const preview = () => {
      lipsShade.textContent = shade.name;
      lipsShade.dataset.preview = "true";
    };
    node.addEventListener("pointerenter", preview);
    node.addEventListener("focus", preview);
    node.addEventListener("pointerleave", () => refresh(look));
    node.addEventListener("blur", () => refresh(look));
    bulletRow.append(node);
    bullets.push(node);
  }
  lipsGroup.append(lipsLabel, bulletRow);

  const actions = el("div", "dock__actions");
  const studioBtn = iconButton("dock__studio", "All products", ICONS.sliders);
  studioBtn.setAttribute("aria-expanded", "false");
  studioBtn.setAttribute("aria-controls", "studio");
  const toolsSlot = el("div", "dock__tools");
  actions.append(studioBtn, toolsSlot);

  dock.append(looksGroup, lipsGroup, actions);

  // ---- the drawer ---------------------------------------------------------

  const drawer = el("div", "studio-layer");
  drawer.dataset.open = "false";
  const panel = el("aside", "studio");
  panel.id = "studio";
  panel.setAttribute("aria-labelledby", "studio-title");
  panel.inert = true;

  const head = el("div", "studio__head");
  const title = el("h2", "studio__title", "Every product");
  title.id = "studio-title";
  const closeBtn = el("button", "studio__close");
  closeBtn.type = "button";
  closeBtn.innerHTML = ICONS.close;
  closeBtn.setAttribute("aria-label", "Close every product");
  head.append(title, closeBtn);

  const tabList = el("div", "tabs");
  tabList.setAttribute("role", "tablist");
  tabList.setAttribute("aria-labelledby", title.id);
  panel.append(head, tabList);

  interface ProductControls {
    meta: ProductMeta;
    tab: HTMLButtonElement;
    tabDot: HTMLElement;
    panel: HTMLElement;
    toggle: HTMLButtonElement;
    shadeName: HTMLElement;
    swatches: HTMLButtonElement[];
    slider: HTMLInputElement;
    sliderValue: HTMLOutputElement;
    finishButtons: Array<{ value: Finish; node: HTMLButtonElement }>;
  }

  const controls: ProductControls[] = [];
  let selected: ProductName = PRODUCTS[0].name;

  for (const meta of PRODUCTS) {
    const tab = el("button", "tab");
    tab.type = "button";
    tab.id = `tab-${meta.name}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `panel-${meta.name}`);
    const tabDot = el("span", "tab__dot");
    tabDot.setAttribute("aria-hidden", "true");
    tab.append(tabDot, el("span", "tab__label", meta.label));
    tabList.append(tab);

    const section = el("div", "product");
    section.id = `panel-${meta.name}`;
    section.dataset.product = meta.name;
    section.setAttribute("role", "tabpanel");
    section.setAttribute("aria-labelledby", tab.id);
    section.tabIndex = -1;

    const productHead = el("div", "product__head");
    const shadeName = el("p", "product__shade");
    shadeName.setAttribute("aria-live", "polite");
    const toggle = el("button", "switch");
    toggle.type = "button";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", "false");
    toggle.setAttribute("aria-label", `Wear ${meta.label.toLowerCase()}`);
    productHead.append(shadeName, toggle);

    const swatchRow = el("div", `swatches swatches--${SWATCH_KIND[meta.name]}`);
    swatchRow.setAttribute("role", "group");
    swatchRow.setAttribute("aria-label", `${meta.label} shades`);
    const swatches: HTMLButtonElement[] = [];
    for (const shade of meta.shades) {
      const node = swatch(SWATCH_KIND[meta.name], shade.hex, shade.name, "swatch");
      node.addEventListener("click", () => pickShade(meta, shade.hex));
      const preview = () => {
        shadeName.dataset.preview = "true";
        shadeName.textContent = shade.name;
      };
      node.addEventListener("pointerenter", preview);
      node.addEventListener("focus", preview);
      node.addEventListener("pointerleave", () => refresh(look));
      node.addEventListener("blur", () => refresh(look));
      swatchRow.append(node);
      swatches.push(node);
    }

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
      look[meta.name].intensity = value;
      if (value > 0) {
        remembered.set(meta.name, value);
      }
      emit("intensity");
    });
    sliderRow.append(sliderLabel, slider, sliderValue);

    section.append(productHead, swatchRow, sliderRow);

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
      section.append(segmented);
    }

    section.append(el("p", "product__blurb", meta.blurb));

    toggle.addEventListener("click", () => {
      const on = look[meta.name].intensity > 0;
      if (on) {
        remembered.set(meta.name, look[meta.name].intensity);
        look[meta.name].intensity = 0;
      } else {
        look[meta.name].intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
      }
      emit("toggle");
    });

    tab.addEventListener("click", () => select(meta.name));
    panel.append(section);
    controls.push({
      meta,
      tab,
      tabDot,
      panel: section,
      toggle,
      shadeName,
      swatches,
      slider,
      sliderValue,
      finishButtons,
    });
  }

  panel.append(el("p", "studio__note", "Every change shows on the mirror as you make it."));
  drawer.append(panel);

  function select(name: ProductName, focusTab = false): void {
    const changed = name !== selected;
    selected = name;
    for (const control of controls) {
      const active = control.meta.name === name;
      control.tab.setAttribute("aria-selected", String(active));
      control.tab.tabIndex = active ? 0 : -1;
      control.panel.hidden = !active;
      if (active && focusTab) {
        control.tab.focus();
      }
    }
    if (changed && drawer.dataset.open === "true") {
      for (const listener of drawerListeners) {
        listener(name);
      }
    }
  }

  // Arrow keys move between tabs, as the tab pattern promises; selection
  // follows focus, since showing a panel costs nothing.
  tabList.addEventListener("keydown", (event) => {
    const index = controls.findIndex((control) => control.meta.name === selected);
    let next = index;
    if (event.key === "ArrowRight") {
      next = (index + 1) % controls.length;
    } else if (event.key === "ArrowLeft") {
      next = (index - 1 + controls.length) % controls.length;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = controls.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    select(controls[next].meta.name, true);
  });

  // The drawer does not block the page: the mirror stays live beside it,
  // so it is a panel rather than a modal dialog. Closed, it is inert --
  // out of the tab order and out of reach of assistive technology.
  function setOpen(open: boolean): void {
    drawer.dataset.open = String(open);
    panel.inert = !open;
    studioBtn.setAttribute("aria-expanded", String(open));
    if (open) {
      for (const listener of drawerListeners) {
        listener(selected);
      }
      const active = controls.find((control) => control.meta.name === selected);
      active?.tab.focus({ preventScroll: true });
    } else if (panel.contains(document.activeElement)) {
      studioBtn.focus({ preventScroll: true });
    }
  }

  studioBtn.addEventListener("click", () => setOpen(drawer.dataset.open !== "true"));
  closeBtn.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && drawer.dataset.open === "true") {
      setOpen(false);
    }
  });

  // ---- keeping every control honest --------------------------------------

  function refresh(next: LookConfig): void {
    look = clone(next);

    for (const { name, node } of presetButtons) {
      node.setAttribute("aria-pressed", String(sameLook(look, PRESETS[name])));
    }

    const lips = look.lipstick;
    const lipsOn = lips.intensity > 0;
    for (const bullet of bullets) {
      bullet.setAttribute(
        "aria-pressed",
        String(lipsOn && bullet.dataset.hex?.toLowerCase() === lips.color.toLowerCase()),
      );
    }
    delete lipsShade.dataset.preview;
    lipsShade.textContent = lipsOn
      ? (lipstick.shades.find((s) => s.hex.toLowerCase() === lips.color.toLowerCase())?.name ?? lips.color)
      : "off";

    for (const control of controls) {
      const product = look[control.meta.name];
      const on = product.intensity > 0;
      const shade = control.meta.shades.find(
        (s) => s.hex.toLowerCase() === product.color.toLowerCase(),
      );
      const name = shade?.name ?? product.color;

      control.panel.dataset.on = String(on);
      control.panel.style.setProperty("--product", product.color);
      control.tab.dataset.on = String(on);
      control.tabDot.style.setProperty("--dot", product.color);
      control.tab.setAttribute("aria-label", `${control.meta.label}: ${on ? name : "off"}`);
      control.toggle.setAttribute("aria-checked", String(on));
      delete control.shadeName.dataset.preview;
      control.shadeName.textContent = on ? name : `${control.meta.label} off`;

      for (const node of control.swatches) {
        node.setAttribute(
          "aria-pressed",
          String(node.dataset.hex?.toLowerCase() === product.color.toLowerCase()),
        );
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
    dock,
    drawer,
    toolsSlot,
    refresh,
    onDrawerFocus(callback) {
      drawerListeners.push(callback);
    },
  };
}
