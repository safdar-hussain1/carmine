/**
 * The controls beside the mirror: four looks, then one product at a time.
 *
 * Every control edits one field of the `LookConfig` the mirror renders, and
 * the rail never trusts a copy of its own: it is handed the current look on
 * every refresh and redraws its selected states from it. That is what keeps
 * a look, a tab's colour dot and a swatch from ever disagreeing about which
 * shade is on the face.
 *
 * Products sit behind tabs rather than in one long column. Six products
 * with a dozen or more shades each is a lot to take in at once; a tab row
 * that names each product with the shade it is wearing gives the whole
 * look at a glance, and opens one product's choices when you want them.
 *
 * Switching a product off sets its intensity to zero (which is what the
 * engine reads), but the rail remembers the intensity it had, so switching
 * back on returns you to your setting rather than to a default.
 */

import { PRESETS, type Finish, type LookConfig, type ProductName } from "../engine/look";
import { FINISHES, PRODUCTS, type ProductMeta } from "./shades";

/** What changed, so the page can decide whether to blend or cut. Sliders
 * cut: a drag already moves continuously, and blending would make the face
 * lag behind the thumb. */
export type Change = "look" | "shade" | "toggle" | "finish" | "intensity";

export interface Rail {
  element: HTMLElement;
  refresh(look: LookConfig): void;
}

interface RailOptions {
  look: LookConfig;
  onChange(look: LookConfig, change: Change): void;
}

interface LookMeta {
  name: string;
  label: string;
  note: string;
}

/** Lightest to boldest, each with the one line that tells them apart. */
const LOOKS: LookMeta[] = [
  { name: "bare", label: "Bare", note: "A tint on lips and brows" },
  { name: "everyday", label: "Everyday", note: "Soft, all-over colour" },
  { name: "velvet", label: "Velvet", note: "Deep matte lip, liner" },
  { name: "glass", label: "Glass", note: "Gloss and glow" },
];

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

export function createRail(options: RailOptions): Rail {
  let look = clone(options.look);
  /** Intensity to restore when a product is switched back on. */
  const remembered = new Map<ProductName, number>();

  const root = el("div", "rail");

  const emit = (change: Change) => {
    options.onChange(clone(look), change);
    refresh(look);
  };

  // ---- looks ------------------------------------------------------------

  const looksGroup = el("section", "rail__group");
  const looksTitle = el("h2", "rail__title", "Start from a look");
  looksTitle.id = "looks-title";
  looksGroup.setAttribute("aria-labelledby", looksTitle.id);
  const presetGrid = el("div", "presets");
  const presetButtons: Array<{ name: string; node: HTMLButtonElement }> = [];

  for (const meta of LOOKS) {
    const preset = PRESETS[meta.name];
    if (!preset) {
      continue;
    }
    const node = el("button", "preset");
    node.type = "button";
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
    node.append(pans, el("span", "preset__name", meta.label), el("span", "preset__note", meta.note));
    node.addEventListener("click", () => {
      look = clone(preset);
      remembered.clear();
      emit("look");
    });
    presetGrid.append(node);
    presetButtons.push({ name: meta.name, node });
  }
  looksGroup.append(looksTitle, presetGrid);
  root.append(looksGroup);

  // ---- products -----------------------------------------------------------

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

  const productsGroup = el("section", "rail__group");
  const productsTitle = el("h2", "rail__title", "Adjust each product");
  productsTitle.id = "products-title";
  productsGroup.setAttribute("aria-labelledby", productsTitle.id);

  const tabList = el("div", "tabs");
  tabList.setAttribute("role", "tablist");
  tabList.setAttribute("aria-labelledby", productsTitle.id);
  productsGroup.append(productsTitle, tabList);

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

    const panel = el("div", "product");
    panel.id = `panel-${meta.name}`;
    panel.dataset.product = meta.name;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tab.id);
    panel.tabIndex = -1;

    const head = el("div", "product__head");
    const shadeName = el("p", "product__shade");
    shadeName.setAttribute("aria-live", "polite");
    const toggle = el("button", "switch");
    toggle.type = "button";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", "false");
    toggle.setAttribute("aria-label", `Wear ${meta.label.toLowerCase()}`);
    head.append(shadeName, toggle);

    const swatchRow = el("div", "swatches");
    swatchRow.setAttribute("role", "group");
    swatchRow.setAttribute("aria-label", `${meta.label} shades`);
    const swatches: HTMLButtonElement[] = [];
    for (const shade of meta.shades) {
      const node = el("button", "swatch");
      node.type = "button";
      node.style.setProperty("--swatch", shade.hex);
      node.title = shade.name;
      node.setAttribute("aria-label", shade.name);
      node.setAttribute("aria-pressed", "false");
      node.dataset.hex = shade.hex;
      node.addEventListener("click", () => {
        look[meta.name].color = shade.hex;
        if (look[meta.name].intensity <= 0) {
          look[meta.name].intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
        }
        emit("shade");
      });
      // Pointing at a swatch previews its name where the worn shade's name
      // sits, so the range can be read without committing to anything.
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

    panel.append(head, swatchRow, sliderRow);

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
      panel.append(segmented);
    }

    panel.append(el("p", "product__blurb", meta.blurb));

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
    productsGroup.append(panel);
    controls.push({
      meta,
      tab,
      tabDot,
      panel,
      toggle,
      shadeName,
      swatches,
      slider,
      sliderValue,
      finishButtons,
    });
  }
  root.append(productsGroup);

  function select(name: ProductName, focusTab = false): void {
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

  function refresh(next: LookConfig): void {
    look = clone(next);

    for (const { name, node } of presetButtons) {
      node.setAttribute("aria-pressed", String(sameLook(look, PRESETS[name])));
    }

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
      control.tab.setAttribute(
        "aria-label",
        `${control.meta.label}: ${on ? name : "off"}`,
      );
      control.toggle.setAttribute("aria-checked", String(on));
      delete control.shadeName.dataset.preview;
      control.shadeName.textContent = on ? name : `${control.meta.label} off`;

      for (const swatch of control.swatches) {
        swatch.setAttribute(
          "aria-pressed",
          String(swatch.dataset.hex?.toLowerCase() === product.color.toLowerCase()),
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

  return { element: root, refresh };
}
