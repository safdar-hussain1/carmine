/**
 * The product rail -- the counter side of the page.
 *
 * Every control here edits one field of the `LookConfig` the mirror renders,
 * and the rail never holds a copy of the look: it is handed the current one
 * on every refresh and redraws its selected states from it. That is what
 * keeps a preset chip, a swatch and the shade card underneath the mirror
 * from ever disagreeing about which shade is on your face.
 *
 * Switching a product off sets its intensity to zero (which is what the
 * engine reads), but the rail remembers the intensity it had so switching
 * back on returns you to your setting rather than to a default.
 */

import { PRESETS, type Finish, type LookConfig, type ProductName } from "../engine/look";
import { FINISHES, PRODUCTS, type ProductMeta } from "./shades";

export interface Rail {
  element: HTMLElement;
  refresh(look: LookConfig): void;
}

interface RailOptions {
  look: LookConfig;
  onChange(look: LookConfig): void;
}

/** Order the preset chips are shown in: lightest look to boldest. */
const PRESET_ORDER = ["bare", "everyday", "velvet", "glass"];

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
  html?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (html !== undefined) {
    node.innerHTML = html;
  }
  return node;
}

export function createRail(options: RailOptions): Rail {
  let look = clone(options.look);
  /** Intensity to restore when a product is switched back on. */
  const remembered = new Map<ProductName, number>();

  const root = el("div", "rail");
  root.setAttribute("aria-label", "Products");

  const emit = () => {
    options.onChange(clone(look));
    refresh(look);
  };

  // ---- preset looks -----------------------------------------------------

  const presetSection = el("section", "rail__section");
  presetSection.append(el("h2", "rail__label", "Looks"));
  const presetGrid = el("div", "presets");
  const presetButtons: Array<{ name: string; node: HTMLButtonElement }> = [];

  for (const name of PRESET_ORDER) {
    const preset = PRESETS[name];
    if (!preset) {
      continue;
    }
    const node = el("button", "preset");
    node.type = "button";
    node.setAttribute("aria-pressed", "false");
    const swatches = el("span", "preset__swatches");
    for (const product of ["lipstick", "eyeshadow", "blush", "highlighter"] as ProductName[]) {
      if (preset[product].intensity <= 0) {
        continue;
      }
      const dot = el("span");
      dot.style.background = preset[product].color;
      swatches.append(dot);
    }
    node.append(swatches, el("span", "preset__name", name));
    node.addEventListener("click", () => {
      look = clone(preset);
      remembered.clear();
      emit();
    });
    presetGrid.append(node);
    presetButtons.push({ name, node });
  }
  presetSection.append(presetGrid);
  root.append(presetSection);

  // ---- products ---------------------------------------------------------

  interface ProductControls {
    meta: ProductMeta;
    section: HTMLElement;
    toggle: HTMLButtonElement;
    shadeLabel: HTMLElement;
    swatches: HTMLButtonElement[];
    slider: HTMLInputElement;
    sliderValue: HTMLElement;
    finishButtons: Array<{ value: Finish; node: HTMLButtonElement }>;
  }

  const controls: ProductControls[] = [];

  for (const meta of PRODUCTS) {
    const section = el("section", "rail__section product");
    section.dataset.product = meta.name;

    const head = el("div", "product__head");
    const heading = el("h3", "product__name", meta.label);
    const shadeLabel = el("span", "product__shade");
    const toggle = el("button", "switch");
    toggle.type = "button";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", "false");
    toggle.setAttribute("aria-label", `Turn ${meta.label.toLowerCase()} on or off`);
    head.append(heading, shadeLabel, toggle);

    const body = el("div", "product__body");

    const swatchRow = el("div", "swatches");
    swatchRow.setAttribute("role", "group");
    swatchRow.setAttribute("aria-label", `${meta.label} shades`);
    const swatches: HTMLButtonElement[] = [];
    for (const shade of meta.shades) {
      const node = el("button", "swatch");
      node.type = "button";
      node.style.setProperty("--shade", shade.hex);
      node.title = shade.name;
      node.setAttribute("aria-label", shade.name);
      node.setAttribute("aria-pressed", "false");
      node.dataset.hex = shade.hex;
      node.addEventListener("click", () => {
        look[meta.name].color = shade.hex;
        if (look[meta.name].intensity <= 0) {
          look[meta.name].intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
        }
        emit();
      });
      swatchRow.append(node);
      swatches.push(node);
    }

    const sliderRow = el("div", "slider");
    const sliderLabel = el("span", undefined, "Intensity");
    const slider = el("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.step = "1";
    slider.setAttribute("aria-label", `${meta.label} intensity`);
    const sliderValue = el("span", "slider__value", "0%");
    slider.addEventListener("input", () => {
      const value = Number(slider.value) / 100;
      look[meta.name].intensity = value;
      if (value > 0) {
        remembered.set(meta.name, value);
      }
      emit();
    });
    sliderRow.append(sliderLabel, slider, sliderValue);

    body.append(swatchRow, sliderRow);

    const finishButtons: Array<{ value: Finish; node: HTMLButtonElement }> = [];
    if (meta.hasFinish) {
      const segmented = el("div", "segmented");
      segmented.setAttribute("role", "group");
      segmented.setAttribute("aria-label", "Lipstick finish");
      for (const finish of FINISHES) {
        const node = el("button");
        node.type = "button";
        node.textContent = finish.label;
        node.title = finish.note;
        node.setAttribute("aria-pressed", "false");
        node.addEventListener("click", () => {
          look[meta.name].finish = finish.value;
          emit();
        });
        segmented.append(node);
        finishButtons.push({ value: finish.value, node });
      }
      body.append(segmented);
    }

    body.append(el("p", "product__blurb", meta.blurb));

    toggle.addEventListener("click", () => {
      const on = look[meta.name].intensity > 0;
      if (on) {
        remembered.set(meta.name, look[meta.name].intensity);
        look[meta.name].intensity = 0;
      } else {
        look[meta.name].intensity = remembered.get(meta.name) ?? meta.defaultIntensity;
      }
      emit();
    });

    section.append(head, body);
    root.append(section);
    controls.push({ meta, section, toggle, shadeLabel, swatches, slider, sliderValue, finishButtons });
  }

  function refresh(next: LookConfig): void {
    look = clone(next);

    for (const { name, node } of presetButtons) {
      node.setAttribute("aria-pressed", String(sameLook(look, PRESETS[name])));
    }

    for (const control of controls) {
      const product = look[control.meta.name];
      const on = product.intensity > 0;
      control.section.dataset.on = String(on);
      control.toggle.setAttribute("aria-checked", String(on));
      const shade = control.meta.shades.find(
        (s) => s.hex.toLowerCase() === product.color.toLowerCase(),
      );
      control.shadeLabel.textContent = on ? (shade?.name ?? product.color) : "off";
      for (const swatch of control.swatches) {
        swatch.setAttribute(
          "aria-pressed",
          String(swatch.dataset.hex?.toLowerCase() === product.color.toLowerCase()),
        );
      }
      const percent = Math.round(product.intensity * 100);
      control.slider.value = String(percent);
      control.sliderValue.textContent = `${percent}%`;
      for (const finish of control.finishButtons) {
        finish.node.setAttribute("aria-pressed", String(finish.value === product.finish));
      }
    }
  }

  refresh(options.look);

  return { element: root, refresh };
}
