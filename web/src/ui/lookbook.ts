/**
 * The lookbook: the four looks, painted on the same face, side by side.
 *
 * Each card is the engine's own render of the sample in one look, made in
 * the browser as soon as the sample is ready, before the opening animation
 * starts, so they never interrupt it. The same renders give the look
 * thumbnails in the shade bar, so they are made once and shared. Choosing a
 * card puts that look on the mirror.
 */

import { PRESETS } from "../engine/look";
import { LOOKS } from "./controls";
import { cropTo, faceCrop, renderLook } from "./offscreen";
import type { PhotoScene } from "./scene";
import { PRODUCTS } from "./shades";

/** Width over height of a lookbook card's picture. */
const CARD_ASPECT = 4 / 5;
/** Size of a look thumbnail in the shade bar, in canvas pixels. */
const THUMB = 160;

export interface Lookbook {
  /** Renders every look once and returns the thumbnails for the bar. */
  render(scene: PhotoScene): Record<string, HTMLCanvasElement>;
}

export function createLookbook(section: HTMLElement, onPick: (name: string) => void): Lookbook {
  const grid = section.querySelector<HTMLElement>(".lookbook__grid");
  const cards = new Map<string, HTMLCanvasElement>();
  for (const meta of LOOKS) {
    const preset = PRESETS[meta.name];
    if (!grid || !preset) {
      continue;
    }
    const card = document.createElement("button");
    card.type = "button";
    card.className = "look-card";
    card.dataset.look = meta.name;
    const frame = document.createElement("span");
    frame.className = "look-card__frame";
    const canvas = document.createElement("canvas");
    canvas.className = "look-card__canvas";
    frame.append(canvas);
    const body = document.createElement("span");
    body.className = "look-card__body";
    const name = document.createElement("span");
    name.className = "look-card__name";
    name.textContent = meta.label;
    const note = document.createElement("span");
    note.className = "look-card__note";
    note.textContent = meta.note;
    const pans = document.createElement("span");
    pans.className = "look-card__pans";
    pans.setAttribute("aria-hidden", "true");
    for (const product of PRODUCTS) {
      if (preset[product.name].intensity > 0) {
        const pan = document.createElement("span");
        pan.style.setProperty("--pan", preset[product.name].color);
        pans.append(pan);
      }
    }
    const action = document.createElement("span");
    action.className = "visually-hidden";
    action.textContent = ", wear it in the mirror";
    body.append(name, note, pans, action);
    card.append(frame, body);
    card.addEventListener("click", () => onPick(meta.name));
    grid.append(card);
    cards.set(meta.name, canvas);
  }

  return {
    render(scene) {
      const thumbs: Record<string, HTMLCanvasElement> = {};
      const portrait = faceCrop(scene, CARD_ASPECT, 1.25);
      const square = faceCrop(scene, 1, 1.05);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const meta of LOOKS) {
        const preset = PRESETS[meta.name];
        if (!preset) {
          continue;
        }
        const painted = renderLook(scene, preset);
        const canvas = cards.get(meta.name);
        if (canvas) {
          const width = Math.round(Math.max(canvas.getBoundingClientRect().width, 240) * dpr);
          const height = Math.round(width / CARD_ASPECT);
          canvas.width = width;
          canvas.height = height;
          canvas.getContext("2d")?.drawImage(cropTo(painted, portrait, width, height), 0, 0);
        }
        thumbs[meta.name] = cropTo(painted, square, THUMB, THUMB);
      }
      section.dataset.drawn = "true";
      return thumbs;
    },
  };
}
