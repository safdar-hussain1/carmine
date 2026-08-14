/**
 * The shade card: what is on the face right now, by name.
 *
 * It reads like the label on the back of a palette, and it only lists
 * products whose intensity is above zero -- so it doubles as the honest
 * answer to "what am I actually looking at", which a row of swatches with
 * selected states cannot give at a glance.
 */

import type { LookConfig } from "../engine/look";
import { PRODUCTS } from "./shades";

export interface ShadeCard {
  element: HTMLElement;
  refresh(look: LookConfig): void;
}

export function createShadeCard(look: LookConfig): ShadeCard {
  const root = document.createElement("div");
  root.className = "shade-card";

  const title = document.createElement("p");
  title.className = "shade-card__title";
  title.textContent = "On your face";

  const list = document.createElement("dl");
  list.className = "shade-card__list";

  root.append(title, list);

  function refresh(next: LookConfig): void {
    list.textContent = "";
    const worn = PRODUCTS.filter((product) => next[product.name].intensity > 0);

    if (worn.length === 0) {
      const empty = document.createElement("dd");
      empty.className = "shade-card__empty";
      empty.textContent = "Nothing on. Switch a product on, or tap a look above.";
      list.append(empty);
      return;
    }

    for (const product of worn) {
      const config = next[product.name];
      const shade = product.shades.find(
        (s) => s.hex.toLowerCase() === config.color.toLowerCase(),
      );

      const item = document.createElement("div");
      item.className = "shade-card__item";

      const chip = document.createElement("span");
      chip.className = "shade-card__chip";
      chip.style.background = config.color;

      const term = document.createElement("dt");
      term.className = "shade-card__for";
      term.textContent = product.label;

      const value = document.createElement("dd");
      value.className = "shade-card__name";
      const finish = product.hasFinish ? ` · ${config.finish}` : "";
      value.textContent = `${shade?.name ?? config.color}${finish}`;

      item.append(chip, term, value);
      list.append(item);
    }
  }

  refresh(look);

  return { element: root, refresh };
}
