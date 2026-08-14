/**
 * Page assembly.
 *
 * One place owns the look; the mirror renders it and the rail edits it. Both
 * are handed a copy on every change, so neither can mutate the other's state
 * behind its back.
 */

import { PRESETS, type LookConfig } from "../engine/look";
import { createMirror } from "./mirror";
import { createRail } from "./rail";
import { createShadeCard } from "./shadeCard";
import { measuredSectionHtml } from "./measured";
import { footerHtml, headerHtml, heroHeadHtml, howItWorksHtml, privacyHtml } from "./sections";
import { initTheme } from "./theme";

/** The look the page opens on. Everyday is the one that reads as makeup
 * rather than as a demo of makeup. */
const OPENING_LOOK = "everyday";

function startingLook(): LookConfig {
  const preset = PRESETS[OPENING_LOOK] ?? Object.values(PRESETS)[0];
  return JSON.parse(JSON.stringify(preset)) as LookConfig;
}

export function mountApp(root: HTMLElement): void {
  let look = startingLook();

  root.innerHTML = `
    ${headerHtml()}
    <main id="top">
      <section class="hero" id="mirror">
        <div class="wrap">
          ${heroHeadHtml()}
          <div class="mirror">
            <div class="mirror__stage"></div>
            <div class="mirror__rail"></div>
          </div>
        </div>
      </section>
      ${howItWorksHtml()}
      ${measuredSectionHtml()}
      ${privacyHtml()}
    </main>
    ${footerHtml()}
  `;

  const mirror = createMirror({ getLook: () => look });
  const shadeCard = createShadeCard(look);
  const rail = createRail({
    look,
    onChange(next) {
      look = next;
      mirror.setLook(next);
      shadeCard.refresh(next);
    },
  });

  mirror.element.append(shadeCard.element);
  root.querySelector(".mirror__stage")?.append(mirror.element);
  root.querySelector(".mirror__rail")?.append(rail.element);

  const toggle = root.querySelector<HTMLButtonElement>("#theme-toggle");
  if (toggle) {
    initTheme(toggle);
  }
}
