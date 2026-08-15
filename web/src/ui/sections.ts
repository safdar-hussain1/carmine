/**
 * The static parts of the page: masthead, the pipeline explainer, privacy,
 * and the footer.
 *
 * The diagrams are inline SVG rather than images for three reasons: they
 * inherit the theme through `currentColor`, they stay sharp at any size, and
 * an image file would be a network request on a page whose whole claim is
 * that it makes none.
 */

import { ICONS } from "./icons";

const REPO_URL = "https://github.com/safdar-hussain1/carmine";

/** Points spaced around an ellipse -- the landmark dots in the first diagram. */
function ellipseDots(cx: number, cy: number, rx: number, ry: number, count: number): string {
  let out = "";
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const x = (cx + Math.cos(angle) * rx).toFixed(1);
    const y = (cy + Math.sin(angle) * ry).toFixed(1);
    out += `<circle cx="${x}" cy="${y}" r="0.9" fill="currentColor" stroke="none"/>`;
  }
  return out;
}

const DIAGRAM_LANDMARKS = `
<svg viewBox="0 0 100 60" fill="none" stroke="currentColor" stroke-width="0.7" stroke-linecap="round">
  <ellipse cx="50" cy="30" rx="16" ry="21" opacity="0.45"/>
  <path d="M38.5 22.5c2.2-1.8 5.4-1.8 7.6 0" opacity="0.7"/>
  <path d="M53.9 22.5c2.2-1.8 5.4-1.8 7.6 0" opacity="0.7"/>
  <ellipse cx="42.3" cy="27.5" rx="4" ry="2.3" opacity="0.7"/>
  <ellipse cx="57.7" cy="27.5" rx="4" ry="2.3" opacity="0.7"/>
  <path d="M42 40.5c4-3.4 12-3.4 16 0-4 4.2-12 4.2-16 0Z" class="accent" opacity="0.9"/>
  ${ellipseDots(50, 40.5, 8.2, 3.4, 18)}
  ${ellipseDots(42.3, 27.5, 4, 2.3, 8)}
  ${ellipseDots(57.7, 27.5, 4, 2.3, 8)}
  <text x="6" y="55" font-size="4" fill="currentColor" stroke="none" opacity="0.75">478 points</text>
</svg>`;

const DIAGRAM_MASKS = `
<svg viewBox="0 0 100 60" fill="none" stroke="currentColor" stroke-width="0.7" stroke-linecap="round">
  <defs>
    <filter id="carmine-feather" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.1"/>
    </filter>
  </defs>
  <path d="M12 30c6-6 20-6 26 0-6 7-20 7-26 0Z" stroke-dasharray="1.6 1.6"/>
  <path d="M62 30c6-6 20-6 26 0-6 7-20 7-26 0Z" class="accent-fill" fill="currentColor" stroke="none" filter="url(#carmine-feather)" opacity="0.85"/>
  <path d="M45 30h8m0 0-2.5-2.5M53 30l-2.5 2.5" opacity="0.7"/>
  <text x="12" y="47" font-size="4" fill="currentColor" stroke="none" opacity="0.75">polygon</text>
  <text x="62" y="47" font-size="4" fill="currentColor" stroke="none" opacity="0.75">feathered</text>
</svg>`;

const DIAGRAM_PIGMENT = `
<svg viewBox="0 0 100 60" fill="none" stroke="currentColor" stroke-width="0.7" stroke-linecap="round">
  <rect x="10" y="12" width="34" height="34" opacity="0.4"/>
  <path d="M10 29h34M27 12v34" opacity="0.4"/>
  <circle cx="31" cy="25" r="1.8" fill="currentColor" stroke="none" opacity="0.7"/>
  <circle cx="39" cy="17" r="2.2" class="accent-fill" fill="currentColor" stroke="none"/>
  <path d="M33 23.5 36.6 20m0 0h-2.6m2.6 0v2.6" class="accent"/>
  <text x="10" y="52" font-size="4" fill="currentColor" stroke="none" opacity="0.75">a·b moves</text>
  <rect x="58" y="16" width="32" height="8" rx="1" opacity="0.4"/>
  <path d="M58 20c3-3 5 3 8 0s5 3 8 0 5 3 8 0 3-1.5 4-2" opacity="0.85"/>
  <rect x="58" y="32" width="32" height="8" rx="1" opacity="0.4"/>
  <path d="M58 36c3-3 5 3 8 0s5 3 8 0 5 3 8 0 3-1.5 4-2" opacity="0.85"/>
  <text x="58" y="52" font-size="4" fill="currentColor" stroke="none" opacity="0.75">L stays put</text>
</svg>`;

const DIAGRAM_PASS = `
<svg viewBox="0 0 100 60" fill="none" stroke="currentColor" stroke-width="0.7" stroke-linecap="round">
  <rect x="6" y="10" width="20" height="14" rx="1.5" opacity="0.55"/>
  <text x="8.5" y="19" font-size="4" fill="currentColor" stroke="none" opacity="0.85">frame</text>
  <rect x="6" y="32" width="20" height="14" rx="1.5" opacity="0.55"/>
  <text x="8.5" y="41" font-size="4" fill="currentColor" stroke="none" opacity="0.85">6 masks</text>
  <path d="M26 17h8l4 8M26 39h8l4-8" opacity="0.6"/>
  <rect x="38" y="19" width="30" height="22" rx="1.5" class="accent"/>
  <text x="41" y="29" font-size="4" class="accent-fill" fill="currentColor" stroke="none">one</text>
  <text x="41" y="35" font-size="4" class="accent-fill" fill="currentColor" stroke="none">shader pass</text>
  <path d="M68 30h6m0 0-2.5-2.5M74 30l-2.5 2.5" opacity="0.6"/>
  <rect x="76" y="19" width="18" height="22" rx="1.5" opacity="0.55"/>
  <text x="78.5" y="32" font-size="4" fill="currentColor" stroke="none" opacity="0.85">mirror</text>
</svg>`;

interface Step {
  title: string;
  body: string;
  art: string;
}

const STEPS: Step[] = [
  {
    title: "Find the face",
    body: "MediaPipe's face landmarker returns 478 points per frame, in video mode, running on wasm bundled with this page. Every measurement that follows is expressed relative to the distance between the eyes, so a webcam frame and a 4000-pixel portrait get makeup in the same relative place.",
    art: DIAGRAM_LANDMARKS,
  },
  {
    title: "Build soft masks",
    body: "Each product is a polygon, a thick line or an ellipse drawn from those points, then feathered. The feather radius is a fraction of the interocular distance, never a pixel count, which is why the edge stays soft as you lean toward the camera instead of turning into a sticker.",
    art: DIAGRAM_MASKS,
  },
  {
    title: "Tint in CIELAB",
    body: "Colour is applied in Lab, where lightness is a separate axis from hue. Chroma moves toward the shade; lightness is only pulled a fraction of the way, capped per product. That is the whole trick: your lip keeps its own highlights and shadow, so tinted skin still reads as skin rather than as a flat sticker.",
    art: DIAGRAM_PIGMENT,
  },
  {
    title: "Paint it in one pass",
    body: "All six products are composited in a single fragment shader invocation per pixel, in the same order the reference engine paints them. No intermediate buffers: on a phone, writing and re-reading a 720p frame six times costs more than all the colour maths put together.",
    art: DIAGRAM_PASS,
  },
];

export function headerHtml(): string {
  return `
  <header class="site-header">
    <div class="wrap site-header__inner">
      <a class="wordmark" href="#top"><span class="wordmark__dot"></span>Carmine</a>
      <nav class="site-nav" aria-label="Sections">
        <a href="#mirror">Mirror</a>
        <a href="#how">How it works</a>
        <a href="#measured">Measured</a>
        <a href="#privacy">Privacy</a>
      </nav>
      <button class="theme-toggle" type="button" id="theme-toggle" aria-label="Switch theme"
        ><span class="theme-toggle__sun">${ICONS.sun}</span><span class="theme-toggle__moon">${ICONS.moon}</span></button
      >
    </div>
  </header>`;
}

export function heroHeadHtml(): string {
  return `
  <div class="hero__head">
    <p class="eyebrow">Live mirror</p>
    <h1>Try a shade on <em>your own face</em>.</h1>
    <p class="lede">
      Not a stock portrait: your camera, your lighting. Carmine tints lips, eyes, brows and cheeks
      while the skin keeps its own texture, and it does the whole thing on your GPU in this tab.
    </p>
  </div>`;
}

export function howItWorksHtml(): string {
  const steps = STEPS.map(
    (step, index) => `
    <article class="step">
      <div class="step__art">${step.art}</div>
      <p class="step__num">${String(index + 1).padStart(2, "0")}</p>
      <h3>${step.title}</h3>
      <p>${step.body}</p>
    </article>`,
  ).join("");

  return `
  <section class="section" id="how">
    <div class="wrap">
      <div class="section__head">
        <p class="eyebrow">How it works</p>
        <h2>Four steps between the camera and the mirror.</h2>
        <p class="lede">
          The same four run for a still photo and for every frame of a live feed. On a live feed
          they run in the few milliseconds between one camera frame and the next.
        </p>
      </div>
      <div class="steps">${steps}</div>
    </div>
  </section>`;
}

export function privacyHtml(): string {
  return `
  <section class="section" id="privacy">
    <div class="wrap">
      <div class="section__head">
        <p class="eyebrow">Privacy</p>
        <h2>Every frame stays on this device.</h2>
        <p class="lede">
          That is a claim about network traffic, so here is how to check it rather than take it on
          trust: after this page has loaded, it makes no requests at all &mdash; not for the model,
          not for fonts, not for analytics, and certainly not for your camera frames.
        </p>
      </div>
      <div class="privacy">
        <article class="privacy__card">
          <h3>What happens to a frame</h3>
          <p>
            It goes from the video element into a GPU texture and back out to the canvas. The only
            data that crosses to JavaScript is 478 landmark coordinates and, for gloss finishes,
            two percentile numbers. Nothing is stored: no frame is written to disk unless you press
            Capture, which saves a PNG through your own browser's download.
          </p>
        </article>
        <article class="privacy__card">
          <h3>How to verify it</h3>
          <ol>
            <li>Open your browser's developer tools and switch to the Network panel.</li>
            <li>Reload this page and let it finish loading.</li>
            <li>Open the mirror, change shades, capture a photo.</li>
            <li>The request list stops growing the moment loading finishes.</li>
          </ol>
          <p>Or the blunter version: switch your network off and use the page anyway.</p>
        </article>
        <article class="privacy__card">
          <h3>What is bundled</h3>
          <p>
            The face landmark model and the wasm runtime that executes it are served from this
            origin instead of a CDN, which is what makes the zero-request claim possible. Type is
            set in system fonts for the same reason &mdash; a web font would be a request to
            somebody else's server on every visit.
          </p>
        </article>
      </div>
    </div>
  </section>`;
}

export function footerHtml(): string {
  return `
  <footer class="site-footer">
    <div class="wrap site-footer__inner">
      <span class="site-footer__mark">Carmine</span>
      <span>MIT licensed</span>
      <a href="${REPO_URL}" rel="noopener">Source on GitHub</a>
      <a href="./DESIGN_CARD.md">Design card</a>
      <a href="./ARCHITECTURE.md">Architecture</a>
      <span>Sample portrait: NASA, public domain</span>
    </div>
  </footer>`;
}
