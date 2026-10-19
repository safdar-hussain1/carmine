/**
 * The page's markup: masthead, the studio screen around the mirror, the
 * lookbook, how it works, privacy, and the footer.
 *
 * Everything here is plain HTML with no behaviour, which is what lets
 * `vite.config.ts` write the same markup into index.html at build time:
 * the served page already carries its headline, its explanation and its
 * one top-level heading before any script runs. `mountApp` slots the live
 * mirror and the controls into it, so nothing moves when it takes over.
 */

import { FACTS, oneDecimal } from "./facts";
import { ICONS } from "./icons";

const REPO_URL = "https://github.com/safdar-hussain1/carmine";

/** One frame of a 30 fps camera, in milliseconds: the time a live frame's
 * work has to fit in. */
const FRAME_BUDGET_MS = 1000 / 30;

export function headerHtml(): string {
  return `
  <header class="site-header">
    <div class="site-header__inner">
      <a class="wordmark" href="#top" aria-label="Carmine, back to the top"><span class="wordmark__dot" aria-hidden="true"></span>Carmine</a>
      <nav class="site-nav" aria-label="Sections">
        <a href="#looks">Looks</a>
        <a href="#how">How it works</a>
        <a href="#privacy">Privacy</a>
      </nav>
      <div class="site-header__actions">
        <a class="icon-link" href="${REPO_URL}" rel="noopener" aria-label="Source code on GitHub">${ICONS.github}</a>
        <button class="theme-toggle" type="button" id="theme-toggle" aria-label="Switch theme"
          ><span class="theme-toggle__sun">${ICONS.sun}</span><span class="theme-toggle__moon">${ICONS.moon}</span></button
        >
      </div>
    </div>
  </header>`;
}

/** The mirror before the script arrives: the sample, in its place. */
function stagePlaceholderHtml(): string {
  return `
  <div class="stage" data-source="sample" data-drawn="false">
    <img class="stage__poster" src="./demo/model.jpg" alt="The sample portrait the demo runs on" width="1600" height="1067" />
  </div>`;
}

function heroHtml(stage: string, tools: string, bar: string): string {
  return `
    <section class="hero" id="demo" aria-label="Try it">
      <div class="screen">
        <div class="screen__media">${stage}</div>
        <div class="screen__scrim" aria-hidden="true"></div>
        <div class="screen__tools">${tools}</div>
      </div>
      <div class="hero__bar">${bar}</div>
      <div class="hero__copy">
        <h1>Virtual makeup that keeps your skin’s texture.</h1>
        <p class="lede">
          Lipstick to highlighter, painted onto your face live through your camera. The colour
          changes; your skin stays yours. It all runs in this tab, and nothing is uploaded.
        </p>
        <div class="cta">
          <button class="btn btn--primary" type="button" data-action="camera">${ICONS.camera}<span>Try it on your face</span></button>
          <button class="btn btn--glass" type="button" data-action="photo">${ICONS.photo}<span>Use a photo</span></button>
        </div>
        <p class="cta__note">Your browser asks before the camera turns on.</p>
      </div>
    </section>`;
}

export function lookbookHtml(): string {
  return `
  <section class="lookbook" id="looks" aria-labelledby="looks-title">
    <div class="wrap">
      <div class="section-head">
        <h2 id="looks-title">Four looks. One face.</h2>
        <p class="lede">
          Each portrait is painted by the engine, live in your browser, from the same photo. Pick one
          to wear it in the mirror.
        </p>
      </div>
      <div class="lookbook__grid"></div>
    </div>
  </section>`;
}

function shiftBars(): string {
  const flat = FACTS.lipShiftFlat;
  const ours = FACTS.lipShiftCarmine;
  const width = (value: number) => `${((value / flat) * 100).toFixed(1)}%`;
  return `
          <figure class="shift">
            <figcaption>Average change in lip brightness over ${FACTS.portraits} portraits</figcaption>
            <div class="shift__row">
              <span class="shift__label">Flat fill</span>
              <span class="shift__track"><span class="shift__bar" style="--w:${width(flat)}"></span></span>
              <span class="shift__value">${oneDecimal(flat)}</span>
            </div>
            <div class="shift__row shift__row--ours">
              <span class="shift__label">Carmine</span>
              <span class="shift__track"><span class="shift__bar" style="--w:${width(ours)}"></span></span>
              <span class="shift__value">${oneDecimal(ours)}</span>
            </div>
            <p class="shift__note">On the 0&ndash;255 lightness scale. Lower keeps more of the real lip.</p>
          </figure>`;
}

function frameBar(): string {
  const share = (ms: number) => `${((ms / FRAME_BUDGET_MS) * 100).toFixed(1)}%`;
  const stages = [
    { key: "detect", label: "Find the face", ms: FACTS.detectMs },
    { key: "masks", label: "Build the masks", ms: FACTS.masksMs },
    { key: "draw", label: "Paint", ms: FACTS.drawMs },
  ];
  return `
          <figure class="frame">
            <figcaption>A 30 fps camera sends a new frame every ${oneDecimal(FRAME_BUDGET_MS)} ms. All of Carmine’s work fits inside one.</figcaption>
            <div class="frame__track">${stages
              .map((stage) => `<span class="frame__seg frame__seg--${stage.key}" style="--w:${share(stage.ms)}"></span>`)
              .join("")}</div>
            <ul class="frame__legend">${stages
              .map(
                (stage) =>
                  `<li><i class="frame__key frame__key--${stage.key}"></i>${stage.label} <b>${oneDecimal(stage.ms)} ms</b></li>`,
              )
              .join("")}<li><i class="frame__key frame__key--spare"></i>Room to spare</li></ul>
          </figure>`;
}

interface Chapter {
  key: string;
  title: string;
  body: string;
  extra?: string;
}

const CHAPTERS: Chapter[] = [
  {
    key: "landmarks",
    title: "Find the face",
    body: "A face landmark model fits a mesh to your face, with a depth for every point. Everything after this is placed from those points and scaled to the distance between your eyes. Drag the face to turn it.",
    extra: `
          <p class="count"><span class="count__value" data-count="${FACTS.landmarks}">${FACTS.landmarks}</span><span class="count__label">points, found again on every frame</span></p>`,
  },
  {
    key: "masks",
    title: "Shape soft masks",
    body: "Lips, lids, lash line, brows, cheeks and highlights each get a soft-edged shape of their own. The edges soften in proportion to your face, so they stay soft as you lean in.",
  },
  {
    key: "texture",
    title: "Change the colour, keep the texture",
    body: "Every pixel of a lip is a point in colour space: lightness up the middle, hue around it. A flat fill lifts the whole lip toward white. Carmine moves the colour and lets the lightness move only a little, so the lip keeps its own light and shadow.",
    extra: shiftBars(),
  },
  {
    key: "final",
    title: "Paint it in one pass",
    body: `All six products are drawn in a single pass on your graphics card, in the order the sheets were stacked. The whole pipeline takes ${oneDecimal(FACTS.frameMs)} ms a frame on an ${FACTS.gpu} laptop, about ${oneDecimal(FACTS.fps)} frames a second.`,
    extra: frameBar(),
  },
];

export function howItWorksHtml(): string {
  const chapters = CHAPTERS.map(
    (chapter, index) => `
        <li class="chapter" data-step="${chapter.key}">
          <span class="chapter__num">${String(index + 1).padStart(2, "0")}</span>
          <h3 class="chapter__title">${chapter.title}</h3>
          <p class="chapter__body">${chapter.body}</p>${chapter.extra ?? ""}
        </li>`,
  ).join("");

  return `
  <section class="story" id="how" aria-labelledby="how-title">
    <div class="wrap">
      <div class="section-head">
        <h2 id="how-title">How it works</h2>
        <p class="lede">
          The same four steps run on every camera frame, in the time between one frame and the next.
          Every picture here is drawn by the engine, live in your browser, from the sample photo.
        </p>
      </div>
      <div class="story__layout">
        <div class="story__visual">
          <div class="story__stage" data-scene="landmarks">
            <canvas class="scene scene--mesh" role="img" aria-label="The face's landmark mesh in 3D. Drag to turn it."></canvas>
            <div class="scene scene--layers" role="img" aria-label="The look as six sheets, one per product, stacked over the face in the order they are painted."></div>
            <div class="scene scene--cloud">
              <canvas class="cloud__canvas" role="img" aria-label="Every pixel of the lip as a dot in colour space, lightness pointing up."></canvas>
              <div class="cloud__chips" role="group" aria-label="Show the lip">
                <button class="cloud__chip" type="button" data-state="original" aria-pressed="true">Your lip</button>
                <button class="cloud__chip" type="button" data-state="flat" aria-pressed="false">Flat fill</button>
                <button class="cloud__chip" type="button" data-state="carmine" aria-pressed="false">Carmine</button>
              </div>
              <div class="cloud__inset" aria-hidden="true"><canvas data-state="original"></canvas><canvas data-state="flat"></canvas><canvas data-state="carmine"></canvas></div>
            </div>
            <div class="story__progress" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
          </div>
        </div>
        <ol class="story__steps">${chapters}
        </ol>
      </div>
      <p class="story__more">
        The <a href="${REPO_URL}#measured" rel="noopener">full benchmark</a> compares Carmine with four
        common shortcuts on every measure, including the two it loses on.
      </p>
    </div>
  </section>`;
}

export function privacyHtml(): string {
  return `
  <section class="privacy" id="privacy" aria-labelledby="privacy-title">
    <div class="wrap">
      <div class="privacy__head">
        <h2 id="privacy-title">Your face never leaves this tab.</h2>
        <p class="lede">
          No server does the work. The page downloads its files from this site, and from then on
          everything happens on your device.
        </p>
      </div>
      <ul class="privacy__points">
        <li>
          <span class="privacy__icon">${ICONS.device}</span>
          <h3>Runs on your device</h3>
          <p>The face model and the colour maths run in your browser, on your own graphics card.</p>
        </li>
        <li>
          <span class="privacy__icon">${ICONS.noUpload}</span>
          <h3>Nothing is uploaded</h3>
          <p>Camera frames go to the screen and nowhere else. Save photo downloads a PNG to your device, and that is the only copy.</p>
        </li>
        <li>
          <span class="privacy__icon">${ICONS.wifiOff}</span>
          <h3>Check it yourself</h3>
          <p>Once the camera is running, turn off your Wi-Fi. It keeps working, because it no longer needs the network.</p>
        </li>
      </ul>
    </div>
  </section>`;
}

export function footerHtml(): string {
  return `
  <footer class="site-footer">
    <div class="wrap site-footer__inner">
      <p class="site-footer__brand">Carmine</p>
      <nav class="site-footer__links" aria-label="Project">
        <a href="https://github.com/safdar-hussain1" rel="author">Built by Safdar Hussain</a>
        <a href="${REPO_URL}" rel="noopener">Source on GitHub</a>
        <a href="./DESIGN_CARD.md">Design card</a>
        <a href="./ARCHITECTURE.md">Architecture</a>
        <span>MIT licensed</span>
      </nav>
      <p class="site-footer__credit">Sample portrait: photo by Andrea Piacquadio on Pexels</p>
    </div>
  </footer>`;
}

/**
 * The whole page, with anything the app mounts slotted in.
 *
 * Called with no arguments this is the script-free page the build writes
 * into index.html: the sample stands where the mirror will go, and empty,
 * reserved slots stand where its buttons and the shade bar will go.
 */
export function shellHtml(
  stage = stagePlaceholderHtml(),
  tools = '<div class="tools tools--pending"></div>',
  bar = '<div class="bar bar--pending"></div>',
): string {
  return `
  ${headerHtml()}
  <main id="top">
    ${heroHtml(stage, tools, bar)}
    ${lookbookHtml()}
    ${howItWorksHtml()}
    ${privacyHtml()}
  </main>
  ${footerHtml()}`;
}
