/**
 * The Measured section, drawn from `reports/benchmark.json`.
 *
 * The JSON is imported, so it is baked into the bundle at build time: the
 * numbers on the page and the numbers in the repository cannot drift apart,
 * and the section costs no network request (which the privacy claim needs).
 *
 * Chart decisions worth stating, because they are the ones that make a
 * comparison honest rather than flattering:
 *
 * - **Two colours, not five.** Method is an identity, but the question the
 *   reader has is "how does this one compare to the standard ways of getting
 *   it wrong", so carmine's bar is carmine and every baseline is the same
 *   neutral. Five categorical hues would imply five things worth telling
 *   apart from each other, and would put colour where a label already is.
 * - **Every panel shows the spread**, min to max across the 26 images, as a
 *   whisker over the bar. A mean alone hides the case where a method is
 *   excellent on average and catastrophic on one face.
 * - **Direction is labelled per panel**, never implied by bar length, since
 *   three of these seven metrics are better when smaller.
 * - **Carmine loses two of these panels** and they are shown at the same
 *   size as the rest. `lip_detail_retention` and `ms_per_image` are in here
 *   because leaving them out would be the dishonest choice.
 */

import benchmark from "../../../reports/benchmark.json";

interface Row {
  method: string;
  spread: Record<string, [number, number]>;
  [key: string]: unknown;
}

interface MetricSpec {
  key: string;
  title: string;
  what: string;
  better: "higher" | "lower";
  format: (value: number) => string;
  /** Fixed upper bound, when the metric has a natural one. */
  max?: number;
}

const OURS = "carmine";

/** Display names: the JSON keys are identifiers, not English. */
const METHOD_LABELS: Record<string, string> = {
  carmine: "carmine",
  opaque_fill: "opaque fill",
  channel_swap: "channel swap",
  mismatched_indices: "wrong indices",
  untrained_gan: "untrained GAN",
};

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const ratio = (value: number) => value.toFixed(3);
const units = (value: number) => value.toFixed(1);
const ms = (value: number) => `${value.toFixed(0)} ms`;

const METRICS: MetricSpec[] = [
  {
    key: "pigment_on_target",
    title: "Pigment on target",
    what: "Share of the pixel change that lands inside the product masks.",
    better: "higher",
    format: percent,
    max: 1,
  },
  {
    key: "background_untouched",
    title: "Background untouched",
    what: "Pixels outside the face left bit-identical to the input.",
    better: "higher",
    format: percent,
    max: 1,
  },
  {
    key: "lip_texture_kept",
    title: "Lip texture kept",
    what: "Correlation of lip lightness before and after, in Lab.",
    better: "higher",
    format: ratio,
    max: 1,
  },
  {
    key: "lip_detail_retention",
    title: "Lip detail retention",
    what: "High-frequency lightness variance surviving inside the lip.",
    better: "higher",
    format: ratio,
  },
  {
    key: "lip_luminance_shift",
    title: "Lip lightness shift",
    what: "How far mean lip brightness moves, in Lab-L units.",
    better: "lower",
    format: units,
  },
  {
    key: "identity_ssim",
    title: "Whole-image similarity",
    what: "Structural similarity of the output to the input.",
    better: "higher",
    format: ratio,
    max: 1,
  },
  {
    key: "ms_per_image",
    title: "CPU reference time",
    what: "Milliseconds per image for the Python reference, not the browser shader.",
    better: "lower",
    format: ms,
  },
];

const ROW_HEIGHT = 26;
const CHART_WIDTH = 320;
const PLOT_X0 = 92;
const PLOT_X1 = 262;
const BAR_HEIGHT = 8;

/**
 * A bar from the baseline at `zero` out to `value`, with a rounded data end
 * and a square baseline end. Bars run left when the value is negative --
 * `lip_texture_kept` has one genuinely negative row, and a metric that can
 * go below zero must be able to show it.
 */
function barPath(zero: number, value: number, y: number, height: number): string {
  const width = Math.abs(value - zero);
  if (width <= 0.5) {
    return `M${zero - 0.25} ${y}h0.5v${height}h-0.5Z`;
  }
  const r = Math.min(height / 2, width);
  const straight = width - r;
  if (value >= zero) {
    return `M${zero} ${y}h${straight}a${r} ${r} 0 0 1 0 ${height}h${-straight}Z`;
  }
  return `M${zero} ${y}h${-straight}a${r} ${r} 0 0 0 0 ${height}h${straight}Z`;
}

function sortedRows(): Row[] {
  const rows = (benchmark.photo.rows as unknown as Row[]).slice();
  // Carmine first: the reader is comparing everything else against it.
  rows.sort((a, b) => (a.method === OURS ? -1 : b.method === OURS ? 1 : 0));
  return rows;
}

function panel(metric: MetricSpec, rows: Row[]): string {
  const values = rows.map((row) => row[metric.key] as number);
  const lows = rows.map((row) => row.spread[metric.key]?.[0] ?? (row[metric.key] as number));
  const highs = rows.map((row) => row.spread[metric.key]?.[1] ?? (row[metric.key] as number));
  const domainMin = Math.min(0, ...values, ...lows);
  const domainMax = metric.max ?? Math.max(...values, ...highs);
  const span = domainMax - domainMin || 1;
  const scale = (value: number) =>
    PLOT_X0 + ((Math.min(Math.max(value, domainMin), domainMax) - domainMin) / span) * (PLOT_X1 - PLOT_X0);
  const zero = scale(Math.max(domainMin, 0));

  const height = rows.length * ROW_HEIGHT + 6;
  const marks = rows
    .map((row, index) => {
      const ours = row.method === OURS;
      const value = row[metric.key] as number;
      const y = index * ROW_HEIGHT + 8;
      const barY = y + (ROW_HEIGHT - BAR_HEIGHT) / 2 - 6;
      const centre = barY + BAR_HEIGHT / 2;
      const x = scale(value);
      const spread = row.spread[metric.key];
      const whisker =
        spread && spread[1] - spread[0] > 1e-9
          ? `<path class="bar-spread" d="M${scale(spread[0])} ${centre - 4}v8M${scale(spread[0])} ${centre}H${scale(spread[1])}M${scale(spread[1])} ${centre - 4}v8" fill="none"/>`
          : "";
      return [
        `<text class="bar-label${ours ? " bar-label--ours" : ""}" x="0" y="${centre + 4}">${METHOD_LABELS[row.method] ?? row.method}</text>`,
        `<rect class="bar-track" x="${PLOT_X0}" y="${barY}" width="${PLOT_X1 - PLOT_X0}" height="${BAR_HEIGHT}" rx="${BAR_HEIGHT / 2}"/>`,
        `<path class="bar-fill${ours ? " bar-fill--ours" : ""}" d="${barPath(zero, x, barY, BAR_HEIGHT)}"/>`,
        whisker,
        `<text class="bar-value${ours ? " bar-value--ours" : ""}" x="${CHART_WIDTH}" y="${centre + 4}" text-anchor="end">${metric.format(value)}</text>`,
      ].join("");
    })
    .join("");

  return `
    <figure class="panel">
      <div class="panel__title">
        <h3>${metric.title}</h3>
        <span class="panel__dir">${metric.better} is better</span>
      </div>
      <p class="panel__what">${metric.what}</p>
      <svg viewBox="0 0 ${CHART_WIDTH} ${height}" role="img" aria-label="${metric.title}: ${rows
        .map((row) => `${METHOD_LABELS[row.method] ?? row.method} ${metric.format(row[metric.key] as number)}`)
        .join(", ")}">${marks}</svg>
    </figure>`;
}

function dataTable(rows: Row[]): string {
  const head = METRICS.map((metric) => `<th scope="col">${metric.title}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr data-ours="${row.method === OURS}"><th scope="row">${METHOD_LABELS[row.method] ?? row.method}</th>${METRICS.map(
          (metric) =>
            `<td>${metric.format(row[metric.key] as number)}<br><span style="opacity:.6">${
              row.spread[metric.key]
                ? `${metric.format(row.spread[metric.key][0])} – ${metric.format(row.spread[metric.key][1])}`
                : "&mdash;"
            }</span></td>`,
        ).join("")}</tr>`,
    )
    .join("");
  return `<div class="table-scroll"><table><caption class="visually-hidden">Every measured value, with its minimum and maximum across the run</caption><thead><tr><th scope="col">Method</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function stabilityChart(): string {
  const stability = benchmark.stability;
  const entries = [
    { label: "no smoothing", value: stability.raw.jitter_px_iod, ours: true },
    {
      label: "library defaults",
      value: stability.one_euro_shipped_defaults.jitter_px_iod,
      ours: false,
    },
    { label: "best tuned", value: stability.one_euro_best_tuned.jitter_px_iod, ours: false },
  ];
  const max = Math.max(...entries.map((entry) => entry.value)) * 1.05;
  const height = entries.length * ROW_HEIGHT + 6;
  const marks = entries
    .map((entry, index) => {
      const y = index * ROW_HEIGHT + 8;
      const barY = y + (ROW_HEIGHT - BAR_HEIGHT) / 2 - 6;
      const centre = barY + BAR_HEIGHT / 2;
      const width = (entry.value / max) * (PLOT_X1 - PLOT_X0);
      return [
        `<text class="bar-label${entry.ours ? " bar-label--ours" : ""}" x="0" y="${centre + 4}">${entry.label}</text>`,
        `<rect class="bar-track" x="${PLOT_X0}" y="${barY}" width="${PLOT_X1 - PLOT_X0}" height="${BAR_HEIGHT}" rx="${BAR_HEIGHT / 2}"/>`,
        `<path class="bar-fill${entry.ours ? " bar-fill--ours" : ""}" d="${barPath(PLOT_X0, PLOT_X0 + width, barY, BAR_HEIGHT)}"/>`,
        `<text class="bar-value${entry.ours ? " bar-value--ours" : ""}" x="${CHART_WIDTH}" y="${centre + 4}" text-anchor="end">${entry.value.toFixed(4)}</text>`,
      ].join("");
    })
    .join("");
  return `<figure class="panel">
      <div class="panel__title">
        <h3>Landmark jitter</h3>
        <span class="panel__dir">lower is better</span>
      </div>
      <p class="panel__what">Motion-compensated jitter, in units of interocular distance.</p>
      <svg viewBox="0 0 ${CHART_WIDTH} ${height}" role="img" aria-label="Landmark jitter: ${entries
        .map((entry) => `${entry.label} ${entry.value.toFixed(4)}`)
        .join(", ")}">${marks}</svg>
    </figure>`;
}

function protocolList(): string {
  const protocol = benchmark.meta.protocol as unknown as Record<string, string>;
  return Object.entries(protocol)
    .map(([key, text]) => `<dt>${key.replace(/_/g, " ")}</dt><dd>${text}</dd>`)
    .join("");
}

/** Builds the whole Measured section. */
export function measuredSectionHtml(): string {
  const rows = sortedRows();
  const meta = benchmark.meta;
  const stability = benchmark.stability;

  return `
  <section class="section" id="measured">
    <div class="wrap">
      <div class="section__head">
        <p class="eyebrow">Measured</p>
        <h2>What the numbers say, including where they say we lose.</h2>
        <p class="lede">
          Every figure below comes from one run over ${meta.n_images} portraits, comparing carmine
          against four standard ways of getting virtual makeup wrong: filling the lip region with a
          flat colour, swapping colour channels, painting the wrong landmark indices, and an
          untrained generative model. The bars are means; the whiskers are the full range across the
          run.
        </p>
      </div>

      <div class="legend">
        <span class="legend__item"><span class="legend__key legend__key--ours"></span> carmine</span>
        <span class="legend__item"><span class="legend__key"></span> failure-mode baselines</span>
        <span class="legend__item"><span class="legend__key" style="background:transparent;box-shadow:inset 0 0 0 1px currentColor"></span> whisker: min to max over ${meta.n_images} images</span>
      </div>

      <div class="metrics">
        ${METRICS.map((metric) => panel(metric, rows)).join("")}
      </div>

      <div class="null-result">
        <div>
          <h3>The smoothing result was a null result.</h3>
          <p>
            One-Euro filtering was expected to steady the landmark stream. On a held-still clip it
            did not: the best configuration that stayed within the deviation constraint cut
            motion-compensated jitter by only
            ${stability.jitter_reduction_pct.toFixed(1)}%, under the
            ${stability.grid_search.improvement_threshold_pct.toFixed(0)}% threshold set before the
            search. The filter library's own defaults were worse than doing nothing at all &mdash;
            ${stability.one_euro_shipped_defaults.deviation_ratio_vs_raw.toFixed(1)}× the deviation of
            the raw stream and ${stability.one_euro_shipped_defaults.jitter_change_pct_vs_raw.toFixed(0)}%
            more jitter.
          </p>
          <p>
            So the engine's defaults were left alone, and the mirror's <em>Steady</em> toggle is
            offered as a preference rather than a fix. MediaPipe's video mode is already doing this
            job.
          </p>
        </div>
        ${stabilityChart()}
      </div>

      <details>
        <summary>Protocol, caveats and the raw table</summary>
        <div class="details-body">
          <p>
            Two settings were overridden for this run only, so that every method was measured on the
            same terms. ${meta.smoothing_override_rationale} ${meta.finish_override_rationale}
          </p>
          <dl>${protocolList()}</dl>
          <p><strong>Where the metrics mislead.</strong> ${meta.protocol.pigment_on_target_caveat}</p>
          <p>${meta.lip_detail_retention_note}</p>
          <p>${meta.lip_luminance_shift_note}</p>
          <p><strong>Stability ground truth.</strong> ${stability.ground_truth_caveat}</p>
          <p>
            <strong>On the timing column.</strong> Those milliseconds are the Python reference
            implementation on the CPU, per image, and carmine is the slowest of the five there. The
            browser path is a different implementation &mdash; one fragment-shader pass &mdash; and its
            speed is the frames-per-second readout on the mirror above, not this column.
          </p>
          ${dataTable(rows)}
        </div>
      </details>
    </div>
  </section>`;
}
