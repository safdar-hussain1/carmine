/**
 * A tiny in-page test registry driven by the `?selftest=1` URL flag.
 *
 * Headless verification (scripts/verify_site.py) loads the built site with
 * that flag, waits for `document.title` to start with "SELFTEST", and reads
 * `window.__carmine_results.selftest` for per-check detail. Checks run
 * sequentially (not in parallel) so a check that depends on a previous
 * check's side effects behaves predictably, and so the first failure can be
 * reported without racing against checks still in flight.
 */

export interface SelftestCheckResult {
  name: string;
  ok: boolean;
  error?: string;
  /** Set when the check could not run because an input it needs is not
   * being served. See SKIPPED below. */
  skipped?: boolean;
  /** Why it skipped, for the run's transcript. */
  reason?: string;
}

export interface SelftestSummary {
  pass: boolean;
  count: number;
  skipped: number;
  results: SelftestCheckResult[];
}

/**
 * A check's way of saying "the input I need is not here".
 *
 * The parity checks compare against fixture renders that are mounted only
 * during local verification (they contain dataset faces and are never
 * shipped), so on the deployed site they have nothing to compare against.
 * Failing there would make the site's own selftest permanently red; silently
 * passing would let a genuinely broken fixture mount look healthy. Skipping
 * explicitly -- counted in the summary and in the page title -- is the
 * honest third option, and `verify_site.py --with-parity` refuses a run
 * where the checks it asked for skipped.
 */
export const SKIPPED = Symbol("selftest-skipped");

export interface SkipResult {
  skip: typeof SKIPPED;
  reason: string;
}

/** Build the value a check returns to declare itself skipped. */
export function skip(reason: string): SkipResult {
  return { skip: SKIPPED, reason };
}

type CheckFn = () => Promise<void | SkipResult>;

function isSkip(value: unknown): value is SkipResult {
  return typeof value === "object" && value !== null && (value as SkipResult).skip === SKIPPED;
}

const checks: Array<{ name: string; fn: CheckFn }> = [];

export function registerCheck(name: string, fn: CheckFn): void {
  checks.push({ name, fn });
}

export async function runSelftest(): Promise<SelftestSummary> {
  const results: SelftestCheckResult[] = [];
  let firstFailure: SelftestCheckResult | null = null;

  for (const check of checks) {
    try {
      const outcome = await check.fn();
      if (isSkip(outcome)) {
        results.push({ name: check.name, ok: true, skipped: true, reason: outcome.reason });
      } else {
        results.push({ name: check.name, ok: true });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: SelftestCheckResult = { name: check.name, ok: false, error: message };
      results.push(result);
      if (firstFailure === null) {
        firstFailure = result;
      }
    }
  }

  const skipped = results.filter((result) => result.skipped).length;
  const summary: SelftestSummary = {
    pass: firstFailure === null,
    count: results.length,
    skipped,
    results,
  };

  document.title =
    firstFailure === null
      ? `SELFTEST PASS n=${results.length} skipped=${skipped}`
      : `SELFTEST FAIL ${firstFailure.name}: ${firstFailure.error}`;

  window.__carmine_results.selftest = summary;
  return summary;
}
