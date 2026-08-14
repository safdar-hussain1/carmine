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
}

export interface SelftestSummary {
  pass: boolean;
  count: number;
  results: SelftestCheckResult[];
}

type CheckFn = () => Promise<void>;

const checks: Array<{ name: string; fn: CheckFn }> = [];

export function registerCheck(name: string, fn: CheckFn): void {
  checks.push({ name, fn });
}

export async function runSelftest(): Promise<SelftestSummary> {
  const results: SelftestCheckResult[] = [];
  let firstFailure: SelftestCheckResult | null = null;

  for (const check of checks) {
    try {
      await check.fn();
      results.push({ name: check.name, ok: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const result: SelftestCheckResult = { name: check.name, ok: false, error: message };
      results.push(result);
      if (firstFailure === null) {
        firstFailure = result;
      }
    }
  }

  const summary: SelftestSummary = {
    pass: firstFailure === null,
    count: results.length,
    results,
  };

  document.title =
    firstFailure === null
      ? `SELFTEST PASS n=${results.length}`
      : `SELFTEST FAIL ${firstFailure.name}: ${firstFailure.error}`;

  window.__carmine_results.selftest = summary;
  return summary;
}
