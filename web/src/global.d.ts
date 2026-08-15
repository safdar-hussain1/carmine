/// <reference types="vite/client" />

import type { ParityResults } from "./lib/parity";
import type { SelftestSummary } from "./lib/selftest";
import type { TimingResults } from "./lib/timing";

export {};

declare global {
  interface CarmineResults {
    selftest?: SelftestSummary;
    /** Cross-surface comparison against the Python engine; see lib/parity.ts. */
    parity?: ParityResults;
    /** Per-stage frame cost; see lib/timing.ts. */
    timing?: TimingResults;
    [key: string]: unknown;
  }

  interface Window {
    __carmine_results: CarmineResults;
  }
}
