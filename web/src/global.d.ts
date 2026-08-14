import type { SelftestSummary } from "./lib/selftest";

export {};

declare global {
  interface CarmineResults {
    selftest?: SelftestSummary;
    [key: string]: unknown;
  }

  interface Window {
    __carmine_results: CarmineResults;
  }
}
