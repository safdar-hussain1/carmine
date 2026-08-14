/**
 * App bootstrap. The real UI lands in a later task; this wires up the
 * selftest harness so `?selftest=1` can be used to verify a build headlessly
 * (see scripts/verify_site.py) and registers the checks that exercise the
 * pieces this task adds: the generated constants and the landmark model.
 */

import { createLandmarker } from "./lib/landmarks";
import { registerCheck, runSelftest } from "./lib/selftest";

window.__carmine_results = {};

const MODEL_URL = "./models/face_landmarker.task";

registerCheck("constants-loaded", async () => {
  const constants = await import("./gen/constants.json");
  const regions = constants.default.regions as Record<string, unknown>;
  const lipsOuter = regions.LIPS_OUTER;
  if (!Array.isArray(lipsOuter) || lipsOuter.length !== 20) {
    throw new Error(`expected LIPS_OUTER to have length 20, got ${JSON.stringify(lipsOuter)}`);
  }

  const presets = constants.default.presets as Record<string, unknown>;
  if (!presets || Object.keys(presets).length === 0) {
    throw new Error("expected constants.presets to be a non-empty object");
  }
});

registerCheck("landmarker-init", async () => {
  await createLandmarker(MODEL_URL);
});

function bootstrap(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (app) {
    app.textContent = "Carmine";
  }

  const params = new URLSearchParams(window.location.search);
  if (params.get("selftest") === "1") {
    void runSelftest();
  }
}

bootstrap();
