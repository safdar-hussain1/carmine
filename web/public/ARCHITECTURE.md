# Carmine — architecture

One algorithm, two surfaces: a Python engine for stills and video files, and a
browser mirror for live camera try-on. Neither is a port of the other in the
usual sense — they are two implementations held together by generated
constants, shared test vectors, and a per-build ΔE comparison against the same
reference renders.

---

## The two surfaces

```
                       ┌──────────────────────────────┐
                       │   carmine/regions.py         │  478-point index tables
                       │   carmine/masks.py           │  geometry fractions
                       │   carmine/pigment.py         │  lightness pulls, finishes
                       │   carmine/look.py            │  the four presets
                       └───────────────┬──────────────┘
                                       │ scripts/export_constants.py
                                       │ scripts/export_test_vectors.py
                                       ▼
                            web/src/gen/constants.json
                            web/src/gen/test_vectors.json

  PYTHON SURFACE                                  BROWSER SURFACE
  ──────────────                                  ───────────────
  image / video file                              camera frame (getUserMedia)
        │                                               │
        ▼                                               ▼
  FaceLandmarker (MediaPipe Tasks)              FaceLandmarker (wasm, bundled)
        │  478 × (x, y) px                             │  478 × (x, y) px
        ▼                                               ▼
  masks.py  →  7 float32 masks                  masks.ts  →  the same 7, as
        │      full resolution                          │      Float32Arrays
        │      true Gaussian feathers                   │      live: half res,
        │                                               │      box-approx feathers
        ▼                                               ▼
  pigment.py: Lab tint / paint /                renderer.ts: ONE WebGL2
  smooth / finish, product by                   fragment pass, all products
  product, NumPy + OpenCV                       per pixel, masks as R8 textures
        │                                               │
        ▼                                               ▼
  BGR uint8 out → file                          canvas → the mirror

                       ▲                                ▲
                       └────────── ΔE parity ───────────┘
                    scripts/verify_site.py --with-parity
                 (3 frames × 3 looks, headless Chrome, CIELAB)
```

Nothing crosses between the surfaces at runtime. They meet only at build time
(generated constants), at test time (shared vectors, parity fixtures), and in
`reports/browser_metrics.json`, where the comparison is written down.

---

## Module map

### Python — `src/carmine/`

| module | responsibility |
| --- | --- |
| `regions.py` | the 478-point index lists: lips, eyes, brows, cheeks, face oval |
| `geometry.py` | interocular distance — the unit every other size is expressed in |
| `landmarks.py` | FaceLandmarker wrapper; model download, sha256 pin, `CARMINE_MODEL` override |
| `masks.py` | seven soft masks, every dimension a fraction of interocular distance |
| `pigment.py` | `tint` (Lab chroma move), `paint` (flat, for eyeliner), `smooth`, `finish_matte`, `finish_gloss` |
| `look.py` | `Product` / `Look` dataclasses, validation, the four presets |
| `engine.py` | `apply_look` (fixed product order) and `VideoEngine` (per-frame + optional One-Euro) |
| `filters.py` | the One-Euro landmark filter |
| `cli.py` | `carmine apply / video / landmarks / looks` |
| `baselines.py` | four standard AR-makeup failure modes, faithfully implemented |
| `metrics.py` | the benchmark scorers — containment, texture, detail, luminance shift, identity |

### TypeScript — `web/src/`

| module | responsibility |
| --- | --- |
| `engine/masks.ts` | the mask port; `exact` and `live` quality paths |
| `engine/blur.ts` | true Gaussian (exact) and box-approximated (live) feathers |
| `engine/color.ts` | sRGB ↔ Lab, matched to OpenCV's conversion |
| `engine/pigment.ts` | the CPU reference port — what parity is measured against |
| `engine/renderer.ts` | the WebGL2 single-pass shader — the live path |
| `engine/look.ts` | presets and product order, read from generated constants |
| `engine/oneEuro.ts` | the filter port, opt-in |
| `lib/landmarks.ts` | wasm landmarker lifecycle, one instance per page |
| `lib/parity.ts` | CPU / GPU / end-to-end ΔE comparisons against fixtures |
| `lib/timing.ts` | per-stage frame cost, medians, fenced draw |
| `lib/selftest.ts` | the `?selftest=1` check registry |
| `lib/camera.ts` | getUserMedia lifecycle |
| `ui/pipeline.ts` | shared landmarker, gloss percentiles, CPU still-render fallback |
| `ui/mirror.ts`, `ui/rail.ts`, `ui/shades.ts`, `ui/measured.ts`, `ui/sections.ts` | the page |

---

## The constants pipeline

Two engines that agree by accident do not stay agreeing. Everything numeric
that both sides need is generated from the Python source rather than typed
twice:

```
carmine/regions.py, filters.py, look.py     ──┐  imported directly
carmine/masks.py, pigment.py, engine.py     ──┤  literals mirrored, each with
  (fractions buried in function bodies)       │  a comment citing its source
                                              ▼
                       scripts/export_constants.py
                                              │
                                              ▼
                            web/src/gen/constants.json
                     (regions, mask fractions, lightness pulls,
                      finish parameters, One-Euro defaults, presets)
```

`tests/test_constants_sync.py` re-runs the generator and diffs its output
against the committed JSON, so a feather radius changed on one side and not
the other fails the Python suite — before it can reach a browser.

`scripts/export_test_vectors.py` does the same job one level down: it records
Python-computed inputs and outputs (colour conversions, mask samples, pigment
results) as `web/src/gen/test_vectors.json`, which the vitest suite asserts
against unit by unit. Constants keep the two sides configured identically;
vectors keep them *computing* identically.

`scripts/export_parity_fixtures.py` writes the third layer: canned frames,
their landmarks, and the Python engine's own renders under three looks. Those
fixtures contain dataset faces, so they are git-ignored and never copied into
the built site — `verify_site.py --with-parity` mounts them from a second
local root, over loopback, for the duration of one verification.

---

## Exact and live mask paths

The mask stage is the expensive part of a frame — seven polygon rasters plus
seven Gaussian feathers, scaling with pixel count — so there are two paths,
and the difference between them is explicit at every call site via a
`quality: "exact" | "live"` flag.

| | exact | live |
| --- | --- | --- |
| processing long side | 720 px | 360 px (quarter the pixels) |
| feather | true Gaussian | box approximation |
| skin smoothing | yes (bilateral) | not in this path |
| matte finish blur | Gaussian of L | frame mip level at a matching radius |
| gloss percentiles | measured after the tint | measured on the incoming frame |
| measured cost (M1 Pro) | 357.6 ms/frame | 7.3 ms/frame |
| used by | still renders, parity checks | the live camera loop |

Every one of those live-path departures is an approximation that trades a
documented amount of fidelity for about a 49× reduction in mask cost, and none
of them is allowed to launder itself into a parity number: `parity-cpu`
compares the exact CPU path against Python and is gated, while the GPU shader
path is measured and published with its renderer string attached but
deliberately not gated — headless verification runs on a software rasteriser,
which is not the hardware anyone uses.

Masks are uploaded as single-channel R8 textures at processing resolution and
sampled bilinearly at output resolution. The 8-bit quantisation costs at most
1/255 of mask weight, far below the tolerance a feathered edge is built with.

---

## The selftest

The built site carries its own acceptance test. `?selftest=1` runs nine checks
in sequence and writes the aggregate into the tab title
(`SELFTEST PASS n=9`), which `scripts/verify_site.py` reads over the DevTools
protocol from headless Chrome. One hard rule: no check may call
`getUserMedia`, because a permission prompt would wedge a headless run — every
check that needs a frame uses the bundled demo portrait.

| check | what only a browser can prove |
| --- | --- |
| `constants-loaded` | the generated constants shipped in the bundle and have the expected shape |
| `landmarker-init` | the wasm runtime instantiates and the bundled model loads |
| `renderer-compiles` | a real GL driver accepts the fragment shader, and no GL error after one frame |
| `presets-valid` | four presets, every colour a valid hex, every intensity in range |
| `ui-mounts` | the rail, the preset chips and the mirror canvas are actually in the DOM |
| `pipeline-canned-frame` | detect → mask → render changes the face by >1% of pixels and leaves the corners bit-stable |
| `parity-cpu` | the CPU path matches the Python renders within the ΔE gates (**gated**) |
| `parity-gpu` | the shader path measured against the same renders (report-only) |
| `timing` | per-stage frame cost, medians, fenced draw (report-only) |

The two parity checks skip explicitly when the fixtures are not mounted, which
is the normal state of the deployed site; `verify_site.py --with-parity`
refuses a run in which a check it asked for skipped, and `--expect-checks`
fails if the registered count ever changes silently.

---

## Where the numbers live

| file | written by | holds |
| --- | --- | --- |
| `reports/benchmark.json` | `scripts/benchmark.py`, `scripts/stability_bench.py` | photo metrics for five methods over 26 images; the stability sweep |
| `reports/browser_metrics.json` | `scripts/verify_site.py --with-parity`, then `--timing-only` | parity ΔE (CPU / GPU / end-to-end), per-stage timing, selftest summary |
| `reports/figures/*.png` | `scripts/make_figures.py`, `scripts/preview_masks.py` | the published figures |
| `notebooks/01_engine_and_benchmarks.ipynb` | `scripts/build_notebook.py` | the write-up, which reads the two JSON files and asserts every number it narrates |

Numbers are never typed into prose. The site's Measured section imports
`benchmark.json` at build time; the notebook reads both files and fails to
execute if the prose has drifted; `tests/test_metrics.py` and
`tests/test_parity_report.py` validate the committed files themselves.
