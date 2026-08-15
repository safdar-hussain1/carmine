# Carmine — design card

What Carmine claims, what each claim was measured against, what the
measurement cannot tell you, and which test fails if the claim stops being
true.

Sources, both committed and machine-readable: `reports/benchmark.json` (photo
quality, video stability) and `reports/browser_metrics.json` (cross-surface
parity, per-stage timing). No figure below appears anywhere in this project
that is not in one of those two files.

---

## C1 — Texture survives the pigment

**Number.** Lip texture correlation **0.988**, the highest of the five methods
benchmarked. Mean lip-lightness shift **13.2** Lab-L units, against **45.7**
for an opaque fill of the same region under the same look — 3.5× less
disturbance of the face's own brightness.

**Protocol.** 26 photographs, no-makeup half of a local paired-portrait set.
One look for every method (velvet, lipstick finish forced to satin, skin
smoothing forced to 0). Texture = Pearson correlation of Lab-L inside the lip
mask before vs after; shift = absolute change in mean Lab-L in the same
region.

**Caveat.** Correlation is invariant to affine rescaling, so an *additive*
flat fill can score high on it too — which is precisely why the lightness
shift is published beside it rather than instead of it. And Carmine's
`lip_detail_retention` is **0.754**, below opaque fill's 0.952: the tint pulls
lip lightness a capped fraction toward the target, which mechanically scales
down high-frequency L variance by roughly that fraction. That metric is a
diagnostic, not a ranking, and the number is published as measured.

**Pinned by.** `tests/test_metrics.py::TestBenchmarkReport` —
`test_lip_luminance_shift_ordering_matches_measured_reality` and
`test_lip_detail_retention_ordering_matches_measured_reality` re-derive the
published orderings from the committed JSON;
`test_carmine_scores_match_expected_shape` floors the texture score at 0.95.
`tests/test_pigment.py` pins the tint's behaviour at the function level.

---

## C2 — Pigment lands where it is aimed, and nowhere else

**Number.** **83.8%** of all pixel-change energy falls inside the product
masks. Background pixels outside the face hull: **1.000** bit-identical. For
comparison, the channel-swap failure mode scores 0.017 on that second column,
because it rewrites every pixel in the frame.

**Protocol.** Containment = share of total change energy inside the union of
the lip, eyeshadow, blush and eyeliner masks, at a strict membership cutoff of
mask value > 0.05. Background = fraction of pixels outside a dilated face hull
that are bit-identical between input and output.

**Caveat.** The cutoff is deliberately adversarial to feathering: it is a hard
membership test, not a soft weighting, so a feathered edge spreads change into
low-mask pixels that are then counted as misses. 0.838 is a soft edge being
penalised by a hard threshold. The opposite trap is on the same axis — the
opaque-fill baseline scores 0.998 because it shares its region geometry with
the scorer. Containment cannot distinguish a hard fill from a soft tint once
the geometry is right; that is C1's job.

**Pinned by.** `tests/test_metrics.py` (scorer unit tests feed each metric the
case it exists to catch, plus `test_carmine_scores_match_expected_shape`,
which floors containment at 0.75 and requires background = 1.0);
`tests/test_masks.py` pins mask geometry and range.

---

## C3 — The browser is the same engine, not a lookalike

**Number.** Given identical landmarks, worst case over nine full-frame
comparisons (3 frames × 3 looks): **mean ΔE 0.747**, p99 2.763, max 11.434,
and **0** pixels changed outside the region either engine painted. A ΔE of
about 1 is the threshold of a just-noticeable colour difference.

End to end, with each side running its own landmark detector: **mean ΔE
2.717**, p99 18.454, max 30.135.

**Protocol.** `scripts/verify_site.py --with-parity` drives the built site in
headless Chrome, renders the committed reference frames in the browser, and
compares against the Python engine's own renders in CIELAB ΔE.

**Caveat.** Both tables belong together. The rendering disagreement is the
first row; the second row is roughly seven times larger and almost none of it
is rendering — it is the two landmarker builds placing the same face a
fraction of a pixel apart. The worst single pixel in row one, 11.434, is 23
pixels out of 388,800, all on the boundary of the two-pixel eyeliner stroke.
The GPU shader path is measured and published too, but is **not** gated: it
takes documented approximations the CPU reference does not, and headless
verification runs on a software rasteriser, which is not the hardware anyone
uses.

**Pinned by.** The browser's own selftest gates `parity-cpu` at mean < 2.0,
p99 < 5.0, max < 12.0, outside = 0 (`web/src/main.ts`), and
`tests/test_parity_report.py` re-applies those same gates to the committed
numbers on every pytest run —
`test_cpu_parity_is_within_the_published_thresholds`,
`test_nothing_was_painted_outside_the_mask_support`. That test never skips.

---

## C4 — It runs live

**Number.** **26.6 ms per frame ≈ 37.6 fps** at 1280×720, on an Apple M1 Pro
through Chrome/ANGLE/Metal. Breakdown: landmark detection 17.5 ms, mask
construction 7.3 ms, fenced GPU draw 1.8 ms.

**Protocol.** 120 frames after 10 warm-up frames; medians, not means, so
shader compilation and a stray garbage collection do not become the headline;
the draw fenced with `gl.finish()` so it measures the frame rather than the
speed of queueing commands. Face close to the camera — interocular ≈ 199px at
processing resolution — because mask cost scales with face size, not frame
size.

**Caveat.** This is one machine, and the number is meaningless without it. The
same measurement on a software rasteriser gives 82.5 ms/frame, almost all of
it detection. The live mask path is an approximation of the reference path —
half the processing long side, box-approximated feathers instead of true
Gaussians — which is how 357.6 ms of mask construction becomes 7.3 ms.
Approximation and reference are separate code paths with an explicit quality
flag, so parity can never be measured against the shortcut.

**Pinned by.** `tests/test_parity_report.py::test_live_path_is_fast_enough_to_be_live`
(live masks under 40 ms and strictly faster than the reference path),
`test_timing_medians_are_finite_and_positive` (resolution, sample counts,
fencing), `test_hardware_timing_attempt_is_recorded_either_way` (a run that
silently fell back to software may not be published as a GPU number).

---

## C5 — Landmark smoothing: a null result

**Number.** Best tuned One-Euro configuration (min_cutoff 1.5, beta 0.5) cut
motion-compensated jitter by **7.6%**, against a **20%** bar it had to clear
to be adopted. The filter's own shipped defaults (1.0, 0.007) were *worse than
no filter* on this clip: **4.6×** the deviation from ground truth and **73%
more** jitter. **No default was changed.**

**Protocol.** 90-frame synthetic clip generated from one still along a known
affine path (pan, zoom, rotation, sensor noise), so tracking lag and jitter
can be separated. Twelve (min_cutoff, beta) pairs swept; a configuration had
to both cut jitter by >20% and not inflate deviation from ground truth beyond
1.5×.

**Caveat.** The clip is clean by construction, and a clean stream is exactly
where a smoothing filter should be expected to lose — it can only add lag.
This result says smoothing is an opt-in for genuinely jittery cameras, not
that the filter is useless. Ground truth is anchored to a real frame-0
detection rather than a synthetic exact position, so the absolute deviations
are mildly optimistic; the raw-vs-filtered comparison is unaffected, since
every stream shares that anchor.

**Pinned by.** `tests/test_metrics.py::TestBenchmarkReport::test_stability_section_schema`,
which asserts the sweep is complete, that a null selection may not claim to
have met the bar, and — explicitly — that the shipped defaults measured worse
than raw, so making that inconvenient fact disappear takes a deliberate edit
rather than a passing test.

---

## Model card — FaceLandmarker

| | |
| --- | --- |
| **Model** | MediaPipe FaceLandmarker (face_landmarker.task, float16) |
| **License** | Apache-2.0 |
| **Size** | 3.76 MB |
| **Output** | 478 3D face landmarks, one face |
| **Where it runs** | on device — CPU/GPU in Python, wasm in the browser |
| **Integrity** | sha256 pinned in `carmine/landmarks.py`; a mismatched download is deleted, not used |
| **Served from** | this origin (`/models/`), never a CDN |

**Data handling.** No image, frame, landmark or capture leaves the device. The
browser build makes zero network requests after load — the model and its wasm
runtime are bundled, type is set in system fonts, there is no analytics — so
the claim is verifiable from the Network panel, or by turning the network off
and using the page anyway. The Python engine touches the network exactly once,
on first run, to fetch the model into `~/.cache/carmine/`; set `CARMINE_MODEL`
to a local file and it makes no requests at all.

**Configuration.** `num_faces=1`, blendshapes and transformation matrices
disabled — none of the three is needed to place makeup, and each is inference
work paid per frame.

**Known limits inherited from the detector.** One face at a time. Landmarks
are geometric, so occlusion is invisible to the pipeline: a hand across the
mouth still gets lipstick painted over it. Detection accuracy varies with
pose, lighting and image quality, and it is by some margin the largest source
of disagreement between the two surfaces (C3) — the pigment maths is not the
risk here.

**Fairness.** Carmine does not train, fine-tune or evaluate the detector, and
publishes no fairness claim about it. The 26-photograph benchmark set is far
too small to support one, and it is not a demographic sample. Any bias in
landmark placement across skin tones, face shapes or ages is inherited whole
from the upstream model.
