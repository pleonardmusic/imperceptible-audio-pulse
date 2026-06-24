# 40Hz ERB Comb Project — Notes

Audio analog of Optoceutics' Invisible Spectral Flicker: mask a 40Hz gamma-entrainment
pulse inside music via ERB-spaced comb filtering + AM crossfade, so it can be listened
to all day instead of via a punishing 1hr/day overt-click protocol. This file is a
durable, colocated record of findings and design decisions — kept alongside the code
specifically so we don't have to rediscover the same things every session.

**Related project — this app's output is tested using a separate EEG measurement
rig.** `/Users/peterleonard/Documents/Claude-Personal-Documents/eeg-experiments`
(iPhone app → `bridge-iphone.js` → browser app, launched via `EEG Monitor.app` on
the Desktop) is what records and analyzes EEG while listening to audio rendered by
*this* app — it has its own colocated `PROJECT_NOTES.md` with the live-vs-legacy
file map for that project. See "Open questions" and the EEG test-run entries below
for what's actually been measured so far; the masked-audio result has not yet
replicated (first test, 2026-06-24, see that project's notes for detail).

## Scientific background (see References section at the bottom of index.html for full citations)
- 40Hz gamma entrainment reduces amyloid/tau in AD mouse models (Iaccarino 2016, Martorell 2019) and produces measurable EEG entrainment in healthy humans (Jones/Galway 2019). Human AD-patient trial evidence (Cognito OVERTURE 2024) is mixed: missed primary endpoint, but real secondary/imaging signals.
- This app's core idea — masking the stimulus rather than reducing its intensity — mirrors Optoceutics' Invisible Spectral Flicker for light (Carstensen 2020), applied to audio instead.
- 40Hz auditory steady-state response (ASSR) is well-established and elicitable from AM-tone carriers across 250Hz-4000Hz, with amplitude *decreasing* as carrier frequency increases (250Hz ~3x larger than 4000Hz). This matters directly for the Min/Max Pulse Frequency sliders — see below.

## How the pipeline works (app.js)
Per channel: `compressDynamics` (optional, light use in practice, default 0.2) → single
whole-track FFT (not block-based STFT — one FFT/IFFT pair per channel, sized to the
next power of 2 above the whole file's sample count) → `buildMask` x2 (complementary
ERB-spaced combs, Track A = odd bands on, Track B = even bands on) → IFFT x2 →
`applyAM` (40Hz crossfade between A and B, shape-adjustable square↔Hanning) → WAV encode.

## Key findings, in order discovered

1. **`edgeSoft` units bug (2026-06-24)**: band-edge softness was originally a flat bin
   count, which becomes vanishingly small in Hz once the FFT spans an entire track
   (bin width = sr/N, and N grows with track length). Fixed: edge softness is now a
   **% of each band's own ERB width** (`edgeSoftFraction`), so it's consistent across
   the spectrum and across track lengths. Note: this fix did NOT turn out to
   meaningfully reduce the audible "gritty/static" artifact on its own — see finding
   #3 below, which was the real cause. Default value history: 15% → 37% (raised same
   session, see "Current defaults" at the time) → back to **15%** (Peter's request,
   2026-06-24, end of session — no specific reason logged, just a revert of the earlier
   bump).

2. **Min/Max Frequency deletion bug (2026-06-24)**: content outside the `freqMin`-
   `freqMax` band range was being silently deleted (mask array initializes to 0 and
   nothing outside the band loop ever wrote to it), not preserved. Fixed: now always
   passes through unmodified above Max Pulse Frequency and below Min Pulse Frequency
   (mask=1 in both A and B, so the AM crossfade's `a+b=1` identity means that content
   never toggles). There used to be on/off checkboxes for this; removed per Peter's
   request — always-preserve is now the only behavior, no legacy-delete option.

3. **Crossfade Shape (`xfadeShape`) is the dominant artifact source, not edge softness
   or band count (found by Peter, 2026-06-24).** This parameter blends between a
   square wave (0: instantaneous on/off switch between Track A/B every 1/40s) and full
   Hanning (1: smooth sinusoidal crossfade). A square-wave switch is a sharp time-domain
   discontinuity happening 40x/sec, which injects broadband harmonic energy on every
   cycle — audible as grit/static regardless of which musical content is being
   toggled. Peter's listening test: gritty below ~50%, "bearable" around 50%, clean at
   1.0 (full Hanning — already the default). **Practical takeaway: keep this at or
   near 1.0; there's little reason to lower it.**

4. **Min Pulse Frequency vs. neural-entrainment strength tradeoff (2026-06-24).**
   Raising Min Pulse Frequency (e.g. 80→500Hz) makes the audio sound much more normal
   (Peter: "almost normal, with a little crunchy artifacts if you're really paying
   attention" at 500Hz). BUT per the ASSR literature above, lower carrier frequencies
   produce *larger* ASSR amplitude — so this trade is real, not free. Excluding bass
   from the pulse likely sounds better while plausibly weakening the neural signal
   somewhat, not improving it. There is no known setting that's better on both axes.
   Caveat: ASSR (a generic, well-replicated 40Hz cortical response to AM tones) is not
   proven equivalent to the specific GENUS/Alzheimer's disease-modifying effect, which
   was characterized using broadband click trains, not narrowband AM tones — so even a
   strong ASSR doesn't guarantee the AD-relevant benefit follows. This was an open
   question before this finding and remains one, just refined.

## Current defaults (2026-06-24) — best combination found so far
**12 bands, Min Pulse Frequency 500Hz, Max Pulse Frequency 5000Hz, Off-band Level 0.1,
Edge Softness 15%, full Hanning crossfade (1.0).** Peter's listening report at 37% edge
softness: music sounds "almost identical to the original" — a meaningfully better result
than earlier in the session. Edge softness was then reverted to 15% at the end of the
session (Peter's request, no reason logged) — **the "almost identical" listening report
above was given at 37%, not yet re-verified at 15%.**

Why this combination works, mechanistically: fewer bands (12 vs the old default 24) means
each band is wider (avg ~375Hz vs ~187Hz across this same 500-5000Hz range, confirmed by
direct calculation) — wide enough to span a whole formant/instrument's harmonic content, so
Track A and Track B are genuinely spectrally distinct from each other rather than finely
interleaved near-duplicates. Peter's own framing: "fewer bands means a more spectrally
different A and B." This connects to an early-session observation (his, from a previous
conversation): fewer bands sounded more pronounced/effective but less pleasant; more bands
sounded closer to the original but likely less effective — a tradeoff that seemed
unavoidable at the time. The combination above is a way to partially escape that tradeoff:
keep the fewer-bands property (preserving more genuine spectral modulation depth, which is
hypothesized — not proven — to matter for entrainment strength) while using the *other*
parameters to neutralize what used to make fewer bands sound bad: narrowing the pulse range
(dodges both the low-frequency AM-sideband problem and the high-frequency noise-jitter
problem identified earlier), high edge softness (smooths the on/off transition so it doesn't
read as a harsh switch — this part of the reasoning was based on 37% edge softness, not the
current 15% default, so it's weaker evidence now), and full Hanning crossfade (no square-wave
grit, see finding #3 above).

**Caveat, same as always: "more spectrally distinct A/B should mean stronger entrainment" is
a reasonable mechanistic hypothesis, not a measured result.** The only way to know whether
this combination is actually better (not just more pleasant) for real 40Hz entrainment is
testing it on Peter's EEG rig.

## Open questions / not yet tested
- Does the masked (ERB comb) stimulus actually preserve 40Hz EEG entrainment relative
  to the overt-click positive control? (Partially tested via Peter's own EEG rig,
  2026-06-24: a 4-condition randomized test — Baseline/Click/No-Pulse/With-Pulse — did
  NOT replicate "pulse beats no-pulse" across 2 runs; attention/arousal confound
  suspected, stronger pattern than any condition effect so far.)
- Does raising Min Pulse Frequency to exclude bass measurably change the EEG response,
  or is it neurally negligible in practice despite the ASSR-amplitude theory above?
  Directly testable with Peter's existing EEG app + condition-tagging system.
- Whether spectral-redistribution masking (this app's approach) preserves entrainment
  as well as simple intensity reduction does in the light literature — still unknown,
  no assumption either way.

## Planned features (not yet built)
- In-app per-condition audio playback buttons (Baseline/Click/No-Pulse/With-Pulse etc.)
  that simultaneously trigger CSV condition-tagging in the EEG Monitor app, replacing
  the current manual external-audio-editing workflow for building test tracks.

## Working conventions for this project
- This is a personal single-user tool now, not something built for handoff — prioritize
  comments that explain *why* a parameter/design choice exists over generic code
  cleanliness.
- Keep this as ONE app with sliders/toggles for every parameter under investigation,
  rather than maintaining forked versions — A/B testing happens by re-rendering with
  different settings in the same tool.
- Backups of pre-fix versions exist as `app.js.bak-pre-edgesoft-fix` /
  `index.html.bak-pre-edgesoft-fix` in this folder (from the 2026-06-24 session, before
  the edgeSoft/passthrough changes).
