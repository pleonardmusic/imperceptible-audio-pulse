// ─────────────────────────────────────────────────────────────────────────────
// "Sound for Aging" / ERB Comb project — audio analog of Optoceutics' Invisible
// Spectral Flicker (light), built to mask a 40Hz gamma-entrainment pulse inside
// music so it can be listened to all day instead of via a punishing 1hr/day
// overt-click session. Background: 40Hz gamma entrainment (GENUS) literature —
// Iaccarino 2016, Martorell 2019 (mouse, light+sound, amyloid/tau reduction via
// microglia), Jones 2019/Galway (human EEG entrainment, light only), Cognito
// Therapeutics OVERTURE trial 2024 (human AD patients, mixed results — missed
// primary endpoint, but positive secondary/imaging signals). See references
// at the bottom of index.html and PROJECT_NOTES.md in this folder for full
// citations and a running log of findings/design decisions.
//
// Pipeline, per channel: compressDynamics (optional dynamics flattening) ->
// single whole-track FFT -> buildMask x2 (complementary ERB combs A/B) ->
// IFFT x2 -> applyAM (40Hz crossfade between A and B) -> encodeWAV.
// This is a single-shot FFT over the ENTIRE file, not block-based STFT — so
// every band's gating is one static mask applied uniformly across the whole
// track's duration, not something that adapts moment-to-moment.
// ─────────────────────────────────────────────────────────────────────────────

// ─── ERB Utilities ────────────────────────────────────────────────────────────
// ERB (Equivalent Rectangular Bandwidth, Glasberg & Moore 1990) spaces bands so
// each one is perceptually equal-width, unlike linear Hz spacing. This matters
// here because the comb's "on"/"off" bands need to alternate in a way that
// sounds like one continuous texture rather than an audibly uneven set of
// gaps — ERB spacing is what makes that plausible.
function hzToErb(hz) { return 21.4 * Math.log10(0.00437 * hz + 1); }
function erbToHz(erb) { return (Math.pow(10, erb / 21.4) - 1) / 0.00437; }

function getErbEdges(n, fMin, fMax) {
  const erbMin = hzToErb(fMin), erbMax = hzToErb(fMax);
  const edges = [];
  for (let i = 0; i <= n; i++) edges.push(erbToHz(erbMin + (erbMax - erbMin) * i / n));
  return edges;
}

// ─── Envelope compression (pre-process) ──────────────────────────────────────
// Peter's usage note (2026-06-24): not using this much in practice — default
// 0.2 is fairly light. Was added to keep the source music's natural loudness
// swings (drum hits, quiet passages) from masking/fighting the 40Hz
// entrainment envelope applied later, but isn't critical at moderate settings.
// One-pole LPF on |signal| gives a slow amplitude envelope; divide signal by
// envelope^amount. amount=0 → bypass; amount=1 → fully flattened. Floor keeps
// silent passages from exploding. Smoothing ~100ms, safely slower than 40Hz
// (25ms) so it can't interact with the entrainment envelope.
function compressDynamics(signal, sr, amount) {
  if (amount <= 0) return signal;
  const L = signal.length;
  const tau = 0.1; // 100ms smoothing
  const a = Math.exp(-1 / (sr * tau));
  const oneMinusA = 1 - a;

  // Forward + backward pass for zero-phase envelope
  const env = new Float32Array(L);
  let e = 0;
  for (let i = 0; i < L; i++) {
    const x = Math.abs(signal[i]);
    e = a * e + oneMinusA * x;
    env[i] = e;
  }
  e = env[L - 1];
  for (let i = L - 1; i >= 0; i--) {
    e = a * e + oneMinusA * Math.abs(signal[i]);
    env[i] = 0.5 * (env[i] + e);
  }

  // Peak envelope value and a 1%-of-peak floor
  let peak = 0;
  for (let i = 0; i < L; i++) if (env[i] > peak) peak = env[i];
  const floor = Math.max(peak * 0.01, 1e-6);

  const out = new Float32Array(L);
  for (let i = 0; i < L; i++) {
    const e2 = Math.max(env[i], floor);
    const divisor = Math.pow(e2, amount);
    out[i] = signal[i] / divisor;
  }
  return out;
}

// ─── Iterative Cooley-Tukey FFT (in-place, Float32Array) ─────────────────────
function fft(re, im) {
  const N = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  // butterflies
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wlen_r = Math.cos(ang), wlen_i = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const ar = re[i + k],           ai = im[i + k];
        const br = re[i + k + half],    bi = im[i + k + half];
        const tr = br * wr - bi * wi;
        const ti = br * wi + bi * wr;
        re[i + k]          = ar + tr;
        im[i + k]          = ai + ti;
        re[i + k + half]   = ar - tr;
        im[i + k + half]   = ai - ti;
        const nwr = wr * wlen_r - wi * wlen_i;
        wi = wr * wlen_i + wi * wlen_r;
        wr = nwr;
      }
    }
  }
}

function ifft(re, im) {
  const N = re.length;
  for (let i = 0; i < N; i++) im[i] = -im[i];
  fft(re, im);
  const inv = 1 / N;
  for (let i = 0; i < N; i++) { re[i] *= inv; im[i] = -im[i] * inv; }
}

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

// ─── Frequency-domain mask (soft-edge band gating) ───────────────────────────
function hzToBin(hz, N, sr) { return Math.round(hz / sr * N); }

function bandWeight(k, low, high, w) {
  if (w <= 0) return (k >= low && k < high) ? 1 : 0;
  if (k < low - w || k > high + w) return 0;
  if (k >= low && k <= high) return 1;
  if (k < low) {
    const x = (k - (low - w)) / w;
    return 0.5 * (1 - Math.cos(Math.PI * x));
  }
  // k > high
  const x = (k - high) / w;
  return 0.5 * (1 + Math.cos(Math.PI * x));
}

// `off` (UI: "Off-band Level") is the core efficacy/comfort tradeoff of the
// whole technique — the level (0-1) that "off" bands are attenuated to,
// rather than fully silenced. 0 = maximum comb contrast (most perceptible,
// most effective in principle); closer to 1 = bands barely alternate (least
// perceptible, weakest masking of the underlying entrainment signal). This is
// the audio analog of Optoceutics never making either light source fully off
// in their Invisible Spectral Flicker.
function buildMask(N, sr, edges, isA, off, edgeSoftFraction) {
  const mask = new Float32Array(N);
  const nb = edges.length - 1;
  const nyq = N >> 1;
  // edgeSoftFraction is a fraction of each band's OWN ERB width, not a flat Hz
  // or bin count — ERB bands are narrow at low frequencies and wide at high
  // frequencies by design, so a flat width disproportionately blurs/merges the
  // low bands (where it's a large % of the band) while barely affecting the
  // high bands (where it's a tiny %). Scaling per-band keeps softening
  // perceptually consistent across the spectrum.
  for (let i = 0; i < nb; i++) {
    const low  = hzToBin(edges[i],     N, sr);
    const high = hzToBin(edges[i + 1], N, sr);
    const bandWidthHz = edges[i + 1] - edges[i];
    const edgeSoft = Math.max(0, Math.round(edgeSoftFraction * bandWidthHz / sr * N));
    const on = isA ? (i % 2 === 0) : (i % 2 === 1);
    const g = on ? 1.0 : off;
    const lo = Math.max(0, low - edgeSoft);
    const hi = Math.min(nyq, high + edgeSoft);
    for (let k = lo; k <= hi; k++) {
      const w = bandWeight(k, low, high, edgeSoft);
      mask[k] += w * g;
    }
  }
  // Below the lowest band edge (Min Pulse Frequency) and above the highest
  // (Max Pulse Frequency), always pass through unmodified: mask=1 in both A
  // and B, so the AM crossfade cancels exactly since a+b=1 always — that
  // content never toggles, full fidelity preserved outside the pulse range.
  // Until 2026-06-24 this range was silently DELETED (mask stayed 0 from
  // initialization) rather than preserved — a real bug, found by Peter
  // noticing artifacts and confirmed by spectrogram. Always-preserve is now
  // the only behavior; there's no toggle to bring the delete-bug back.
  //
  // Where to set Min/Max Pulse Frequency is a real tradeoff, not just a
  // cosmetic choice — see 40Hz auditory steady-state response (ASSR)
  // literature: ASSR amplitude is reliably measurable for AM carrier tones
  // from 250Hz-4000Hz, but DECREASES as carrier frequency increases (250Hz
  // carrier ASSR amplitude ~3x larger than at 4000Hz). So narrowing the pulse
  // range upward (e.g. Min Pulse Frequency 80->500Hz, which Peter found
  // sounds much cleaner) trades away some of this signal's theoretical
  // neural-entrainment strength for less audible artifact — it does not
  // improve entrainment, even though it sounds better. There is no known
  // setting that is simply better on both axes. See PROJECT_NOTES.md.
  const bottomBin = hzToBin(edges[0], N, sr);
  for (let k = 0; k < bottomBin; k++) mask[k] = 1.0;
  const topBin = hzToBin(edges[nb], N, sr);
  for (let k = topBin; k <= nyq; k++) mask[k] = 1.0;
  // mirror to negative frequencies for real-signal conjugate symmetry
  for (let k = 1; k < nyq; k++) mask[N - k] = mask[k];
  return mask;
}

// ─── Process one channel: returns { A, B } each length = signal.length ───────
async function processChannelFFT(signal, sr, edges, off, edgeSoft, progressCb) {
  const L = signal.length;
  const N = nextPow2(L);

  if (progressCb) progressCb(0.0, 'FFT forward...');
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(signal);
  await new Promise(r => setTimeout(r, 0));
  fft(re, im);

  if (progressCb) progressCb(0.35, 'Building masks...');
  await new Promise(r => setTimeout(r, 0));
  const maskA = buildMask(N, sr, edges, true,  off, edgeSoft);
  const maskB = buildMask(N, sr, edges, false, off, edgeSoft);

  if (progressCb) progressCb(0.45, 'IFFT Track A...');
  const reA = new Float32Array(N), imA = new Float32Array(N);
  for (let i = 0; i < N; i++) { reA[i] = re[i] * maskA[i]; imA[i] = im[i] * maskA[i]; }
  await new Promise(r => setTimeout(r, 0));
  ifft(reA, imA);

  if (progressCb) progressCb(0.7, 'IFFT Track B...');
  // reuse re/im buffers for B — we no longer need the raw spectrum copy
  for (let i = 0; i < N; i++) { re[i] = re[i] * maskB[i]; im[i] = im[i] * maskB[i]; }
  await new Promise(r => setTimeout(r, 0));
  ifft(re, im);

  return {
    A: reA.subarray(0, L),
    B: re.subarray(0, L),
  };
}

// ─── AM envelopes (A and B sum to 1 at every sample, for any shape) ──────────
// `shape` blends between a square wave (0: Track A and B swap with an
// instantaneous on/off step, every 1/modFreq seconds) and a full Hanning/
// raised-cosine crossfade (1: smooth sinusoidal transition, no discontinuity).
//
// FINDING (Peter, 2026-06-24): this parameter, not edge softness or band
// count, was the dominant source of the "gritty/clicky" artifact. A square
// wave has a sharp discontinuity at every transition; sharp discontinuities
// inject broadband harmonic energy on every cycle (40 times/sec), which is
// audible as grit/static independent of which musical content is being
// toggled. A smooth Hanning crossfade has no discontinuity in its derivative,
// so it doesn't generate that broadband splatter. Peter found: gritty below
// ~50%, "bearable" around 50%, clean at 1.0 (full Hanning, the default).
// Net effect: there is little reason to ever lower this below ~0.5, and the
// previously-investigated edge-softness/passthrough fixes were real
// correctness improvements but likely minor compared to this one.
function envAatSample(i, period, shape) {
  const phase = (i / period) % 1.0;
  if (shape >= 1.0) return 0.5 * (1 + Math.cos(2 * Math.PI * phase));
  if (shape <= 0.0) return phase < 0.5 ? 1.0 : 0.0;
  const sq = phase < 0.5 ? 1.0 : 0.0;
  const hn = 0.5 * (1 + Math.cos(2 * Math.PI * phase));
  return sq * (1 - shape) + hn * shape;
}

function applyAM(trackA, trackB, sr, modFreq, shape) {
  const L = trackA.length;
  const period = sr / modFreq;
  const outA = new Float32Array(L);
  const outB = new Float32Array(L);
  const outMix = new Float32Array(L);
  for (let i = 0; i < L; i++) {
    const a = envAatSample(i, period, shape);
    const b = 1 - a;
    const va = trackA[i] * a;
    const vb = trackB[i] * b;
    outA[i] = va;
    outB[i] = vb;
    outMix[i] = va + vb;
  }
  return { A: outA, B: outB, mix: outMix };
}

// ─── WAV encoding (16-bit PCM, mono or stereo, shared normalization) ─────────
function encodeWAV(channels, sampleRate, sharedScale) {
  const numCh = channels.length;
  const numFrames = channels[0].length;
  const byteRate = sampleRate * numCh * 2;
  const blockAlign = numCh * 2;
  const dataSize = numFrames * numCh * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  ws(8, 'WAVE'); ws(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  ws(36, 'data');
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i] * sharedScale));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
  }
  return buffer;
}

function peakOfAll(arrays) {
  let max = 0;
  for (const arr of arrays) for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]); if (a > max) max = a;
  }
  return max;
}

// ─── UI state ────────────────────────────────────────────────────────────────
let audioBuffer = null;
let audioCtx = null;
let blobUrls = [];

function getAudioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function log(msg, type = 'info') {
  const el = document.getElementById('log');
  el.style.display = 'block';
  el.innerHTML += `<div class="log-${type}">${msg}</div>`;
  el.scrollTop = el.scrollHeight;
}

function setProgress(pct, label) {
  document.getElementById('progressContainer').style.display = 'block';
  document.getElementById('progressFill').style.width = (pct * 100) + '%';
  document.getElementById('progressLabel').textContent = label;
}

function drawWaveform(channelData) {
  const canvas = document.getElementById('waveformCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  ctx.fillStyle = '#1a1a24';
  ctx.fillRect(0, 0, W, H);
  const step = Math.ceil(channelData.length / W);
  ctx.strokeStyle = '#7fffb2';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < W; x++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const v = channelData[x * step + j] || 0;
      if (v < min) min = v; if (v > max) max = v;
    }
    ctx.moveTo(x, (1 - (min + 1) / 2) * H);
    ctx.lineTo(x, (1 - (max + 1) / 2) * H);
  }
  ctx.stroke();
}

function drawComb(canvasId, isA, numBands, offLevel, color) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * devicePixelRatio;
  canvas.height = 32 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = canvas.offsetWidth, H = 32;
  ctx.fillStyle = '#1a1a24';
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < numBands; i++) {
    const isOn = isA ? (i % 2 === 0) : (i % 2 === 1);
    const x = i / numBands * W;
    const w = W / numBands - 1;
    const level = isOn ? 1.0 : offLevel;
    ctx.fillStyle = isOn ? color : 'rgba(100,100,120,0.4)';
    ctx.fillRect(x, H - level * H, w, level * H);
  }
}

function updateViz() {
  const n = parseInt(document.getElementById('numBands').value);
  const off = parseFloat(document.getElementById('offLevel').value);
  drawComb('combCanvasA', true, n, off, '#7fffb2');
  drawComb('combCanvasB', false, n, off, '#b27fff');
}

function setupSliders() {
  const sliders = [
    { id: 'numBands',   valId: 'numBandsVal',   fmt: v => v, viz: true },
    { id: 'freqMin',    valId: 'freqMinVal',    fmt: v => v + ' Hz' },
    { id: 'freqMax',    valId: 'freqMaxVal',    fmt: v => v + ' Hz' },
    { id: 'offLevel',   valId: 'offLevelVal',   fmt: v => parseFloat(v).toFixed(2), viz: true },
    { id: 'edgeSoft',   valId: 'edgeSoftVal',   fmt: v => v + '%' },
    { id: 'modFreq',    valId: 'modFreqVal',    fmt: v => v + ' Hz' },
    { id: 'compression',valId: 'compressionVal',fmt: v => parseFloat(v).toFixed(2) },
    { id: 'xfadeShape', valId: 'xfadeShapeVal', fmt: v => parseFloat(v).toFixed(2) },
  ];
  sliders.forEach(({ id, valId, fmt, viz }) => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      document.getElementById(valId).textContent = fmt(el.value);
      if (viz) updateViz();
    });
  });
}

// ─── File load ───────────────────────────────────────────────────────────────
document.getElementById('audioFile').addEventListener('change', async function () {
  const file = this.files[0];
  if (!file) return;
  document.getElementById('fileName').style.display = 'block';
  document.getElementById('fileName').textContent = '▸ ' + file.name;
  const ctx = getAudioContext();
  const ab = await file.arrayBuffer();
  audioBuffer = await ctx.decodeAudioData(ab);
  document.getElementById('waveformContainer').style.display = 'block';
  drawWaveform(audioBuffer.getChannelData(0));
  document.getElementById('processBtn').disabled = false;
  log(`Loaded: ${file.name} — ${audioBuffer.duration.toFixed(1)}s @ ${audioBuffer.sampleRate}Hz, ${audioBuffer.numberOfChannels} ch`, 'ok');
});

const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    const input = document.getElementById('audioFile');
    const dt = new DataTransfer(); dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
  }
});

// ─── Main process ────────────────────────────────────────────────────────────
document.getElementById('processBtn').addEventListener('click', async () => {
  if (!audioBuffer) return;

  document.getElementById('processBtn').disabled = true;
  document.getElementById('players').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('log').innerHTML = '';

  // Revoke any previous blob URLs
  blobUrls.forEach(u => URL.revokeObjectURL(u));
  blobUrls = [];

  const sr = audioBuffer.sampleRate;
  const numCh = audioBuffer.numberOfChannels;
  const numBands = parseInt(document.getElementById('numBands').value);
  const fMin = parseInt(document.getElementById('freqMin').value);
  const fMax = parseInt(document.getElementById('freqMax').value);
  const offLevel = parseFloat(document.getElementById('offLevel').value);
  const edgeSoft = parseInt(document.getElementById('edgeSoft').value) / 100; // % -> fraction of each band's own width
  const modFreq = parseInt(document.getElementById('modFreq').value);
  const xfadeShape = parseFloat(document.getElementById('xfadeShape').value);
  const compression = parseFloat(document.getElementById('compression').value);

  const edges = getErbEdges(numBands, fMin, fMax);

  log(`FFT masking — ${numBands} bands, ${fMin}-${fMax}Hz, off=${offLevel}, softness=${edgeSoft} (audio outside ${fMin}-${fMax}Hz always preserved unmodified)`, 'info');
  log(`Processing ${numCh} channel(s) @ ${sr}Hz, ${audioBuffer.length} samples`, 'info');

  const t0 = performance.now();

  // Process each channel
  const chAm = []; // [{A, B, mix}, ...] per channel, each channel same length
  for (let c = 0; c < numCh; c++) {
    const rawSignal = audioBuffer.getChannelData(c);
    const chStart = c / numCh;
    const chSpan = 1 / numCh;
    setProgress(chStart, `Ch ${c+1}/${numCh}: compression...`);
    await new Promise(r => setTimeout(r, 0));
    const signal = compressDynamics(rawSignal, sr, compression);
    log(`Channel ${c}: compression=${compression.toFixed(2)}, then forward FFT + masking...`, 'info');
    const { A, B } = await processChannelFFT(
      signal, sr, edges, offLevel, edgeSoft,
      (p, label) => setProgress(chStart + p * chSpan * 0.9, `Ch ${c+1}/${numCh}: ${label}`)
    );
    setProgress(chStart + chSpan * 0.95, `Ch ${c+1}/${numCh}: applying 40Hz AM...`);
    await new Promise(r => setTimeout(r, 0));
    const am = applyAM(A, B, sr, modFreq, xfadeShape);
    chAm.push(am);
  }

  setProgress(0.98, 'Encoding WAVs...');
  await new Promise(r => setTimeout(r, 0));

  // Shared peak normalization across all output tracks so relative levels are preserved
  const allArrays = [];
  for (const am of chAm) { allArrays.push(am.A, am.B, am.mix); }
  const peak = peakOfAll(allArrays);
  const scale = peak > 0 ? 0.95 / peak : 1;

  // Pack into {mix, A, B} each as multi-channel arrays
  const mixChannels = chAm.map(a => a.mix);
  const aChannels   = chAm.map(a => a.A);
  const bChannels   = chAm.map(a => a.B);

  const mixWav = encodeWAV(mixChannels, sr, scale);
  const aWav   = encodeWAV(aChannels,   sr, scale);
  const bWav   = encodeWAV(bChannels,   sr, scale);

  const mixUrl = URL.createObjectURL(new Blob([mixWav], { type: 'audio/wav' }));
  const aUrl   = URL.createObjectURL(new Blob([aWav],   { type: 'audio/wav' }));
  const bUrl   = URL.createObjectURL(new Blob([bWav],   { type: 'audio/wav' }));
  blobUrls = [mixUrl, aUrl, bUrl];

  document.getElementById('audioMix').src = mixUrl;
  document.getElementById('audioA').src = aUrl;
  document.getElementById('audioB').src = bUrl;
  document.getElementById('dlMix').href = mixUrl;
  document.getElementById('dlA').href = aUrl;
  document.getElementById('dlB').href = bUrl;

  // Stats
  document.getElementById('results').style.display = 'block';
  document.getElementById('statDuration').textContent = audioBuffer.duration.toFixed(2) + 's';
  document.getElementById('statSampleRate').textContent = sr + ' Hz';
  document.getElementById('statChannels').textContent = numCh === 1 ? 'Mono' : numCh === 2 ? 'Stereo' : numCh + ' ch';
  document.getElementById('statCompression').textContent = compression.toFixed(2);
  document.getElementById('statBands').textContent = numBands + ' (' + Math.ceil(numBands/2) + ' per comb)';
  document.getElementById('statOffLevel').textContent = offLevel.toFixed(2);
  document.getElementById('statEdge').textContent = (edgeSoft * 100).toFixed(0) + '%';
  document.getElementById('statMod').textContent = modFreq + 'Hz / ' + (xfadeShape >= 1 ? 'Hanning' : xfadeShape <= 0 ? 'Square' : 'Blend ' + xfadeShape.toFixed(2));

  document.getElementById('players').style.display = 'block';
  setProgress(1.0, 'Complete.');
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  log(`Done in ${secs}s. Scroll down to listen.`, 'ok');
  document.getElementById('processBtn').disabled = false;
});

// Init
setupSliders();
updateViz();
window.addEventListener('resize', () => {
  updateViz();
  if (audioBuffer) drawWaveform(audioBuffer.getChannelData(0));
});
