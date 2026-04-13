/**
 * NanoPitch C inference engine.
 *
 * Implements mel spectrogram computation and GRU neural network inference.
 * Compiles to WASM for browser deployment with Emscripten.
 *
 * This file is a complete, self-contained pitch detection engine written in
 * plain C.  It takes raw 16 kHz audio in, and produces a fundamental-
 * frequency (f0) estimate in Hz for every 10 ms frame.  The pipeline is:
 *
 *   raw audio  -->  Hann window  -->  FFT  -->  mel filterbank  -->  log
 *        -->  1-D convolutions  -->  3x stacked GRU  -->  dense heads
 *        -->  pitch posterior + VAD  -->  online Viterbi  -->  f0 (Hz)
 *
 * KEY CONCEPTS COVERED IN THIS FILE:
 *
 *   1. Mel filterbank (HTK mel scale, triangular filters)
 *      - Maps perceptual pitch spacing to a compact representation.
 *
 *   2. FFT (radix-2 Cooley-Tukey decimation-in-time)
 *      - Efficiently converts time-domain audio into frequency-domain.
 *
 *   3. Hann window (raised cosine, spectral leakage prevention)
 *      - Smoothly tapers the edges of each analysis frame.
 *
 *   4. Streaming overlap-save mel computation
 *      - Allows frame-by-frame processing without storing the full signal.
 *
 *   5. GRU (Gated Recurrent Unit) cells
 *      - Recurrent neural network layer that captures temporal context.
 *
 *   6. Sigmoid activation as a probability function
 *      - Squashes unbounded logits into the [0,1] range.
 *
 *   7. Online Viterbi pitch tracker
 *      - Dynamic programming that temporally smooths pitch estimates.
 *
 *   8. Weight loading from a flat array (zero-copy memory layout)
 *      - Maps a contiguous block of floats to individual weight matrices.
 *
 * The code targets WASM (WebAssembly) via Emscripten so it can run in the
 * browser at real-time speed.  Every design choice (small model, in-place
 * memory reuse, causal convolutions) supports sub-millisecond latency on
 * a modern device.
 */

#include "nanopitch.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ════════════════════════════════════════════════════════════════════
 * Mel filterbank (auto-generated values matching Python exactly)
 * ════════════════════════════════════════════════════════════════════
 *
 * HTK MEL SCALE AND TRIANGULAR FILTERBANKS
 * -------------------------------------------------------------
 *
 * Human hearing perceives pitch on a roughly logarithmic scale: the
 * perceptual distance between 100 Hz and 200 Hz feels about the same
 * as between 1000 Hz and 2000 Hz.  The mel scale was introduced by
 * Stevens, Volkmann, and Newman (1937) to quantify this.  The "HTK"
 * variant (used in the Hidden Markov Toolkit and adopted as a de facto
 * standard) defines the mapping:
 *
 *     mel(f) = 2595 * log10(1 + f / 700)
 *
 * Its inverse is:
 *
 *     hz(m) = 700 * (10^(m / 2595) - 1)
 *
 * To build a mel filterbank:
 *
 *   (a) Choose how many filters you want (here NC_N_MELS = 40) and a
 *       frequency range (0 Hz to 8000 Hz, which is the Nyquist
 *       frequency for 16 kHz audio).
 *
 *   (b) Convert the low and high frequencies to mel, then place
 *       (NC_N_MELS + 2) points evenly spaced on the mel axis.
 *       The "+2" is because each triangular filter needs a left edge,
 *       a center, and a right edge, and adjacent filters share edges.
 *       So NC_N_MELS filters need NC_N_MELS + 2 boundary points.
 *
 *   (c) Convert those mel-spaced points back to Hz, then to FFT bin
 *       indices (fractional) via:   bin = hz * N_FFT / sample_rate
 *
 *   (d) For each filter m, construct a triangle that rises linearly
 *       from 0 at the left edge to 1 at the center, then falls back
 *       to 0 at the right edge.  Concretely:
 *
 *          filter_m[k] = (k - left)   / (center - left)    if left <= k < center
 *                      = (right - k)  / (right  - center)  if center <= k <= right
 *                      = 0                                  otherwise
 *
 *       This means the filters are wider (cover more FFT bins) at
 *       high frequencies, mirroring the decreased frequency resolution
 *       of human hearing in that region.
 *
 *   (e) Multiply each filter by the power spectrum and sum to get the
 *       energy in that mel band.  Take the log to compress the huge
 *       dynamic range of audio energy (easily 10^6 between quiet and
 *       loud) into a compact, roughly perceptually-uniform scale.
 *
 * The result -- a 40-dimensional log-mel vector per frame -- is the
 * standard input representation for speech and music neural networks.
 */

/* HTK mel scale -- convert frequency in Hz to mel units */
static float hz_to_mel(float f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}

/* Inverse: convert mel units back to Hz */
static float mel_to_hz(float m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

/*
 * Pre-computed mel filterbank matrix.
 *
 * mel_fb[m][k] stores the weight of FFT bin k for mel filter m.
 * Dimensions: [NC_N_MELS (40)][NC_N_FREQS (257)].
 *
 * NC_N_FREQS = N_FFT/2 + 1 = 257 because the FFT of a real signal is
 * conjugate-symmetric, so only the first half (plus DC and Nyquist)
 * carries unique information.
 *
 * This is "lazily initialized" -- computed once on the first call, then
 * cached in a static array for subsequent frames.  This avoids startup
 * cost if the module is loaded but not used.
 */
static float mel_fb[NC_N_MELS][NC_N_FREQS];
static int mel_fb_initialized = 0;

static void init_mel_filterbank(void) {
    if (mel_fb_initialized) return;

    /* Step (a): Convert frequency range endpoints to mel */
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(8000.0f);

    /*
     * Step (b): Place NC_N_MELS + 2 points uniformly in mel space.
     *
     * mel_points[] -- the evenly-spaced mel values
     * hz_points[]  -- those same points converted back to Hz
     * bin_points[] -- the corresponding (fractional) FFT bin indices
     *
     * Example: if NC_N_MELS = 40, we create 42 boundary points.
     * Filter m uses points [m], [m+1], [m+2] as left/center/right.
     */
    float mel_points[NC_N_MELS + 2];
    float hz_points[NC_N_MELS + 2];
    float bin_points[NC_N_MELS + 2];

    for (int i = 0; i < NC_N_MELS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (NC_N_MELS + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        /* bin = frequency * (N_FFT / sample_rate).  This gives a fractional
         * FFT bin index, which is fine -- the triangular filter weight is
         * computed via linear interpolation between bins anyway. */
        bin_points[i] = hz_points[i] * NC_N_FFT / NC_SAMPLE_RATE;
    }

    /* Start with all zeros -- bins outside any triangle remain zero */
    memset(mel_fb, 0, sizeof(mel_fb));

    /*
     * Step (d): Build each triangular filter.
     *
     * For filter m:
     *   f_left   = bin_points[m]       (left foot of the triangle)
     *   f_center = bin_points[m + 1]   (peak of the triangle, weight = 1)
     *   f_right  = bin_points[m + 2]   (right foot)
     *
     * The ascending slope runs from f_left to f_center (0 -> 1).
     * The descending slope runs from f_center to f_right (1 -> 0).
     * Adjacent triangles overlap: the right foot of filter m is the
     * center of filter m+1, ensuring that every FFT bin contributes
     * to at least one filter (no gaps in frequency coverage).
     */
    for (int m = 0; m < NC_N_MELS; m++) {
        float f_left   = bin_points[m];
        float f_center = bin_points[m + 1];
        float f_right  = bin_points[m + 2];

        for (int k = 0; k < NC_N_FREQS; k++) {
            float kf = (float)k;
            if (kf >= f_left && kf < f_center) {
                /* Ascending slope: linearly ramp from 0 at f_left to 1
                 * at f_center.  This is the "left edge" of the triangle. */
                mel_fb[m][k] = (kf - f_left) / (f_center - f_left);
            } else if (kf >= f_center && kf <= f_right) {
                /* Descending slope: linearly ramp from 1 at f_center to
                 * 0 at f_right.  This is the "right edge." */
                mel_fb[m][k] = (f_right - kf) / (f_right - f_center);
            }
            /* else: stays 0 (bin is outside this filter's triangle) */
        }
    }

    mel_fb_initialized = 1;
}

/* ════════════════════════════════════════════════════════════════════
 * FFT -- simple radix-2 DIT for N=512
 * ════════════════════════════════════════════════════════════════════
 *
 * HOW THE FFT WORKS (RADIX-2 COOLEY-TUKEY)
 * -------------------------------------------------------------
 *
 * The Discrete Fourier Transform (DFT) converts N time-domain samples
 * into N frequency-domain coefficients.  A naive DFT is O(N^2), but the
 * Fast Fourier Transform (FFT) reduces this to O(N log N) by exploiting
 * the symmetry and periodicity of the complex exponential "twiddle
 * factors" W_N^k = e^{-j 2 pi k / N}.
 *
 * The Cooley-Tukey algorithm (1965) is the most widely used FFT variant.
 * The "radix-2 decimation-in-time" (DIT) version requires N to be a
 * power of 2 (here N = 512 = 2^9).  It works by:
 *
 *   1. DIVIDE: Split the N-point DFT into two N/2-point DFTs --
 *      one on the even-indexed samples, one on the odd-indexed samples.
 *
 *   2. CONQUER: Recursively compute those smaller DFTs.
 *
 *   3. COMBINE: Merge results using "butterfly" operations:
 *        X[k]       = E[k] + W_N^k * O[k]
 *        X[k + N/2] = E[k] - W_N^k * O[k]
 *      where E[k] and O[k] are the DFTs of even and odd samples.
 *
 * In practice, the recursion is "unrolled" into an iterative algorithm
 * with two phases:
 *
 *   Phase 1 -- BIT-REVERSAL PERMUTATION:
 *     The recursive splitting means the input must be reordered so that
 *     element x[i] is moved to position bit_reverse(i).  For example,
 *     with N=8 (3 bits), index 3 (binary 011) maps to 6 (binary 110).
 *     After this reordering, the butterflies can be performed entirely
 *     in-place without further data movement.
 *
 *   Phase 2 -- BUTTERFLY STAGES:
 *     There are log2(N) stages (here 9 stages for N=512).  Stage s
 *     operates on groups of size 2^s.  Within each group, pairs of
 *     elements are combined using a complex "twiddle factor":
 *
 *       u = x[i]
 *       v = W * x[i + half]       (complex multiply)
 *       x[i]        = u + v       (top butterfly output)
 *       x[i + half] = u - v       (bottom butterfly output)
 *
 *     where W = e^{-j 2 pi k / group_size} rotates through the unit
 *     circle as k increases.
 *
 * For a 512-point FFT on real audio:
 *   - We get 257 unique frequency bins (0 through Nyquist).
 *   - Bin spacing = sample_rate / N_FFT = 16000/512 = 31.25 Hz.
 *   - The highest representable frequency is sample_rate/2 = 8000 Hz.
 *
 * This particular implementation works on complex input (cpx struct)
 * and is not optimized for real-only input.  For N=512 in WASM, it is
 * already fast enough for real-time use.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Complex number type used by the FFT */
typedef struct { float re, im; } cpx;

static void fft512(cpx *x) {
    const int N = NC_N_FFT;  /* 512 */

    /*
     * PHASE 1: Bit-reversal permutation.
     *
     * This loop visits every index i from 1 to N-1 and computes the
     * corresponding bit-reversed index j.  When i < j, the elements
     * are swapped.  The "i < j" condition ensures each pair is swapped
     * only once.
     *
     * The bit-reversal is computed incrementally (not by reversing all
     * bits each time).  The inner while loop "increments" j in
     * bit-reversed order by toggling bits from the MSB downward,
     * analogous to how a normal increment propagates carries upward.
     */
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            cpx tmp = x[i]; x[i] = x[j]; x[j] = tmp;
        }
    }

    /*
     * PHASE 2: Cooley-Tukey butterfly stages.
     *
     * Outer loop: `len` is the current butterfly group size.
     * It doubles each iteration: 2, 4, 8, 16, ..., 512.
     * That gives log2(512) = 9 stages.
     *
     * `wlen` is the principal twiddle factor for groups of size `len`:
     *     wlen = e^{-j * 2*pi / len} = cos(-2*pi/len) + j*sin(-2*pi/len)
     *
     * Middle loop: iterates over groups of size `len`.
     * Inner loop: performs len/2 butterfly operations within each group.
     *
     * Each butterfly computes:
     *     u = x[i + j]                           (even element)
     *     v = w * x[i + j + len/2]               (odd element, rotated)
     *     x[i + j]          = u + v
     *     x[i + j + len/2]  = u - v
     *
     * After all stages, x[] holds the DFT coefficients.
     */
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        cpx wlen = { cosf(ang), sinf(ang) };
        for (int i = 0; i < N; i += len) {
            cpx w = { 1.0f, 0.0f };  /* w starts at 1 and rotates by wlen */
            for (int j = 0; j < len / 2; j++) {
                cpx u = x[i + j];
                /* Complex multiplication: v = w * x[i + j + len/2]
                 *   (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
                cpx v = {
                    w.re * x[i + j + len/2].re - w.im * x[i + j + len/2].im,
                    w.re * x[i + j + len/2].im + w.im * x[i + j + len/2].re
                };
                x[i + j].re = u.re + v.re;
                x[i + j].im = u.im + v.im;
                x[i + j + len/2].re = u.re - v.re;
                x[i + j + len/2].im = u.im - v.im;
                /* Advance the twiddle factor: w = w * wlen
                 * This rotates w by one step around the unit circle. */
                float wnew_re = w.re * wlen.re - w.im * wlen.im;
                float wnew_im = w.re * wlen.im + w.im * wlen.re;
                w.re = wnew_re;
                w.im = wnew_im;
            }
        }
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Hann window
 * ════════════════════════════════════════════════════════════════════
 *
 * HOW THE HANN WINDOW PREVENTS SPECTRAL LEAKAGE
 * ------------------------------------------------------------------
 *
 * When we extract a short frame of audio for spectral analysis, we are
 * implicitly multiplying the infinite signal by a rectangular window
 * (1 inside the frame, 0 outside).  In the frequency domain, this is
 * equivalent to convolving the true spectrum with a sinc function -- a
 * function with wide sidelobes that "smear" energy across bins.  This
 * artifact is called SPECTRAL LEAKAGE.
 *
 * For pitch detection, leakage is devastating: a pure 440 Hz tone
 * would produce phantom energy spread across many FFT bins, making it
 * harder for the neural network to identify the true fundamental.
 *
 * The Hann window (also called Hanning, after Julius von Hann) is a
 * raised-cosine taper:
 *
 *     w[n] = 0.5 * (1 - cos(2*pi*n / N))
 *
 * It smoothly brings the signal amplitude to zero at both edges of the
 * frame.  In the frequency domain, its "mainlobe" is wider than the
 * rectangular window's (so frequency resolution is slightly reduced),
 * but its "sidelobes" are dramatically suppressed (about -32 dB for
 * the first sidelobe, vs. -13 dB for rectangular).  This trade-off --
 * slightly wider mainlobe but much cleaner sidelobes -- is ideal for
 * most music and speech applications.
 *
 * Our window length is NC_WIN_LENGTH = 400 samples = 25 ms at 16 kHz.
 * The hop length is NC_HOP_LENGTH = 160 samples = 10 ms.  The overlap
 * ratio is (400 - 160) / 400 = 60%, which ensures that the windows
 * add up to a nearly constant value across time (COLA -- Constant
 * Overlap-Add -- property), important for synthesis but also a sign
 * that no audio information is "lost" between frames.
 *
 * Like the mel filterbank, the window is computed once and cached.
 */
static float hann_win[NC_WIN_LENGTH];
static int hann_initialized = 0;

static void init_hann(void) {
    if (hann_initialized) return;
    for (int i = 0; i < NC_WIN_LENGTH; i++) {
        /*
         * The classic Hann formula.  Note: some references use (N-1) in
         * the denominator (the "periodic" vs. "symmetric" Hann window).
         * Using N (as here) gives the "periodic" variant, which is
         * preferred for FFT-based spectral analysis because it maintains
         * perfect COLA overlap-add reconstruction.
         */
        hann_win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / NC_WIN_LENGTH));
    }
    hann_initialized = 1;
}

/* ════════════════════════════════════════════════════════════════════
 * Mel spectrogram computation
 * ════════════════════════════════════════════════════════════════════
 *
 * STREAMING OVERLAP-SAVE MEL COMPUTATION
 * -----------------------------------------------------------
 *
 * In a traditional (offline) mel spectrogram, you have the entire audio
 * signal in memory and can extract overlapping frames at will.  In a
 * real-time (streaming) system, audio arrives one small chunk at a time
 * -- here, NC_HOP_LENGTH = 160 samples (10 ms) per call.
 *
 * But each analysis window needs NC_WIN_LENGTH = 400 samples (25 ms).
 * So we need 400 - 160 = 240 samples of "history" from the previous
 * chunk.  This is the OVERLAP-SAVE approach:
 *
 *   analysis_mem[]:   [....... 400 samples from previous window .......]
 *
 *   To form the current window, we take:
 *     - The last 240 samples from analysis_mem (the "overlap")
 *     - The 160 new samples from the current audio_frame
 *
 *   That gives us 240 + 160 = 400 samples, exactly one window's worth.
 *
 *   We then save the full 400 samples into analysis_mem for next time.
 *
 * This is extremely memory-efficient: we store only ONE window's worth
 * of history (400 floats = 1.6 KB), regardless of how long the audio
 * stream is.  This is critical for WASM deployment where memory is
 * limited.
 *
 * After assembling the window:
 *   1. Apply the Hann window (element-wise multiply)
 *   2. Zero-pad from 400 to 512 samples (the FFT size)
 *   3. Compute the 512-point FFT
 *   4. Compute the power spectrum: |X[k]|^2 = re^2 + im^2
 *      (Only the first 257 bins, since the rest are conjugate mirrors)
 *   5. Apply the mel filterbank: energy_m = sum_k(filter[m][k] * power[k])
 *   6. Take the log: log_mel_m = log(energy_m + epsilon)
 *      (The epsilon = 1e-10 prevents log(0) = -infinity)
 */
void nanopitch_compute_mel(NanoPitchState *st, const float *audio_frame,
                           float *out_mel) {
    init_mel_filterbank();
    init_hann();

    /*
     * Build the analysis window using the overlap-save method.
     *
     * analysis_mem contains the previous window's 400 samples.
     * We shift out the oldest NC_HOP_LENGTH (160) samples and append
     * the new 160 samples from audio_frame.
     *
     * Before:  analysis_mem = [ old_240 | old_160 ]   (400 total)
     * After:   window_buf   = [ old_160..old_399 | new_0..new_159 ]
     *                       = [ overlap (240)    | new hop (160)  ]
     */
    float window_buf[NC_WIN_LENGTH];
    int overlap = NC_WIN_LENGTH - NC_HOP_LENGTH;  /* 400 - 160 = 240 */

    /* Copy the "overlap" portion: the last 240 samples from analysis_mem */
    memcpy(window_buf, st->analysis_mem + NC_HOP_LENGTH,
           overlap * sizeof(float));
    /* Append the new hop of 160 fresh samples */
    memcpy(window_buf + overlap, audio_frame,
           NC_HOP_LENGTH * sizeof(float));

    /* Save the assembled window for next frame's overlap */
    memcpy(st->analysis_mem, window_buf, NC_WIN_LENGTH * sizeof(float));

    /* Apply the Hann window: element-wise multiply to taper frame edges.
     * This is the critical step that suppresses spectral leakage. */
    float windowed[NC_WIN_LENGTH];
    for (int i = 0; i < NC_WIN_LENGTH; i++) {
        windowed[i] = window_buf[i] * hann_win[i];
    }

    /*
     * Zero-pad from NC_WIN_LENGTH (400) to NC_N_FFT (512) and compute FFT.
     *
     * Why zero-pad?  The FFT size must be a power of 2 for radix-2, and
     * making it larger than the window length gives finer frequency
     * resolution (more interpolation between bins) at no extra cost in
     * spectral leakage.  400 -> 512 is a common choice.
     */
    cpx fft_buf[NC_N_FFT];
    memset(fft_buf, 0, sizeof(fft_buf));
    for (int i = 0; i < NC_WIN_LENGTH; i++) {
        fft_buf[i].re = windowed[i];
        fft_buf[i].im = 0.0f;  /* input is real-valued audio */
    }
    fft512(fft_buf);

    /*
     * Power spectrum: |X[k]|^2 = re(X[k])^2 + im(X[k])^2
     *
     * We only need the first NC_N_FREQS = 257 bins (0 through Nyquist).
     * Bins 258..511 are the complex conjugate mirror and carry no new
     * information for real-valued input.
     */
    float power[NC_N_FREQS];
    for (int i = 0; i < NC_N_FREQS; i++) {
        power[i] = fft_buf[i].re * fft_buf[i].re
                 + fft_buf[i].im * fft_buf[i].im;
    }

    /*
     * Apply the mel filterbank and take the log.
     *
     * For each mel band m, compute:
     *     energy_m = sum over k of (mel_fb[m][k] * power[k])
     *     out_mel[m] = log(energy_m + NC_LOG_OFFSET)
     *
     * NC_LOG_OFFSET (1e-10) is a tiny constant that prevents log(0)
     * when a mel band has zero energy (e.g., in silence).
     *
     * The log compression is essential: raw power values span many
     * orders of magnitude (a whisper might have energy ~0.001, while a
     * shout could be ~1000).  Logarithm maps this to a ~linear
     * perceptual loudness scale, making the neural network's job easier.
     */
    for (int m = 0; m < NC_N_MELS; m++) {
        float energy = 0.0f;
        for (int k = 0; k < NC_N_FREQS; k++) {
            energy += mel_fb[m][k] * power[k];
        }
        out_mel[m] = logf(energy + NC_LOG_OFFSET);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Neural network operations
 * ════════════════════════════════════════════════════════════════════ */

/*
 * SIGMOID ACTIVATION AS A PROBABILITY
 * --------------------------------------------------------
 *
 * The sigmoid (logistic) function maps any real number to the open
 * interval (0, 1):
 *
 *     sigma(x) = 1 / (1 + e^{-x})
 *
 * Properties that make it useful in neural networks:
 *
 *   - As x -> +infinity, sigma(x) -> 1
 *   - As x -> -infinity, sigma(x) -> 0
 *   - sigma(0) = 0.5
 *   - It is monotonically increasing and differentiable everywhere
 *   - Its output can be interpreted as a probability
 *
 * In this model, sigmoid is used in two places:
 *
 *   1. GRU GATES (reset and update):
 *      Gates need to be in [0,1] to act as "soft switches."  A gate
 *      value of 0 means "completely block," 1 means "completely pass."
 *      Sigmoid provides exactly this range.
 *
 *   2. OUTPUT HEADS (VAD and pitch posterior):
 *      The voice-activity detector (VAD) outputs a single sigmoid value
 *      interpreted as P(voice active).  The pitch head outputs one
 *      sigmoid per bin, giving an independent probability for each
 *      pitch.  (This is NOT softmax; each bin is independent, which
 *      allows the model to express uncertainty by activating several
 *      neighboring bins simultaneously.)
 */
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Thin wrapper around tanhf for readability.  tanh maps R -> (-1, 1)
 * and is used as the activation for the GRU candidate hidden state. */
static float tanhf_safe(float x) {
    return tanhf(x);
}

/*
 * HOW GRU CELLS WORK STEP BY STEP
 * ----------------------------------------------------
 *
 * A GRU (Gated Recurrent Unit, Cho et al. 2014) is a type of recurrent
 * neural network cell that maintains a "hidden state" vector h across
 * time steps.  At each step, it reads a new input x and updates h.
 *
 * The GRU has two "gates" -- learned functions that control information
 * flow -- and a "candidate" hidden state:
 *
 *   STEP 1 -- RESET GATE (r):
 *     r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
 *
 *     The reset gate decides how much of the previous hidden state to
 *     "forget" when computing the candidate.  When r ≈ 0, the candidate
 *     ignores the old hidden state entirely and just looks at the current
 *     input, like starting fresh.  When r ≈ 1, the full old state is used.
 *
 *   STEP 2 -- UPDATE GATE (z):
 *     z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
 *
 *     The update gate controls how much of the candidate to mix into the
 *     final hidden state.  Think of it as a "blend dial" between old and
 *     new.  When z ≈ 1, the old hidden state is kept unchanged (useful for
 *     carrying information over long spans).  When z ≈ 0, the new candidate
 *     completely replaces the old state.
 *
 *   STEP 3 -- CANDIDATE HIDDEN STATE (n):
 *     n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
 *
 *     This is the "proposed" new hidden state.  The reset gate r appears
 *     here, multiplying the old-hidden-state contribution.  The tanh
 *     activation squashes values to [-1, 1], keeping magnitudes bounded.
 *
 *   STEP 4 -- FINAL HIDDEN STATE:
 *     h_new = (1 - z) * n + z * h_old
 *
 *     This is the key equation.  It linearly interpolates between the
 *     candidate n and the old hidden state h, controlled by the update
 *     gate z.  Note the convention: z=1 means "keep old," z=0 means
 *     "use new."  (Some references swap this; PyTorch uses this convention.)
 *
 * WHY GRU INSTEAD OF LSTM?
 *   GRUs have fewer parameters (2 gates vs. 3) and are slightly faster
 *   to compute, with similar performance on many tasks.  For a tiny
 *   real-time model running in WASM, every saved multiply counts.
 *
 * WEIGHT LAYOUT (PyTorch convention):
 *   W_ih = [W_ir; W_iz; W_in] stacked vertically, shape [3*hidden, input]
 *   W_hh = [W_hr; W_hz; W_hn] stacked vertically, shape [3*hidden, hidden]
 *   b_ih = [b_ir; b_iz; b_in] concatenated, length 3*hidden
 *   b_hh = [b_hr; b_hz; b_hn] concatenated, length 3*hidden
 *
 *   The first `hidden_size` rows correspond to the reset gate (r),
 *   the next `hidden_size` rows to the update gate (z),
 *   and the last `hidden_size` rows to the candidate (n).
 */

/**
 * GRU cell: single time step.
 *
 * h_new = GRU(x, h_old) where x is [input_size] and h is [hidden_size].
 *
 * PyTorch GRU equations:
 *   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
 *   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
 *   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
 *   h_new = (1 - z) * n + z * h
 *
 * W_ih = [W_ir; W_iz; W_in] of shape [3*hidden, input]
 * W_hh = [W_hr; W_hz; W_hn] of shape [3*hidden, hidden]
 */
static void gru_step(const float *x, float *h, int input_size, int hidden_size,
                     const float *w_ih, const float *w_hh,
                     const float *b_ih, const float *b_hh) {
    int h3 = 3 * hidden_size;
    float *gates_ih = (float *)calloc(h3, sizeof(float));
    float *gates_hh = (float *)calloc(h3, sizeof(float));
    float *h_new = (float *)calloc(hidden_size, sizeof(float));

    /*
     * Compute gates_ih = W_ih @ x + b_ih
     *
     * This is a matrix-vector product.  W_ih has shape [3*hidden, input],
     * so gates_ih has length 3*hidden.  The three sections are:
     *   gates_ih[0..hidden-1]           -> input contribution to reset gate
     *   gates_ih[hidden..2*hidden-1]    -> input contribution to update gate
     *   gates_ih[2*hidden..3*hidden-1]  -> input contribution to candidate
     *
     * Memory layout: W_ih is stored row-major, so W_ih[i][j] is at
     * offset i * input_size + j.
     */
    for (int i = 0; i < h3; i++) {
        float sum = b_ih[i];
        for (int j = 0; j < input_size; j++) {
            sum += w_ih[i * input_size + j] * x[j];
        }
        gates_ih[i] = sum;
    }

    /*
     * Compute gates_hh = W_hh @ h + b_hh
     *
     * Same structure, but operates on the previous hidden state h instead
     * of the input x.  W_hh has shape [3*hidden, hidden].
     */
    for (int i = 0; i < h3; i++) {
        float sum = b_hh[i];
        for (int j = 0; j < hidden_size; j++) {
            sum += w_hh[i * hidden_size + j] * h[j];
        }
        gates_hh[i] = sum;
    }

    /*
     * Combine the two sets of pre-activations to form r, z, n gates,
     * then compute the new hidden state.
     *
     * For each hidden unit i:
     *   r[i] = sigmoid(gates_ih[i] + gates_hh[i])               -- reset gate
     *   z[i] = sigmoid(gates_ih[H+i] + gates_hh[H+i])          -- update gate
     *   n[i] = tanh(gates_ih[2H+i] + r[i] * gates_hh[2H+i])   -- candidate
     *   h_new[i] = (1 - z[i]) * n[i] + z[i] * h[i]            -- interpolate
     *
     * Note: the reset gate r multiplies only the hidden-state contribution
     * (gates_hh) in the candidate computation, NOT the input contribution.
     * This is the PyTorch convention (mode="default").
     */
    for (int i = 0; i < hidden_size; i++) {
        float r = sigmoid(gates_ih[i] + gates_hh[i]);
        float z = sigmoid(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
        float n = tanhf_safe(gates_ih[2 * hidden_size + i]
                             + r * gates_hh[2 * hidden_size + i]);
        h_new[i] = (1.0f - z) * n + z * h[i];
    }

    memcpy(h, h_new, hidden_size * sizeof(float));
    free(gates_ih);
    free(gates_hh);
    free(h_new);
}

/*
 * Dense (fully connected) layer with sigmoid activation.
 *
 *   out[i] = sigmoid(sum_j(weight[i][j] * x[j]) + bias[i])
 *
 * This is used for the output heads.  The sigmoid makes each output an
 * independent probability in [0, 1] (see the sigmoid note above).
 *
 * Weight layout: row-major, weight[i * in_size + j] = W[i][j].
 */
/** Dense layer: out = sigmoid(W @ x + b) */
static void dense_sigmoid(const float *x, float *out, int in_size, int out_size,
                          const float *weight, const float *bias) {
    for (int i = 0; i < out_size; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_size; j++) {
            sum += weight[i * in_size + j] * x[j];
        }
        out[i] = sigmoid(sum);
    }
}

/* ════════════════════════════════════════════════════════════════════
 * Online Viterbi pitch tracking
 * ════════════════════════════════════════════════════════════════════
 *
 * HOW THE ONLINE VITERBI PITCH TRACKER WORKS
 * ---------------------------------------------------------------
 *
 * The neural network outputs a "pitch posterior" -- a probability for
 * each of NC_PITCH_BINS (360) bins spanning from ~32 Hz (PITCH_FMIN)
 * upward in 20-cent steps.  (20 cents = 1/5 of a semitone, so there
 * are about 5 bins per semitone and 60 bins per octave.)  We could
 * just pick the bin with the highest probability each frame, but that
 * would produce jittery, noisy pitch tracks.
 *
 * Instead, we use a Viterbi-style dynamic programming algorithm that
 * considers both the neural network's frame-level predictions AND the
 * smoothness of the resulting pitch trajectory.  The classic Viterbi
 * algorithm requires seeing the entire sequence before backtracking,
 * but this is an ONLINE variant that outputs a decision every frame
 * without buffering future frames.  (The tradeoff is that it can only
 * look backward, not forward, so it may occasionally be less accurate
 * than offline Viterbi.)
 *
 * STATE SPACE:
 *   - States 0 through N-1 (N = NC_PITCH_BINS = 360): voiced at that
 *     pitch bin.  Bin b maps to frequency:
 *         f0 = PITCH_FMIN * 2^(b * PITCH_CENTS_PER_BIN / 1200)
 *     This is a standard cents-to-Hz formula (1200 cents = 1 octave).
 *
 *   - State N: unvoiced (no pitch).
 *
 *   Total: N + 1 = 361 states.
 *
 * OBSERVATION PROBABILITIES:
 *   - For voiced state s: log(posterior[s] + epsilon)
 *   - For unvoiced state: log(1 - max(posterior) + epsilon)
 *     i.e., the more confident the network is about any pitch, the
 *     less likely it is that the frame is unvoiced.
 *
 * TRANSITION PROBABILITIES:
 *   - Voiced -> voiced: only allowed within a +/- VITERBI_TRANSITION_WIDTH
 *     (12) bin neighborhood.  This prevents the pitch from jumping more
 *     than 12 * 20 = 240 cents (2.4 semitones) per frame (10 ms).
 *     Within the allowed range, all transitions are equally likely
 *     (uniform cost).
 *
 *   - Voiced -> unvoiced: incurs a penalty of -0.75 (in log space).
 *     This discourages frequent voiced/unvoiced switches (flutter).
 *
 *   - Unvoiced -> voiced: same -0.75 penalty ("onset" cost).
 *
 *   - Unvoiced -> unvoiced: free (no penalty).
 *
 * ALGORITHM (per frame):
 *   For each state s in the current frame:
 *     1. Find the best predecessor from the previous frame, considering
 *        only states within the allowed transition neighborhood.
 *     2. Also consider transitioning from the unvoiced state (with
 *        penalty).
 *     3. Pick whichever predecessor gives the higher accumulated score.
 *     4. Add the observation log-probability for the current frame.
 *
 *   The state with the highest accumulated score is the output for this
 *   frame.  (A full offline Viterbi would store backpointers and trace
 *   back at the end; here we output greedily each frame.)
 */

#define PITCH_FMIN 31.7f
#define PITCH_CENTS_PER_BIN 20.0f
#define VITERBI_TRANSITION_WIDTH 12
#define VITERBI_VOICING_THRESHOLD 0.3f
#define VITERBI_ONSET_PENALTY 0.75f

/*
 * Convert a pitch bin index to a frequency in Hz.
 *
 * The bin spacing is 20 cents (PITCH_CENTS_PER_BIN).  Since there are
 * 1200 cents per octave, the formula is:
 *     f0 = f_min * 2^(bin * 20 / 1200)
 *
 * With PITCH_FMIN = 31.7 Hz and 360 bins, the range covers:
 *   bin 0:    31.7 Hz  (approximately B0)
 *   bin 359:  31.7 * 2^(359*20/1200) = ~31.7 * 2^5.98 = ~2006 Hz
 *
 * This comfortably spans the fundamental frequency range of the human
 * singing voice (roughly 80 Hz to 1100 Hz for most singers).
 */
static float bin_to_f0(int bin) {
    return PITCH_FMIN * powf(2.0f, bin * PITCH_CENTS_PER_BIN / 1200.0f);
}

/*
 * Perform one step of the online Viterbi algorithm.
 *
 * Inputs:
 *   st        -- persistent state, including viterbi_prev[N+1] which holds
 *                the accumulated log-scores from the previous frame.
 *   posterior -- the neural network's pitch posterior for the current frame,
 *                an array of NC_PITCH_BINS probabilities in [0, 1].
 *
 * Output:
 *   f0_out    -- the estimated fundamental frequency in Hz, or 0.0 if
 *                the best state is "unvoiced."
 */
static void viterbi_step(NanoPitchState *st, const float *posterior, float *f0_out) {
    int N = NC_PITCH_BINS;
    float *prev = st->viterbi_prev;  /* [N+1]: 0..N-1 pitched, N unvoiced */
    float *curr = (float *)calloc(N + 1, sizeof(float));
    int *best_from = (int *)calloc(N + 1, sizeof(int));

    /*
     * Compute the observation probability for the "unvoiced" state.
     * If the network's best pitch bin has probability 0.9, then
     * unvoiced_obs = log(1 - 0.9 + 1e-10) = log(0.1) ≈ -2.3.
     * High confidence in some pitch -> low score for unvoiced.
     */
    float max_post = 0.0f;
    for (int i = 0; i < N; i++) {
        if (posterior[i] > max_post) max_post = posterior[i];
    }

    float unvoiced_obs = logf(1.0f - max_post + 1e-10f);

    /*
     * For each voiced state s (0..N-1):
     *   1. Search the allowed predecessor neighborhood [s-12, s+12]
     *      to find the previous voiced state with the highest score.
     *   2. Also check if coming from "unvoiced" (state N) with the
     *      onset penalty would be better.
     *   3. Choose the better of the two and add the current frame's
     *      observation: log(posterior[s] + epsilon).
     */
    for (int s = 0; s < N; s++) {
        float log_obs = logf(posterior[s] + 1e-10f);

        /* Clamp transition neighborhood to valid bin range */
        int lo = s - VITERBI_TRANSITION_WIDTH;
        if (lo < 0) lo = 0;
        int hi = s + VITERBI_TRANSITION_WIDTH + 1;
        if (hi > N) hi = N;

        /* Find the best voiced predecessor within the neighborhood */
        float best_val = -1e30f;
        int best_idx = lo;
        for (int p = lo; p < hi; p++) {
            if (prev[p] > best_val) {
                best_val = prev[p];
                best_idx = p;
            }
        }

        /* Transition from unvoiced, with the tuned "onset" penalty */
        float from_unvoiced = prev[N] - VITERBI_ONSET_PENALTY;

        if (best_val >= from_unvoiced) {
            curr[s] = best_val + log_obs;
            best_from[s] = best_idx;
        } else {
            curr[s] = from_unvoiced + log_obs;
            best_from[s] = N;
        }
    }

    /*
     * Unvoiced state: can come from any voiced state (with the same offset
     * penalty) or from the previous unvoiced state (free transition).
     */
    float best_voiced = -1e30f;
    int best_voiced_idx = 0;
    for (int p = 0; p < N; p++) {
        if (prev[p] > best_voiced) {
            best_voiced = prev[p];
            best_voiced_idx = p;
        }
    }
    best_voiced -= VITERBI_ONSET_PENALTY;

    if (prev[N] >= best_voiced) {
        curr[N] = prev[N] + unvoiced_obs;
        best_from[N] = N;
    } else {
        curr[N] = best_voiced + unvoiced_obs;
        best_from[N] = best_voiced_idx;
    }

    /*
     * GREEDY OUTPUT: pick the state with the highest accumulated score.
     *
     * In a full offline Viterbi we would store best_from[] for every
     * frame and do a backtrace at the end of the sequence.  Here we
     * take the argmax of curr[] immediately, which is the "online" or
     * "streaming" approximation.  This adds no latency but sacrifices
     * some accuracy compared to the full algorithm.
     */
    float best_score = -1e30f;
    int best_state = N;
    for (int s = 0; s <= N; s++) {
        if (curr[s] > best_score) {
            best_score = curr[s];
            best_state = s;
        }
    }

    /* Map the winning state to a frequency.
     * If the best state is "unvoiced" (index N), output 0.0 Hz. */
    if (best_state < N) {
        *f0_out = bin_to_f0(best_state);
    } else {
        *f0_out = 0.0f;
    }

    /* Save the current scores as the previous frame for next time */
    memcpy(prev, curr, (N + 1) * sizeof(float));
    free(curr);
    free(best_from);
}

/* ════════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════════ */

/*
 * HOW WEIGHTS ARE LOADED FROM A FLAT ARRAY
 * -------------------------------------------------------------
 *
 * Neural network frameworks (PyTorch, TensorFlow) store model weights
 * as named tensors with various shapes.  For deployment in C/WASM, we
 * flatten ALL weights into a single contiguous array of floats:
 *
 *   [ conv1_weight | conv1_bias | conv2_weight | conv2_bias |
 *     gru1_w_ih | gru1_w_hh | gru1_b_ih | gru1_b_hh |
 *     gru2_...  | gru3_... |
 *     dense_vad_weight | dense_vad_bias |
 *     dense_pitch_weight | dense_pitch_bias ]
 *
 * The Python export script writes them in this exact order.  The C code
 * then walks through the flat array with a moving pointer `p`, assigning
 * each weight field to point directly into the data buffer (zero-copy).
 *
 * ZERO-COPY DESIGN:
 *   The NanoPitchWeights struct does NOT allocate separate memory for
 *   each weight matrix.  Instead, each field (e.g., w->gru1_w_ih) is a
 *   raw pointer into the caller-provided `data` buffer.  This means:
 *     - No memcpy overhead at load time
 *     - The caller MUST keep the `data` buffer alive as long as the
 *       weights are in use (the weights struct does not own the memory)
 *     - nanopitch_free_weights() only frees the struct, not the data
 *
 * SIZE VALIDATION:
 *   Before assigning pointers, we compute the expected total number of
 *   floats and check that n_floats is large enough.  This catches
 *   mismatches between the Python export and C import (e.g., if model
 *   architecture changes but the weight file is stale).
 *
 * MEMORY LAYOUT EXAMPLE (for gru_size=96, cond_size=64):
 *   conv1_weight: 64 * 40 * 3 = 7680 floats  (output_ch * input_ch * kernel)
 *   conv1_bias:   64 floats
 *   conv2_weight: 96 * 64 * 3 = 18432 floats
 *   conv2_bias:   96 floats
 *   gru1_w_ih:    288 * 96 = 27648 floats  (3*hidden * input)
 *   gru1_w_hh:    288 * 96 = 27648 floats  (3*hidden * hidden)
 *   gru1_b_ih:    288 floats
 *   gru1_b_hh:    288 floats
 *   ... (gru2 and gru3 same sizes as gru1) ...
 *   dense_vad_weight:   1 * 384 = 384 floats  (1 * 4*gru_size)
 *   dense_vad_bias:     1 float
 *   dense_pitch_weight: 360 * 384 = 138240 floats
 *   dense_pitch_bias:   360 floats
 */
NanoPitchWeights* nanopitch_load_weights(const float *data, int n_floats,
                                          int cond_size, int gru_size) {
    NanoPitchWeights *w = (NanoPitchWeights *)calloc(1, sizeof(NanoPitchWeights));
    if (!w) return NULL;

    w->cond_size = cond_size;
    w->gru_size = gru_size;

    /* Reject models that exceed stack buffer limits */
    if (cond_size > NC_MAX_LAYER_SIZE || gru_size > NC_MAX_LAYER_SIZE) {
        free(w);
        return NULL;
    }

    /* cat_size = 4 * gru_size because the dense heads receive the
     * concatenation of [conv2_out, gru1_h, gru2_h, gru3_h], each of
     * size gru_size. */
    int cat_size = 4 * gru_size;
    /* h3 = 3 * gru_size because the GRU weight matrices stack the
     * reset, update, and candidate sub-matrices vertically. */
    int h3 = 3 * gru_size;

    /* Calculate expected total number of floats across all layers */
    int expected = 0;
    expected += cond_size * NC_N_MELS * 3 + cond_size;    /* conv1 */
    expected += gru_size * cond_size * 3 + gru_size;       /* conv2 */
    expected += 3 * (h3 * gru_size + h3 * gru_size + h3 + h3); /* 3 GRUs */
    expected += 1 * cat_size + 1;                           /* dense_vad */
    expected += NC_PITCH_BINS * cat_size + NC_PITCH_BINS;  /* dense_pitch */

    /* Safety check: bail if the flat array is too small */
    if (n_floats < expected) {
        free(w);
        return NULL;
    }

    /*
     * Walk through the flat array with a moving pointer `p`.
     * Each ASSIGN macro sets a struct field to point at the current
     * position in `data`, then advances `p` by the layer's size.
     *
     * This is the zero-copy trick: no data is copied, the struct just
     * holds pointers into the original buffer.
     */
    const float *p = data;

    #define ASSIGN(field, size) do { w->field = (float *)p; p += (size); } while(0)

    /* Conv1: 1-D convolution, kernel_size=3, in_channels=n_mels, out_channels=cond_size
     * Weight shape: [cond_size][n_mels][3], stored row-major */
    ASSIGN(conv1_weight, cond_size * NC_N_MELS * 3);
    ASSIGN(conv1_bias, cond_size);

    /* Conv2: 1-D convolution, kernel_size=3, in_channels=cond_size, out_channels=gru_size */
    ASSIGN(conv2_weight, gru_size * cond_size * 3);
    ASSIGN(conv2_bias, gru_size);

    /* GRU 1: W_ih [3*gru_size, gru_size], W_hh [3*gru_size, gru_size],
     *         b_ih [3*gru_size],           b_hh [3*gru_size] */
    ASSIGN(gru1_w_ih, h3 * gru_size);
    ASSIGN(gru1_w_hh, h3 * gru_size);
    ASSIGN(gru1_b_ih, h3);
    ASSIGN(gru1_b_hh, h3);

    /* GRU 2: same layout as GRU 1 */
    ASSIGN(gru2_w_ih, h3 * gru_size);
    ASSIGN(gru2_w_hh, h3 * gru_size);
    ASSIGN(gru2_b_ih, h3);
    ASSIGN(gru2_b_hh, h3);

    /* GRU 3: same layout as GRU 1 */
    ASSIGN(gru3_w_ih, h3 * gru_size);
    ASSIGN(gru3_w_hh, h3 * gru_size);
    ASSIGN(gru3_b_ih, h3);
    ASSIGN(gru3_b_hh, h3);

    /* Dense VAD head: 1 output neuron, input size = 4 * gru_size */
    ASSIGN(dense_vad_weight, 1 * cat_size);
    ASSIGN(dense_vad_bias, 1);

    /* Dense pitch head: NC_PITCH_BINS (360) output neurons */
    ASSIGN(dense_pitch_weight, NC_PITCH_BINS * cat_size);
    ASSIGN(dense_pitch_bias, NC_PITCH_BINS);

    #undef ASSIGN

    return w;
}

/*
 * Free the weights struct.  Note: we only free the struct itself, NOT the
 * underlying float data, because the weight pointers are zero-copy views
 * into the caller's data buffer (see nanopitch_load_weights above).
 */
void nanopitch_free_weights(NanoPitchWeights *w) {
    /* Weights are not owned (point into external buffer), just free struct */
    free(w);
}

/*
 * Allocate and zero-initialize the inference state.
 *
 * The state holds:
 *   - Hidden states for the 3 GRU layers (persistent across frames)
 *   - The Viterbi log-score vector (persistent across frames)
 *   - The overlap-save analysis memory for mel computation
 *   - The conv ring buffer for the two causal 1-D convolution layers
 *
 * Everything starts at zero (or a uniform low score for Viterbi),
 * meaning the model begins with no memory of previous audio.
 */
NanoPitchState* nanopitch_create_state(int gru_size) {
    NanoPitchState *st = (NanoPitchState *)calloc(1, sizeof(NanoPitchState));
    if (!st) return NULL;

    st->gru1_h = (float *)calloc(gru_size, sizeof(float));
    st->gru2_h = (float *)calloc(gru_size, sizeof(float));
    st->gru3_h = (float *)calloc(gru_size, sizeof(float));
    st->viterbi_prev = (float *)calloc(NC_PITCH_BINS + 1, sizeof(float));

    /*
     * Initialize Viterbi with a uniform prior: all states start with
     * equal (low) log-probability.  The value -10.0 is arbitrary but
     * ensures that initial scores are dominated by actual observations
     * after the first few frames.
     */
    for (int i = 0; i <= NC_PITCH_BINS; i++) {
        st->viterbi_prev[i] = -10.0f;
    }

    return st;
}

void nanopitch_free_state(NanoPitchState *st) {
    if (!st) return;
    free(st->gru1_h);
    free(st->gru2_h);
    free(st->gru3_h);
    free(st->viterbi_prev);
    free(st);
}

/*
 * Reset all state to initial values.  Call this when starting a new
 * audio stream (e.g., the user stops and restarts recording) to avoid
 * contaminating the new stream with old hidden state or Viterbi history.
 */
void nanopitch_reset_state(NanoPitchState *st, int gru_size) {
    memset(st->conv_buf, 0, sizeof(st->conv_buf));
    st->conv_buf_pos = 0;
    memset(st->gru1_h, 0, gru_size * sizeof(float));
    memset(st->gru2_h, 0, gru_size * sizeof(float));
    memset(st->gru3_h, 0, gru_size * sizeof(float));
    memset(st->analysis_mem, 0, sizeof(st->analysis_mem));
    st->frame_count = 0;
    for (int i = 0; i <= NC_PITCH_BINS; i++) {
        st->viterbi_prev[i] = -10.0f;
    }
    st->last_f0 = 0.0f;
}

/*
 * ────────────────────────────────────────────────────────────────────
 * nanopitch_process_frame -- the main per-frame inference entry point
 * ────────────────────────────────────────────────────────────────────
 *
 * This function is called once every 10 ms with NC_HOP_LENGTH (160)
 * new audio samples.  It runs the full pipeline:
 *
 *   1. MEL SPECTROGRAM: overlap-save -> Hann window -> FFT -> mel -> log
 *
 *   2. CAUSAL CONVOLUTIONS: two 1-D conv layers (kernel=3 each) act as
 *      a "conditioning" network that extracts local spectral features.
 *      Two stacked k=3 convolutions have an effective receptive field
 *      of 5 frames (50 ms), achieved using a ring buffer that stores
 *      the most recent 5 mel frames.
 *
 *      Why "causal"?  In real-time streaming, we cannot use future
 *      frames.  The convolution only looks at the current frame and
 *      past frames, introducing no lookahead latency.
 *
 *      The first NC_CONV_CONTEXT (4) frames produce no output because
 *      the ring buffer is not yet full enough for two k=3 convolutions.
 *
 *   3. GRU STACK: three GRU layers in sequence.  Each layer's hidden
 *      state persists across frames, giving the network memory of the
 *      entire audio history.  The stack of 3 GRUs allows hierarchical
 *      temporal abstraction: GRU1 captures short-range patterns, GRU3
 *      captures longer-range context.
 *
 *   4. CONCATENATION: the conv2 output and all three GRU hidden states
 *      are concatenated into one vector of size 4*gru_size.  This gives
 *      the output heads access to features at multiple timescales.
 *
 *   5. OUTPUT HEADS: two dense (fully connected) layers with sigmoid:
 *      - VAD (voice activity detection): 1 output in [0,1], where 1
 *        means "voice is present" and 0 means "silence/noise."
 *      - Pitch posterior: NC_PITCH_BINS (360) outputs, each in [0,1],
 *        representing the probability of each pitch bin being active.
 *
 *   6. VITERBI DECODING: the pitch posterior is fed into the online
 *      Viterbi tracker, which outputs a single f0 in Hz (or 0 if
 *      unvoiced).
 *
 * Returns 1 when the output is valid, 0 during the initial warmup
 * (first NC_CONV_CONTEXT frames where the conv buffer is filling up).
 */
int nanopitch_process_frame(const NanoPitchWeights *w,
                            NanoPitchState *st,
                            const float *audio_frame,
                            NanoPitchOutput *out) {
    int gs = w->gru_size;    /* e.g. 96 */
    int cs = w->cond_size;   /* e.g. 64 */
    int cat_size = 4 * gs;   /* concatenated feature size: 4 * 96 = 384 */

    /* ── Step 1: Compute mel spectrogram for this frame ─────────── */
    float mel[NC_N_MELS];
    nanopitch_compute_mel(st, audio_frame, mel);
    memcpy(out->mel, mel, NC_N_MELS * sizeof(float));

    /*
     * ── Step 2: Push mel into the conv ring buffer ─────────────
     *
     * The ring buffer stores the 5 most recent mel frames, which is
     * exactly the receptive field needed for two stacked k=3 causal
     * convolutions (k=3 + k=3 - 1 = 5 frames).
     *
     * The modulo-5 indexing wraps around so old frames are overwritten
     * automatically without explicit shifting.
     */
    memcpy(st->conv_buf[st->conv_buf_pos % 5], mel, NC_N_MELS * sizeof(float));
    st->conv_buf_pos++;
    st->frame_count++;

    /* Need at least 5 frames for the two k=3 conv layers.
     * During warmup, output zeros. */
    if (st->frame_count <= NC_CONV_CONTEXT) {
        out->vad = 0.0f;
        memset(out->pitch_posterior, 0, sizeof(out->pitch_posterior));
        out->f0_hz = 0.0f;
        return 0;
    }

    /*
     * ── Step 3: Causal 1-D convolutions ────────────────────────
     *
     * Conv1: kernel_size=3, maps [n_mels] -> [cond_size] per time step.
     *        Applied at 3 consecutive positions (t=0,1,2) over the 5
     *        stored mel frames, producing 3 output frames.
     *
     * Conv2: kernel_size=3, maps [cond_size] -> [gru_size] per time step.
     *        Applied at 1 position over the 3 conv1 outputs, producing
     *        1 final output frame.
     *
     * Together: 5 mel frames -> 3 intermediate frames -> 1 conditioning
     * vector.  Activation: tanh (bounds output to [-1,1]).
     */

    /* Retrieve the 5 most recent mel frames in chronological order
     * from the ring buffer. */
    float input_frames[5][NC_N_MELS];
    for (int i = 0; i < 5; i++) {
        int idx = (st->conv_buf_pos - 5 + i) % 5;
        if (idx < 0) idx += 5;
        memcpy(input_frames[i], st->conv_buf[idx], NC_N_MELS * sizeof(float));
    }

    /*
     * Conv1: for each of 3 output time steps (t=0,1,2), convolve over
     * mel frames [t, t+1, t+2] with kernel size 3.
     *
     * The convolution for output channel o at time t is:
     *   conv1_out[t][o] = bias[o] + sum over k=0..2, c=0..n_mels-1 of
     *                     weight[o][c][k] * input_frames[t+k][c]
     *
     * Weight layout: conv1_weight[(o * NC_N_MELS + c) * 3 + k]
     * This is the row-major layout for a [cond_size][n_mels][3] tensor.
     */
    float conv1_out[3][NC_MAX_LAYER_SIZE];
    for (int t = 0; t < 3; t++) {
        for (int o = 0; o < cs; o++) {
            float sum = w->conv1_bias[o];
            for (int k = 0; k < 3; k++) {
                for (int c = 0; c < NC_N_MELS; c++) {
                    sum += w->conv1_weight[(o * NC_N_MELS + c) * 3 + k]
                           * input_frames[t + k][c];
                }
            }
            conv1_out[t][o] = tanhf_safe(sum);
        }
    }

    /* Conv2: convolve the 3 conv1 output frames down to 1 frame.
     * Same structure as conv1, but in_channels=cond_size, out_channels=gru_size.
     * Weight layout: conv2_weight[(o * cs + c) * 3 + k] */
    float conv2_out[NC_MAX_LAYER_SIZE];
    for (int o = 0; o < gs; o++) {
        float sum = w->conv2_bias[o];
        for (int k = 0; k < 3; k++) {
            for (int c = 0; c < cs; c++) {
                sum += w->conv2_weight[(o * cs + c) * 3 + k]
                       * conv1_out[k][c];
            }
        }
        conv2_out[o] = tanhf_safe(sum);
    }

    /*
     * ── Step 4: GRU layers ─────────────────────────────────────
     *
     * Three stacked GRU layers, each with input_size = hidden_size = gru_size.
     *
     * The hidden state for each GRU persists across frames (stored in
     * st->gruN_h).  This is what makes the model "recurrent" -- it can
     * build up context over the entire audio stream.
     *
     * Data flow: conv2_out -> GRU1 -> GRU2 -> GRU3
     * Each GRU reads the previous GRU's output as input.
     */
    float gru1_out[NC_MAX_LAYER_SIZE];
    memcpy(gru1_out, st->gru1_h, gs * sizeof(float));
    gru_step(conv2_out, gru1_out, gs, gs,
             w->gru1_w_ih, w->gru1_w_hh, w->gru1_b_ih, w->gru1_b_hh);
    memcpy(st->gru1_h, gru1_out, gs * sizeof(float));

    float gru2_out[NC_MAX_LAYER_SIZE];
    memcpy(gru2_out, st->gru2_h, gs * sizeof(float));
    gru_step(gru1_out, gru2_out, gs, gs,
             w->gru2_w_ih, w->gru2_w_hh, w->gru2_b_ih, w->gru2_b_hh);
    memcpy(st->gru2_h, gru2_out, gs * sizeof(float));

    float gru3_out[NC_MAX_LAYER_SIZE];
    memcpy(gru3_out, st->gru3_h, gs * sizeof(float));
    gru_step(gru2_out, gru3_out, gs, gs,
             w->gru3_w_ih, w->gru3_w_hh, w->gru3_b_ih, w->gru3_b_hh);
    memcpy(st->gru3_h, gru3_out, gs * sizeof(float));

    /*
     * ── Step 5: Concatenate multi-scale features ───────────────
     *
     * cat = [conv2_out | gru1_out | gru2_out | gru3_out]
     *
     * Size: 4 * gru_size (e.g., 384).
     *
     * This concatenation gives the output heads access to both the
     * immediate spectral features (conv2_out) and the recurrent
     * context at three levels of temporal depth (gru1, gru2, gru3).
     */
    float cat[NC_MAX_LAYER_SIZE * 4];  /* 4 * max gru_size */
    memcpy(cat, conv2_out, gs * sizeof(float));
    memcpy(cat + gs, gru1_out, gs * sizeof(float));
    memcpy(cat + 2 * gs, gru2_out, gs * sizeof(float));
    memcpy(cat + 3 * gs, gru3_out, gs * sizeof(float));

    /*
     * ── Step 6: Dense output heads ─────────────────────────────
     *
     * VAD head: 384 -> 1 with sigmoid.  Output is P(voice active).
     *
     * Pitch head: 384 -> 360 with sigmoid.  Each output is the
     * independent probability that the corresponding pitch bin is
     * the fundamental frequency.  Using sigmoid (not softmax) means
     * the 360 values do not have to sum to 1 -- the model can hedge
     * by activating several neighboring bins, or express confident
     * silence by making all bins near 0.
     */
    dense_sigmoid(cat, &out->vad, cat_size, 1,
                  w->dense_vad_weight, w->dense_vad_bias);
    dense_sigmoid(cat, out->pitch_posterior, cat_size, NC_PITCH_BINS,
                  w->dense_pitch_weight, w->dense_pitch_bias);

    /*
     * ── Step 7: Viterbi pitch tracking ─────────────────────────
     *
     * Feed the 360-bin posterior into the online Viterbi decoder,
     * which outputs a single f0 in Hz (or 0 if unvoiced).  The
     * Viterbi tracker smooths the pitch trajectory over time,
     * preventing octave jumps and spurious voiced/unvoiced switches.
     */
    viterbi_step(st, out->pitch_posterior, &out->f0_hz);

    return 1;
}
