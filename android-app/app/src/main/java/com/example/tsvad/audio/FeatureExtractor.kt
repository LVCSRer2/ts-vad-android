package com.example.tsvad.audio

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * Mel-Filterbank feature extractor in pure Kotlin.
 * Uses pre-computed librosa mel filterbank for exact compatibility.
 *
 * librosa compatibility:
 *   - center=True padding (n_fft/2 on each side)
 *   - Power spectrum = |S|^2 (no division by n_fft)
 *   - Slaney-normalized mel filterbank (loaded from file)
 */
class FeatureExtractor(
    private val sampleRate: Int = 16000,
    private val nFft: Int = 400,
    private val hopLength: Int = 160,
    private val nMels: Int = 40,
    private val applyLog: Boolean = true,
    private val melFilterbank: FloatArray,  // (nMels * specSize) row-major
) {
    private val specSize = nFft / 2 + 1  // 201
    private val window: FloatArray = hanningWindow(nFft)

    // Bluestein chirp factors for exact N-point FFT
    private val bluesteinM: Int = run { var n = 1; while (n < 2 * nFft - 1) n *= 2; n }
    private val chirpReal = FloatArray(nFft)
    private val chirpImag = FloatArray(nFft)
    private val bReal = FloatArray(bluesteinM)
    private val bImag = FloatArray(bluesteinM)
    private val BReal: FloatArray  // pre-computed FFT of b
    private val BImag: FloatArray

    init {
        // Pre-compute chirp sequence: w_k = exp(-j*pi*k^2/N)
        for (k in 0 until nFft) {
            val angle = -PI * k.toLong() * k.toLong() / nFft
            chirpReal[k] = cos(angle).toFloat()
            chirpImag[k] = sin(angle).toFloat()
        }
        // Pre-compute b sequence (conjugate chirp, zero-padded)
        bReal[0] = chirpReal[0]; bImag[0] = -chirpImag[0]
        for (k in 1 until nFft) {
            bReal[k] = chirpReal[k]; bImag[k] = -chirpImag[k]
            bReal[bluesteinM - k] = chirpReal[k]; bImag[bluesteinM - k] = -chirpImag[k]
        }
        val (br, bi) = radix2Fft(bReal.copyOf(), bImag.copyOf(), bluesteinM)
        BReal = br; BImag = bi
    }

    companion object {
        /**
         * Load pre-computed librosa mel filterbank from assets.
         */
        fun loadMelFilterbank(context: Context, nMels: Int = 40, specSize: Int = 201): FloatArray {
            val bytes = context.assets.open("mel_filterbank.bin").readBytes()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val floats = FloatArray(nMels * specSize)
            buf.asFloatBuffer().get(floats)
            return floats
        }

        fun create(
            context: Context,
            sampleRate: Int = 16000,
            nFft: Int = 400,
            hopLength: Int = 160,
            nMels: Int = 40,
            applyLog: Boolean = true,
        ): FeatureExtractor {
            val filterbank = loadMelFilterbank(context, nMels, nFft / 2 + 1)
            return FeatureExtractor(sampleRate, nFft, hopLength, nMels, applyLog, filterbank)
        }
    }

    /**
     * Extract mel features from audio (librosa-compatible, batch).
     */
    fun extract(audio: FloatArray): Array<FloatArray> {
        // librosa center=True: pad n_fft/2 on each side
        val pad = nFft / 2
        val padded = FloatArray(audio.size + 2 * pad)
        audio.copyInto(padded, pad)

        val numFrames = max(0, (padded.size - nFft) / hopLength + 1)
        if (numFrames == 0) return emptyArray()

        val features = Array(numFrames) { FloatArray(nMels) }

        for (i in 0 until numFrames) {
            val start = i * hopLength
            val frame = FloatArray(nFft)
            for (j in 0 until nFft) {
                val idx = start + j
                if (idx < padded.size) frame[j] = padded[idx] * window[j]
            }

            val (real, imag) = fft(frame)
            val powerSpec = FloatArray(specSize) { k ->
                real[k] * real[k] + imag[k] * imag[k]
            }

            for (m in 0 until nMels) {
                var sum = 0f
                val fbOffset = m * specSize
                for (k in 0 until specSize) {
                    sum += melFilterbank[fbOffset + k] * powerSpec[k]
                }
                features[i][m] = if (applyLog) log10(max(sum, 1e-6f)) else sum
            }
        }

        return features
    }

    /**
     * Extract features from a single frame (for streaming VAD).
     */
    fun extractFrame(frame: FloatArray): FloatArray {
        val windowed = FloatArray(nFft) { i ->
            if (i < frame.size) frame[i] * window[i] else 0f
        }

        val (real, imag) = fft(windowed)
        val powerSpec = FloatArray(specSize) { k ->
            real[k] * real[k] + imag[k] * imag[k]
        }

        return FloatArray(nMels) { m ->
            var sum = 0f
            val fbOffset = m * specSize
            for (k in 0 until specSize) {
                sum += melFilterbank[fbOffset + k] * powerSpec[k]
            }
            if (applyLog) log10(max(sum, 1e-6f)) else sum
        }
    }

    private fun hanningWindow(size: Int): FloatArray {
        return FloatArray(size) { n ->
            0.5f * (1 - cos(2.0 * PI * n / size).toFloat())
        }
    }

    /**
     * Exact N-point FFT using Bluestein (Chirp-Z) algorithm.
     * Converts arbitrary-size DFT into convolution via radix-2 FFT.
     */
    private fun fft(input: FloatArray): Pair<FloatArray, FloatArray> {
        val n = nFft
        val m = bluesteinM

        // a_k = x_k * chirp_k
        val aReal = FloatArray(m)
        val aImag = FloatArray(m)
        for (k in 0 until n) {
            val x = if (k < input.size) input[k] else 0f
            aReal[k] = x * chirpReal[k]
            aImag[k] = x * chirpImag[k]
        }

        // FFT(a)
        val (arF, aiF) = radix2Fft(aReal, aImag, m)

        // Pointwise multiply: C = FFT(a) * B
        val cReal = FloatArray(m)
        val cImag = FloatArray(m)
        for (k in 0 until m) {
            cReal[k] = arF[k] * BReal[k] - aiF[k] * BImag[k]
            cImag[k] = arF[k] * BImag[k] + aiF[k] * BReal[k]
        }

        // IFFT(C) = conjugate(FFT(conjugate(C))) / M
        for (k in 0 until m) cImag[k] = -cImag[k]
        val (rr, ri) = radix2Fft(cReal, cImag, m)
        val invM = 1f / m
        for (k in 0 until m) { rr[k] *= invM; ri[k] = -ri[k] * invM }

        // result_k = ifft_k * chirp_k
        val outReal = FloatArray(n)
        val outImag = FloatArray(n)
        for (k in 0 until n) {
            outReal[k] = rr[k] * chirpReal[k] - ri[k] * chirpImag[k]
            outImag[k] = rr[k] * chirpImag[k] + ri[k] * chirpReal[k]
        }

        return Pair(outReal, outImag)
    }

    /**
     * In-place radix-2 FFT (power-of-2 size only).
     */
    private fun radix2Fft(real: FloatArray, imag: FloatArray, n: Int): Pair<FloatArray, FloatArray> {
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
            var m = n / 2
            while (m >= 1 && j >= m) { j -= m; m /= 2 }
            j += m
        }

        var step = 2
        while (step <= n) {
            val halfStep = step / 2
            val angle = -2.0 * PI / step
            for (i in 0 until n step step) {
                for (k in 0 until halfStep) {
                    val theta = angle * k
                    val wr = cos(theta).toFloat()
                    val wi = sin(theta).toFloat()
                    val idx1 = i + k
                    val idx2 = i + k + halfStep
                    val tr = wr * real[idx2] - wi * imag[idx2]
                    val ti = wr * imag[idx2] + wi * real[idx2]
                    real[idx2] = real[idx1] - tr
                    imag[idx2] = imag[idx1] - ti
                    real[idx1] += tr
                    imag[idx1] += ti
                }
            }
            step *= 2
        }

        return Pair(real, imag)
    }
}
