package com.example.tsvad.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.example.tsvad.audio.FeatureExtractor
import java.nio.FloatBuffer

/**
 * Resemblyzer-compatible speaker encoder (GE2E d-vector).
 * Extracts 256-dim speaker embedding from enrollment audio.
 *
 * Input: 40-dim mel spectrogram (NOT log-mel), fixed 160 frames per partial
 * Output: 256-dim L2-normalized d-vector
 */
class SpeakerEncoder(context: Context) : AutoCloseable {

    companion object {
        private const val PARTIALS_N_FRAMES = 160  // Resemblyzer fixed input size
        private const val EMBED_DIM = 256
    }

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val featureExtractor = FeatureExtractor.create(
        context = context,
        sampleRate = 16000,
        nFft = 400,
        hopLength = 160,
        nMels = 40,
        applyLog = false,  // Resemblyzer uses mel, NOT log-mel
    )

    init {
        val modelBytes = context.assets.open("speaker_encoder.onnx").readBytes()
        session = ortEnv.createSession(modelBytes)
    }

    /**
     * Extract speaker embedding from audio waveform.
     * Splits mel into 160-frame partials, encodes each, and averages.
     *
     * @param audio 16kHz mono float audio
     * @return 256-dim speaker embedding (L2-normalized)
     */
    fun extractEmbedding(audio: FloatArray): FloatArray {
        val mel = featureExtractor.extract(audio)
        if (mel.isEmpty()) return FloatArray(EMBED_DIM)

        val numMels = mel[0].size

        // Split into 160-frame partials (with 50% overlap for better coverage)
        val partials = mutableListOf<Array<FloatArray>>()
        val step = PARTIALS_N_FRAMES / 2  // 80 frame overlap
        var start = 0
        while (start + PARTIALS_N_FRAMES <= mel.size) {
            partials.add(mel.sliceArray(start until start + PARTIALS_N_FRAMES))
            start += step
        }
        // If no full partial, pad the last one
        if (partials.isEmpty()) {
            val padded = Array(PARTIALS_N_FRAMES) { i ->
                if (i < mel.size) mel[i] else FloatArray(numMels)
            }
            partials.add(padded)
        }

        // Encode each partial and average
        val sumEmbedding = FloatArray(EMBED_DIM)
        for (partial in partials) {
            val emb = encodePartial(partial)
            for (i in emb.indices) sumEmbedding[i] += emb[i]
        }

        // Average and L2-normalize
        val n = partials.size.toFloat()
        for (i in sumEmbedding.indices) sumEmbedding[i] /= n
        return normalize(sumEmbedding)
    }

    private fun encodePartial(frames: Array<FloatArray>): FloatArray {
        val numMels = frames[0].size
        val flatData = FloatArray(PARTIALS_N_FRAMES * numMels)
        for (i in frames.indices) {
            frames[i].copyInto(flatData, i * numMels)
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(flatData),
            longArrayOf(1, PARTIALS_N_FRAMES.toLong(), numMels.toLong()),
        )

        inputTensor.use { tensor ->
            val inputName = session.inputNames.first()
            val results = session.run(mapOf(inputName to tensor))
            results.use { output ->
                @Suppress("UNCHECKED_CAST")
                return (output[0].value as Array<FloatArray>)[0]
            }
        }
    }

    private fun normalize(vec: FloatArray): FloatArray {
        var norm = 0f
        for (v in vec) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm < 1e-8f) return vec
        return FloatArray(vec.size) { vec[it] / norm }
    }

    override fun close() {
        session.close()
    }
}
