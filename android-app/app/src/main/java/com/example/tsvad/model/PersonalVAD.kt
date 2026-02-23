package com.example.tsvad.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * Personal VAD: Speaker-conditioned voice activity detection.
 *
 * Input: 40-dim log-fbank + 256-dim speaker embedding = 296-dim per frame
 * Output: 3-class (silence / non-target / target) per frame
 *
 * Maintains LSTM hidden state across calls for streaming inference.
 */
class PersonalVAD(context: Context) : AutoCloseable {

    companion object {
        const val NUM_CLASSES = 3
        const val FBANK_DIM = 40
        const val EMBED_DIM = 256
        const val INPUT_DIM = FBANK_DIM + EMBED_DIM  // 296
        private const val HIDDEN_DIM = 64
        private const val NUM_LAYERS = 2

        const val CLASS_SILENCE = 0
        const val CLASS_NON_TARGET = 1
        const val CLASS_TARGET = 2
    }

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // LSTM hidden state (persisted across frames)
    private var h0 = FloatArray(NUM_LAYERS * HIDDEN_DIM)
    private var c0 = FloatArray(NUM_LAYERS * HIDDEN_DIM)

    init {
        val modelBytes = context.assets.open("personal_vad.onnx").readBytes()
        session = ortEnv.createSession(modelBytes)
    }

    /**
     * Run inference on a batch of frames.
     *
     * @param fbankFrames array of 40-dim log-fbank features
     * @param speakerEmbedding 256-dim speaker embedding
     * @return array of class probabilities (numFrames, 3)
     */
    fun infer(fbankFrames: Array<FloatArray>, speakerEmbedding: FloatArray): Array<FloatArray> {
        val numFrames = fbankFrames.size
        if (numFrames == 0) return emptyArray()

        // Concatenate fbank + embedding for each frame → (1, numFrames, 296)
        val inputData = FloatArray(numFrames * INPUT_DIM)
        for (i in 0 until numFrames) {
            val offset = i * INPUT_DIM
            fbankFrames[i].copyInto(inputData, offset)
            speakerEmbedding.copyInto(inputData, offset + FBANK_DIM)
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, numFrames.toLong(), INPUT_DIM.toLong()),
        )
        val h0Tensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(h0),
            longArrayOf(NUM_LAYERS.toLong(), 1, HIDDEN_DIM.toLong()),
        )
        val c0Tensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(c0),
            longArrayOf(NUM_LAYERS.toLong(), 1, HIDDEN_DIM.toLong()),
        )

        val inputs = mapOf("input" to inputTensor, "h0" to h0Tensor, "c0" to c0Tensor)

        inputTensor.use { _ ->
            h0Tensor.use { _ ->
                c0Tensor.use { _ ->
                    val results = session.run(inputs)
                    results.use { output ->
                        // Update hidden state
                        val hn = output["hn"].get().value
                        val cn = output["cn"].get().value
                        if (hn is Array<*>) {
                            @Suppress("UNCHECKED_CAST")
                            val hnArr = hn as Array<Array<FloatArray>>
                            flattenState(hnArr, h0)
                        }
                        if (cn is Array<*>) {
                            @Suppress("UNCHECKED_CAST")
                            val cnArr = cn as Array<Array<FloatArray>>
                            flattenState(cnArr, c0)
                        }

                        // Parse logits → softmax
                        val logits = output["logits"].get().value
                        @Suppress("UNCHECKED_CAST")
                        val logitsArr = (logits as Array<Array<FloatArray>>)[0]
                        return Array(numFrames) { i -> softmax(logitsArr[i]) }
                    }
                }
            }
        }
    }

    /**
     * Reset LSTM hidden state (call when starting a new session).
     */
    fun resetState() {
        h0.fill(0f)
        c0.fill(0f)
    }

    private fun flattenState(state: Array<Array<FloatArray>>, target: FloatArray) {
        var offset = 0
        for (layer in state) {
            for (batch in layer) {
                batch.copyInto(target, offset)
                offset += batch.size
            }
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { kotlin.math.exp(logits[it] - max) }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    override fun close() {
        session.close()
    }
}
