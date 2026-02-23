package com.example.tsvad

import android.Manifest
import android.app.Application
import androidx.annotation.RequiresPermission
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.tsvad.audio.AudioCapturer
import com.example.tsvad.audio.FeatureExtractor
import com.example.tsvad.data.EmbeddingStore
import com.example.tsvad.model.PersonalVAD
import com.example.tsvad.model.SpeakerEncoder
import com.example.tsvad.ui.EnrollState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val SPEAKER_NAME = "default"
        private const val ENROLL_DURATION_SEC = 5f

        // Process audio in chunks of N frames for VAD
        // 25 frames = 250ms of audio (with 10ms hop)
        private const val VAD_CHUNK_FRAMES = 25

        // Context frames fed before current chunk for LSTM warmup (stateless mode)
        // 50 frames = 500ms of context
        private const val CONTEXT_FRAMES = 50
    }

    private val context = application.applicationContext

    // Models (lazy init)
    private var speakerEncoder: SpeakerEncoder? = null
    private var personalVAD: PersonalVAD? = null

    private val audioCapturer = AudioCapturer()
    private val vadFeatureExtractor = FeatureExtractor.create(
        context = context,
        sampleRate = 16000, nFft = 400, hopLength = 160, nMels = 40,
    )
    private val embeddingStore = EmbeddingStore(context)

    // Speaker embedding
    private var speakerEmbedding: FloatArray? = null

    // Enrollment state
    val enrollState = MutableStateFlow<EnrollState>(
        if (embeddingStore.exists(SPEAKER_NAME)) EnrollState.Done else EnrollState.Idle
    )

    // Detection state
    val isDetecting = MutableStateFlow(false)
    val currentClass = MutableStateFlow(PersonalVAD.CLASS_SILENCE)
    val confidence = MutableStateFlow(0f)
    val silenceProb = MutableStateFlow(0f)
    val nonTargetProb = MutableStateFlow(0f)
    val targetProb = MutableStateFlow(0f)
    val recentEvents = MutableStateFlow<List<String>>(emptyList())

    // Audio frame accumulator for VAD
    private val frameAccumulator = mutableListOf<FloatArray>()
    private val contextBuffer = mutableListOf<FloatArray>()  // sliding window context
    private val audioBuffer = ArrayDeque<Float>()
    private val nFft = 400
    private val hopLength = 160

    init {
        // Load saved embedding if available
        embeddingStore.load(SPEAKER_NAME)?.let {
            speakerEmbedding = it
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startEnrollment() {
        enrollState.value = EnrollState.Recording

        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Record audio
                val audio = audioCapturer.recordSeconds(ENROLL_DURATION_SEC)
                enrollState.value = EnrollState.Processing

                // Init speaker encoder if needed
                if (speakerEncoder == null) {
                    speakerEncoder = SpeakerEncoder(context)
                }

                // Extract embedding
                val embedding = speakerEncoder!!.extractEmbedding(audio)
                speakerEmbedding = embedding
                embeddingStore.save(SPEAKER_NAME, embedding)

                enrollState.value = EnrollState.Done
            } catch (e: Exception) {
                enrollState.value = EnrollState.Error(e.message ?: "Unknown error")
            }
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startDetection() {
        val embedding = speakerEmbedding ?: return

        viewModelScope.launch(Dispatchers.IO) {
            // Init VAD model if needed
            if (personalVAD == null) {
                personalVAD = PersonalVAD(context)
            }
            personalVAD!!.resetState()
            frameAccumulator.clear()
            contextBuffer.clear()
            audioBuffer.clear()

            isDetecting.value = true

            audioCapturer.start { samples ->
                processAudioChunk(samples, embedding)
            }
        }
    }

    fun stopDetection() {
        isDetecting.value = false
        audioCapturer.stop()
        currentClass.value = PersonalVAD.CLASS_SILENCE
        confidence.value = 0f
        silenceProb.value = 0f
        nonTargetProb.value = 0f
        targetProb.value = 0f
    }

    fun resetEnrollment() {
        stopDetection()
        embeddingStore.delete(SPEAKER_NAME)
        speakerEmbedding = null
        enrollState.value = EnrollState.Idle
        recentEvents.value = emptyList()
    }

    private fun processAudioChunk(samples: FloatArray, embedding: FloatArray) {
        // Add samples to buffer
        for (s in samples) audioBuffer.addLast(s)

        // Extract frames when we have enough samples
        while (audioBuffer.size >= nFft) {
            val frame = FloatArray(nFft) { audioBuffer.elementAt(it) }
            val fbank = vadFeatureExtractor.extractFrame(frame)
            frameAccumulator.add(fbank)

            // Remove hop_length samples (slide window)
            repeat(hopLength) { audioBuffer.removeFirst() }
        }

        // Run VAD with sliding window (stateless): context + current chunk
        // Reset LSTM each time to prevent drift; context provides warmup
        while (frameAccumulator.size >= VAD_CHUNK_FRAMES) {
            val currentFrames = Array(VAD_CHUNK_FRAMES) { frameAccumulator[it] }
            frameAccumulator.subList(0, VAD_CHUNK_FRAMES).clear()

            // Update context buffer: keep last CONTEXT_FRAMES
            for (f in currentFrames) contextBuffer.add(f)
            if (contextBuffer.size > CONTEXT_FRAMES) {
                contextBuffer.subList(0, contextBuffer.size - CONTEXT_FRAMES).clear()
            }

            try {
                // Stateless sliding window: reset state, replay context in 25-frame chunks, then run current
                personalVAD!!.resetState()

                // Replay context for LSTM warmup (results discarded)
                val ctxSize = contextBuffer.size - VAD_CHUNK_FRAMES  // exclude current frames just added
                if (ctxSize > 0) {
                    var offset = 0
                    while (offset + VAD_CHUNK_FRAMES <= ctxSize) {
                        val ctxChunk = Array(VAD_CHUNK_FRAMES) { contextBuffer[offset + it] }
                        personalVAD!!.infer(ctxChunk, embedding)  // warmup, discard result
                        offset += VAD_CHUNK_FRAMES
                    }
                    // Handle remaining context frames < 25: pad with zeros
                    val remaining = ctxSize - offset
                    if (remaining > 0) {
                        val padChunk = Array(VAD_CHUNK_FRAMES) { i ->
                            if (i < remaining) contextBuffer[offset + i] else FloatArray(PersonalVAD.FBANK_DIM)
                        }
                        personalVAD!!.infer(padChunk, embedding)
                    }
                }

                // Run current chunk (this result we use)
                val probs = personalVAD!!.infer(currentFrames, embedding)
                if (probs.isNotEmpty()) {
                    // Use the last frame's prediction (from current chunk, not context)
                    val lastProb = probs.last()
                    val maxIdx = lastProb.indices.maxByOrNull { lastProb[it] } ?: 0

                    silenceProb.value = lastProb[PersonalVAD.CLASS_SILENCE]
                    nonTargetProb.value = lastProb[PersonalVAD.CLASS_NON_TARGET]
                    targetProb.value = lastProb[PersonalVAD.CLASS_TARGET]
                    currentClass.value = maxIdx
                    confidence.value = lastProb[maxIdx]

                    // Log target speech events
                    if (maxIdx == PersonalVAD.CLASS_TARGET && lastProb[maxIdx] > 0.5f) {
                        val timestamp = System.currentTimeMillis()
                        val timeStr = java.text.SimpleDateFormat(
                            "HH:mm:ss.SSS", java.util.Locale.getDefault()
                        ).format(java.util.Date(timestamp))
                        val event = "$timeStr - Target detected (${(lastProb[maxIdx] * 100).toInt()}%)"
                        recentEvents.value = (listOf(event) + recentEvents.value).take(50)
                    }
                }
            } catch (e: Exception) {
                // Log but don't crash
                val event = "Error: ${e.message}"
                recentEvents.value = (listOf(event) + recentEvents.value).take(50)
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        audioCapturer.stop()
        speakerEncoder?.close()
        personalVAD?.close()
    }
}
