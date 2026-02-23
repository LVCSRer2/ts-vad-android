package com.example.tsvad.audio

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Real-time audio capture at 16kHz mono PCM16.
 */
class AudioCapturer(
    private val sampleRate: Int = 16000,
    private val frameSizeMs: Int = 10,
) {
    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null

    val frameSizeSamples: Int = sampleRate * frameSizeMs / 1000  // 160 samples per 10ms

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(onFrames: (FloatArray) -> Unit) {
        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            ),
            frameSizeSamples * 2 * 4  // at least 4 frames buffer
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        ).also { it.startRecording() }

        captureJob = CoroutineScope(Dispatchers.IO).launch {
            val shortBuffer = ShortArray(frameSizeSamples)
            while (isActive) {
                val read = audioRecord?.read(shortBuffer, 0, frameSizeSamples) ?: 0
                if (read > 0) {
                    val floatBuffer = FloatArray(read) { shortBuffer[it] / 32768.0f }
                    onFrames(floatBuffer)
                }
            }
        }
    }

    fun stop() {
        captureJob?.cancel()
        captureJob = null
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    /**
     * Record a fixed duration for enrollment.
     * Returns the full audio as a FloatArray.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun recordSeconds(durationSec: Float): FloatArray {
        val totalSamples = (sampleRate * durationSec).toInt()
        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            ),
            totalSamples * 2
        )

        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )

        val shortBuffer = ShortArray(totalSamples)
        record.startRecording()
        var offset = 0
        while (offset < totalSamples) {
            val read = record.read(shortBuffer, offset, totalSamples - offset)
            if (read > 0) offset += read else break
        }
        record.stop()
        record.release()

        return FloatArray(offset) { shortBuffer[it] / 32768.0f }
    }
}
