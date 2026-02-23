package com.example.tsvad.data

import android.content.Context
import java.io.File

/**
 * Stores and loads speaker embeddings from local files.
 */
class EmbeddingStore(private val context: Context) {

    private val dir = File(context.filesDir, "embeddings").also { it.mkdirs() }

    fun save(name: String, embedding: FloatArray) {
        val file = File(dir, "$name.bin")
        file.outputStream().buffered().use { out ->
            val bytes = java.nio.ByteBuffer.allocate(embedding.size * 4)
            bytes.asFloatBuffer().put(embedding)
            out.write(bytes.array())
        }
    }

    fun load(name: String): FloatArray? {
        val file = File(dir, "$name.bin")
        if (!file.exists()) return null

        val bytes = file.readBytes()
        val floats = FloatArray(bytes.size / 4)
        java.nio.ByteBuffer.wrap(bytes).asFloatBuffer().get(floats)
        return floats
    }

    fun exists(name: String): Boolean = File(dir, "$name.bin").exists()

    fun listSpeakers(): List<String> {
        return dir.listFiles()
            ?.filter { it.extension == "bin" }
            ?.map { it.nameWithoutExtension }
            ?: emptyList()
    }

    fun delete(name: String) {
        File(dir, "$name.bin").delete()
    }
}
