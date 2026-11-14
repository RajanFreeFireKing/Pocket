package com.shounak.localmeshai.core.streamvex.native

import android.util.Log

fun interface StreamVexTokenCallback {
    fun onToken(token: String)
}

/**
 * Low-level JNI interface to libstreamvex.so.
 *
 * Provides direct, thread-safe access to the StreamVex C++ engine,
 * Resource Intelligence Manager, Layer Streamer, and Voice AI subsystem.
 */
object StreamVexNativeBridge {
    private const val TAG = "StreamVexNativeBridge"
    var isLibraryLoaded: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("streamvex")
            isLibraryLoaded = true
            Log.i(TAG, "Successfully loaded native libstreamvex.so")
        } catch (t: Throwable) {
            Log.w(TAG, "Native libstreamvex.so could not be loaded: ${t.message}")
        }
    }

    external fun initEngine(modelPath: String): Int
    external fun streamToken(prompt: String, callback: StreamVexTokenCallback): Int
    external fun getMemoryUsage(): Double
    external fun isEngineReady(): Boolean
    external fun releaseEngine()
    external fun abortGeneration()
    external fun onTrimMemory(trimLevel: Int)
    external fun setBackgroundMode(isBackground: Boolean)
    external fun setPriority(priorityLevel: Int): Int
    external fun getResourceTelemetry(): String?
    external fun getDeviceProfile(): String?
    external fun validateModel(modelPath: String, contextLen: Int): String?
    external fun parseGgufMetadata(modelPath: String): String?
    external fun computeGemvBenchmark(rows: Int, cols: Int): String?
    external fun benchmarkTripleBuffering(filePath: String): String?
    external fun getTelemetry(): String?
    external fun processVadFrame(pcmSamples: ShortArray, numSamples: Int): Boolean
    external fun synthesizeSpeech(text: String): ShortArray?
    external fun transcribeSpeech(pcmSamples: ShortArray, numSamples: Int): String?
}
