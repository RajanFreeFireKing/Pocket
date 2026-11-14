package com.shounak.localmeshai.core.streamvex.native

import android.util.Log
import androidx.annotation.Keep

@Keep
interface StreamVexTokenCallback {
    @Keep
    fun onToken(token: String)

    @Keep
    fun onTokenReceived(token: String) {
        onToken(token)
    }
}

/**
 * Low-level JNI interface to libstreamvex.so.
 *
 * Provides direct, thread-safe access to the StreamVex C++ engine,
 * Resource Intelligence Manager, Layer Streamer, and Voice AI subsystem.
 */
@Keep
object StreamVexNativeBridge {
    private const val TAG = "StreamVexNativeBridge"

    @Keep
    @Volatile
    var isLibraryLoaded: Boolean = false
        private set

    @Keep
    @Volatile
    var activeTokenCallback: StreamVexTokenCallback? = null

    init {
        try {
            System.loadLibrary("streamvex")
            isLibraryLoaded = true
            Log.i(TAG, "Successfully loaded native libstreamvex.so")
        } catch (t: Throwable) {
            Log.w(TAG, "Native libstreamvex.so could not be loaded: ${t.message}")
        }
    }

    /**
     * Fallback callback invoked by JNI if invoked on the bridge object directly.
     */
    @Keep
    @JvmStatic
    fun onTokenReceived(token: String) {
        activeTokenCallback?.onToken(token)
    }

    @Keep
    @JvmStatic
    fun onToken(token: String) {
        activeTokenCallback?.onToken(token)
    }

    @Keep external fun initEngine(modelPath: String): Int
    @Keep external fun streamToken(prompt: String, callback: StreamVexTokenCallback): Int
    @Keep external fun getMemoryUsage(): Double
    @Keep external fun isEngineReady(): Boolean
    @Keep external fun releaseEngine()
    @Keep external fun abortGeneration()
    @Keep external fun onTrimMemory(trimLevel: Int)
    @Keep external fun setBackgroundMode(isBackground: Boolean)
    @Keep external fun setPriority(priorityLevel: Int): Int
    @Keep external fun getResourceTelemetry(): String?
    @Keep external fun getDeviceProfile(): String?
    @Keep external fun validateModel(modelPath: String, contextLen: Int): String?
    @Keep external fun parseGgufMetadata(modelPath: String): String?
    @Keep external fun computeGemvBenchmark(rows: Int, cols: Int): String?
    @Keep external fun benchmarkTripleBuffering(filePath: String): String?
    @Keep external fun getTelemetry(): String?
    @Keep external fun processVadFrame(pcmSamples: ShortArray, numSamples: Int): Boolean
    @Keep external fun synthesizeSpeech(text: String): ShortArray?
    @Keep external fun transcribeSpeech(pcmSamples: ShortArray, numSamples: Int): String?
}
