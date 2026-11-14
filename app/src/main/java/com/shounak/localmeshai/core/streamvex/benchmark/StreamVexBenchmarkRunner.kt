package com.shounak.localmeshai.core.streamvex.benchmark

import com.shounak.localmeshai.core.streamvex.api.StreamVexBenchmarkResult
import com.shounak.localmeshai.core.streamvex.diagnostics.StreamVexDiagnostics
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

object StreamVexBenchmarkRunner {
    suspend fun runBenchmark(
        modelPath: String,
        testPrompt: String = "Explain the fundamental principles of quantum computing in two concise sentences.",
        onProgress: (String) -> Unit = {}
    ): StreamVexBenchmarkResult = withContext(Dispatchers.IO) {
        if (!StreamVexNativeBridge.isLibraryLoaded) {
            return@withContext StreamVexBenchmarkResult()
        }

        onProgress("1/4 Initializing engine & measuring model load time...")
        val loadStart = System.currentTimeMillis()
        val initRes = StreamVexNativeBridge.initEngine(modelPath)
        val loadDuration = System.currentTimeMillis() - loadStart

        if (initRes != 0) {
            StreamVexDiagnostics.log("Benchmark", "Engine initialization failed: $initRes")
            return@withContext StreamVexBenchmarkResult(modelLoadDurationMs = loadDuration)
        }

        onProgress("2/4 Measuring autoregressive generation & TTFT...")
        var firstTokenMs = 0.0
        val genStart = System.currentTimeMillis()
        var tokenCount = 0

        val benchCallback = object : com.shounak.localmeshai.core.streamvex.native.StreamVexTokenCallback {
            override fun onToken(token: String) {
                if (tokenCount == 0) {
                    firstTokenMs = (System.currentTimeMillis() - genStart).toDouble()
                }
                tokenCount++
            }
            override fun onTokenReceived(token: String) {
                onToken(token)
            }
        }
        StreamVexNativeBridge.activeTokenCallback = benchCallback
        try {
            StreamVexNativeBridge.streamToken(testPrompt, benchCallback)
        } finally {
            StreamVexNativeBridge.activeTokenCallback = null
        }
        val genDurationSec = (System.currentTimeMillis() - genStart) / 1000.0

        onProgress("3/4 Querying authentic engine telemetry & RAM...")
        val teleJson = StreamVexNativeBridge.getTelemetry() ?: "{}"
        val teleObj = JSONObject(teleJson)
        val measuredTtft = if (teleObj.has("ttft_ms")) teleObj.optDouble("ttft_ms", firstTokenMs) else firstTokenMs
        val measuredTokensPerSec = if (teleObj.has("tokens_per_sec")) teleObj.optDouble("tokens_per_sec", 0.0) else (tokenCount / genDurationSec)
        val measuredPrefillRate = teleObj.optDouble("prefill_tokens_per_sec", 0.0)
        val peakRam = teleObj.optDouble("peak_ram_mb", StreamVexNativeBridge.getMemoryUsage())

        onProgress("4/4 Running SIMD GEMV and Double vs Triple Buffering benchmarks...")
        var q4Gflops = 0.0
        var f32Gflops = 0.0
        val gemvJson = StreamVexNativeBridge.computeGemvBenchmark(1024, 1024)
        if (!gemvJson.isNullOrBlank()) {
            val gemvObj = JSONObject(gemvJson)
            q4Gflops = gemvObj.optDouble("q4_gflops", 0.0)
            f32Gflops = gemvObj.optDouble("f32_gflops", 0.0)
        }

        var doubleThroughput = 0.0
        var tripleThroughput = 0.0
        var recommendedMode = "double"
        val bufferJson = StreamVexNativeBridge.benchmarkTripleBuffering(modelPath)
        if (!bufferJson.isNullOrBlank()) {
            val bufferObj = JSONObject(bufferJson)
            val dBuf = bufferObj.optJSONObject("double_buffer")
            val tBuf = bufferObj.optJSONObject("triple_buffer")
            doubleThroughput = dBuf?.optDouble("throughput_mb_s", 0.0) ?: 0.0
            tripleThroughput = tBuf?.optDouble("throughput_mb_s", 0.0) ?: 0.0
            recommendedMode = bufferObj.optString("recommended_mode", "double")
        }

        onProgress("Benchmark completed successfully.")
        StreamVexBenchmarkResult(
            modelLoadDurationMs = loadDuration,
            ttftMs = measuredTtft,
            tokensPerSec = measuredTokensPerSec,
            prefillTokensPerSec = measuredPrefillRate,
            peakRamMb = peakRam,
            totalTokens = tokenCount,
            wallDurationSec = genDurationSec,
            doubleBufferThroughputMbS = doubleThroughput,
            tripleBufferThroughputMbS = tripleThroughput,
            recommendedBufferingMode = recommendedMode,
            q4Gflops = q4Gflops,
            f32Gflops = f32Gflops
        )
    }
}
