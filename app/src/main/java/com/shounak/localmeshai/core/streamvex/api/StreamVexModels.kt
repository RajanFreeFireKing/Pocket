package com.shounak.localmeshai.core.streamvex.api

data class StreamVexConfig(
    val threads: Int = 3,
    val contextLength: Int = 2048,
    val temperature: Float = 0.7f,
    val topP: Float = 0.95f,
    val topK: Int = 40,
    val priorityLevel: Int = 1 // 1: P1 Foreground Interactive, 4: P4 Background
)

sealed interface StreamVexEngineState {
    object Uninitialized : StreamVexEngineState
    data class Loading(val modelPath: String, val stage: String = "Loading tensors") : StreamVexEngineState
    data class Ready(val modelPath: String, val residentRamMb: Double) : StreamVexEngineState
    data class Generating(val prompt: String, val tokensGenerated: Int = 0) : StreamVexEngineState
    data class Error(val message: String, val cause: Throwable? = null) : StreamVexEngineState
}

data class StreamVexTelemetry(
    val model: String = "",
    val architecture: String = "",
    val ttftMs: Double = 0.0,
    val tokensPerSec: Double = 0.0,
    val prefillTokensPerSec: Double = 0.0,
    val totalTokens: Int = 0,
    val residentRamMb: Double = 0.0,
    val peakRamMb: Double = 0.0,
    val isRamBounded: Boolean = true
)

data class StreamVexResourceDiagnostics(
    val totalRamMb: Double = 0.0,
    val availableRamMb: Double = 0.0,
    val processRssMb: Double = 0.0,
    val processPeakRssMb: Double = 0.0,
    val totalCores: Int = 0,
    val onlineCores: Int = 0,
    val bigCores: Int = 0,
    val littleCores: Int = 0,
    val thermalCelsius: Double = 0.0,
    val thermalStatus: Int = 0,
    val batteryPct: Int = 100
)

data class StreamVexDeviceProfile(
    val tier: Int = 1,
    val tierName: String = "Mid-Range Mobile",
    val totalRamMb: Double = 0.0,
    val availableRamMb: Double = 0.0,
    val totalCores: Int = 8,
    val bigCores: Int = 2,
    val hasDotprod: Boolean = true,
    val hasFp16: Boolean = true,
    val maxRecommendedModelSizeMb: Double = 2500.0,
    val maxRecommendedContext: Int = 2048,
    val recommendedQuantization: String = "Q4_K_M",
    val recommendedThreads: Int = 3
)

data class StreamVexValidationResult(
    val isCompatible: Boolean = true,
    val message: String = "",
    val suggestedModel: String = "",
    val suggestedContext: Int = 2048
)

data class StreamVexBenchmarkResult(
    val modelLoadDurationMs: Long = 0L,
    val ttftMs: Double = 0.0,
    val tokensPerSec: Double = 0.0,
    val prefillTokensPerSec: Double = 0.0,
    val peakRamMb: Double = 0.0,
    val totalTokens: Int = 0,
    val wallDurationSec: Double = 0.0,
    val doubleBufferThroughputMbS: Double = 0.0,
    val tripleBufferThroughputMbS: Double = 0.0,
    val recommendedBufferingMode: String = "double",
    val q4Gflops: Double = 0.0,
    val f32Gflops: Double = 0.0
)

data class StreamVexToken(
    val text: String,
    val isFirstToken: Boolean = false,
    val isComplete: Boolean = false,
    val errorMessage: String? = null
)
