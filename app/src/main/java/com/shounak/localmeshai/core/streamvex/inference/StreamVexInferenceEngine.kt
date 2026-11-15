package com.shounak.localmeshai.core.streamvex.inference

import com.shounak.localmeshai.core.streamvex.api.StreamVexConfig
import com.shounak.localmeshai.core.streamvex.api.StreamVexTelemetry
import com.shounak.localmeshai.core.streamvex.api.StreamVexToken
import com.shounak.localmeshai.core.streamvex.diagnostics.StreamVexDiagnostics
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class StreamVexInferenceEngine {
    private val isGenerating = AtomicBoolean(false)

    fun streamInference(
        prompt: String,
        config: StreamVexConfig = StreamVexConfig()
    ): Flow<StreamVexToken> = callbackFlow {
        if (!StreamVexNativeBridge.isLibraryLoaded || !StreamVexNativeBridge.isEngineReady()) {
            trySend(StreamVexToken(text = "", isComplete = true, errorMessage = "StreamVex engine is not ready or loaded"))
            close()
            return@callbackFlow
        }

        if (isGenerating.getAndSet(true)) {
            trySend(StreamVexToken(text = "", isComplete = true, errorMessage = "Inference already in progress"))
            close()
            return@callbackFlow
        }

        StreamVexDiagnostics.log("Inference", "Starting token generation for prompt (${prompt.length} chars)")
        var firstTokenReceived = false

        try {
            val callback = object : com.shounak.localmeshai.core.streamvex.native.StreamVexTokenCallback {
                override fun onToken(token: String) {
                    val cleanTok = com.shounak.localmeshai.utils.ModelOutputSanitizer.clean(token)
                    if (cleanTok.isEmpty()) return
                    if (!firstTokenReceived) {
                        firstTokenReceived = true
                        trySend(StreamVexToken(text = cleanTok, isFirstToken = true))
                    } else {
                        trySend(StreamVexToken(text = cleanTok))
                    }
                }

                override fun onTokenReceived(token: String) {
                    onToken(token)
                }
            }
            StreamVexNativeBridge.activeTokenCallback = callback
            val returnCode = StreamVexNativeBridge.streamToken(prompt, callback)

            if (returnCode == 0) {
                StreamVexDiagnostics.log("Inference", "Generation completed successfully")
                trySend(StreamVexToken(text = "", isComplete = true))
            } else if (returnCode == 1 || returnCode == -10) {
                StreamVexDiagnostics.log("Inference", "Generation stopped or cancelled: code $returnCode")
                trySend(StreamVexToken(text = "", isComplete = true, errorMessage = "Cancelled"))
            } else {
                StreamVexDiagnostics.log("Inference", "Generation returned non-zero code: $returnCode")
                trySend(StreamVexToken(text = "", isComplete = true, errorMessage = "Inference stopped with code $returnCode"))
            }
        } catch (c: CancellationException) {
            StreamVexDiagnostics.log("Inference", "Generation cancelled by caller")
            StreamVexNativeBridge.abortGeneration()
            trySend(StreamVexToken(text = "", isComplete = true, errorMessage = "Cancelled"))
            throw c
        } catch (t: Throwable) {
            StreamVexDiagnostics.log("Inference", "Generation threw exception: ${t.message}")
            trySend(StreamVexToken(text = "", isComplete = true, errorMessage = t.message))
        } finally {
            StreamVexNativeBridge.activeTokenCallback = null
            isGenerating.set(false)
            close()
        }

        awaitClose {
            if (isGenerating.get()) {
                StreamVexDiagnostics.log("Inference", "Flow collection aborted, signalling native abort")
                StreamVexNativeBridge.abortGeneration()
                isGenerating.set(false)
            }
        }
    }.flowOn(Dispatchers.IO)

    fun abort() {
        if (isGenerating.get()) {
            StreamVexDiagnostics.log("Inference", "Explicit abort requested")
            StreamVexNativeBridge.abortGeneration()
        }
    }

    suspend fun sampleLastTelemetry(): StreamVexTelemetry = withContext(Dispatchers.IO) {
        if (!StreamVexNativeBridge.isLibraryLoaded) return@withContext StreamVexTelemetry()
        try {
            val jsonStr = StreamVexNativeBridge.getTelemetry() ?: return@withContext StreamVexTelemetry()
            val root = JSONObject(jsonStr)
            StreamVexTelemetry(
                model = root.optString("model", ""),
                architecture = root.optString("architecture", ""),
                ttftMs = root.optDouble("ttft_ms", 0.0),
                tokensPerSec = root.optDouble("tokens_per_sec", 0.0),
                prefillTokensPerSec = root.optDouble("prefill_tokens_per_sec", 0.0),
                totalTokens = root.optInt("total_tokens", 0),
                residentRamMb = root.optDouble("resident_ram_mb", 0.0),
                peakRamMb = root.optDouble("peak_ram_mb", 0.0),
                isRamBounded = root.optBoolean("is_ram_bounded", true)
            )
        } catch (t: Throwable) {
            StreamVexDiagnostics.log("Inference", "Error reading telemetry: ${t.message}")
            StreamVexTelemetry()
        }
    }
}
