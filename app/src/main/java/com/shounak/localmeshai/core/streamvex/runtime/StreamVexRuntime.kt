package com.shounak.localmeshai.core.streamvex.runtime

import com.shounak.localmeshai.core.streamvex.api.*
import com.shounak.localmeshai.core.streamvex.diagnostics.StreamVexDiagnostics
import com.shounak.localmeshai.core.streamvex.inference.StreamVexInferenceEngine
import com.shounak.localmeshai.core.streamvex.model.StreamVexModelManager
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Central runtime orchestrator for StreamVex Core Engine.
 *
 * Thread-safe singleton managing the native engine lifecycle,
 * concurrent access serialization, memory monitoring, and telemetry polling.
 */
class StreamVexRuntime private constructor() : StreamVexApi {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val lifecycleMutex = Mutex()
    private val inferenceEngine = StreamVexInferenceEngine()

    private val _state = MutableStateFlow<StreamVexEngineState>(StreamVexEngineState.Uninitialized)
    override val state: StateFlow<StreamVexEngineState> = _state.asStateFlow()

    private val _telemetry = MutableStateFlow(StreamVexTelemetry())
    override val telemetry: StateFlow<StreamVexTelemetry> = _telemetry.asStateFlow()

    override val isReady: Boolean
        get() = StreamVexNativeBridge.isLibraryLoaded && StreamVexNativeBridge.isEngineReady()

    override suspend fun initialize(modelPath: String, config: StreamVexConfig): Result<Unit> = lifecycleMutex.withLock {
        withContext(Dispatchers.IO) {
            if (!StreamVexNativeBridge.isLibraryLoaded) {
                val err = "libstreamvex.so is not available on this platform"
                StreamVexDiagnostics.log("Runtime", err)
                _state.value = StreamVexEngineState.Error(err)
                return@withContext Result.failure(IllegalStateException(err))
            }

            // If already loaded with the same model, reuse
            if (isReady && (_state.value as? StreamVexEngineState.Ready)?.modelPath == modelPath) {
                StreamVexDiagnostics.log("Runtime", "Model already loaded and ready: $modelPath")
                return@withContext Result.success(Unit)
            }

            // Unload any existing model first
            if (isReady) {
                StreamVexDiagnostics.log("Runtime", "Releasing previous model before loading new one")
                StreamVexNativeBridge.releaseEngine()
            }

            _state.value = StreamVexEngineState.Loading(modelPath)
            StreamVexDiagnostics.log("Runtime", "Initializing engine with model: $modelPath")

            try {
                // Set priority before loading
                StreamVexNativeBridge.setPriority(config.priorityLevel)

                val loadStart = System.currentTimeMillis()
                val res = StreamVexNativeBridge.initEngine(modelPath)
                val duration = System.currentTimeMillis() - loadStart

                if (res != 0) {
                    val err = "Engine initialization failed with code: $res"
                    StreamVexDiagnostics.log("Runtime", err)
                    _state.value = StreamVexEngineState.Error(err)
                    return@withContext Result.failure(RuntimeException(err))
                }

                val ramUsage = StreamVexNativeBridge.getMemoryUsage()
                StreamVexDiagnostics.log("Runtime", "Engine loaded in ${duration}ms, resident RAM: ${ramUsage}MB")

                _state.value = StreamVexEngineState.Ready(modelPath, ramUsage)
                updateTelemetry()
                Result.success(Unit)
            } catch (t: Throwable) {
                StreamVexDiagnostics.log("Runtime", "Exception during initEngine: ${t.message}")
                _state.value = StreamVexEngineState.Error(t.message ?: "Unknown init error", t)
                Result.failure(t)
            }
        }
    }

    override fun generate(prompt: String): Flow<StreamVexToken> = flow {
        val currentState = _state.value
        if (currentState !is StreamVexEngineState.Ready) {
            emit(StreamVexToken(text = "", isComplete = true, errorMessage = "Engine is not ready"))
            return@flow
        }

        _state.value = StreamVexEngineState.Generating(prompt)
        var tokenCount = 0

        inferenceEngine.streamInference(prompt)
            .onEach { token ->
                if (token.text.isNotEmpty()) {
                    tokenCount++
                    _state.value = StreamVexEngineState.Generating(prompt, tokenCount)
                }
            }
            .collect { emit(it) }

        // Restore ready state and refresh telemetry
        val ram = StreamVexNativeBridge.getMemoryUsage()
        _state.value = StreamVexEngineState.Ready(currentState.modelPath, ram)
        updateTelemetry()
    }

    override fun cancelGeneration() {
        StreamVexDiagnostics.log("Runtime", "Cancelling generation")
        inferenceEngine.abort()
    }

    fun resetContext() {
        if (StreamVexNativeBridge.isLibraryLoaded) {
            StreamVexDiagnostics.log("Runtime", "Resetting KV cache context")
            StreamVexNativeBridge.resetContext()
        }
    }

    override suspend fun shutdown() = lifecycleMutex.withLock {
        withContext(Dispatchers.IO) {
            StreamVexDiagnostics.log("Runtime", "Shutting down engine")
            if (StreamVexNativeBridge.isLibraryLoaded) {
                try {
                    StreamVexNativeBridge.abortGeneration()
                    StreamVexNativeBridge.releaseEngine()
                } catch (t: Throwable) {
                    StreamVexDiagnostics.log("Runtime", "Error releasing engine: ${t.message}")
                }
            }
            _state.value = StreamVexEngineState.Uninitialized
        }
    }

    override fun trimMemory(level: Int) {
        if (StreamVexNativeBridge.isLibraryLoaded) {
            StreamVexDiagnostics.log("Runtime", "Forwarding trimMemory level: $level")
            StreamVexNativeBridge.onTrimMemory(level)
            updateTelemetry()
        }
    }

    override fun setPriority(priorityLevel: Int) {
        if (StreamVexNativeBridge.isLibraryLoaded) {
            StreamVexDiagnostics.log("Runtime", "Setting priority level: $priorityLevel")
            StreamVexNativeBridge.setPriority(priorityLevel)
        }
    }

    override fun getDiagnostics(): StreamVexResourceDiagnostics =
        StreamVexDiagnostics.sampleResourceDiagnostics()

    override fun getDeviceProfile(): StreamVexDeviceProfile =
        StreamVexDiagnostics.sampleDeviceProfile()

    override fun validateModel(modelPath: String, contextLen: Int): StreamVexValidationResult =
        StreamVexModelManager.validateModelFile(modelPath, contextLen)

    private fun updateTelemetry() {
        scope.launch(Dispatchers.IO) {
            _telemetry.value = inferenceEngine.sampleLastTelemetry()
        }
    }

    companion object {
        @Volatile
        private var instance: StreamVexRuntime? = null

        fun getInstance(): StreamVexRuntime {
            return instance ?: synchronized(this) {
                instance ?: StreamVexRuntime().also { instance = it }
            }
        }
    }
}
