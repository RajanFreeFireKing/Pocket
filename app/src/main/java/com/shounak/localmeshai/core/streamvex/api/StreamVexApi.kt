package com.shounak.localmeshai.core.streamvex.api

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/**
 * Universal Mobile AI Runtime API interface for StreamVex Core Engine.
 */
interface StreamVexApi {
    val state: StateFlow<StreamVexEngineState>
    val telemetry: StateFlow<StreamVexTelemetry>
    val isReady: Boolean

    suspend fun initialize(modelPath: String, config: StreamVexConfig = StreamVexConfig()): Result<Unit>
    fun generate(prompt: String): Flow<StreamVexToken>
    fun cancelGeneration()
    suspend fun shutdown()

    fun trimMemory(level: Int)
    fun setPriority(priorityLevel: Int)
    fun getDiagnostics(): StreamVexResourceDiagnostics
    fun getDeviceProfile(): StreamVexDeviceProfile
    fun validateModel(modelPath: String, contextLen: Int = 2048): StreamVexValidationResult
}
