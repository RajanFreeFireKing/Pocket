package com.shounak.localmeshai.core.streamvex.model

import com.shounak.localmeshai.core.streamvex.api.StreamVexValidationResult
import com.shounak.localmeshai.core.streamvex.diagnostics.StreamVexDiagnostics
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import org.json.JSONObject
import java.io.File

object StreamVexModelManager {
    fun validateModelFile(modelPath: String, contextLen: Int = 2048): StreamVexValidationResult {
        val file = File(modelPath)
        if (!file.exists() || !file.canRead()) {
            return StreamVexValidationResult(
                isCompatible = false,
                message = "Model file does not exist or cannot be read: $modelPath"
            )
        }

        if (file.length() < 1024 * 1024) {
            return StreamVexValidationResult(
                isCompatible = false,
                message = "Model file is too small to be a valid GGUF model (< 1 MB)"
            )
        }

        if (!StreamVexNativeBridge.isLibraryLoaded) {
            return StreamVexValidationResult(
                isCompatible = false,
                message = "StreamVex native library libstreamvex.so is not available on this device"
            )
        }

        return try {
            val jsonStr = StreamVexNativeBridge.validateModel(file.absolutePath, contextLen)
            if (jsonStr.isNullOrBlank()) {
                StreamVexValidationResult(isCompatible = false, message = "Empty response from validator")
            } else {
                val obj = JSONObject(jsonStr)
                StreamVexValidationResult(
                    isCompatible = obj.optBoolean("is_compatible", false),
                    message = obj.optString("message", "Validation completed"),
                    suggestedModel = obj.optString("suggested_model", ""),
                    suggestedContext = obj.optInt("suggested_context", 2048)
                )
            }
        } catch (t: Throwable) {
            StreamVexDiagnostics.log("ModelManager", "Validation error: ${t.message}")
            StreamVexValidationResult(isCompatible = false, message = "Validation failed: ${t.message}")
        }
    }

    fun inspectGguf(modelPath: String): JSONObject {
        if (!StreamVexNativeBridge.isLibraryLoaded) return JSONObject()
        return try {
            val jsonStr = StreamVexNativeBridge.parseGgufMetadata(modelPath)
            if (jsonStr.isNullOrBlank()) JSONObject() else JSONObject(jsonStr)
        } catch (t: Throwable) {
            StreamVexDiagnostics.log("ModelManager", "Error parsing GGUF metadata: ${t.message}")
            JSONObject()
        }
    }
}
