package com.shounak.localmeshai.core.streamvex.diagnostics

import android.util.Log
import com.shounak.localmeshai.core.streamvex.api.StreamVexDeviceProfile
import com.shounak.localmeshai.core.streamvex.api.StreamVexResourceDiagnostics
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedDeque

object StreamVexDiagnostics {
    private const val TAG = "StreamVexDiagnostics"
    private const val MAX_LOGS = 200
    private val logQueue = ConcurrentLinkedDeque<String>()
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)

    fun log(tag: String, message: String) {
        val entry = "[${dateFormat.format(Date())}] [$tag] $message"
        logQueue.addLast(entry)
        while (logQueue.size > MAX_LOGS) {
            logQueue.pollFirst()
        }
        Log.i("StreamVex:$tag", message)
    }

    fun getLogs(): List<String> = logQueue.toList()

    fun clearLogs() {
        logQueue.clear()
    }

    fun sampleResourceDiagnostics(): StreamVexResourceDiagnostics {
        if (!StreamVexNativeBridge.isLibraryLoaded) return StreamVexResourceDiagnostics()
        return try {
            val jsonStr = StreamVexNativeBridge.getResourceTelemetry() ?: return StreamVexResourceDiagnostics()
            val root = JSONObject(jsonStr)
            val mem = root.optJSONObject("memory")
            val cpu = root.optJSONObject("cpu")
            val therm = root.optJSONObject("thermal")
            val batt = root.optJSONObject("battery")

            StreamVexResourceDiagnostics(
                totalRamMb = mem?.optDouble("total_mb", 0.0) ?: 0.0,
                availableRamMb = mem?.optDouble("available_mb", 0.0) ?: 0.0,
                processRssMb = mem?.optDouble("process_rss_mb", 0.0) ?: 0.0,
                processPeakRssMb = mem?.optDouble("process_peak_rss_mb", 0.0) ?: 0.0,
                totalCores = cpu?.optInt("total_cores", 8) ?: 8,
                onlineCores = cpu?.optInt("online_cores", 8) ?: 8,
                bigCores = cpu?.optInt("big_cores", 2) ?: 2,
                littleCores = cpu?.optInt("little_cores", 6) ?: 6,
                thermalCelsius = therm?.optDouble("celsius", 0.0) ?: 0.0,
                thermalStatus = therm?.optInt("status", 0) ?: 0,
                batteryPct = batt?.optInt("capacity_pct", 100) ?: 100
            )
        } catch (t: Throwable) {
            log("Diagnostics", "Error parsing resource telemetry: ${t.message}")
            StreamVexResourceDiagnostics()
        }
    }

    fun sampleDeviceProfile(): StreamVexDeviceProfile {
        if (!StreamVexNativeBridge.isLibraryLoaded) return StreamVexDeviceProfile()
        return try {
            val jsonStr = StreamVexNativeBridge.getDeviceProfile() ?: return StreamVexDeviceProfile()
            val root = JSONObject(jsonStr)

            StreamVexDeviceProfile(
                tier = root.optInt("tier", 1),
                tierName = root.optString("tier_name", "Mid-Range Mobile"),
                totalRamMb = root.optDouble("total_ram_mb", 0.0),
                availableRamMb = root.optDouble("available_ram_mb", 0.0),
                totalCores = root.optInt("total_cores", 8),
                bigCores = root.optInt("big_cores", 2),
                hasDotprod = root.optBoolean("has_dotprod", true),
                hasFp16 = root.optBoolean("has_fp16", true),
                maxRecommendedModelSizeMb = root.optDouble("max_recommended_model_size_mb", 2500.0),
                maxRecommendedContext = root.optInt("max_recommended_context", 2048),
                recommendedQuantization = root.optString("recommended_quantization", "Q4_K_M"),
                recommendedThreads = root.optInt("recommended_threads", 3)
            )
        } catch (t: Throwable) {
            log("Diagnostics", "Error parsing device profile: ${t.message}")
            StreamVexDeviceProfile()
        }
    }
}
