package com.shounak.localmeshai.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.shounak.localmeshai.core.streamvex.api.*
import com.shounak.localmeshai.core.streamvex.benchmark.StreamVexBenchmarkRunner
import com.shounak.localmeshai.core.streamvex.diagnostics.StreamVexDiagnostics
import com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge
import com.shounak.localmeshai.core.streamvex.runtime.StreamVexRuntime
import com.shounak.localmeshai.ui.components.SlidingGlassTab
import com.shounak.localmeshai.ui.components.SlidingGlassTabs
import com.shounak.localmeshai.utils.GlassDispersionCard
import com.shounak.localmeshai.utils.LiquidGlassButton
import dev.chrisbanes.haze.HazeState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StreamVexEngineDialog(
    hazeState: HazeState,
    initialModelPath: String? = null,
    onDismissRequest: () -> Unit
) {
    Dialog(
        onDismissRequest = onDismissRequest,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        val runtime = remember { StreamVexRuntime.getInstance() }
        val engineState by runtime.state.collectAsState()
        val telemetry by runtime.telemetry.collectAsState()
        val context = LocalContext.current
        val coroutineScope = rememberCoroutineScope()

        var selectedTab by remember { mutableIntStateOf(0) }
        val tabs = listOf("Status & Controls", "Hardware Benchmark", "Diagnostics & Logs")

        var isBenchmarking by remember { mutableStateOf(false) }
        var benchmarkProgress by remember { mutableStateOf("") }
        var benchmarkResult by remember { mutableStateOf<StreamVexBenchmarkResult?>(null) }
        var modelPathInput by remember {
            mutableStateOf(
                initialModelPath ?: (engineState as? StreamVexEngineState.Ready)?.modelPath ?: ""
            )
        }

        var liveDiagnostics by remember { mutableStateOf(runtime.getDiagnostics()) }
        var liveProfile by remember { mutableStateOf(runtime.getDeviceProfile()) }

        // Periodic live polling
        LaunchedEffect(Unit) {
            while (true) {
                liveDiagnostics = runtime.getDiagnostics()
                liveProfile = runtime.getDeviceProfile()
                delay(1500)
            }
        }

        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background.copy(alpha = 0.95f)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .navigationBarsPadding()
            ) {
                // Top App Bar
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = onDismissRequest,
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    Spacer(modifier = Modifier.width(12.dp))

                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "StreamVex Core Engine",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Text(
                            text = "v2.0 Universal Mobile Runtime (ARM64 NEON)",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }

                    // Native status badge
                    val isNativeLoaded = StreamVexNativeBridge.isLibraryLoaded
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = if (isNativeLoaded) Color(0xFF10B981).copy(alpha = 0.15f) else Color(0xFFEF4444).copy(alpha = 0.15f),
                        border = androidx.compose.foundation.BorderStroke(
                            1.dp,
                            if (isNativeLoaded) Color(0xFF10B981) else Color(0xFFEF4444)
                        )
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(if (isNativeLoaded) Color(0xFF10B981) else Color(0xFFEF4444))
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = if (isNativeLoaded) "LIB READY" else "UNAVAILABLE",
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (isNativeLoaded) Color(0xFF10B981) else Color(0xFFEF4444)
                            )
                        }
                    }
                }

                // Tab Selector
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 6.dp)
                ) {
                    tabs.forEachIndexed { index, title ->
                        val isSelected = selectedTab == index
                        Surface(
                            modifier = Modifier
                                .weight(1f)
                                .padding(horizontal = 3.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .clickable { selectedTab = index },
                            color = if (isSelected) MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                            else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp,
                                if (isSelected) MaterialTheme.colorScheme.primary else Color.Transparent
                            )
                        ) {
                            Box(
                                modifier = Modifier.padding(vertical = 10.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = title,
                                    fontSize = 11.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                                    color = if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Tab Content
                Box(modifier = Modifier.weight(1f)) {
                    when (selectedTab) {
                        0 -> StatusAndControlsTab(
                            engineState = engineState,
                            telemetry = telemetry,
                            diagnostics = liveDiagnostics,
                            modelPathInput = modelPathInput,
                            onModelPathChange = { modelPathInput = it },
                            onInitialize = { path ->
                                coroutineScope.launch {
                                    val res = runtime.initialize(path)
                                    if (res.isSuccess) {
                                        Toast.makeText(context, "StreamVex model initialized successfully", Toast.LENGTH_SHORT).show()
                                    } else {
                                        Toast.makeText(context, "Init error: ${res.exceptionOrNull()?.message}", Toast.LENGTH_LONG).show()
                                    }
                                }
                            },
                            onShutdown = {
                                coroutineScope.launch {
                                    runtime.shutdown()
                                    Toast.makeText(context, "StreamVex engine unloaded", Toast.LENGTH_SHORT).show()
                                }
                            },
                            onTrimMemory = {
                                runtime.trimMemory(15) // TRIM_MEMORY_RUNNING_CRITICAL
                                Toast.makeText(context, "Triggered TRIM_MEMORY_RUNNING_CRITICAL", Toast.LENGTH_SHORT).show()
                            },
                            onPriorityChanged = { p ->
                                runtime.setPriority(p)
                                Toast.makeText(context, "Priority set to $p", Toast.LENGTH_SHORT).show()
                            }
                        )
                        1 -> BenchmarkTab(
                            modelPath = modelPathInput,
                            isBenchmarking = isBenchmarking,
                            progress = benchmarkProgress,
                            result = benchmarkResult,
                            onRunBenchmark = {
                                if (modelPathInput.isBlank()) {
                                    Toast.makeText(context, "Enter a valid model path first", Toast.LENGTH_SHORT).show()
                                    return@BenchmarkTab
                                }
                                isBenchmarking = true
                                coroutineScope.launch {
                                    benchmarkResult = StreamVexBenchmarkRunner.runBenchmark(
                                        modelPath = modelPathInput,
                                        onProgress = { p -> benchmarkProgress = p }
                                    )
                                    isBenchmarking = false
                                }
                            }
                        )
                        2 -> DiagnosticsAndLogsTab(
                            diagnostics = liveDiagnostics,
                            profile = liveProfile
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusAndControlsTab(
    engineState: StreamVexEngineState,
    telemetry: StreamVexTelemetry,
    diagnostics: StreamVexResourceDiagnostics,
    modelPathInput: String,
    onModelPathChange: (String) -> Unit,
    onInitialize: (String) -> Unit,
    onShutdown: () -> Unit,
    onTrimMemory: () -> Unit,
    onPriorityChanged: (Int) -> Unit
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // Status Overview Card
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "RUNTIME STATUS",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    val (statusText, statusColor) = when (engineState) {
                        is StreamVexEngineState.Uninitialized -> "UNINITIALIZED" to Color(0xFF9CA3AF)
                        is StreamVexEngineState.Loading -> "LOADING" to Color(0xFFF59E0B)
                        is StreamVexEngineState.Ready -> "READY" to Color(0xFF10B981)
                        is StreamVexEngineState.Generating -> "GENERATING" to Color(0xFF6366F1)
                        is StreamVexEngineState.Error -> "ERROR" to Color(0xFFEF4444)
                    }

                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = statusColor.copy(alpha = 0.15f),
                        border = androidx.compose.foundation.BorderStroke(1.dp, statusColor)
                    ) {
                        Text(
                            text = statusText,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            color = statusColor
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                MetricRow("Backend", "ARM64 NEON + Adaptive Scheduler (P0-P4)")
                MetricRow("Process RSS RAM", "${String.format("%.1f", diagnostics.processRssMb)} MB")
                MetricRow("System Available RAM", "${String.format("%.1f", diagnostics.availableRamMb)} MB")
                MetricRow("Context Size", "2048 tokens")
                MetricRow("Tokens / Sec", if (telemetry.tokensPerSec > 0) "${String.format("%.2f", telemetry.tokensPerSec)} tok/s" else "Idle")
                MetricRow("Time to First Token", if (telemetry.ttftMs > 0) "${String.format("%.1f", telemetry.ttftMs)} ms" else "Idle")

                if (engineState is StreamVexEngineState.Ready) {
                    MetricRow("Loaded Model", engineState.modelPath.substringAfterLast('/'))
                } else if (engineState is StreamVexEngineState.Error) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Error: ${engineState.message}",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        // Model Path & Initialization Controls
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "MODEL CONFIGURATION",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = modelPathInput,
                    onValueChange = onModelPathChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Model File Absolute Path (.gguf)") },
                    textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                    singleLine = true,
                    shape = RoundedCornerShape(12.dp)
                )

                Spacer(modifier = Modifier.height(14.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Button(
                        onClick = { onInitialize(modelPathInput) },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        enabled = modelPathInput.isNotBlank() && engineState !is StreamVexEngineState.Loading
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Initialize")
                    }

                    OutlinedButton(
                        onClick = onShutdown,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        enabled = engineState is StreamVexEngineState.Ready || engineState is StreamVexEngineState.Generating
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Shutdown")
                    }
                }
            }
        }

        // Adaptive Resource Controls
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "ADAPTIVE INTELLIGENCE & RECOVERY",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(modifier = Modifier.height(10.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { onPriorityChanged(1) }, // P1 Foreground Interactive
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("P1: Foreground", fontSize = 11.sp)
                    }

                    OutlinedButton(
                        onClick = { onPriorityChanged(4) }, // P4 Background Maintenance
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("P4: Background", fontSize = 11.sp)
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = onTrimMemory,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.tertiary),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(Icons.Default.CleaningServices, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Simulate Android Memory Pressure Trim", fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun BenchmarkTab(
    modelPath: String,
    isBenchmarking: Boolean,
    progress: String,
    result: StreamVexBenchmarkResult?,
    onRunBenchmark: () -> Unit
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Card(
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
            border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "REAL HARDWARE BENCHMARK SUITE",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Executes authentic token prefill, autoregressive generation, NEON GEMV kernels, and double vs. triple buffering flash storage benchmarks directly on this ARM64 device.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(14.dp))

                Button(
                    onClick = onRunBenchmark,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    enabled = !isBenchmarking && modelPath.isNotBlank()
                ) {
                    if (isBenchmarking) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Benchmarking...", fontSize = 13.sp)
                    } else {
                        Icon(Icons.Default.Speed, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Run Real Hardware Benchmark", fontSize = 13.sp)
                    }
                }

                if (progress.isNotBlank()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = progress,
                        style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                        color = MaterialTheme.colorScheme.secondary
                    )
                }
            }
        }

        if (result != null) {
            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "EMPIRICAL BENCHMARK RESULTS",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    MetricRow("Model Load Duration", "${result.modelLoadDurationMs} ms")
                    MetricRow("Time to First Token (TTFT)", "${String.format("%.1f", result.ttftMs)} ms")
                    MetricRow("Generation Throughput", "${String.format("%.2f", result.tokensPerSec)} tokens/sec")
                    MetricRow("Prefill Throughput", "${String.format("%.2f", result.prefillTokensPerSec)} tokens/sec")
                    MetricRow("Peak Resident RAM", "${String.format("%.1f", result.peakRamMb)} MB")
                    MetricRow("Total Generated Tokens", "${result.totalTokens} tokens")
                    MetricRow("Wall Generation Duration", "${String.format("%.2f", result.wallDurationSec)} sec")

                    Divider(modifier = Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))

                    MetricRow("Double-Buffer Throughput", "${String.format("%.1f", result.doubleBufferThroughputMbS)} MB/s")
                    MetricRow("Triple-Buffer Throughput", "${String.format("%.1f", result.tripleBufferThroughputMbS)} MB/s")
                    MetricRow("Recommended Buffering Mode", result.recommendedBufferingMode.uppercase())

                    Divider(modifier = Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))

                    MetricRow("ARM NEON Q4_0 Kernel", "${String.format("%.2f", result.q4Gflops)} GFLOPS")
                    MetricRow("Standard Float32 Kernel", "${String.format("%.2f", result.f32Gflops)} GFLOPS")
                }
            }
        }
    }
}

@Composable
private fun DiagnosticsAndLogsTab(
    diagnostics: StreamVexResourceDiagnostics,
    profile: StreamVexDeviceProfile
) {
    val logs = remember { mutableStateListOf<String>() }
    val context = LocalContext.current

    LaunchedEffect(Unit) {
        logs.clear()
        logs.addAll(StreamVexDiagnostics.getLogs())
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        item {
            Card(
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f)),
                border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f))
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "HARDWARE PROFILE & TOPOLOGY",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    MetricRow("Device Tier", profile.tierName)
                    MetricRow("CPU Core Topology", "${profile.totalCores} Cores (${profile.bigCores} Big + ${profile.totalCores - profile.bigCores} Little)")
                    MetricRow("ARM64 Dot-Product (ASIMDDP)", if (profile.hasDotprod) "Supported (Hardware Accelerated)" else "Emulated")
                    MetricRow("ARM64 FP16 Vectorization", if (profile.hasFp16) "Supported" else "Emulated")
                    MetricRow("Max Safe Model Budget", "${String.format("%.0f", profile.maxRecommendedModelSizeMb)} MB")
                    MetricRow("Thermal Status", "${String.format("%.1f", diagnostics.thermalCelsius)} °C (Level: ${diagnostics.thermalStatus})")
                    MetricRow("Battery Level", "${diagnostics.batteryPct}%")
                }
            }
        }

        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "LIVE ENGINE LOGS (${logs.size})",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IconButton(
                        onClick = {
                            val allLogs = logs.joinToString("\n")
                            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            cm.setPrimaryClip(ClipData.newPlainText("StreamVex Logs", allLogs))
                            Toast.makeText(context, "Logs copied to clipboard", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(Icons.Default.ContentCopy, contentDescription = "Copy Logs", modifier = Modifier.size(16.dp))
                    }

                    IconButton(
                        onClick = {
                            StreamVexDiagnostics.clearLogs()
                            logs.clear()
                        },
                        modifier = Modifier.size(32.dp)
                    ) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear Logs", modifier = Modifier.size(16.dp))
                    }
                }
            }
        }

        items(logs) { logLine ->
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(8.dp),
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.5f)
            ) {
                Text(
                    text = logLine,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(text = value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface)
    }
}
