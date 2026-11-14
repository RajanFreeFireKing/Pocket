#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include <mutex>
#include <memory>
#include <iostream>

#include "streamvex.h"

namespace {

static JavaVM* g_jvm = nullptr;

/**
 * RAII Guard for attaching native/worker threads to JavaVM.
 */
class JniThreadAttachment {
public:
    JniThreadAttachment() {
        if (!g_jvm) return;
        jint status = g_jvm->GetEnv(reinterpret_cast<void**>(&m_env), JNI_VERSION_1_6);
        if (status == JNI_EDETACHED) {
#if defined(__ANDROID__)
            if (g_jvm->AttachCurrentThread(&m_env, nullptr) == 0) {
                m_attached = true;
            }
#else
            if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&m_env), nullptr) == 0) {
                m_attached = true;
            }
#endif
        }
    }

    ~JniThreadAttachment() {
        if (m_attached && g_jvm) {
            g_jvm->DetachCurrentThread();
        }
    }

    JNIEnv* env() const { return m_env; }
    bool is_valid() const { return m_env != nullptr; }

private:
    JNIEnv* m_env = nullptr;
    bool m_attached = false;
};

/**
 * Thread-safe callback dispatcher for streaming tokens back to Java/Kotlin.
 */
class JniStreamingDispatcher {
public:
    JniStreamingDispatcher(JNIEnv* env, jobject thiz, jmethodID mid)
        : m_mid(mid) {
        m_global_thiz = env->NewGlobalRef(thiz);
    }

    ~JniStreamingDispatcher() {
        if (m_global_thiz) {
            JniThreadAttachment guard;
            if (guard.is_valid()) {
                guard.env()->DeleteGlobalRef(m_global_thiz);
            }
            m_global_thiz = nullptr;
        }
    }

    void emit_token(const char* token) {
        if (!token || !m_global_thiz || !m_mid) return;
        JniThreadAttachment guard;
        if (!guard.is_valid()) return;

        JNIEnv* env = guard.env();
        jstring jtok = env->NewStringUTF(token);
        if (jtok) {
            env->CallVoidMethod(m_global_thiz, m_mid, jtok);
            env->DeleteLocalRef(jtok);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
        }
    }

private:
    jobject m_global_thiz = nullptr;
    jmethodID m_mid = nullptr;
};

static thread_local std::shared_ptr<JniStreamingDispatcher> tls_dispatcher;

static void active_jni_token_callback(const char* token) {
    if (tls_dispatcher) {
        tls_dispatcher->emit_token(token);
    }
}

} // anonymous namespace

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void)reserved;
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    (void)vm;
    (void)reserved;
    g_jvm = nullptr;
}

// ============================================================================
// Original NativeBridge APIs (Preserved & Fully Backwards-Compatible)
// ============================================================================

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_initEngine(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    (void)thiz;
    if (!modelPath) return -1;
    const char* c_path = env->GetStringUTFChars(modelPath, nullptr);
    if (!c_path) return -2;

    int res = -1;
    try {
        res = init_engine(c_path);
    } catch (...) {
        res = -99;
    }
    env->ReleaseStringUTFChars(modelPath, c_path);
    return static_cast<jint>(res);
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_nativeInitEngine(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    return Java_com_vexdesk_client_NativeBridge_initEngine(env, thiz, modelPath);
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_streamPrompt(
    JNIEnv* env, jobject thiz, jstring prompt) {
    if (!prompt) return -1;

    jclass clazz = env->GetObjectClass(thiz);
    jmethodID mid = env->GetMethodID(clazz, "onTokenReceived", "(Ljava/lang/String;)V");
    if (!mid) return -2;

    tls_dispatcher = std::make_shared<JniStreamingDispatcher>(env, thiz, mid);

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    int res = -1;
    try {
        res = stream_token(c_prompt, active_jni_token_callback);
    } catch (...) {
        res = -99;
    }
    env->ReleaseStringUTFChars(prompt, c_prompt);

    tls_dispatcher.reset();
    return static_cast<jint>(res);
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_nativeStreamPrompt(
    JNIEnv* env, jobject thiz, jstring prompt) {
    return Java_com_vexdesk_client_NativeBridge_streamPrompt(env, thiz, prompt);
}

JNIEXPORT jdouble JNICALL Java_com_vexdesk_client_NativeBridge_getMemoryUsage(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    return static_cast<jdouble>(get_memory_usage());
}

JNIEXPORT jdouble JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetMemoryUsage(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getMemoryUsage(env, thiz);
}

JNIEXPORT jboolean JNICALL Java_com_vexdesk_client_NativeBridge_isEngineReady(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    return static_cast<jboolean>(is_engine_ready() != 0);
}

JNIEXPORT jboolean JNICALL Java_com_vexdesk_client_NativeBridge_nativeIsEngineReady(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_isEngineReady(env, thiz);
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_releaseEngine(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    try {
        release_engine();
    } catch (...) {}
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_nativeReleaseEngine(
    JNIEnv* env, jobject thiz) {
    Java_com_vexdesk_client_NativeBridge_releaseEngine(env, thiz);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_getTelemetry(
    JNIEnv* env, jobject thiz) {
    (void)thiz;
    char buf[2048] = {0};
    int res = get_last_generation_telemetry(buf, sizeof(buf));
    if (res != 0) {
        return env->NewStringUTF("{}");
    }
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetTelemetry(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getTelemetry(env, thiz);
}

// ============================================================================
// Version 2.0 Extended JNI APIs (Adaptive Background, Voice, Diagnostics)
// ============================================================================

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_nativeAbortGeneration(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    streamvex_abort_generation();
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_nativeOnTrimMemory(
    JNIEnv* env, jobject thiz, jint trimLevel) {
    (void)env;
    (void)thiz;
    streamvex_on_trim_memory(static_cast<int>(trimLevel));
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_nativeSetBackgroundMode(
    JNIEnv* env, jobject thiz, jboolean isBackground) {
    (void)env;
    (void)thiz;
    streamvex_set_background_mode(isBackground ? 1 : 0);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetResourceTelemetry(
    JNIEnv* env, jobject thiz) {
    (void)thiz;
    char buf[4096] = {0};
    int res = streamvex_get_resource_telemetry(buf, sizeof(buf));
    if (res != 0) return env->NewStringUTF("{}");
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetDeviceProfile(
    JNIEnv* env, jobject thiz) {
    (void)thiz;
    char buf[4096] = {0};
    int res = streamvex_get_device_profile(buf, sizeof(buf));
    if (res != 0) return env->NewStringUTF("{}");
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeValidateModel(
    JNIEnv* env, jobject thiz, jstring modelPath, jint contextLen) {
    (void)thiz;
    if (!modelPath) return env->NewStringUTF("{\"is_compatible\": false, \"message\": \"Null model path\"}");
    const char* c_path = env->GetStringUTFChars(modelPath, nullptr);
    if (!c_path) return env->NewStringUTF("{\"is_compatible\": false, \"message\": \"Failed to read model path\"}");

    char buf[4096] = {0};
    streamvex_validate_model(c_path, static_cast<int>(contextLen), buf, sizeof(buf));
    env->ReleaseStringUTFChars(modelPath, c_path);
    return env->NewStringUTF(buf);
}

JNIEXPORT jboolean JNICALL Java_com_vexdesk_client_NativeBridge_nativeProcessVadFrame(
    JNIEnv* env, jobject thiz, jshortArray pcmSamples, jint numSamples) {
    (void)thiz;
    if (!pcmSamples || numSamples <= 0) return JNI_FALSE;

    jshort* samples = env->GetShortArrayElements(pcmSamples, nullptr);
    if (!samples) return JNI_FALSE;

    int is_speech = 0;
    float prob = 0.0f;
    float energy_db = 0.0f;
    streamvex_voice_process_vad(samples, static_cast<size_t>(numSamples), &is_speech, &prob, &energy_db);

    env->ReleaseShortArrayElements(pcmSamples, samples, JNI_ABORT);
    return (is_speech != 0) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jshortArray JNICALL Java_com_vexdesk_client_NativeBridge_nativeSynthesizeSpeech(
    JNIEnv* env, jobject thiz, jstring text) {
    (void)thiz;
    if (!text) return nullptr;

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    if (!c_text) return nullptr;

    std::vector<int16_t> audio_buf(65536);
    int actual_samples = 0;
    int res = streamvex_voice_synthesize(c_text, audio_buf.data(), audio_buf.size(), &actual_samples);
    env->ReleaseStringUTFChars(text, c_text);

    if (res != 0 || actual_samples <= 0) {
        return nullptr;
    }

    jshortArray jarr = env->NewShortArray(actual_samples);
    if (jarr) {
        env->SetShortArrayRegion(jarr, 0, actual_samples, audio_buf.data());
    }
    return jarr;
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeTranscribeSpeech(
    JNIEnv* env, jobject thiz, jshortArray pcmSamples, jint numSamples) {
    (void)thiz;
    if (!pcmSamples || numSamples <= 0) return env->NewStringUTF("");

    jshort* samples = env->GetShortArrayElements(pcmSamples, nullptr);
    if (!samples) return env->NewStringUTF("");

    char text_buf[1024] = {0};
    streamvex_voice_transcribe(samples, static_cast<size_t>(numSamples), text_buf, sizeof(text_buf));
    env->ReleaseShortArrayElements(pcmSamples, samples, JNI_ABORT);

    return env->NewStringUTF(text_buf);
}

// ============================================================================
// Solus StreamVex Integration JNI Bindings (com.shounak.localmeshai.core.streamvex.native.StreamVexNativeBridge)
// ============================================================================

JNIEXPORT jint JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_initEngine(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    return Java_com_vexdesk_client_NativeBridge_initEngine(env, thiz, modelPath);
}

JNIEXPORT jint JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_streamToken(
    JNIEnv* env, jobject thiz, jstring prompt, jobject callback) {
    (void)thiz;
    if (!prompt || !callback) return -1;

    jclass callbackClass = env->GetObjectClass(callback);
    jmethodID mid = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    if (!mid) {
        env->ExceptionClear();
        mid = env->GetMethodID(callbackClass, "onTokenReceived", "(Ljava/lang/String;)V");
    }
    if (!mid) return -2;

    tls_dispatcher = std::make_shared<JniStreamingDispatcher>(env, callback, mid);

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    int res = -1;
    try {
        res = stream_token(c_prompt, active_jni_token_callback);
    } catch (...) {
        res = -99;
    }
    env->ReleaseStringUTFChars(prompt, c_prompt);
    tls_dispatcher.reset();
    return static_cast<jint>(res);
}

JNIEXPORT jdouble JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_getMemoryUsage(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getMemoryUsage(env, thiz);
}

JNIEXPORT jboolean JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_isEngineReady(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_isEngineReady(env, thiz);
}

JNIEXPORT void JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_releaseEngine(
    JNIEnv* env, jobject thiz) {
    Java_com_vexdesk_client_NativeBridge_releaseEngine(env, thiz);
}

JNIEXPORT void JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_abortGeneration(
    JNIEnv* env, jobject thiz) {
    Java_com_vexdesk_client_NativeBridge_nativeAbortGeneration(env, thiz);
}

JNIEXPORT void JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_onTrimMemory(
    JNIEnv* env, jobject thiz, jint trimLevel) {
    Java_com_vexdesk_client_NativeBridge_nativeOnTrimMemory(env, thiz, trimLevel);
}

JNIEXPORT void JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_setBackgroundMode(
    JNIEnv* env, jobject thiz, jboolean isBackground) {
    Java_com_vexdesk_client_NativeBridge_nativeSetBackgroundMode(env, thiz, isBackground);
}

JNIEXPORT jint JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_setPriority(
    JNIEnv* env, jobject thiz, jint priorityLevel) {
    (void)env;
    (void)thiz;
    streamvex_set_background_mode(priorityLevel >= 3 ? 1 : 0);
    return 0;
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_getResourceTelemetry(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_nativeGetResourceTelemetry(env, thiz);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_getDeviceProfile(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_nativeGetDeviceProfile(env, thiz);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_validateModel(
    JNIEnv* env, jobject thiz, jstring modelPath, jint contextLen) {
    return Java_com_vexdesk_client_NativeBridge_nativeValidateModel(env, thiz, modelPath, contextLen);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_parseGgufMetadata(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    (void)thiz;
    if (!modelPath) return env->NewStringUTF("{}");
    const char* c_path = env->GetStringUTFChars(modelPath, nullptr);
    if (!c_path) return env->NewStringUTF("{}");

    char buf[16384] = {0};
    int res = parse_gguf_metadata(c_path, buf, sizeof(buf));
    env->ReleaseStringUTFChars(modelPath, c_path);
    if (res != 0) return env->NewStringUTF("{}");
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_computeGemvBenchmark(
    JNIEnv* env, jobject thiz, jint rows, jint cols) {
    (void)thiz;
    char buf[4096] = {0};
    int res = compute_gemv_benchmark(static_cast<int>(rows), static_cast<int>(cols), buf, sizeof(buf));
    if (res != 0) return env->NewStringUTF("{}");
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_benchmarkTripleBuffering(
    JNIEnv* env, jobject thiz, jstring filePath) {
    (void)thiz;
    if (!filePath) return env->NewStringUTF("{}");
    const char* c_path = env->GetStringUTFChars(filePath, nullptr);
    if (!c_path) return env->NewStringUTF("{}");

    char buf[4096] = {0};
    int res = streamvex_benchmark_triple_buffering(c_path, 4, 10, buf, sizeof(buf));
    env->ReleaseStringUTFChars(filePath, c_path);
    if (res != 0) return env->NewStringUTF("{}");
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_getTelemetry(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getTelemetry(env, thiz);
}

JNIEXPORT jboolean JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_processVadFrame(
    JNIEnv* env, jobject thiz, jshortArray pcmSamples, jint numSamples) {
    return Java_com_vexdesk_client_NativeBridge_nativeProcessVadFrame(env, thiz, pcmSamples, numSamples);
}

JNIEXPORT jshortArray JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_synthesizeSpeech(
    JNIEnv* env, jobject thiz, jstring text) {
    return Java_com_vexdesk_client_NativeBridge_nativeSynthesizeSpeech(env, thiz, text);
}

JNIEXPORT jstring JNICALL Java_com_shounak_localmeshai_core_streamvex_native_StreamVexNativeBridge_transcribeSpeech(
    JNIEnv* env, jobject thiz, jshortArray pcmSamples, jint numSamples) {
    return Java_com_vexdesk_client_NativeBridge_nativeTranscribeSpeech(env, thiz, pcmSamples, numSamples);
}

} // extern "C"
