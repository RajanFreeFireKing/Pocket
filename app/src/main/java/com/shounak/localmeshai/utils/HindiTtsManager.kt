package com.shounak.localmeshai.utils

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import java.util.Locale
import java.util.UUID

/**
 * Native Android Hindi Text-To-Speech (TTS) Manager.
 * Configured with Hindi locale (Locale("hi", "IN")), with automatic fallback to system default.
 * Provides markdown cleaning and instant speech start / stop controls.
 */
object HindiTtsManager {
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private val _speakingMessageId = mutableStateOf<String?>(null)
    val speakingMessageId: State<String?> = _speakingMessageId

    fun init(context: Context) {
        if (tts != null) return
        val appContext = context.applicationContext
        tts = TextToSpeech(appContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val hindiLocale = Locale("hi", "IN")
                val result = tts?.setLanguage(hindiLocale)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    // Fallback to default locale if Hindi language pack is unavailable
                    tts?.language = Locale.getDefault()
                }
                isInitialized = true
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        _speakingMessageId.value = null
                    }
                    override fun onError(utteranceId: String?) {
                        _speakingMessageId.value = null
                    }
                    override fun onStop(utteranceId: String?, interrupted: Boolean) {
                        _speakingMessageId.value = null
                    }
                })
            }
        }
    }

    fun isSpeaking(messageId: String): Boolean {
        return _speakingMessageId.value == messageId
    }

    fun speak(context: Context, messageId: String, rawText: String) {
        init(context)
        if (_speakingMessageId.value == messageId) {
            stop()
            return
        }
        stop()

        val cleanText = stripMarkdownForSpeech(rawText)
        if (cleanText.isBlank()) return

        val utteranceId = UUID.randomUUID().toString()
        _speakingMessageId.value = messageId
        tts?.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }

    fun stop() {
        tts?.stop()
        _speakingMessageId.value = null
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
        _speakingMessageId.value = null
    }

    /**
     * Strips markdown symbols (**, `, #, _, >, etc.) for natural voice speech.
     */
    fun stripMarkdownForSpeech(text: String): String {
        return text
            // Remove code blocks
            .replace(Regex("```[a-zA-Z0-9]*\\n([\\s\\S]*?)```"), "$1")
            // Remove inline code
            .replace(Regex("`([^`]+)`"), "$1")
            // Remove bold/italics
            .replace(Regex("\\*\\*([^*]+)\\*\\*"), "$1")
            .replace(Regex("\\*([^*]+)\\*"), "$1")
            .replace(Regex("__([^_]+)__"), "$1")
            .replace(Regex("_([^_]+)_"), "$1")
            // Remove headers (#, ##, ###)
            .replace(Regex("(?m)^#{1,6}\\s+"), "")
            // Remove blockquotes (> )
            .replace(Regex("(?m)^>\\s*"), "")
            // Remove markdown links [text](url) -> text
            .replace(Regex("\\[([^\\]]+)\\]\\([^)]+\\)"), "$1")
            // Remove bullet points (*, -, +)
            .replace(Regex("(?m)^[\\*\\-\\+]\\s+"), "")
            // Remove numbered lists (1. 2.)
            .replace(Regex("(?m)^\\d+\\.\\s+"), "")
            // Remove tildes (~~strikethrough~~)
            .replace(Regex("~~([^~]+)~~"), "$1")
            // Remove excess blank lines
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
    }
}
