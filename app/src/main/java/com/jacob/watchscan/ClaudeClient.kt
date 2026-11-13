package com.jacob.watchscan

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.URL
import javax.net.ssl.HttpsURLConnection

/** Sends the scanned page to the Claude Messages API and returns the answer text. */
object ClaudeClient {
    val MODELS = listOf(
        "claude-sonnet-5-5" to "Sonnet 5.5",
        "claude-opus-5-5" to "Opus 5.5",
    )

    private const val SYSTEM = """You are looking at a sheet of paper that the user scanned with a smartwatch camera. Read the question or questions on it and answer them.

Your reply is shown on a tiny round watch screen and may be read aloud, so:
- Start with the answer itself (for multiple choice, give the letter and the option text).
- Then add a short explanation only if it helps: at most three short sentences.
- If there are several questions, number the answers to match.
- Plain text only. No Markdown, no tables, no LaTeX.
- If part of the page is unreadable or cut off, say which part, and answer what you can."""

    private fun imageBlock(jpeg: ByteArray) = JSONObject()
        .put("type", "image")
        .put(
            "source", JSONObject()
                .put("type", "base64")
                .put("media_type", "image/jpeg")
                .put("data", Base64.encodeToString(jpeg, Base64.NO_WRAP))
        )

    /** [page] is the stitched full page; [closeUps] are sharp close-up photos of parts of it. */
    fun ask(apiKey: String, model: String, page: ByteArray, closeUps: List<ByteArray> = emptyList()): String {
        val content = JSONArray()
        content.put(JSONObject().put("type", "text").put("text", "Full page (stitched from many close-up frames):"))
        content.put(imageBlock(page))
        if (closeUps.isNotEmpty()) {
            content.put(
                JSONObject().put("type", "text").put(
                    "text",
                    "Close-up photos of parts of the same page (sharper, use them to read small or unclear text):"
                )
            )
            for (c in closeUps) content.put(imageBlock(c))
        }
        content.put(JSONObject().put("type", "text").put("text", "Answer the question(s) on this page."))
        val body = JSONObject()
            .put("model", model)
            .put("max_tokens", 1500)
            .put("system", SYSTEM)
            .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", content)))

        val conn = URL("https://api.anthropic.com/v1/messages").openConnection() as HttpsURLConnection
        try {
            conn.requestMethod = "POST"
            conn.connectTimeout = 20_000
            conn.readTimeout = 180_000
            conn.doOutput = true
            conn.setRequestProperty("x-api-key", apiKey)
            conn.setRequestProperty("anthropic-version", "2023-06-01")
            conn.setRequestProperty("content-type", "application/json")
            conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val resp = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            if (code !in 200..299) {
                val msg = try {
                    JSONObject(resp).getJSONObject("error").optString("message", resp)
                } catch (e: Exception) {
                    resp.take(200)
                }
                throw IOException(
                    when (code) {
                        401 -> "The API key was rejected. Scan a new key in settings."
                        429 -> "Rate limited or out of credit: $msg"
                        529 -> "Claude is overloaded right now. Try again in a moment."
                        else -> "Claude API error $code: $msg"
                    }
                )
            }
            val json = JSONObject(resp)
            val content = json.optJSONArray("content") ?: JSONArray()
            val sb = StringBuilder()
            for (i in 0 until content.length()) {
                val block = content.getJSONObject(i)
                if (block.optString("type") == "text") sb.append(block.optString("text"))
            }
            val answer = sb.toString().trim()
            return answer.ifEmpty {
                "(No answer returned. Stop reason: ${json.optString("stop_reason", "unknown")})"
            }
        } finally {
            conn.disconnect()
        }
    }
}
