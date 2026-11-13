package com.jacob.watchscan

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Self-update from the GitHub "latest" release. The APK is downloaded into the app's own
 * cache (not the Downloads folder) and removed again on the next start.
 */
object Updater {
    private const val RELEASE_API = "https://api.github.com/repos/jacob71-collab/watchscan/releases/tags/latest"

    class Release(val build: Long, val apkUrl: String)

    fun currentBuild(ctx: Context): Long {
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        @Suppress("DEPRECATION")
        return if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode else pi.versionCode.toLong()
    }

    private fun updatesDir(ctx: Context) = File(ctx.cacheDir, "updates").apply { mkdirs() }

    /** Deletes any downloaded update files (call on start). */
    fun cleanUp(ctx: Context) {
        updatesDir(ctx).listFiles()?.forEach { it.delete() }
    }

    /** Newest published build, or throws. Network call: run off the main thread. */
    fun latest(): Release {
        val conn = URL(RELEASE_API).openConnection() as HttpURLConnection
        try {
            conn.connectTimeout = 15_000; conn.readTimeout = 20_000
            conn.setRequestProperty("Accept", "application/vnd.github+json")
            conn.setRequestProperty("User-Agent", "WatchScan")
            if (conn.responseCode != 200) throw IOException("GitHub answered ${conn.responseCode}")
            val json = JSONObject(conn.inputStream.bufferedReader().use { it.readText() })
            val build = Regex("build (\\d+)").find(json.optString("name"))?.groupValues?.get(1)?.toLong()
                ?: throw IOException("couldn't read the build number")
            val assets = json.getJSONArray("assets")
            var url: String? = null
            for (i in 0 until assets.length()) {
                val a = assets.getJSONObject(i)
                if (a.optString("name").endsWith(".apk")) url = a.optString("browser_download_url")
            }
            return Release(build, url ?: throw IOException("no APK in the release"))
        } finally {
            conn.disconnect()
        }
    }

    /** Downloads the APK into the app's cache, reporting progress 0..100. */
    fun download(ctx: Context, r: Release, progress: (Int) -> Unit): File {
        val out = File(updatesDir(ctx), "WatchScan-${r.build}.apk")
        var url = URL(r.apkUrl)
        var conn: HttpURLConnection
        var hops = 0
        while (true) { // GitHub redirects to its file host
            conn = url.openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 15_000; conn.readTimeout = 60_000
            conn.setRequestProperty("User-Agent", "WatchScan")
            val code = conn.responseCode
            if (code in 300..399 && hops < 5) {
                url = URL(url, conn.getHeaderField("Location")); conn.disconnect(); hops++; continue
            }
            if (code != 200) throw IOException("download failed ($code)")
            break
        }
        try {
            val total = conn.contentLengthLong
            conn.inputStream.use { input ->
                out.outputStream().use { o ->
                    val buf = ByteArray(64 * 1024)
                    var done = 0L; var last = -1
                    while (true) {
                        val n = input.read(buf); if (n < 0) break
                        o.write(buf, 0, n); done += n
                        if (total > 0) { val p = (100 * done / total).toInt(); if (p != last) { last = p; progress(p) } }
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
        return out
    }

    /** True if Android lets this app open the installer (Android 8+ asks once per app). */
    fun canInstall(ctx: Context): Boolean =
        Build.VERSION.SDK_INT < 26 || ctx.packageManager.canRequestPackageInstalls()

    /** Opens the system screen where the user allows WatchScan to install updates. */
    fun askInstallPermission(ctx: Context) {
        if (Build.VERSION.SDK_INT >= 26) {
            ctx.startActivity(
                Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${ctx.packageName}"))
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }
    }

    /** Opens Android's installer for the downloaded APK. */
    fun install(ctx: Context, apk: File) {
        val uri = FileProvider.getUriForFile(ctx, "com.jacob.watchscan.files", apk)
        val i = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        ctx.startActivity(i)
    }
}
