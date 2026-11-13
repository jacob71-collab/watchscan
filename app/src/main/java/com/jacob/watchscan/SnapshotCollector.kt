package com.jacob.watchscan

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Takes periodic snapshots while the user sweeps over a page.
 *
 * Within each time window it keeps the sharpest frame; at the end of the window that
 * frame becomes a snapshot if it is reasonably sharp and not a near-duplicate of the
 * previous snapshot (e.g. the user paused). Snapshots are lighting-corrected and
 * stored as JPEGs sized for Claude.
 *
 * Call from one thread.
 */
class SnapshotCollector {
    companion object {
        const val WINDOW_MS = 800L
        const val MAX_KEEP = 45
    }

    /**
     * [full] is handed to the stitcher, which releases it. [px], [py] is where the
     * camera had moved to (tracked frame to frame, in full-frame pixels); [unc] grows
     * when tracking was unsure.
     */
    class Snap(val full: Mat, val thumb: Mat, val sharp: Double, val px: Double, val py: Double, val unc: Double)

    /** [dark] = share of the view that is in deep shadow or blocked (0..1). */
    class Info(val sharp: Double, val maxSharp: Double, val newSnap: Snap?, val count: Int, val dark: Double)

    val snaps = ArrayList<Snap>()
    private var maxSharp = 0.0
    private var windowStart = 0L
    private var cand: Mat? = null
    private var candSharp = 0.0
    private var lastSig: Mat? = null

    // Frame-to-frame motion tracking (phase correlation on small frames)
    private var prevSmall: Mat? = null
    private val win = Mat()
    private var posX = 0.0
    private var posY = 0.0
    private var unc = 0.0
    private var candX = 0.0
    private var candY = 0.0
    private var candUnc = 0.0

    var lastDark = 0.0; private set
    private var candDark = 0.0

    /** Share of a small frame much darker than the paper (hand shadow, watch strap in view). */
    private fun darkShare(small8: Mat): Double {
        val n = small8.total().toInt()
        val a = ByteArray(n); small8.get(0, 0, a)
        val hist = IntArray(256); for (v in a) hist[v.toInt() and 0xff]++
        var acc = 0; var p95 = 255
        for (v in 0 until 256) { acc += hist[v]; if (acc >= n * 0.95) { p95 = v; break } }
        var dark = 0; for (v in 0 until (p95 / 2)) dark += hist[v]
        return dark.toDouble() / max(1, n)
    }

    private fun track(frame: Mat) {
        val tw = 256
        val th = max(16, (256.0 * frame.rows() / frame.cols()).toInt())
        val small = Mat()
        Imgproc.resize(frame, small, Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        lastDark = darkShare(small)
        small.convertTo(small, org.opencv.core.CvType.CV_32F)
        if (win.empty() || win.cols() != tw || win.rows() != th) {
            Imgproc.createHanningWindow(win, Size(tw.toDouble(), th.toDouble()), org.opencv.core.CvType.CV_32F)
        }
        val prev = prevSmall
        if (prev != null && prev.cols() == tw && prev.rows() == th) {
            val resp = DoubleArray(1)
            val sh = Imgproc.phaseCorrelate(prev, small, win, resp)
            if (resp[0] > 0.02) {
                posX -= sh.x * frame.cols() / tw
                posY -= sh.y * frame.rows() / th
            } else {
                unc += 0.05 * hypot(frame.cols().toDouble(), frame.rows().toDouble())
            }
        }
        prev?.release()
        prevSmall = small
    }

    fun offer(frame: Mat, now: Long): Info {
        val w = frame.cols(); val h = frame.rows()
        val cr = Rect(max(0, w / 2 - 300), max(0, h / 2 - 300), min(600, w), min(600, h))
        val sharp = laplaceVar(frame.submat(cr))
        maxSharp = max(maxSharp * 0.985, sharp)
        if (windowStart == 0L) windowStart = now
        track(frame)

        if (cand == null || sharp > candSharp) {
            cand?.release()
            cand = frame.clone()
            candSharp = sharp
            candX = posX; candY = posY; candUnc = unc; candDark = lastDark
        }
        var made: Snap? = null
        if (now - windowStart >= WINDOW_MS) {
            val c = cand
            // Skip views mostly covered by shadow or blocked: they add little and confuse the joining.
            if (c != null && candSharp >= 0.45 * maxSharp && candDark < 0.55 && snaps.size < MAX_KEEP) {
                val sig = signature(c)
                val prev = lastSig
                if (prev == null || difference(sig, prev) > 7.0) {
                    made = makeSnap(c, candSharp)
                    snaps.add(made)
                    prev?.release()
                    lastSig = sig
                } else {
                    sig.release()
                }
            }
            c?.release()
            cand = null
            candSharp = 0.0
            windowStart = now
        }
        return Info(sharp, maxSharp, made, snaps.size, lastDark)
    }

    /** Small blurred thumbnail used to spot near-duplicate snapshots. */
    private fun signature(img: Mat): Mat {
        val s = Mat()
        Imgproc.resize(img, s, Size(64.0, 48.0), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(s, s, Size(3.0, 3.0), 0.0)
        Core.normalize(s, s, 0.0, 255.0, Core.NORM_MINMAX)
        return s
    }

    private fun difference(a: Mat, b: Mat): Double {
        val d = Mat()
        Core.absdiff(a, b, d)
        val m = Core.mean(d).`val`[0]
        d.release()
        return m
    }

    private fun makeSnap(frame: Mat, sharp: Double): Snap {
        val small = Mat()
        Imgproc.resize(frame, small, Size(240.0, 240.0 * frame.rows() / frame.cols()), 0.0, 0.0, Imgproc.INTER_AREA)
        val thumb = flatten(small)
        small.release()
        return Snap(frame.clone(), thumb, sharp, candX, candY, candUnc)
    }

    /** All snapshot thumbnails tiled into one image, numbered in order. */
    fun contactSheet(): Mat {
        val n = max(1, snaps.size)
        val cols = min(4, n)
        val rows = (n + cols - 1) / cols
        val tw = 240
        val th = snaps.firstOrNull()?.thumb?.rows() ?: 180
        val sheet = Mat(rows * (th + 4) + 4, cols * (tw + 4) + 4, org.opencv.core.CvType.CV_8UC1, Scalar(40.0))
        for ((i, s) in snaps.withIndex()) {
            val x = 4 + (i % cols) * (tw + 4)
            val y = 4 + (i / cols) * (th + 4)
            val roi = sheet.submat(Rect(x, y, min(tw, s.thumb.cols()), min(th, s.thumb.rows())))
            s.thumb.submat(Rect(0, 0, roi.cols(), roi.rows())).copyTo(roi)
            Imgproc.putText(sheet, "${i + 1}", org.opencv.core.Point(x + 6.0, y + 26.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0.0), 4)
            Imgproc.putText(sheet, "${i + 1}", org.opencv.core.Point(x + 6.0, y + 26.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255.0), 2)
        }
        return sheet
    }

    fun release() {
        for (s in snaps) s.thumb.release()
        snaps.clear()
        cand?.release(); cand = null
        lastSig?.release(); lastSig = null
        prevSmall?.release(); prevSmall = null
        win.release()
    }
}
