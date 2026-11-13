package com.jacob.watchscan

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

// ---------------------------------------------------------------------------
// Shared image helpers

fun laplaceVar(img: Mat): Double {
    val lap = Mat()
    Imgproc.Laplacian(img, lap, CvType.CV_16S)
    val mean = MatOfDouble(); val sd = MatOfDouble()
    Core.meanStdDev(lap, mean, sd)
    lap.release()
    val v = sd.toArray()[0]
    return v * v
}

/** Removes uneven lighting / vignetting; paper becomes ~245. */
fun flatten(gray: Mat): Mat {
    val w = gray.cols(); val h = gray.rows()
    val small = Mat()
    Imgproc.resize(gray, small, Size(max(1, w / 8).toDouble(), max(1, h / 8).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
    Imgproc.dilate(small, small, Mat.ones(7, 7, CvType.CV_8U))
    val med = Mat()
    Imgproc.medianBlur(small, med, 21)
    val bg = Mat()
    Imgproc.resize(med, bg, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
    val out = Mat()
    Core.divide(gray, bg, out, 245.0)
    small.release(); med.release(); bg.release()
    return out
}

fun enhance(gray: Mat): Mat {
    val out = flatten(gray)
    val n = (out.total() * out.channels()).toInt()
    val data = ByteArray(n)
    out.get(0, 0, data)
    val hist = IntArray(256)
    var cnt = 0
    var i = 0
    while (i < n) { hist[data[i].toInt() and 0xff]++; cnt++; i += 7 }
    var acc = 0
    var lo = 0
    while (lo < 255) { acc += hist[lo]; if (acc >= cnt * 0.02) break; lo++ }
    val alpha = 255.0 / max(1, 255 - lo)
    Core.convertScaleAbs(out, out, alpha, -lo * alpha)
    return out
}

fun isConvex(x: DoubleArray, y: DoubleArray): Boolean {
    var sign = 0
    for (i in 0 until 4) {
        val j = (i + 1) % 4; val k = (i + 2) % 4
        val cross = (x[j] - x[i]) * (y[k] - y[j]) - (y[j] - y[i]) * (x[k] - x[j])
        val sg = if (cross > 0) 1 else if (cross < 0) -1 else 0
        if (sg == 0) return false
        if (sign == 0) sign = sg else if (sg != sign) return false
    }
    return true
}

fun polyArea(x: DoubleArray, y: DoubleArray): Double {
    var a = 0.0
    for (i in x.indices) { val j = (i + 1) % x.size; a += x[i] * y[j] - x[j] * y[i] }
    return abs(a) / 2
}

/** Output size (w, h) with long side [long] for a detected page, snapping to A4 / Letter / Legal shapes. */
fun pageSize(q: Array<Point>, long: Int): Pair<Int, Int> {
    fun d(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
    val w = (d(q[0], q[1]) + d(q[3], q[2])) / 2
    val h = (d(q[0], q[3]) + d(q[1], q[2])) / 2
    var ar = h / max(1.0, w)
    for (std in doubleArrayOf(1.4142, 1.2941, 1.6471)) {
        for (a in doubleArrayOf(std, 1 / std)) if (abs(ar - a) / a < 0.07) ar = a
    }
    return if (ar >= 1) Pair((long / ar).roundToInt(), long)
    else Pair(long, (long * ar).roundToInt())
}

object PageDetector {
    /**
     * Returns page corners (TL, TR, BR, BL) in image coordinates, or null.
     * Candidates are 4-sided outlines from a brightness threshold and from edges; the
     * winner is the large one that is clearly brighter inside than just outside it.
     * [valid] (optional, 255 = captured) excludes outlines whose outside was never seen.
     */
    fun detect(gray: Mat, valid: Mat? = null): Array<Point>? {
        val w = gray.cols(); val h = gray.rows()
        val s = 640.0 / max(w, h)
        val sz = Size((w * s).toInt().toDouble(), (h * s).toInt().toDouble())
        val small = Mat()
        Imgproc.resize(gray, small, sz, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(5.0, 5.0), 0.0)
        var vs: Mat? = null
        if (valid != null) { vs = Mat(); Imgproc.resize(valid, vs, sz, 0.0, 0.0, Imgproc.INTER_NEAREST) }
        val areaMin = 0.15 * small.rows() * small.cols()
        val th = Mat()
        Imgproc.threshold(small, th, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.morphologyEx(th, th, Imgproc.MORPH_CLOSE, Mat.ones(9, 9, CvType.CV_8U))
        val ed = Mat()
        Imgproc.Canny(small, ed, 30.0, 90.0)
        Imgproc.dilate(ed, ed, Mat.ones(3, 3, CvType.CV_8U))
        val cands = ArrayList<Pair<Double, Array<Point>>>()
        for (m in listOf(th, ed)) {
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(m, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            for (c in contours) {
                if (Imgproc.contourArea(c) < areaMin) continue
                val hullIdx = MatOfInt()
                Imgproc.convexHull(c, hullIdx)
                val pts = c.toArray()
                val hull = MatOfPoint2f(*hullIdx.toArray().map { pts[it] }.toTypedArray())
                val peri = Imgproc.arcLength(hull, true)
                for (eps in doubleArrayOf(0.02, 0.03, 0.05, 0.08)) {
                    val ap = MatOfPoint2f()
                    Imgproc.approxPolyDP(hull, ap, eps * peri, true)
                    val a = ap.toArray()
                    if (a.size == 4) {
                        val xs = DoubleArray(4) { a[it].x }; val ys = DoubleArray(4) { a[it].y }
                        if (isConvex(xs, ys)) cands.add(Pair(polyArea(xs, ys), a))
                        break
                    }
                }
            }
        }
        var best: Array<Point>? = null
        var bestScore = 0.0
        val inside = Mat(); val dil = Mat(); val inv = Mat(); val ring = Mat()
        val kernel = Mat.ones(25, 25, CvType.CV_8U)
        for ((area, q) in cands) {
            inside.create(small.size(), CvType.CV_8UC1); inside.setTo(org.opencv.core.Scalar(0.0))
            Imgproc.fillConvexPoly(inside, MatOfPoint(*q.map { Point(it.x, it.y) }.toTypedArray()), org.opencv.core.Scalar(255.0))
            Imgproc.dilate(inside, dil, kernel)
            Core.bitwise_not(inside, inv)
            Core.bitwise_and(dil, inv, ring)
            if (vs != null) {
                val all = Core.countNonZero(ring)
                Core.bitwise_and(ring, vs, ring)
                if (all > 0 && Core.countNonZero(ring) < 0.3 * all) continue
            }
            val contrast = if (Core.countNonZero(ring) == 0) 20.0
            else Core.mean(small, inside).`val`[0] - Core.mean(small, ring).`val`[0]
            val score = area * max(0.0, contrast)
            if (score > bestScore) { bestScore = score; best = q }
        }
        small.release(); th.release(); ed.release(); vs?.release()
        inside.release(); dil.release(); inv.release(); ring.release()
        val b = best ?: return null
        return order(b.map { Point(it.x / s, it.y / s) }.toTypedArray())
    }

    fun order(q: Array<Point>): Array<Point> {
        val cx = q.sumOf { it.x } / 4; val cy = q.sumOf { it.y } / 4
        val sorted = q.sortedBy { atan2(it.y - cy, it.x - cx) }
        val i = sorted.indices.minByOrNull { sorted[it].x + sorted[it].y } ?: 0
        return Array(4) { sorted[(i + it) % 4] }
    }
}

/** Image size that fits Claude's high-resolution vision limits (long edge 2576, 4784 visual tokens). */
fun sizeForClaude(width: Int, height: Int, maxEdge: Int = 2576, maxTokens: Int = 4784): Pair<Int, Int> {
    fun fits(w: Int, h: Int) = ceil(w / 28.0) * 28 <= maxEdge && ceil(h / 28.0) * 28 <= maxEdge &&
        ceil(w / 28.0) * ceil(h / 28.0) <= maxTokens
    if (fits(width, height)) return Pair(width, height)
    if (height > width) { val (a, b) = sizeForClaude(height, width, maxEdge, maxTokens); return Pair(b, a) }
    val ar = width.toDouble() / height
    var lo = 1; var hi = width
    while (lo + 1 < hi) {
        val mid = (lo + hi) / 2
        if (fits(mid, max(1, (mid / ar).roundToInt()))) lo = mid else hi = mid
    }
    return Pair(lo, max(1, (lo / ar).roundToInt()))
}

fun encodeJpeg(img: Mat, quality: Int): ByteArray {
    val buf = MatOfByte()
    Imgcodecs.imencode(".jpg", img, buf, MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality))
    return buf.toArray()
}
