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
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

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

/**
 * Removes uneven lighting, vignetting and soft shadows; paper becomes ~245.
 * The background is estimated by erasing text (local max at about stroke size),
 * then smoothing, at a working size of at most 900 px.
 */
fun flatten(gray: Mat): Mat {
    val w = gray.cols(); val h = gray.rows()
    val s = min(1.0, 900.0 / max(w, h))
    val small = Mat()
    if (s < 1.0) Imgproc.resize(gray, small, Size(w * s, h * s), 0.0, 0.0, Imgproc.INTER_AREA) else gray.copyTo(small)
    val k = max(5, (min(small.cols(), small.rows()) / 60.0).roundToInt() or 1)
    val bg = Mat()
    Imgproc.dilate(small, bg, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble())))
    val med = Mat()
    Imgproc.medianBlur(bg, med, (3 * k) or 1)
    Imgproc.GaussianBlur(med, bg, Size(0.0, 0.0), k.toDouble())
    val full = Mat()
    if (s < 1.0) Imgproc.resize(bg, full, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR) else bg.copyTo(full)
    val out = Mat()
    Core.divide(gray, full, out, 245.0)
    small.release(); bg.release(); med.release(); full.release()
    return out
}

/** Final touch for a sheet built from lighting-corrected photos: stretch contrast. */
fun enhance(gray: Mat): Mat {
    val out = gray.clone()
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

// ---------------------------------------------------------------------------
// Page orientation (works out which way up the text is)

object Orientation {
    /** Character-sized dark marks only (drops shadows, background texture and big blobs). */
    private fun inkMask(gray: Mat): Mat {
        val s = 1000.0 / max(gray.cols(), gray.rows())
        val g = Mat()
        Imgproc.resize(gray, g, Size(gray.cols() * s, gray.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        val m = Mat()
        Imgproc.adaptiveThreshold(g, m, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 25, 15.0)
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(m, labels, stats, cents, 8, CvType.CV_32S)
        val w = m.cols(); val h = m.rows()
        val keep = BooleanArray(n)
        for (i in 1 until n) {
            val ch = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0]; val cw = stats.get(i, Imgproc.CC_STAT_WIDTH)[0]
            val a = stats.get(i, Imgproc.CC_STAT_AREA)[0]
            keep[i] = ch < 0.035 * h && cw < 0.08 * w && a >= 4 && ch >= 2
        }
        val lab = IntArray(w * h); labels.get(0, 0, lab)
        val out = ByteArray(w * h)
        val bx = (0.04 * w).toInt(); val by = (0.03 * h).toInt() // ignore marks at the very edges
        for (y in by until h - by) for (x in bx until w - bx) {
            if (keep[lab[y * w + x]]) out[y * w + x] = 1
        }
        val res = Mat(h, w, CvType.CV_8UC1); res.put(0, 0, out)
        g.release(); m.release(); labels.release(); stats.release(); cents.release()
        return res
    }

    private fun rowSums(m: Mat): DoubleArray {
        val w = m.cols(); val h = m.rows()
        val a = ByteArray(w * h); m.get(0, 0, a)
        return DoubleArray(h) { y -> var s = 0.0; for (x in 0 until w) s += a[y * w + x]; s }
    }

    /** How strongly the marks form horizontal lines (high when text runs left to right). */
    private fun lineScore(m: Mat): Double {
        val p = rowSums(m)
        val mean = p.average()
        val sd = sqrt(p.sumOf { (it - mean) * (it - mean) } / max(1, p.size))
        return sd / (mean + 1e-6)
    }

    /** Positive when lines start more evenly than they end (normal left-aligned text). */
    private fun leftAlignment(m: Mat): Double {
        val w = m.cols(); val h = m.rows()
        val a = ByteArray(w * h); m.get(0, 0, a)
        val p = rowSums(m)
        val pmax = p.maxOrNull() ?: 0.0
        val lines = ArrayList<Pair<Int, Int>>()
        var start = -1
        for (y in 0..h) {
            val on = y < h && p[y] > 0.02 * pmax
            if (on && start < 0) start = y
            if (!on && start >= 0) { if (y - start >= 3) lines.add(start to y); start = -1 }
        }
        val left = ArrayList<Int>(); val right = ArrayList<Int>()
        for ((y0, y1) in lines) {
            var first = -1; var last = -1
            for (x in 0 until w) {
                var any = false
                for (y in y0 until y1) if (a[y * w + x].toInt() != 0) { any = true; break }
                if (any) { if (first < 0) first = x; last = x }
            }
            if (first >= 0 && last - first > 10) { left.add(first); right.add(last) }
        }
        if (left.size < 4) return 0.0
        val tol = 0.012 * w
        fun clustered(v: List<Int>): Double = v.maxOf { c -> v.count { abs(it - c) < tol } }.toDouble() / v.size
        return clustered(left) - clustered(right)
    }

    private fun rot(m: Mat, k: Int): Mat {
        val o = Mat()
        when (k % 4) {
            1 -> Core.rotate(m, o, Core.ROTATE_90_COUNTERCLOCKWISE)
            2 -> Core.rotate(m, o, Core.ROTATE_180)
            3 -> Core.rotate(m, o, Core.ROTATE_90_CLOCKWISE)
            else -> m.copyTo(o)
        }
        return o
    }

    /** Quarter turns (counter-clockwise) that make the page upright. */
    fun quarterTurns(page: Mat): Int {
        val m = inkMask(page)
        val r1 = rot(m, 1)
        var k = if (lineScore(r1) > lineScore(m)) 1 else 0
        val mk = if (k == 1) r1 else m
        if (leftAlignment(mk) < 0) k += 2
        m.release(); r1.release()
        return k % 4
    }

    /** Returns the page turned upright (a new Mat). */
    fun upright(page: Mat): Mat = rot(page, quarterTurns(page))
}
