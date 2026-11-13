package com.jacob.watchscan

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Builds a high-resolution scan of a page from many close-up camera frames.
 *
 * 1. start(): an overview frame of the whole page is flattened into a low-res
 *    "reference" (REF_LONG px) and an empty high-res canvas (K x reference).
 * 2. process(): each close-up frame is located on the reference (ORB features +
 *    homography), refined with ECC against the canvas, and its sharp pixels are
 *    copied into the grid cells it fully covers, if they are sharper than what
 *    the cell already holds and agree with it (guards against misplacement).
 * 3. result(): the canvas with lighting evened out and contrast stretched.
 *
 * All methods must be called from one thread.
 */
class ScanEngine {
    companion object {
        const val REF_LONG = 1100
        const val K = 3
        const val CELL = 330
        const val GOOD_F = 1.35       // canvas px per frame px for a cell to count as "good"
        const val ZMIN = 0.6          // min agreement between a new patch and the canvas
        const val FINE_MAXW = 400.0   // template width for the ECC refinement

        const val EMPTY = 0
        const val LOW = 1
        const val GOOD = 2
        const val BLANK = 3
    }

    class Info(
        val ok: Boolean,
        val accepted: Boolean,
        val f: Double,
        val updated: Int,
        val good: Int,
        val total: Int,
        val done: Boolean,
        val status: IntArray,
        val cols: Int,
        val rows: Int,
        /** Frame corners on the page, normalised 0..1 (8 values) or null. */
        val footprint: FloatArray?,
        /** Page corners in the frame, normalised 0..1 (8 values) or null. */
        val pageInFrame: FloatArray?,
    )

    private val orbRef = ORB.create(4000, 1.2f, 8, 31, 0, 2, 0, 31, 8)
    private val orb = ORB.create(1500, 1.2f, 8, 31, 0, 2, 0, 31, 8)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)

    private lateinit var ref: Mat
    private lateinit var refDes: Mat
    private lateinit var refPts: Array<Point>
    lateinit var canvas: Mat
        private set
    private lateinit var canvasSmall: Mat
    var rw = 0; private set
    var rh = 0; private set
    var cw = 0; private set
    var ch = 0; private set
    var cols = 0; private set
    var rows = 0; private set
    private lateinit var score: DoubleArray
    private lateinit var status: IntArray
    private var s = 0.5
    private var lost = 0
    private var maxSharp = 0.0
    var started = false; private set

    fun start(frame: Mat, quad: Array<Point>) {
        val (w, h) = pageSize(quad)
        rw = w; rh = h
        val dst = MatOfPoint2f(Point(0.0, 0.0), Point(w.toDouble(), 0.0), Point(w.toDouble(), h.toDouble()), Point(0.0, h.toDouble()))
        val src = MatOfPoint2f(*quad)
        val hqMat = Imgproc.getPerspectiveTransform(src, dst)
        val hq = H.fromMat(hqMat)
        ref = Mat()
        Imgproc.warpPerspective(frame, ref, hqMat, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_LINEAR)

        val kp = MatOfKeyPoint()
        refDes = Mat()
        orbRef.detectAndCompute(ref, Mat(), kp, refDes)
        refPts = kp.toArray().map { it.pt }.toTypedArray()

        cw = rw * K; ch = rh * K
        val flatRef = flatten(ref)
        canvas = Mat()
        Imgproc.resize(flatRef, canvas, Size(cw.toDouble(), ch.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
        canvasSmall = Mat()
        Imgproc.resize(canvas, canvasSmall, Size(rw.toDouble(), rh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        cols = max(2, (cw.toDouble() / CELL).roundToInt())
        rows = max(2, (ch.toDouble() / CELL).roundToInt())
        val n = cols * rows
        score = DoubleArray(n)
        status = IntArray(n)
        val ink = Mat()
        Imgproc.GaussianBlur(flatRef, ink, Size(3.0, 3.0), 0.0)
        val tmp = Mat()
        for (i in 0 until n) {
            val r = cellRect(i)
            score[i] = laplaceVar(canvas.submat(r))
            val rr = Rect(r.x / K, r.y / K, max(1, r.width / K), max(1, r.height / K))
            val rc = clip(rr, rw, rh)
            Imgproc.threshold(ink.submat(rc), tmp, 169.0, 255.0, Imgproc.THRESH_BINARY_INV)
            if (Core.countNonZero(tmp) < 0.005 * rc.area()) status[i] = BLANK
        }
        tmp.release(); ink.release(); flatRef.release()

        val cx = quad.sumOf { it.x } / 4
        val cy = quad.sumOf { it.y } / 4
        s = min(1.0, hq.localScale(cx, cy))
        lost = 0
        maxSharp = 0.0
        started = true
    }

    private fun clip(r: Rect, w: Int, h: Int): Rect {
        val x = r.x.coerceIn(0, w - 1)
        val y = r.y.coerceIn(0, h - 1)
        return Rect(x, y, min(r.width, w - x), min(r.height, h - y))
    }

    fun cellRect(i: Int): Rect {
        val c = i % cols
        val r = i / cols
        val x0 = (c * cw.toDouble() / cols).roundToInt()
        val x1 = ((c + 1) * cw.toDouble() / cols).roundToInt()
        val y0 = (r * ch.toDouble() / rows).roundToInt()
        val y1 = ((r + 1) * ch.toDouble() / rows).roundToInt()
        return Rect(x0, y0, x1 - x0, y1 - y0)
    }

    /** Locates a full-resolution frame on the reference. Returns frame -> ref. */
    private fun register(frame: Mat, scale: Double): H? {
        val w = frame.cols()
        val h = frame.rows()
        val sw = max(32, (w * scale).toInt())
        val sh = max(32, (h * scale).toInt())
        val small = Mat()
        Imgproc.resize(frame, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val kp = MatOfKeyPoint()
        val des = Mat()
        orb.detectAndCompute(small, Mat(), kp, des)
        small.release()
        val kps = kp.toArray()
        if (des.empty() || kps.size < 30) return null
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(des, refDes, knn, 2)
        val src = ArrayList<Point>()
        val dst = ArrayList<Point>()
        for (mm in knn) {
            val a = mm.toArray()
            if (a.size == 2 && a[0].distance < 0.8f * a[1].distance) {
                src.add(kps[a[0].queryIdx].pt)
                dst.add(refPts[a[0].trainIdx])
            }
        }
        des.release()
        if (src.size < 20) return null
        val mask = Mat()
        val hm = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()), Calib3d.RANSAC, 3.0, mask)
        if (hm.empty()) return null
        val inl = Core.countNonZero(mask)
        if (inl < 20 || inl < 0.2 * src.size) return null
        val hs = H.fromMat(hm)
        // Sanity: frame outline must map to a plausible convex quad.
        val xs = doubleArrayOf(0.0, sw.toDouble(), sw.toDouble(), 0.0)
        val ys = doubleArrayOf(0.0, 0.0, sh.toDouble(), sh.toDouble())
        val px = DoubleArray(4) { hs.mapX(xs[it], ys[it]) }
        val py = DoubleArray(4) { hs.mapY(xs[it], ys[it]) }
        if (!isConvex(px, py)) return null
        val area = polyArea(px, py)
        if (area < 0.005 * rw * rh || area > 4.0 * rw * rh) return null
        return hs * H.resize(sw.toDouble() / w, sh.toDouble() / h)
    }

    /** ECC refinement of frame -> canvas against the canvas at reference resolution. */
    private fun refine(frame: Mat, hc: H): H {
        val w = frame.cols()
        val h = frame.rows()
        val f = hc.localScale(w / 2.0, h / 2.0)
        val t = minOf(1.0, f / K, FINE_MAXW / w)
        val tw = max(16, (w * t).toInt())
        val th = max(16, (h * t).toInt())
        val tmpl = Mat()
        Imgproc.resize(frame, tmpl, Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val d = H.resize(1.0 / K, 1.0 / K)
        val down = H.resize(tw.toDouble() / w, th.toDouble() / h)
        val w0 = d * hc * down.inv()
        try {
            val warp = w0.toMat32()
            Video.findTransformECC(
                tmpl, canvasSmall, warp, Video.MOTION_HOMOGRAPHY,
                TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 20, 1e-4), Mat(), 3
            )
            val wr = H.fromMat(warp)
            var maxD = 0.0
            for ((x, y) in listOf(0.0 to 0.0, tw.toDouble() to 0.0, tw.toDouble() to th.toDouble(), 0.0 to th.toDouble())) {
                maxD = max(maxD, hypot(wr.mapX(x, y) - w0.mapX(x, y), wr.mapY(x, y) - w0.mapY(x, y)))
            }
            if (maxD < 12.0) return d.inv() * wr * down
        } catch (e: Exception) {
            // ECC did not converge; keep the feature-based estimate.
        } finally {
            tmpl.release()
        }
        return hc
    }

    fun process(frame: Mat): Info {
        val w = frame.cols()
        val h = frame.rows()
        val cr = Rect(max(0, w / 2 - 256), max(0, h / 2 - 256), min(512, w), min(512, h))
        val sharp = laplaceVar(frame.submat(cr))
        maxSharp = max(maxSharp * 0.97, sharp)

        val tries = if (lost < 2) listOf(s) else listOf(s, s * 0.6, s * 1.6, s * 0.35)
        var hr: H? = null
        for (sc in tries) {
            hr = register(frame, sc.coerceIn(0.05, 1.0))
            if (hr != null) break
        }
        if (hr == null) {
            lost++
            return info(false, false, 0.0, 0, null, null)
        }
        lost = 0
        s = hr.localScale(w / 2.0, h / 2.0).coerceIn(0.05, 1.0)
        val hc = refine(frame, H.resize(K.toDouble(), K.toDouble()) * hr)
        val f = hc.localScale(w / 2.0, h / 2.0)
        val accepted = sharp >= 0.55 * maxSharp
        var updated = 0
        if (accepted) {
            val flat = flatten(frame)
            val hinv = hc.inv()
            val mx = 0.04 * w
            val my = 0.04 * h
            val patch = Mat()
            for (i in 0 until cols * rows) {
                val r = cellRect(i)
                val cx = doubleArrayOf(r.x.toDouble(), (r.x + r.width).toDouble(), (r.x + r.width).toDouble(), r.x.toDouble())
                val cy = doubleArrayOf(r.y.toDouble(), r.y.toDouble(), (r.y + r.height).toDouble(), (r.y + r.height).toDouble())
                var inside = true
                for (k in 0 until 4) {
                    val fx = hinv.mapX(cx[k], cy[k])
                    val fy = hinv.mapY(cx[k], cy[k])
                    if (fx < mx || fy < my || fx > w - mx || fy > h - my) { inside = false; break }
                }
                if (!inside) continue
                val m = (H.translate(-r.x.toDouble(), -r.y.toDouble()) * hc).toMat64()
                Imgproc.warpPerspective(flat, patch, m, Size(r.width.toDouble(), r.height.toDouble()), Imgproc.INTER_LINEAR)
                m.release()
                val sp = laplaceVar(patch)
                if (sp <= score[i] * 1.05) continue
                val cur = canvas.submat(r)
                if (!agrees(patch, cur)) continue
                patch.copyTo(cur)
                score[i] = sp
                if (status[i] != BLANK || f <= GOOD_F) {
                    status[i] = if (f <= GOOD_F) GOOD else max(status[i], LOW)
                }
                updated++
            }
            patch.release()
            flat.release()
            if (updated > 0) {
                Imgproc.resize(canvas, canvasSmall, Size(rw.toDouble(), rh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            }
        }
        // Overlay geometry
        val fx = doubleArrayOf(0.0, w.toDouble(), w.toDouble(), 0.0)
        val fy = doubleArrayOf(0.0, 0.0, h.toDouble(), h.toDouble())
        val foot = FloatArray(8)
        for (k in 0 until 4) {
            foot[2 * k] = (hc.mapX(fx[k], fy[k]) / cw).toFloat()
            foot[2 * k + 1] = (hc.mapY(fx[k], fy[k]) / ch).toFloat()
        }
        val hinv = hc.inv()
        val px = doubleArrayOf(0.0, cw.toDouble(), cw.toDouble(), 0.0)
        val py = doubleArrayOf(0.0, 0.0, ch.toDouble(), ch.toDouble())
        val page = FloatArray(8)
        for (k in 0 until 4) {
            page[2 * k] = (hinv.mapX(px[k], py[k]) / w).toFloat()
            page[2 * k + 1] = (hinv.mapY(px[k], py[k]) / h).toFloat()
        }
        return info(true, accepted, f, updated, foot, page)
    }

    private fun info(ok: Boolean, acc: Boolean, f: Double, upd: Int, foot: FloatArray?, page: FloatArray?): Info {
        val good = status.count { it == GOOD || it == BLANK }
        return Info(ok, acc, f, upd, good, status.size, good == status.size, status.copyOf(), cols, rows, foot, page)
    }

    /** Does the new patch show the same content as what the canvas already has there? */
    private fun agrees(patch: Mat, cur: Mat): Boolean {
        val sz = Size(max(8, patch.cols() / K).toDouble(), max(8, patch.rows() / K).toDouble())
        val a = Mat(); val b = Mat()
        Imgproc.resize(patch, a, sz, 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(cur, b, sz, 0.0, 0.0, Imgproc.INTER_AREA)
        a.convertTo(a, CvType.CV_32F); b.convertTo(b, CvType.CV_32F)
        Imgproc.GaussianBlur(a, a, Size(0.0, 0.0), 1.0)
        Imgproc.GaussianBlur(b, b, Size(0.0, 0.0), 1.0)
        val ma = MatOfDouble(); val sa = MatOfDouble(); val mb = MatOfDouble(); val sb = MatOfDouble()
        Core.meanStdDev(a, ma, sa); Core.meanStdDev(b, mb, sb)
        val sda = sa.toArray()[0]; val sdb = sb.toArray()[0]
        if (sda <= 4.0 || sdb <= 4.0) { a.release(); b.release(); return true }
        Core.subtract(a, org.opencv.core.Scalar(ma.toArray()[0]), a)
        Core.subtract(b, org.opencv.core.Scalar(mb.toArray()[0]), b)
        Core.multiply(a, b, a)
        val z = Core.mean(a).`val`[0] / (sda * sdb)
        a.release(); b.release()
        return z >= ZMIN
    }

    /** Final image: grayscale, even lighting, stretched contrast. */
    fun result(): Mat = enhance(canvas)

    fun release() {
        if (!started) return
        ref.release(); refDes.release(); canvas.release(); canvasSmall.release()
        started = false
    }
}

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

/** Reference size (w, h) for a detected page, snapping to A4 / Letter / Legal shapes. */
fun pageSize(q: Array<Point>): Pair<Int, Int> {
    fun d(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
    val w = (d(q[0], q[1]) + d(q[3], q[2])) / 2
    val h = (d(q[0], q[3]) + d(q[1], q[2])) / 2
    var ar = h / max(1.0, w)
    for (std in doubleArrayOf(1.4142, 1.2941, 1.6471)) {
        for (a in doubleArrayOf(std, 1 / std)) if (abs(ar - a) / a < 0.07) ar = a
    }
    return if (ar >= 1) Pair((ScanEngine.REF_LONG / ar).roundToInt(), ScanEngine.REF_LONG)
    else Pair(ScanEngine.REF_LONG, (ScanEngine.REF_LONG * ar).roundToInt())
}

object PageDetector {
    /** Returns page corners (TL, TR, BR, BL) in image coordinates, or null. */
    fun detect(gray: Mat): Array<Point>? {
        val w = gray.cols(); val h = gray.rows()
        val s = 640.0 / max(w, h)
        val small = Mat()
        Imgproc.resize(gray, small, Size((w * s).toInt().toDouble(), (h * s).toInt().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(5.0, 5.0), 0.0)
        val areaMin = 0.15 * small.rows() * small.cols()
        val th = Mat()
        Imgproc.threshold(small, th, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.morphologyEx(th, th, Imgproc.MORPH_CLOSE, Mat.ones(9, 9, CvType.CV_8U))
        val ed = Mat()
        Imgproc.Canny(small, ed, 30.0, 90.0)
        Imgproc.dilate(ed, ed, Mat.ones(3, 3, CvType.CV_8U))
        var best: Array<Point>? = null
        var bestArea = 0.0
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
                        if (isConvex(xs, ys)) {
                            val area = polyArea(xs, ys)
                            if (area > bestArea) { bestArea = area; best = a }
                        }
                        break
                    }
                }
            }
        }
        small.release(); th.release(); ed.release()
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
