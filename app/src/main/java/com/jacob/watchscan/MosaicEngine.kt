package com.jacob.watchscan

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Builds a page scan from a sweep of close-up frames, with no overview photo.
 *
 * The first sharp frame defines the mosaic's coordinates. Each later frame is matched
 * (ORB features) against the previous located frame and the nearest keyframes, giving a
 * homography frame -> mosaic. Frames that are sharp and show enough new area become
 * keyframes. At the end the keyframes are blended (each pixel from the keyframe whose
 * centre is nearest), the paper is found and straightened, and lighting is evened out.
 *
 * All methods must be called from one thread.
 */
class MosaicEngine {
    companion object {
        const val FEAT_SCALE = 0.5
        const val CELL = 16.0          // coverage grid, in mosaic px
        const val THUMB = 8            // keyframe thumbnails are 1/THUMB size
        const val MAX_KEYS = 45
        const val OUT_LONG = 3000      // final mosaic long side
    }

    class Info(
        val ok: Boolean,
        val how: String,
        val matches: Int,
        val inliers: Int,
        val newKey: Boolean,
        val keys: Int,
        val sharp: Double,
        val accepted: Boolean,
        /** Live preview of the mosaic so far (only when it changed). */
        val preview: Mat?,
        /** Current frame outline on the preview, normalised 0..1 (8 floats). */
        val footprint: FloatArray?,
    )

    private class Feat(val pts: Array<Point>, val des: Mat)

    private class Key(
        val jpeg: ByteArray,
        val thumb: Mat,
        val h: H,               // frame -> mosaic
        val pts: Array<Point>,  // features in mosaic coords
        val des: Mat,
        val sharp: Double,
        val cx: Double,
        val cy: Double,
        val qx: DoubleArray,
        val qy: DoubleArray,
    )

    private val orb = ORB.create(1500, 1.2f, 8, 31, 0, 2, 0, 31, 10)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)

    private val keys = ArrayList<Key>()
    private var lastPts: Array<Point>? = null
    private var lastDes: Mat? = null
    private var lastH: H? = null
    private var maxSharp = 0.0
    private var lost = 0
    private val covered = HashSet<Long>()
    var fw = 0; private set
    var fh = 0; private set

    // Preview geometry (mosaic -> preview px)
    private var prevT: H? = null
    private var prevW = 0
    private var prevH = 0

    val keyCount get() = keys.size

    // ------------------------------------------------------------------ features & matching

    private fun feats(frame: Mat): Feat? {
        val w = frame.cols(); val h = frame.rows()
        val sw = (w * FEAT_SCALE).toInt(); val sh = (h * FEAT_SCALE).toInt()
        val small = Mat()
        Imgproc.resize(frame, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val kp = MatOfKeyPoint()
        val des = Mat()
        orb.detectAndCompute(small, Mat(), kp, des)
        small.release()
        val up = H.resize(sw.toDouble() / w, sh.toDouble() / h).inv()
        val pts = kp.toArray().map { Point(up.mapX(it.pt.x, it.pt.y), up.mapY(it.pt.x, it.pt.y)) }.toTypedArray()
        if (des.empty() || pts.size < 30) { des.release(); return null }
        return Feat(pts, des)
    }

    private class Cand(val pts: Array<Point>, val des: Mat)

    private fun candidates(all: Boolean): List<Cand> {
        val out = ArrayList<Cand>()
        val lp = lastPts; val ld = lastDes
        if (!all && lp != null && ld != null) out.add(Cand(lp, ld))
        val lh = lastH
        val chosen: List<Key> = if (all || lh == null) keys else {
            val cx = lh.mapX(fw / 2.0, fh / 2.0); val cy = lh.mapY(fw / 2.0, fh / 2.0)
            val diag = hypot(fw.toDouble(), fh.toDouble())
            val near = keys.sortedBy { hypot(it.cx - cx, it.cy - cy) }
                .filter { hypot(it.cx - cx, it.cy - cy) < 1.2 * diag }.take(4).toMutableList()
            if (keys.isNotEmpty() && !near.contains(keys.last())) near.add(keys.last())
            near
        }
        for (k in chosen) out.add(Cand(k.pts, k.des))
        return out
    }

    /** Frame -> mosaic homography, matching each candidate separately then re-fitting on all. */
    private fun locate(f: Feat, cands: List<Cand>, diag: IntArray): H? {
        var best: H? = null
        var bestInl = 0
        val allSrc = ArrayList<Point>(); val allDst = ArrayList<Point>()
        for (c in cands) {
            if (c.des.empty()) continue
            val knn = ArrayList<MatOfDMatch>()
            matcher.knnMatch(f.des, c.des, knn, 2)
            val src = ArrayList<Point>(); val dst = ArrayList<Point>()
            for (mm in knn) {
                val a = mm.toArray()
                if (a.size == 2 && a[0].distance < 0.8f * a[1].distance) {
                    src.add(f.pts[a[0].queryIdx]); dst.add(c.pts[a[0].trainIdx])
                }
            }
            allSrc.addAll(src); allDst.addAll(dst)
            if (src.size < 20) continue
            val mask = Mat()
            val hm = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()), Calib3d.RANSAC, 5.0, mask)
            if (hm.empty()) continue
            val inl = Core.countNonZero(mask)
            if (inl > bestInl && inl >= 25 && inl >= 0.15 * src.size) { best = H.fromMat(hm); bestInl = inl }
        }
        diag[0] = allSrc.size; diag[1] = bestInl
        val b = best ?: return null
        // Re-fit with every correspondence that agrees with the best estimate.
        val s2 = ArrayList<Point>(); val d2 = ArrayList<Point>()
        for (i in allSrc.indices) {
            val p = allSrc[i]; val q = allDst[i]
            if (hypot(b.mapX(p.x, p.y) - q.x, b.mapY(p.x, p.y) - q.y) < 5.0) { s2.add(p); d2.add(q) }
        }
        diag[1] = max(bestInl, s2.size)
        if (s2.size >= 25) {
            val hm = Calib3d.findHomography(MatOfPoint2f(*s2.toTypedArray()), MatOfPoint2f(*d2.toTypedArray()), 0, 3.0)
            if (!hm.empty()) return H.fromMat(hm)
        }
        return b
    }

    private fun quad(h: H): Pair<DoubleArray, DoubleArray> {
        val xs = doubleArrayOf(0.0, fw.toDouble(), fw.toDouble(), 0.0)
        val ys = doubleArrayOf(0.0, 0.0, fh.toDouble(), fh.toDouble())
        return Pair(DoubleArray(4) { h.mapX(xs[it], ys[it]) }, DoubleArray(4) { h.mapY(xs[it], ys[it]) })
    }

    private fun sane(h: H): Boolean {
        val (qx, qy) = quad(h)
        if (!isConvex(qx, qy)) return false
        val a = polyArea(qx, qy) / (fw.toDouble() * fh)
        return a > 0.4 && a < 2.5
    }

    // ------------------------------------------------------------------ coverage

    private fun inQuad(x: Double, y: Double, qx: DoubleArray, qy: DoubleArray): Boolean {
        var sign = 0
        for (i in 0 until 4) {
            val j = (i + 1) % 4
            val c = (qx[j] - qx[i]) * (y - qy[i]) - (qy[j] - qy[i]) * (x - qx[i])
            val s = if (c > 0) 1 else if (c < 0) -1 else 0
            if (s == 0) continue
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return true
    }

    private fun cells(h: H): List<Long> {
        val (qx, qy) = quad(h)
        val x0 = floor(qx.min() / CELL).toInt(); val x1 = ceil(qx.max() / CELL).toInt()
        val y0 = floor(qy.min() / CELL).toInt(); val y1 = ceil(qy.max() / CELL).toInt()
        val out = ArrayList<Long>()
        for (y in y0 until y1) for (x in x0 until x1) {
            if (inQuad((x + 0.5) * CELL, (y + 0.5) * CELL, qx, qy)) out.add((x.toLong() shl 32) xor (y.toLong() and 0xffffffffL))
        }
        return out
    }

    private fun newFraction(h: H): Double {
        val cs = cells(h)
        if (cs.isEmpty()) return 0.0
        return cs.count { it !in covered }.toDouble() / cs.size
    }

    // ------------------------------------------------------------------ per frame

    fun process(frame: Mat): Info {
        fw = frame.cols(); fh = frame.rows()
        val cr = Rect(max(0, fw / 2 - 256), max(0, fh / 2 - 256), min(512, fw), min(512, fh))
        val sharp = laplaceVar(frame.submat(cr))
        maxSharp = max(maxSharp * 0.97, sharp)
        val f = feats(frame) ?: run {
            lost++
            return Info(false, "few features", 0, 0, false, keys.size, sharp, false, null, null)
        }
        if (keys.isEmpty()) {
            if (sharp < 0.5 * maxSharp) {
                f.des.release()
                return Info(false, "waiting for a sharp view", 0, 0, false, 0, sharp, false, null, null)
            }
            val h = H(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            addKey(frame, h, f, sharp)
            setLast(f, h)
            val prev = renderPreview()
            return Info(true, "start", 0, 0, true, 1, sharp, true, prev, footprint(h))
        }
        val diag = IntArray(2)
        var h = locate(f, candidates(false), diag)
        var how = "track"
        if (h == null || !sane(h)) {
            h = null
            if (lost >= 2 && keys.size > 1 && lost % 2 == 0) {
                h = locate(f, candidates(true), diag)
                how = "relocate"
                if (h != null && !sane(h)) h = null
            }
        }
        if (h == null) {
            lost++
            f.des.release()
            return Info(false, "lost", diag[0], diag[1], false, keys.size, sharp, false, null, null)
        }
        lost = 0
        val accepted = sharp >= 0.6 * maxSharp
        var newKey = false
        var prev: Mat? = null
        if (accepted && keys.size < MAX_KEYS && newFraction(h) > 0.25) {
            addKey(frame, h, f, sharp)
            setLast(f, h)
            newKey = true
            prev = renderPreview()
        } else {
            setLast(f, h)
        }
        return Info(true, how, diag[0], diag[1], newKey, keys.size, sharp, accepted, prev, footprint(h))
    }

    /** Remember the last located frame (features in mosaic coords) for the next match. */
    private fun setLast(f: Feat, h: H) {
        val old = lastDes
        if (old != null && keys.none { it.des === old }) old.release()
        lastPts = f.pts.map { Point(h.mapX(it.x, it.y), h.mapY(it.x, it.y)) }.toTypedArray()
        lastDes = f.des
        lastH = h
    }

    private fun addKey(frame: Mat, h: H, f: Feat, sharp: Double) {
        val mp = f.pts.map { Point(h.mapX(it.x, it.y), h.mapY(it.x, it.y)) }.toTypedArray()
        val thumb = Mat()
        Imgproc.resize(frame, thumb, Size((frame.cols() / THUMB).toDouble(), (frame.rows() / THUMB).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val (qx, qy) = quad(h)
        keys.add(
            Key(
                encodeJpeg(frame, 93), thumb, h, mp, f.des, sharp,
                h.mapX(fw / 2.0, fh / 2.0), h.mapY(fw / 2.0, fh / 2.0), qx, qy
            )
        )
        covered.addAll(cells(h))
    }

    // ------------------------------------------------------------------ preview

    private fun bounds(): DoubleArray {
        var x0 = Double.MAX_VALUE; var y0 = Double.MAX_VALUE; var x1 = -Double.MAX_VALUE; var y1 = -Double.MAX_VALUE
        for (k in keys) for (i in 0 until 4) {
            x0 = min(x0, k.qx[i]); x1 = max(x1, k.qx[i]); y0 = min(y0, k.qy[i]); y1 = max(y1, k.qy[i])
        }
        return doubleArrayOf(x0, y0, x1, y1)
    }

    /** Composite of keyframe thumbnails at true brightness. Returns image + valid mask, and the mosaic -> image map. */
    private fun composeThumbs(maxLong: Int): Triple<Mat, Mat, H> {
        val b = bounds()
        val sc = maxLong / max(b[2] - b[0], b[3] - b[1])
        val w = max(8, ((b[2] - b[0]) * sc).toInt() + 1)
        val hh = max(8, ((b[3] - b[1]) * sc).toInt() + 1)
        val t = H(doubleArrayOf(sc, 0.0, -b[0] * sc, 0.0, sc, -b[1] * sc, 0.0, 0.0, 1.0))
        val img = Mat.zeros(hh, w, CvType.CV_8UC1)
        val valid = Mat.zeros(hh, w, CvType.CV_8UC1)
        val tmp = Mat(); val tmpMask = Mat()
        for (k in keys) {
            val down = H.resize(k.thumb.cols().toDouble() / fw, k.thumb.rows().toDouble() / fh)
            val m = (t * k.h * down.inv()).toMat64()
            Imgproc.warpPerspective(k.thumb, tmp, m, Size(w.toDouble(), hh.toDouble()), Imgproc.INTER_LINEAR)
            val ones = Mat(k.thumb.rows(), k.thumb.cols(), CvType.CV_8UC1, Scalar(255.0))
            Imgproc.warpPerspective(ones, tmpMask, m, Size(w.toDouble(), hh.toDouble()), Imgproc.INTER_NEAREST)
            tmp.copyTo(img, tmpMask)
            Core.bitwise_or(valid, tmpMask, valid)
            ones.release(); m.release()
        }
        tmp.release(); tmpMask.release()
        return Triple(img, valid, t)
    }

    private fun renderPreview(): Mat {
        val (img, valid, t) = composeThumbs(300)
        valid.release()
        prevT = t; prevW = img.cols(); prevH = img.rows()
        return img
    }

    private fun footprint(h: H): FloatArray? {
        val t = prevT ?: return null
        val (qx, qy) = quad(h)
        return FloatArray(8) {
            val i = it / 2
            val x = t.mapX(qx[i], qy[i]); val y = t.mapY(qx[i], qy[i])
            if (it % 2 == 0) (x / prevW).toFloat() else (y / prevH).toFloat()
        }
    }

    // ------------------------------------------------------------------ result

    /** Blends the keyframes, finds and straightens the paper, evens the lighting. */
    fun result(): Mat {
        if (keys.isEmpty()) throw IllegalStateException("Nothing was captured. Hold the watch steady over the page, then sweep.")
        val b = bounds()
        val sc = min(1.0, OUT_LONG / max(b[2] - b[0], b[3] - b[1]))
        val w = ((b[2] - b[0]) * sc).toInt() + 1
        val hh = ((b[3] - b[1]) * sc).toInt() + 1
        val t = H(doubleArrayOf(sc, 0.0, -b[0] * sc, 0.0, sc, -b[1] * sc, 0.0, 0.0, 1.0))
        val out = Mat(hh, w, CvType.CV_8UC1, Scalar(0.0))
        val best = Mat.zeros(hh, w, CvType.CV_8UC1)

        // Weight: distance to the frame edge (prefer frame centres).
        val border = Mat(fh, fw, CvType.CV_8UC1, Scalar(255.0))
        Imgproc.rectangle(border, Point(0.0, 0.0), Point(fw - 1.0, fh - 1.0), Scalar(0.0), 1)
        val dist = Mat()
        Imgproc.distanceTransform(border, dist, Imgproc.DIST_L1, 3)
        val wmap = Mat()
        dist.convertTo(wmap, CvType.CV_8UC1, 250.0 / (min(fw, fh) / 2.0), 1.0)
        border.release(); dist.release()

        val img = Mat(); val wt = Mat(); val sel = Mat()
        for (k in keys) {
            val raw = Imgcodecs.imdecode(MatOfByte(*k.jpeg), Imgcodecs.IMREAD_GRAYSCALE)
            val flat = flatten(raw)
            raw.release()
            val m = (t * k.h).toMat64()
            Imgproc.warpPerspective(flat, img, m, Size(w.toDouble(), hh.toDouble()), Imgproc.INTER_LINEAR)
            val kw = Mat()
            Core.multiply(wmap, Scalar(sqrt(k.sharp / max(1e-6, maxSharp)).coerceIn(0.3, 1.0)), kw)
            Imgproc.warpPerspective(kw, wt, m, Size(w.toDouble(), hh.toDouble()), Imgproc.INTER_LINEAR)
            Core.compare(wt, best, sel, Core.CMP_GT)
            img.copyTo(out, sel)
            wt.copyTo(best, sel)
            flat.release(); kw.release(); m.release()
        }
        img.release(); wt.release(); sel.release(); wmap.release()

        // Find the paper on a small true-brightness composite.
        val (raw, valid, tl) = composeThumbs(800)
        val validPx = Core.countNonZero(valid)
        var page: Array<Point>? = null
        if (validPx > 0) {
            val fill = percentile(raw, valid, 0.05)
            val inv = Mat()
            Core.bitwise_not(valid, inv)
            raw.setTo(Scalar(fill), inv)
            inv.release()
            val q = PageDetector.detect(raw)
            if (q != null) {
                // Reject a quad that is mostly outside what was captured.
                val qm = Mat.zeros(raw.rows(), raw.cols(), CvType.CV_8UC1)
                Imgproc.fillConvexPoly(qm, MatOfPoint(*q.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
                val qa = Core.countNonZero(qm)
                Core.bitwise_and(qm, valid, qm)
                if (qa > 0 && Core.countNonZero(qm) >= 0.85 * qa) {
                    // thumbnail-composite px -> output px
                    val toOut = t * tl.inv()
                    page = q.map { Point(toOut.mapX(it.x, it.y), toOut.mapY(it.x, it.y)) }.toTypedArray()
                }
                qm.release()
            }
        }
        raw.release(); valid.release()

        val straight: Mat
        if (page != null) {
            val (pw, ph) = pageSize(page, max(w, hh))
            val dst = MatOfPoint2f(Point(0.0, 0.0), Point(pw.toDouble(), 0.0), Point(pw.toDouble(), ph.toDouble()), Point(0.0, ph.toDouble()))
            val hq = Imgproc.getPerspectiveTransform(MatOfPoint2f(*page), dst)
            straight = Mat()
            Imgproc.warpPerspective(out, straight, hq, Size(pw.toDouble(), ph.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        } else {
            // No clear paper outline: keep everything captured, gaps white.
            val gaps = Mat()
            Core.compare(best, Scalar(0.0), gaps, Core.CMP_EQ)
            out.setTo(Scalar(255.0), gaps)
            gaps.release()
            straight = out.clone()
        }
        out.release(); best.release()
        val e = enhance(straight)
        straight.release()
        return e
    }

    private fun percentile(img: Mat, mask: Mat, p: Double): Double {
        val n = (img.total()).toInt()
        val a = ByteArray(n); val m = ByteArray(n)
        img.get(0, 0, a); mask.get(0, 0, m)
        val hist = IntArray(256); var cnt = 0
        for (i in 0 until n) if (m[i].toInt() != 0) { hist[a[i].toInt() and 0xff]++; cnt++ }
        var acc = 0
        for (v in 0 until 256) { acc += hist[v]; if (acc >= cnt * p) return v.toDouble() }
        return 0.0
    }

    /** Up to [n] sharp keyframes spread over the page, re-encoded for Claude. */
    fun closeUps(n: Int): List<ByteArray> {
        val diag = hypot(fw.toDouble(), fh.toDouble())
        val chosen = ArrayList<Key>()
        for (k in keys.sortedByDescending { it.sharp }) {
            if (chosen.size >= n) break
            if (chosen.all { hypot(it.cx - k.cx, it.cy - k.cy) > 0.5 * diag }) chosen.add(k)
        }
        return chosen.sortedWith(compareBy({ it.cy }, { it.cx })).map { k ->
            val raw = Imgcodecs.imdecode(MatOfByte(*k.jpeg), Imgcodecs.IMREAD_GRAYSCALE)
            val flat = flatten(raw)
            val (tw, th) = sizeForClaude(flat.cols(), flat.rows(), 1568, 2400)
            val small = Mat()
            Imgproc.resize(flat, small, Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val j = encodeJpeg(small, 85)
            raw.release(); flat.release(); small.release()
            j
        }
    }

    fun release() {
        for (k in keys) { k.thumb.release(); k.des.release() }
        keys.clear()
        lastDes?.release(); lastDes = null
    }
}
