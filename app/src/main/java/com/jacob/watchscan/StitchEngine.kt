package com.jacob.watchscan

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Joins snapshots of one page into a single sheet, on the device.
 *
 * - Each new snapshot is matched (ORB) against recent snapshots and ones placed nearby.
 *   A link is kept only if it passes strict checks: enough consistent matches, spread
 *   over the photo, and a plausible geometry for two handheld shots of a flat page.
 * - All links are solved together (least squares, rotation+scale+shift per photo), so
 *   one bad link can't throw a photo far off; links that disagree with the solution are
 *   dropped and the solve repeated. Photos that can't be joined are left out rather than
 *   pasted in the wrong place.
 * - Each photo then gets a full perspective fit to its placed neighbours.
 * - The result keeps, for every spot, the photo whose centre is nearest; then the paper's
 *   edges are found, the page is straightened and the lighting evened out.
 *
 * All methods must be called from one thread.
 */
class StitchEngine {
    companion object {
        const val STORE_LONG = 2048
        const val FEAT_LONG = 960.0    // features are found on the photo resized to this long side (enlarged if smaller)
        const val THUMB_LONG = 400.0
        const val OUT_LONG = 3000
        const val FUSE_LONG = 2400     // long side of the fused sheet (Claude reads up to 2576)
        const val FUSE_SCALE = 2.0     // fused sheet is rendered at up to 2x the camera's resolution
    }

    private class Img(
        val jpeg: ByteArray, val w: Int, val h: Int,
        val pts: Array<Point>, val des: Mat, val thumb: Mat,
        val pos: Point?, val unc: Double,       // tracked camera position (stored-photo px)
        val wthumb: Mat,                        // how much this photo can contribute, per spot (thumb size)
    )

    /** Live coverage map: RGB image plus how much of the page is well covered / missing. */
    class Coverage(val rgb: Mat, val pageFound: Boolean, val goodPct: Int, val weakPct: Int, val missingPct: Int)

    private class Edge(val j: Int, val i: Int, val pj: Array<Point>, val pi: Array<Point>)

    private val orb = ORB.create(2500, 1.2f, 8, 31, 0, 2, 0, 31, 10)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
    private val imgs = ArrayList<Img>()
    private val edges = ArrayList<Edge>()
    private var placedH = HashMap<Int, H>()   // photo -> sheet
    private var byMotion = HashSet<Int>()      // placed from tracked motion only
    var joined = 0; private set

    val count get() = imgs.size
    val placed get() = placedH.size

    // ------------------------------------------------------------------ adding photos

    /** Adds a snapshot (full-res grayscale). Returns the live coverage map. */
    fun add(frame: Mat, px: Double = Double.NaN, py: Double = Double.NaN, unc: Double = 0.0): Coverage? {
        val s = min(1.0, STORE_LONG.toDouble() / max(frame.cols(), frame.rows()))
        val full = Mat()
        if (s < 1.0) Imgproc.resize(frame, full, Size(frame.cols() * s, frame.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        else frame.copyTo(full)
        val w = full.cols(); val h = full.rows()
        // Small camera frames are enlarged before finding features: on a 480x640 frame,
        // text is too fine for the detector at its native size.
        val fs = min(1.5, FEAT_LONG / max(w, h))
        val sw = (w * fs).toInt(); val sh = (h * fs).toInt()
        val small = Mat()
        val evened = flatten(full) // shadows removed first, so shadow edges don't become "features"
        Imgproc.resize(evened, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, if (fs < 1.0) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC)
        evened.release()
        val kp = MatOfKeyPoint(); val des = Mat()
        orb.detectAndCompute(small, Mat(), kp, des)
        small.release()
        val up = H.resize(sw.toDouble() / w, sh.toDouble() / h).inv()
        val pts = kp.toArray().map { Point(up.mapX(it.pt.x, it.pt.y), up.mapY(it.pt.x, it.pt.y)) }.toTypedArray()
        val thumb = Mat()
        val ts = min(1.0, THUMB_LONG / max(w, h))
        Imgproc.resize(full, thumb, Size(max(8.0, w * ts), max(8.0, h * ts)), 0.0, 0.0, Imgproc.INTER_AREA)
        val i = imgs.size
        val pos = if (px.isNaN()) null else Point(px * s, py * s)
        val wthumb = centrality(thumb.cols(), thumb.rows())
        litWeight(thumb).also { Core.multiply(wthumb, it, wthumb); it.release() }
        imgs.add(Img(encodeJpeg(full, 92), w, h, pts, des, thumb, pos, unc * s, wthumb))
        full.release()

        // Candidates: the last few photos, plus any placed near where the previous one is.
        val cands = sortedSetOf<Int>()
        for (j in max(0, i - 4) until i) cands.add(j)
        val prev = (i - 1 downTo 0).firstOrNull { placedH.containsKey(it) }
        if (prev != null) {
            val ph = placedH[prev]!!
            val cx = ph.mapX(w / 2.0, h / 2.0); val cy = ph.mapY(w / 2.0, h / 2.0)
            val diag = hypot(w.toDouble(), h.toDouble())
            placedH.entries
                .map { (k, hk) -> k to hypot(hk.mapX(imgs[k].w / 2.0, imgs[k].h / 2.0) - cx, hk.mapY(imgs[k].w / 2.0, imgs[k].h / 2.0) - cy) }
                .filter { it.second < 1.1 * diag }.sortedBy { it.second }.take(4).forEach { cands.add(it.first) }
        }
        cands.remove(i)
        for (j in cands) link(i, j)?.let { edges.add(it) }
        solve()
        return coverage(i)
    }

    /** Strictly verified correspondences between photos i (new) and j, or null. */
    private fun link(i: Int, j: Int): Edge? {
        val a = imgs[i]; val b = imgs[j]
        if (a.des.empty() || b.des.empty() || a.pts.size < 30 || b.pts.size < 30) return null
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(a.des, b.des, knn, 2)
        val pa = ArrayList<Point>(); val pb = ArrayList<Point>()
        for (mm in knn) {
            val m = mm.toArray()
            if (m.size == 2 && m[0].distance < 0.8f * m[1].distance) { pa.add(a.pts[m[0].queryIdx]); pb.add(b.pts[m[0].trainIdx]) }
        }
        // Use the tracked motion: keep only matches near where the camera movement says they should land.
        var offX = 0.0; var offY = 0.0; var tol = -1.0
        val ap = a.pos; val bp = b.pos
        if (ap != null && bp != null) {
            offX = ap.x - bp.x; offY = ap.y - bp.y
            tol = min(0.15, 0.07 + 0.015 * (kotlin.math.abs(i - j) - 1)) * hypot(a.w.toDouble(), a.h.toDouble()) + kotlin.math.abs(a.unc - b.unc)
            val fa = ArrayList<Point>(); val fb = ArrayList<Point>()
            for (k in pa.indices) {
                if (hypot(pb[k].x - (pa[k].x + offX), pb[k].y - (pa[k].y + offY)) < tol) { fa.add(pa[k]); fb.add(pb[k]) }
            }
            pa.clear(); pa.addAll(fa); pb.clear(); pb.addAll(fb)
        }
        if (pa.size < 25) return null
        val mask = Mat()
        val hm = Calib3d.findHomography(MatOfPoint2f(*pa.toTypedArray()), MatOfPoint2f(*pb.toTypedArray()), Calib3d.RANSAC, 5.0, mask, 3000, 0.995)
        if (hm.empty()) return null
        val n = Core.countNonZero(mask)
        if (n < 25 || n < 0.2 * pa.size) return null
        val hl = H.fromMat(hm)
        if (!plausible(hl, a.w, a.h)) return null
        if (tol > 0) {
            val cx = hl.mapX(a.w / 2.0, a.h / 2.0); val cy = hl.mapY(a.w / 2.0, a.h / 2.0)
            if (hypot(cx - (a.w / 2.0 + offX), cy - (a.h / 2.0 + offY)) > tol) return null
        }
        val m = ByteArray(pa.size); mask.get(0, 0, m)
        val ia = ArrayList<Point>(); val ib = ArrayList<Point>()
        for (k in pa.indices) if (m[k].toInt() != 0) { ia.add(pa[k]); ib.add(pb[k]) }
        // Matches must be spread over the photo, not clumped in one spot.
        val sx = std(ia.map { it.x }); val sy = std(ia.map { it.y })
        if (min(sx, sy) < 0.06 * min(a.w, a.h)) return null
        return Edge(j, i, ib.toTypedArray(), ia.toTypedArray())
    }

    private fun std(v: List<Double>): Double {
        val mean = v.average()
        return sqrt(v.sumOf { (it - mean) * (it - mean) } / max(1, v.size))
    }

    /** Two handheld shots of a flat page: scale change and tilt within generous limits. */
    private fun plausible(h: H, w: Int, hh: Int): Boolean {
        val xs = doubleArrayOf(0.0, w.toDouble(), w.toDouble(), 0.0)
        val ys = doubleArrayOf(0.0, 0.0, hh.toDouble(), hh.toDouble())
        val qx = DoubleArray(4) { h.mapX(xs[it], ys[it]) }; val qy = DoubleArray(4) { h.mapY(xs[it], ys[it]) }
        if (!isConvex(qx, qy)) return false
        val r = polyArea(qx, qy) / (w.toDouble() * hh)
        if (r < 0.25 || r > 4.0) return false
        // Opposite sides may differ by up to 2x: allows holding the watch at a fair tilt.
        val side = DoubleArray(4) { hypot(qx[(it + 1) % 4] - qx[it], qy[(it + 1) % 4] - qy[it]) }
        return max(side[0], side[2]) / min(side[0], side[2]) < 2.0 && max(side[1], side[3]) / min(side[1], side[3]) < 2.0
    }

    // ------------------------------------------------------------------ global placement

    private fun solve() {
        val live = ArrayList(edges)
        var pose: Map<Int, DoubleArray> = emptyMap()
        for (iter in 0 until 3) {
            // Largest group of linked photos; its first photo is the anchor.
            val adj = HashMap<Int, MutableSet<Int>>()
            for (e in live) { adj.getOrPut(e.i) { mutableSetOf() }.add(e.j); adj.getOrPut(e.j) { mutableSetOf() }.add(e.i) }
            var comp: Set<Int> = setOf(0)
            val seen = HashSet<Int>()
            for (st in imgs.indices) {
                if (st in seen) continue
                val c = HashSet<Int>(); val stack = ArrayDeque<Int>(); c.add(st); stack.add(st)
                while (stack.isNotEmpty()) { val u = stack.removeLast(); for (v in adj[u] ?: emptySet<Int>()) if (c.add(v)) stack.add(v) }
                seen.addAll(c)
                if (c.size > comp.size) comp = c
            }
            val ids = comp.sorted()
            val idx = HashMap<Int, Int>(); ids.forEachIndexed { k, id -> idx[id] = k }
            val n = 4 * ids.size
            // Normal equations for similarity (a, b, tx, ty) per photo.
            val ata = DoubleArray(n * n); val atb = DoubleArray(n)
            fun addRow(cols: IntArray, vals: DoubleArray, rhs: Double) {
                for (p in cols.indices) {
                    atb[cols[p]] += vals[p] * rhs
                    for (q in cols.indices) ata[cols[p] * n + cols[q]] += vals[p] * vals[q]
                }
            }
            for (e in live) {
                val kj = idx[e.j] ?: continue; val ki = idx[e.i] ?: continue
                val cnt = e.pj.size
                val step = max(1, cnt / 60)
                var k = 0
                while (k < cnt) {
                    val q = e.pj[k]; val p = e.pi[k]
                    val cj = intArrayOf(4 * kj, 4 * kj + 1, 4 * kj + 2, 4 * kj + 3)
                    val ci = intArrayOf(4 * ki, 4 * ki + 1, 4 * ki + 2, 4 * ki + 3)
                    addRow(cj + ci, doubleArrayOf(q.x, -q.y, 1.0, 0.0, -p.x, p.y, -1.0, 0.0), 0.0)
                    addRow(cj + ci, doubleArrayOf(q.y, q.x, 0.0, 1.0, -p.y, -p.x, 0.0, -1.0), 0.0)
                    k += step
                }
            }
            val anchor = idx[ids[0]]!!
            val prior = doubleArrayOf(1.0, 0.0, 0.0, 0.0)
            for (d in 0 until 4) addRow(intArrayOf(4 * anchor + d), doubleArrayOf(1e4), prior[d] * 1e4)
            for (d in 0 until n) ata[d * n + d] += 1e-6
            val A = Mat(n, n, CvType.CV_64F); A.put(0, 0, *ata)
            val B = Mat(n, 1, CvType.CV_64F); B.put(0, 0, *atb)
            val X = Mat()
            Core.solve(A, B, X, Core.DECOMP_SVD)
            val x = DoubleArray(n); X.get(0, 0, x)
            A.release(); B.release(); X.release()
            pose = ids.associateWith { id -> val k = idx[id]!!; doubleArrayOf(x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]) }
            // Drop links that disagree with the joint solution, then solve again.
            val bad = live.filter { e ->
                val pj = pose[e.j]; val pi = pose[e.i]
                if (pj == null || pi == null) false else {
                    val r = e.pj.indices.map { k ->
                        val q = e.pj[k]; val p = e.pi[k]
                        hypot(
                            (pj[0] * q.x - pj[1] * q.y + pj[2]) - (pi[0] * p.x - pi[1] * p.y + pi[2]),
                            (pj[1] * q.x + pj[0] * q.y + pj[3]) - (pi[1] * p.x + pi[0] * p.y + pi[3])
                        )
                    }.sorted()
                    r[r.size / 2] > 30.0
                }
            }
            if (bad.isEmpty()) break
            live.removeAll(bad.toSet())
        }
        val hs = HashMap<Int, H>()
        for ((id, p) in pose) hs[id] = H(doubleArrayOf(p[0], -p[1], p[2], p[1], p[0], p[3], 0.0, 0.0, 1.0))
        refine(hs, live)
        joined = hs.size
        // Photos with nothing to match (e.g. blank paper) are placed from tracked motion next to a joined neighbour.
        val motion = HashSet<Int>()
        val joinedIds = hs.keys.toList()
        for (i in imgs.indices) {
            if (hs.containsKey(i)) continue
            val pi = imgs[i].pos ?: continue
            val j = joinedIds.filter { kotlin.math.abs(it - i) <= 3 && imgs[it].pos != null }.minByOrNull { kotlin.math.abs(it - i) } ?: continue
            val pj = imgs[j].pos!!
            hs[i] = hs[j]!! * H(doubleArrayOf(1.0, 0.0, pi.x - pj.x, 0.0, 1.0, pi.y - pj.y, 0.0, 0.0, 1.0))
            motion.add(i)
        }
        placedH = hs
        byMotion = motion

    }

    /** Full perspective fit of each photo to where its neighbours were placed. */
    private fun refine(hs: HashMap<Int, H>, live: List<Edge>) {
        val order = hs.keys.sortedByDescending { id -> live.count { it.i == id || it.j == id } }
        repeat(2) {
            for (id in order) {
                val src = ArrayList<Point>(); val dst = ArrayList<Point>()
                for (e in live) {
                    if (e.i == id) { val hj = hs[e.j] ?: continue; for (k in e.pi.indices) { src.add(e.pi[k]); dst.add(Point(hj.mapX(e.pj[k].x, e.pj[k].y), hj.mapY(e.pj[k].x, e.pj[k].y))) } }
                    else if (e.j == id) { val hi = hs[e.i] ?: continue; for (k in e.pj.indices) { src.add(e.pj[k]); dst.add(Point(hi.mapX(e.pi[k].x, e.pi[k].y), hi.mapY(e.pi[k].x, e.pi[k].y))) } }
                }
                if (src.size < 30) continue
                val hm = Calib3d.findHomography(MatOfPoint2f(*src.toTypedArray()), MatOfPoint2f(*dst.toTypedArray()), Calib3d.RANSAC, 8.0)
                if (hm.empty()) continue
                val hn = H.fromMat(hm)
                if (plausible(hs[id]!!.inv() * hn, imgs[id].w, imgs[id].h)) hs[id] = hn
            }
        }
    }

    // ------------------------------------------------------------------ drawing the sheet

    private fun bounds(): DoubleArray {
        var x0 = Double.MAX_VALUE; var y0 = Double.MAX_VALUE; var x1 = -Double.MAX_VALUE; var y1 = -Double.MAX_VALUE
        for ((id, h) in placedH) {
            val im = imgs[id]
            for ((x, y) in listOf(0.0 to 0.0, im.w.toDouble() to 0.0, im.w.toDouble() to im.h.toDouble(), 0.0 to im.h.toDouble())) {
                val u = h.mapX(x, y); val v = h.mapY(x, y)
                x0 = min(x0, u); x1 = max(x1, u); y0 = min(y0, v); y1 = max(y1, v)
            }
        }
        return doubleArrayOf(x0, y0, x1, y1)
    }

    private fun edgeWeight(w: Int, h: Int): Mat {
        val border = Mat(h, w, CvType.CV_8UC1, Scalar(255.0))
        Imgproc.rectangle(border, Point(0.0, 0.0), Point(w - 1.0, h - 1.0), Scalar(0.0), 1)
        val dist = Mat()
        Imgproc.distanceTransform(border, dist, Imgproc.DIST_L1, 3)
        val wm = Mat()
        dist.convertTo(wm, CvType.CV_8UC1, 250.0 / (min(w, h) / 2.0), 1.0)
        border.release(); dist.release()
        return wm
    }

    /**
     * Composite of the placed photos. Thumbnails at true (exposure-equalised) brightness
     * for previews/paper finding, or full photos with lighting flattened for the result.
     * Returns (image, captured-mask, sheet -> image transform).
     */
    private fun compose(maxLong: Int, full: Boolean): Triple<Mat, Mat, H> {
        val b = bounds()
        val sc = min(1.0, maxLong / max(b[2] - b[0], b[3] - b[1]))
        val w = max(8, ((b[2] - b[0]) * sc).toInt() + 1)
        val hh = max(8, ((b[3] - b[1]) * sc).toInt() + 1)
        val t = H(doubleArrayOf(sc, 0.0, -b[0] * sc, 0.0, sc, -b[1] * sc, 0.0, 0.0, 1.0))
        val out = Mat.zeros(hh, w, CvType.CV_8UC1)
        val best = Mat.zeros(hh, w, CvType.CV_8UC1)
        val img = Mat(); val wt = Mat(); val sel = Mat()
        var wm: Mat? = null
        for ((id, h) in placedH) {
            val im = imgs[id]
            val src: Mat
            if (full) {
                val raw = Imgcodecs.imdecode(MatOfByte(*im.jpeg), Imgcodecs.IMREAD_GRAYSCALE)
                src = flatten(raw); raw.release()
            } else {
                src = Mat()
                val p95 = percentile(im.thumb, 0.95)
                Core.convertScaleAbs(im.thumb, src, 230.0 / max(1.0, p95), 0.0)
            }
            if (wm == null || wm.cols() != src.cols() || wm.rows() != src.rows()) { wm?.release(); wm = edgeWeight(src.cols(), src.rows()) }
            val f = im.w.toDouble() / src.cols()
            val m = (t * h * H(doubleArrayOf(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0))).toMat64()
            val size = Size(w.toDouble(), hh.toDouble())
            Imgproc.warpPerspective(src, img, m, size, Imgproc.INTER_LINEAR)
            Imgproc.warpPerspective(wm!!, wt, m, size, Imgproc.INTER_LINEAR)
            Core.compare(wt, best, sel, Core.CMP_GT)
            img.copyTo(out, sel)
            wt.copyTo(best, sel)
            src.release(); m.release()
        }
        img.release(); wt.release(); sel.release(); wm?.release()
        val valid = Mat()
        Core.compare(best, Scalar(0.0), valid, Core.CMP_GT)
        best.release()
        return Triple(out, valid, t)
    }

    private fun percentile(img: Mat, p: Double): Double {
        val n = img.total().toInt()
        val a = ByteArray(n); img.get(0, 0, a)
        val hist = IntArray(256); for (v in a) hist[v.toInt() and 0xff]++
        var acc = 0
        for (v in 0 until 256) { acc += hist[v]; if (acc >= n * p) return v.toDouble() }
        return 255.0
    }

    // ------------------------------------------------------------------ finding the paper

    /**
     * The paper on a small true-brightness composite: the largest region that is smooth
     * (low texture, unlike a wood or stone surface) and not dark, then the 4-sided
     * outline that overlaps that region best. Returns TL, TR, BR, BL or null.
     */
    private fun findPaper(raw: Mat, valid: Mat): Array<Point>? {
        val g = Mat(); raw.convertTo(g, CvType.CV_32F)
        val k = Size(15.0, 15.0)
        val m = Mat(); Imgproc.blur(g, m, k)
        val g2 = Mat(); Core.multiply(g, g, g2)
        val m2 = Mat(); Imgproc.blur(g2, m2, k)
        val mm = Mat(); Core.multiply(m, m, mm)
        val variance = Mat(); Core.subtract(m2, mm, variance)
        val smooth = Mat(); Imgproc.threshold(variance, smooth, 144.0, 255.0, Imgproc.THRESH_BINARY_INV) // local sd < 12
        smooth.convertTo(smooth, CvType.CV_8U)
        val m8 = Mat(); m.convertTo(m8, CvType.CV_8U)
        val bright = Mat(); Imgproc.threshold(m8, bright, percentileMasked(m8, valid, 0.15), 255.0, Imgproc.THRESH_BINARY)
        val mask = Mat()
        Core.bitwise_and(smooth, bright, mask); Core.bitwise_and(mask, valid, mask)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, Mat.ones(25, 25, CvType.CV_8U)) // fill in the text
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, Mat.ones(9, 9, CvType.CV_8U))
        for (x in listOf(g, m, g2, m2, mm, variance, smooth, m8, bright)) x.release()
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, cents)
        var bestLab = -1; var bestArea = 0.0
        for (i in 1 until n) { val a = stats.get(i, Imgproc.CC_STAT_AREA)[0]; if (a > bestArea) { bestArea = a; bestLab = i } }
        if (bestLab < 0 || bestArea < 0.15 * mask.total()) { mask.release(); labels.release(); return null }
        val comp = Mat()
        Core.compare(labels, Scalar(bestLab.toDouble()), comp, Core.CMP_EQ)
        labels.release(); stats.release(); cents.release(); mask.release()
        val cs = ArrayList<org.opencv.core.MatOfPoint>()
        Imgproc.findContours(comp.clone(), cs, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val c = cs.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        val hullIdx = org.opencv.core.MatOfInt()
        Imgproc.convexHull(c, hullIdx)
        val cp = c.toArray()
        val hull = MatOfPoint2f(*hullIdx.toArray().map { cp[it] }.toTypedArray())
        val peri = Imgproc.arcLength(hull, true)
        val cands = ArrayList<Array<Point>>()
        for (eps in doubleArrayOf(0.01, 0.02, 0.03, 0.05, 0.08, 0.12)) {
            val ap = MatOfPoint2f(); Imgproc.approxPolyDP(hull, ap, eps * peri, true)
            val a = ap.toArray(); if (a.size == 4) cands.add(a)
        }
        val box = Array(4) { Point() }; Imgproc.minAreaRect(hull).points(box); cands.add(box)
        // Pick the outline that overlaps the paper region best.
        var best: Array<Point>? = null; var bestIou = 0.0
        val qm = Mat(); val inter = Mat(); val uni = Mat()
        for (q in cands) {
            qm.create(comp.size(), CvType.CV_8UC1); qm.setTo(Scalar(0.0))
            Imgproc.fillConvexPoly(qm, org.opencv.core.MatOfPoint(*q.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
            Core.bitwise_and(qm, comp, inter); Core.bitwise_or(qm, comp, uni)
            val iou = Core.countNonZero(inter).toDouble() / max(1, Core.countNonZero(uni))
            if (iou > bestIou) { bestIou = iou; best = q }
        }
        qm.release(); inter.release(); uni.release(); comp.release()
        return best?.let { PageDetector.order(it) }
    }

    // ------------------------------------------------------------------ live coverage map

    /**
     * The page so far, coloured by how well each spot has been captured:
     * red = not captured (or only in shadow / at a photo's edge), yellow = weak, green = good.
     * Once the paper's outline is visible, everything inside it counts, so gaps show up red.
     */
    private fun coverage(latest: Int): Coverage? {
        if (placedH.isEmpty()) return null
        val (img, valid, t) = compose(320, full = false)
        val rows = img.rows(); val cols = img.cols(); val size = img.size()
        val ws = Mat.zeros(rows, cols, CvType.CV_32F); val wi = Mat()
        for ((id, h) in placedH) {
            val im = imgs[id]
            val f = im.w.toDouble() / im.wthumb.cols()
            val m = (t * h * H(doubleArrayOf(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0))).toMat64()
            Imgproc.warpPerspective(im.wthumb, wi, m, size, Imgproc.INTER_LINEAR)
            Core.add(ws, wi, ws)
            m.release()
        }
        wi.release()
        // Paper outline, found on a larger composite.
        var quad: Array<Point>? = null
        val (big, bvalid, tb) = compose(800, full = false)
        try {
            findPaper(big, bvalid)?.let { q ->
                // Trust the outline only where it follows real paper edges: just outside it
                // should mostly be captured background, not the not-yet-scanned unknown.
                val qm = Mat.zeros(big.rows(), big.cols(), CvType.CV_8UC1)
                Imgproc.fillConvexPoly(qm, org.opencv.core.MatOfPoint(*q.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
                val ring = Mat(); Imgproc.dilate(qm, ring, Mat.ones(21, 21, CvType.CV_8U))
                Core.subtract(ring, qm, ring)
                val all = Core.countNonZero(ring)
                Core.bitwise_and(ring, bvalid, ring)
                val seen = Core.countNonZero(ring)
                qm.release(); ring.release()
                if (all > 0 && seen >= 0.6 * all) {
                    val toP = t * tb.inv()
                    quad = q.map { Point(toP.mapX(it.x, it.y), toP.mapY(it.x, it.y)) }.toTypedArray()
                }
            }
        } catch (_: Throwable) {
        }
        big.release(); bvalid.release()
        val n = rows * cols
        val pm = ByteArray(n)
        val qd = quad
        if (qd != null) {
            val pageMask = Mat.zeros(rows, cols, CvType.CV_8UC1)
            Imgproc.fillConvexPoly(pageMask, org.opencv.core.MatOfPoint(*qd.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
            pageMask.get(0, 0, pm); pageMask.release()
        }
        val g = ByteArray(n); img.get(0, 0, g)
        val v = ByteArray(n); valid.get(0, 0, v)
        val w = FloatArray(n); ws.get(0, 0, w)
        img.release(); valid.release(); ws.release()
        val out = ByteArray(n * 3)
        var good = 0; var weak = 0; var missing = 0
        for (i in 0 until n) {
            val gray = g[i].toInt() and 0xff
            val inPage = if (qd != null) pm[i].toInt() != 0 else v[i].toInt() != 0
            var r: Int; var gr: Int; var b: Int
            if (!inPage) {
                r = gray * 35 / 100; gr = r; b = r
            } else if (w[i] < 0.15f) {
                missing++; r = (gray * 40 + 255 * 60) / 100; gr = gray * 40 / 100 + 20; b = gray * 40 / 100 + 20
            } else if (w[i] < 0.8f) {
                weak++; r = (gray * 55 + 255 * 45) / 100; gr = (gray * 55 + 200 * 45) / 100; b = gray * 55 / 100
            } else {
                good++; r = gray * 80 / 100; gr = (gray * 80 + 255 * 20) / 100; b = gray * 80 / 100
            }
            out[3 * i] = r.coerceIn(0, 255).toByte(); out[3 * i + 1] = gr.coerceIn(0, 255).toByte(); out[3 * i + 2] = b.coerceIn(0, 255).toByte()
        }
        val rgb = Mat(rows, cols, CvType.CV_8UC3); rgb.put(0, 0, out)
        if (qd != null) {
            Imgproc.polylines(rgb, listOf(org.opencv.core.MatOfPoint(*qd.map { Point(it.x, it.y) }.toTypedArray())), true, Scalar(0.0, 220.0, 255.0), 2)
        }
        // Outline of the newest photo, so you can see where you are.
        placedH[latest]?.let { h ->
            val im = imgs[latest]
            val c = listOf(0.0 to 0.0, im.w.toDouble() to 0.0, im.w.toDouble() to im.h.toDouble(), 0.0 to im.h.toDouble())
                .map { (x, y) -> val m = t * h; Point(m.mapX(x, y), m.mapY(x, y)) }
            Imgproc.polylines(rgb, listOf(org.opencv.core.MatOfPoint(*c.toTypedArray())), true, Scalar(255.0, 255.0, 255.0), 2)
        }
        val tot = max(1, good + weak + missing)
        return Coverage(rgb, qd != null, 100 * good / tot, 100 * weak / tot, 100 * missing / tot)
    }

    // ------------------------------------------------------------------ multi-frame fusion

    /** Weight that favours the middle of a photo over its edges. */
    private fun centrality(w: Int, h: Int): Mat {
        val border = Mat(h, w, CvType.CV_8UC1, Scalar(255.0))
        Imgproc.rectangle(border, Point(0.0, 0.0), Point(w - 1.0, h - 1.0), Scalar(0.0), 1)
        val d = Mat()
        Imgproc.distanceTransform(border, d, Imgproc.DIST_L2, 3)
        Core.multiply(d, Scalar(1.0 / (0.15 * min(w, h))), d)
        Imgproc.threshold(d, d, 1.0, 1.0, Imgproc.THRESH_TRUNC)
        Core.multiply(d, d, d)
        Core.add(d, Scalar(1e-3), d)
        border.release()
        return d
    }

    /** Weight that favours well-lit parts of a photo over shadowed ones: (relative brightness)^4. */
    private fun litWeight(src: Mat): Mat {
        val p = max(1.0, percentile(src, 0.95))
        val d = Mat()
        Imgproc.dilate(src, d, Mat.ones(7, 7, CvType.CV_8U)) // ignore the text itself
        val r = Mat()
        d.convertTo(r, CvType.CV_32F, 1.0 / p)
        Imgproc.threshold(r, r, 1.0, 1.0, Imgproc.THRESH_TRUNC)
        Imgproc.GaussianBlur(r, r, Size(0.0, 0.0), 6.0)
        Core.multiply(r, r, r); Core.multiply(r, r, r)
        d.release()
        return r
    }

    /** Fine-tunes each photo's placement against the joined sheet (sub-pixel), for sharp fusion. */
    private fun refineForFusion() {
        val b = bounds()
        val ext = max(b[2] - b[0], b[3] - b[1])
        if (ext > 4000) return // unusually large: skip rather than use lots of memory
        val (ref8, rv, t1) = compose(ext.toInt() + 2, full = true)
        rv.release()
        val ref = Mat(); ref8.convertTo(ref, CvType.CV_32F); ref8.release()
        val crit = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 40, 1e-5)
        for (id in placedH.keys.toList()) {
            val hi = placedH[id]!!
            val raw = Imgcodecs.imdecode(MatOfByte(*imgs[id].jpeg), Imgcodecs.IMREAD_GRAYSCALE)
            val f8 = flatten(raw); raw.release()
            val tmpl = Mat(); f8.convertTo(tmpl, CvType.CV_32F); f8.release()
            val w0 = t1 * hi
            try {
                val warp = w0.toMat32()
                Video.findTransformECC(tmpl, ref, warp, Video.MOTION_HOMOGRAPHY, crit, Mat(), 3)
                val wr = H.fromMat(warp)
                var dev = 0.0
                val w = imgs[id].w.toDouble(); val h = imgs[id].h.toDouble()
                for ((x, y) in listOf(0.0 to 0.0, w to 0.0, w to h, 0.0 to h)) {
                    dev = max(dev, hypot(wr.mapX(x, y) - w0.mapX(x, y), wr.mapY(x, y) - w0.mapY(x, y)))
                }
                if (dev < 6.0) placedH[id] = t1.inv() * wr
                warp.release()
            } catch (_: Exception) {
                // keep the feature-based placement
            }
            tmpl.release()
        }
        ref.release()
    }

    /**
     * Blends all overlapping photos instead of picking one per spot: each spot is a weighted
     * average of every photo that saw it, favouring well-lit and central views. Averaging
     * many slightly shifted low-res views gives a cleaner, sharper sheet than any single photo.
     * Returns (sheet, captured-mask, sheet-coords -> output transform).
     */
    private fun fuse(): Triple<Mat, Mat, H> {
        refineForFusion()
        val b = bounds()
        val sc = min(FUSE_SCALE, FUSE_LONG / max(b[2] - b[0], b[3] - b[1]))
        val w = ((b[2] - b[0]) * sc).toInt() + 1
        val hh = ((b[3] - b[1]) * sc).toInt() + 1
        val t = H(doubleArrayOf(sc, 0.0, -b[0] * sc, 0.0, sc, -b[1] * sc, 0.0, 0.0, 1.0))
        val size = Size(w.toDouble(), hh.toDouble())
        val acc = Mat.zeros(hh, w, CvType.CV_32F)
        val ws = Mat.zeros(hh, w, CvType.CV_32F)
        val fi = Mat(); val wi = Mat(); val prod = Mat()
        for ((id, h) in placedH) {
            val raw = Imgcodecs.imdecode(MatOfByte(*imgs[id].jpeg), Imgcodecs.IMREAD_GRAYSCALE)
            val f8 = flatten(raw)
            val f = Mat(); f8.convertTo(f, CvType.CV_32F); f8.release()
            val cw = centrality(raw.cols(), raw.rows())
            val lw = litWeight(raw)
            Core.multiply(cw, lw, cw)
            raw.release(); lw.release()
            val m = (t * h).toMat64()
            Imgproc.warpPerspective(f, fi, m, size, Imgproc.INTER_CUBIC)
            Imgproc.warpPerspective(cw, wi, m, size, Imgproc.INTER_LINEAR)
            Core.multiply(fi, wi, prod)
            Core.add(acc, prod, acc)
            Core.add(ws, wi, ws)
            f.release(); cw.release(); m.release()
        }
        fi.release(); prod.release()
        val valid = Mat()
        Core.compare(ws, Scalar(1e-4), valid, Core.CMP_GT)
        Core.max(ws, Scalar(1e-6), wi)
        Core.divide(acc, wi, acc)
        ws.release(); wi.release()
        // Mild sharpening to recover detail softened by averaging.
        val blur = Mat()
        Imgproc.GaussianBlur(acc, blur, Size(0.0, 0.0), 1.0 * sc)
        Core.addWeighted(acc, 1.7, blur, -0.7, 0.0, acc)
        blur.release()
        val out = Mat()
        acc.convertTo(out, CvType.CV_8U)
        acc.release()
        return Triple(out, valid, t)
    }

    /** The finished sheet: joined, fused, paper found and straightened. */
    fun result(): Mat {
        if (placedH.isEmpty()) throw IllegalStateException("No photos could be joined.")
        val (mos, valid, t) = fuse()
        val (raw, rvalid, tl) = compose(800, full = false)
        var page: Array<Point>? = null
        if (Core.countNonZero(rvalid) > 0) {
            val q = findPaper(raw, rvalid)
            if (q != null) {
                val toOut = t * tl.inv()
                page = q.map { Point(toOut.mapX(it.x, it.y), toOut.mapY(it.x, it.y)) }.toTypedArray()
            }
        }
        raw.release(); rvalid.release()
        val straight: Mat
        if (page != null) {
            val (pw, ph) = pageSize(page, max(mos.cols(), mos.rows()))
            val dst = MatOfPoint2f(Point(0.0, 0.0), Point(pw.toDouble(), 0.0), Point(pw.toDouble(), ph.toDouble()), Point(0.0, ph.toDouble()))
            val hq = Imgproc.getPerspectiveTransform(MatOfPoint2f(*page), dst)
            straight = Mat()
            Imgproc.warpPerspective(mos, straight, hq, Size(pw.toDouble(), ph.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(255.0))
        } else {
            val gaps = Mat(); Core.bitwise_not(valid, gaps); mos.setTo(Scalar(255.0), gaps); gaps.release()
            straight = mos.clone()
        }
        mos.release(); valid.release()
        val e = enhance(straight)
        straight.release()
        if (page == null) return e // without a clean page outline, background bits could mislead the check
        // Photos may have been taken with the watch upside down or sideways: turn the page upright.
        val up = try { Orientation.upright(e) } catch (_: Throwable) { return e }
        e.release()
        return up
    }

    private fun percentileMasked(img: Mat, mask: Mat, p: Double): Double {
        val n = img.total().toInt()
        val a = ByteArray(n); val m = ByteArray(n)
        img.get(0, 0, a); mask.get(0, 0, m)
        val hist = IntArray(256); var cnt = 0
        for (k in 0 until n) if (m[k].toInt() != 0) { hist[a[k].toInt() and 0xff]++; cnt++ }
        var acc = 0
        for (v in 0 until 256) { acc += hist[v]; if (acc >= cnt * p) return v.toDouble() }
        return 0.0
    }

    /** Photos that couldn't be joined (as JPEGs sized for Claude), so nothing is lost. */
    fun unplacedPhotos(maxN: Int): List<ByteArray> =
        imgs.indices.filter { !placedH.containsKey(it) }.take(maxN).map { id ->
            val raw = Imgcodecs.imdecode(MatOfByte(*imgs[id].jpeg), Imgcodecs.IMREAD_GRAYSCALE)
            val flat = flatten(raw)
            val (tw, th) = sizeForClaude(flat.cols(), flat.rows(), 1568, 2400)
            val small = Mat()
            Imgproc.resize(flat, small, Size(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            val j = encodeJpeg(small, 85)
            raw.release(); flat.release(); small.release()
            j
        }

    /** Stored photos, for the troubleshooting zip. */
    fun photoJpegs(): List<ByteArray> = imgs.map { it.jpeg }

    fun describe(): String = "photos=${imgs.size} joined=$joined byMotion=${byMotion.size} links=${edges.size}: " +
        edges.joinToString(" ") { "${it.j}-${it.i}(${it.pi.size})" }

    fun release() {
        for (im in imgs) { im.des.release(); im.thumb.release(); im.wthumb.release() }
        imgs.clear(); edges.clear(); placedH.clear()
    }
}
