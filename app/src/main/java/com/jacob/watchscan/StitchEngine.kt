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
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
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
        const val FEAT_SCALE = 0.5
        const val OUT_LONG = 3000
    }

    private class Img(
        val jpeg: ByteArray, val w: Int, val h: Int,
        val pts: Array<Point>, val des: Mat, val thumb: Mat,
        val pos: Point?, val unc: Double,       // tracked camera position (stored-photo px)
        val corners: List<Corner>,              // paper corners seen in this photo
    )

    /** A paper corner at (x, y); (dx, dy) points into the paper. */
    class Corner(val x: Double, val y: Double, val dx: Double, val dy: Double)

    private class Edge(val j: Int, val i: Int, val pj: Array<Point>, val pi: Array<Point>)

    private val orb = ORB.create(2500, 1.2f, 8, 31, 0, 2, 0, 31, 10)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
    private val imgs = ArrayList<Img>()
    private val edges = ArrayList<Edge>()
    private var placedH = HashMap<Int, H>()   // photo -> sheet
    private var byMotion = HashSet<Int>()      // placed from tracked motion only
    var joined = 0; private set
    var cornersFound = 0; private set

    val count get() = imgs.size
    val placed get() = placedH.size

    // ------------------------------------------------------------------ adding photos

    /** Adds a snapshot (full-res grayscale). Returns the live preview of the sheet. */
    fun add(frame: Mat, px: Double = Double.NaN, py: Double = Double.NaN, unc: Double = 0.0): Mat? {
        val s = min(1.0, STORE_LONG.toDouble() / max(frame.cols(), frame.rows()))
        val full = Mat()
        if (s < 1.0) Imgproc.resize(frame, full, Size(frame.cols() * s, frame.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        else frame.copyTo(full)
        val w = full.cols(); val h = full.rows()
        val sw = (w * FEAT_SCALE).toInt(); val sh = (h * FEAT_SCALE).toInt()
        val small = Mat()
        Imgproc.resize(full, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val kp = MatOfKeyPoint(); val des = Mat()
        orb.detectAndCompute(small, Mat(), kp, des)
        small.release()
        val up = H.resize(sw.toDouble() / w, sh.toDouble() / h).inv()
        val pts = kp.toArray().map { Point(up.mapX(it.pt.x, it.pt.y), up.mapY(it.pt.x, it.pt.y)) }.toTypedArray()
        val thumb = Mat()
        Imgproc.resize(full, thumb, Size((w / 8).toDouble(), (h / 8).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val i = imgs.size
        val pos = if (px.isNaN()) null else Point(px * s, py * s)
        imgs.add(Img(encodeJpeg(full, 92), w, h, pts, des, thumb, pos, unc * s, findCorners(thumb, w.toDouble() / thumb.cols())))
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
        if (placedH.isEmpty()) return null
        val (pv, valid, _) = compose(300, full = false)
        valid.release()
        return pv
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

    /** Two handheld shots of a flat page from a similar distance: moderate scale change, mild perspective. */
    private fun plausible(h: H, w: Int, hh: Int): Boolean {
        val xs = doubleArrayOf(0.0, w.toDouble(), w.toDouble(), 0.0)
        val ys = doubleArrayOf(0.0, 0.0, hh.toDouble(), hh.toDouble())
        val qx = DoubleArray(4) { h.mapX(xs[it], ys[it]) }; val qy = DoubleArray(4) { h.mapY(xs[it], ys[it]) }
        if (!isConvex(qx, qy)) return false
        val r = polyArea(qx, qy) / (w.toDouble() * hh)
        if (r < 0.36 || r > 2.8) return false
        val side = DoubleArray(4) { hypot(qx[(it + 1) % 4] - qx[it], qy[(it + 1) % 4] - qy[it]) }
        return max(side[0], side[2]) / min(side[0], side[2]) < 1.5 && max(side[1], side[3]) / min(side[1], side[3]) < 1.5
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
        cornersFound = cornerGroups().size
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

    // ------------------------------------------------------------------ paper corners

    /** Paper corners visible in a photo thumbnail: crisp bright-paper vs darker-surroundings corners. */
    private fun findCorners(thumb: Mat, up: Double): List<Corner> {
        val w = thumb.cols(); val h = thumb.rows()
        val b = Mat(); Imgproc.GaussianBlur(thumb, b, Size(5.0, 5.0), 0.0)
        val m = Mat(); Imgproc.threshold(b, m, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        try {
            val fgN = Core.countNonZero(m)
            if (fgN == 0 || (w * h - fgN) < 0.03 * w * h) return emptyList()
            val inv = Mat(); Core.bitwise_not(m, inv)
            val fg = Core.mean(b, m).`val`[0]; val bg = Core.mean(b, inv).`val`[0]
            inv.release()
            if (fg - bg < 25) return emptyList()
            Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, Mat.ones(5, 5, CvType.CV_8U))
            val cs = ArrayList<org.opencv.core.MatOfPoint>()
            Imgproc.findContours(m, cs, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val c = cs.maxByOrNull { Imgproc.contourArea(it) } ?: return emptyList()
            if (Imgproc.contourArea(c) < 0.15 * w * h) return emptyList()
            val c2 = MatOfPoint2f(*c.toArray())
            val ap = MatOfPoint2f()
            Imgproc.approxPolyDP(c2, ap, 0.02 * Imgproc.arcLength(c2, true), true)
            val v = ap.toArray()
            val out = ArrayList<Corner>()
            val mx = 0.04 * w; val my = 0.04 * h
            for (k in v.indices) {
                val p = v[k]
                if (p.x < mx || p.y < my || p.x > w - mx || p.y > h - my) continue // on the photo border
                val pv = v[(k - 1 + v.size) % v.size]; val nx = v[(k + 1) % v.size]
                val ax = pv.x - p.x; val ay = pv.y - p.y; val bx = nx.x - p.x; val by = nx.y - p.y
                val la = hypot(ax, ay); val lb = hypot(bx, by)
                if (min(la, lb) < 0.12 * min(w, h)) continue // both edges clearly visible
                val cos = ((ax * bx + ay * by) / (la * lb)).coerceIn(-1.0, 1.0)
                val ang = Math.toDegrees(kotlin.math.acos(cos))
                if (ang < 65 || ang > 115) continue
                val dx = ax / la + bx / lb; val dy = ay / la + by / lb
                if (sharpEdges(b, p, ax / la, ay / la, la, dx, dy) && sharpEdges(b, p, bx / lb, by / lb, lb, dx, dy)) {
                    out.add(Corner(p.x * up, p.y * up, dx, dy))
                }
            }
            return out
        } finally {
            b.release(); m.release()
        }
    }

    /** A real paper edge is a crisp brightness step, unlike soft lens vignetting. */
    private fun sharpEdges(img: Mat, p: Point, ux: Double, uy: Double, len: Double, dx: Double, dy: Double): Boolean {
        var nx = -uy; var ny = ux
        if (nx * dx + ny * dy < 0) { nx = -nx; ny = -ny } // into the paper
        val steps = ArrayList<Double>()
        for (f in doubleArrayOf(0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)) {
            val qx = p.x + ux * min(len, 60.0) * f; val qy = p.y + uy * min(len, 60.0) * f
            val ix = (qx + nx * 3).toInt(); val iy = (qy + ny * 3).toInt()
            val ox = (qx - nx * 3).toInt(); val oy = (qy - ny * 3).toInt()
            if (ix < 0 || iy < 0 || ox < 0 || oy < 0 || ix >= img.cols() || ox >= img.cols() || iy >= img.rows() || oy >= img.rows()) continue
            steps.add(img.get(iy, ix)[0] - img.get(oy, ox)[0])
        }
        if (steps.size < 4) return false
        steps.sort()
        return steps[steps.size / 2] >= 20.0
    }

    /** Detected corners on the sheet, grouped as 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left. */
    private fun cornerGroups(): Map<Int, List<Point>> {
        val found = HashMap<Int, MutableList<Point>>()
        for ((id, h) in placedH) for (c in imgs[id].corners) {
            val px = h.mapX(c.x, c.y); val py = h.mapY(c.x, c.y)
            val qx = h.mapX(c.x + c.dx * 50, c.y + c.dy * 50) - px
            val qy = h.mapY(c.x + c.dx * 50, c.y + c.dy * 50) - py
            val kind = when {
                qx > 0 && qy > 0 -> 0
                qx <= 0 && qy > 0 -> 1
                qx <= 0 -> 2
                else -> 3
            }
            found.getOrPut(kind) { ArrayList() }.add(Point(px, py))
        }
        return found
    }

    /** Page outline from the paper's corners (TL, TR, BR, BL on the sheet) if at least 3 were seen. */
    private fun pageFromCorners(): Array<Point>? {
        val g = cornerGroups()
        if (g.size < 3) return null
        val q = arrayOfNulls<Point>(4)
        for ((k, pts) in g) {
            val xs = pts.map { it.x }.sorted(); val ys = pts.map { it.y }.sorted()
            q[k] = Point(xs[xs.size / 2], ys[ys.size / 2])
        }
        for (miss in 0 until 4) if (q[miss] == null) {
            val a = q[(miss + 1) % 4]!!; val c = q[(miss + 3) % 4]!!; val o = q[(miss + 2) % 4]!!
            q[miss] = Point(a.x + c.x - o.x, a.y + c.y - o.y)
        }
        val xs = DoubleArray(4) { q[it]!!.x }; val ys = DoubleArray(4) { q[it]!!.y }
        if (!isConvex(xs, ys) || polyArea(xs, ys) < 1000.0) return null
        return Array(4) { q[it]!! }
    }

    /** The finished sheet: joined, paper found and straightened, lighting evened out. */
    fun result(): Mat {
        if (placedH.isEmpty()) throw IllegalStateException("No photos could be joined.")
        val (mos, valid, t) = compose(OUT_LONG, full = true)
        val (raw, rvalid, tl) = compose(800, full = false)
        var page: Array<Point>? = pageFromCorners()?.map { Point(t.mapX(it.x, it.y), t.mapY(it.x, it.y)) }?.toTypedArray()
        if (page == null && Core.countNonZero(rvalid) > 0) {
            // Fill gaps with a background-ish tone so they don't look like paper.
            val fill = percentileMasked(raw, rvalid, 0.05)
            val inv = Mat(); Core.bitwise_not(rvalid, inv); raw.setTo(Scalar(fill), inv); inv.release()
            val q = PageDetector.detect(raw, rvalid)
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
        return e
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

    fun describe(): String = "photos=${imgs.size} joined=$joined byMotion=${byMotion.size} corners=$cornersFound links=${edges.size}: " +
        edges.joinToString(" ") { "${it.j}-${it.i}(${it.pi.size})" }

    fun release() {
        for (im in imgs) { im.des.release(); im.thumb.release() }
        imgs.clear(); edges.clear(); placedH.clear()
    }
}
