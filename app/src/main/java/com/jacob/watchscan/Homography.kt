package com.jacob.watchscan

import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.abs
import kotlin.math.sqrt

/** Small 3x3 homography helper (row-major). */
class H(val m: DoubleArray) {
    init {
        require(m.size == 9)
    }

    operator fun times(o: H): H {
        val r = DoubleArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0.0
            for (k in 0 until 3) s += m[i * 3 + k] * o.m[k * 3 + j]
            r[i * 3 + j] = s
        }
        return H(r)
    }

    fun inv(): H {
        val a = m
        val c00 = a[4] * a[8] - a[5] * a[7]
        val c01 = -(a[3] * a[8] - a[5] * a[6])
        val c02 = a[3] * a[7] - a[4] * a[6]
        val det = a[0] * c00 + a[1] * c01 + a[2] * c02
        val d = if (abs(det) < 1e-12) 1e-12 else det
        return H(
            doubleArrayOf(
                c00 / d, -(a[1] * a[8] - a[2] * a[7]) / d, (a[1] * a[5] - a[2] * a[4]) / d,
                c01 / d, (a[0] * a[8] - a[2] * a[6]) / d, -(a[0] * a[5] - a[2] * a[3]) / d,
                c02 / d, -(a[0] * a[7] - a[1] * a[6]) / d, (a[0] * a[4] - a[1] * a[3]) / d
            )
        )
    }

    fun mapX(x: Double, y: Double): Double = (m[0] * x + m[1] * y + m[2]) / (m[6] * x + m[7] * y + m[8])
    fun mapY(x: Double, y: Double): Double = (m[3] * x + m[4] * y + m[5]) / (m[6] * x + m[7] * y + m[8])

    /** sqrt(|det J|) of the mapping at (x, y): local linear scale factor. */
    fun localScale(x: Double, y: Double): Double {
        val d = m[6] * x + m[7] * y + m[8]
        val u = (m[0] * x + m[1] * y + m[2]) / d
        val v = (m[3] * x + m[4] * y + m[5]) / d
        val j00 = (m[0] - u * m[6]) / d
        val j01 = (m[1] - u * m[7]) / d
        val j10 = (m[3] - v * m[6]) / d
        val j11 = (m[4] - v * m[7]) / d
        return sqrt(abs(j00 * j11 - j01 * j10))
    }

    fun toMat64(): Mat {
        val mat = Mat(3, 3, CvType.CV_64F)
        mat.put(0, 0, *m)
        return mat
    }

    fun toMat32(): Mat {
        val m64 = toMat64()
        val m32 = Mat()
        m64.convertTo(m32, CvType.CV_32F)
        m64.release()
        return m32
    }

    companion object {
        fun fromMat(mat: Mat): H {
            val m64 = Mat()
            mat.convertTo(m64, CvType.CV_64F)
            val arr = DoubleArray(9)
            m64.get(0, 0, arr)
            m64.release()
            return H(arr)
        }

        /** The exact pixel-centre mapping cv::resize uses for a scale of (sx, sy). */
        fun resize(sx: Double, sy: Double) =
            H(doubleArrayOf(sx, 0.0, 0.5 * sx - 0.5, 0.0, sy, 0.5 * sy - 0.5, 0.0, 0.0, 1.0))

        fun translate(tx: Double, ty: Double) =
            H(doubleArrayOf(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0))
    }
}
