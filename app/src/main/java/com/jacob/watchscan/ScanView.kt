package com.jacob.watchscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import kotlin.math.min

/** Viewfinder with page outline, plus a mini-map of which parts of the page are captured. */
class ScanView(context: Context) : View(context) {
    private var frame: Bitmap? = null
    private var still: Bitmap? = null

    /** Page outline in the frame, normalised (8 floats), and whether it is a confident detection. */
    private var outline: FloatArray? = null
    private var outlineColor = Color.GREEN

    private var grid: IntArray? = null
    private var cols = 0
    private var rows = 0
    private var pageAspect = 1.29f // height / width
    private var footprint: FloatArray? = null

    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val fill = Paint()
    private val frameRect = RectF()

    fun setFrame(bmp: Bitmap) {
        frame = bmp
        invalidate()
    }

    fun setStill(bmp: Bitmap?) {
        still = bmp
        invalidate()
    }

    fun setOutline(o: FloatArray?, color: Int) {
        outline = o
        outlineColor = color
        invalidate()
    }

    fun setGrid(status: IntArray?, cols: Int, rows: Int, aspect: Float, foot: FloatArray?) {
        grid = status
        this.cols = cols
        this.rows = rows
        pageAspect = aspect
        footprint = foot
        invalidate()
    }

    fun clearOverlays() {
        outline = null
        grid = null
        footprint = null
        invalidate()
    }

    private fun fitRect(bw: Int, bh: Int, out: RectF) {
        val s = min(width.toFloat() / bw, height.toFloat() / bh)
        val dw = bw * s
        val dh = bh * s
        out.set((width - dw) / 2, (height - dh) / 2, (width + dw) / 2, (height + dh) / 2)
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.BLACK)
        val st = still
        if (st != null) {
            fitRect(st.width, st.height, frameRect)
            canvas.drawBitmap(st, null, frameRect, bmpPaint)
            return
        }
        val fr = frame ?: return
        fitRect(fr.width, fr.height, frameRect)
        canvas.drawBitmap(fr, null, frameRect, bmpPaint)

        outline?.let { o ->
            val p = Path()
            for (k in 0 until 4) {
                val x = frameRect.left + o[2 * k] * frameRect.width()
                val y = frameRect.top + o[2 * k + 1] * frameRect.height()
                if (k == 0) p.moveTo(x, y) else p.lineTo(x, y)
            }
            p.close()
            line.color = outlineColor
            canvas.drawPath(p, line)
        }

        val g = grid ?: return
        if (cols == 0 || rows == 0) return
        // Mini-map on the right, vertically centred (safe on round screens).
        val maxW = width * 0.30f
        val maxH = height * 0.42f
        var mw = maxW
        var mh = mw * pageAspect
        if (mh > maxH) { mh = maxH; mw = mh / pageAspect }
        val left = width * 0.93f - mw
        val top = (height - mh) / 2
        fill.color = Color.argb(150, 0, 0, 0)
        canvas.drawRect(left - 4, top - 4, left + mw + 4, top + mh + 4, fill)
        val cwid = mw / cols
        val chgt = mh / rows
        for (i in g.indices) {
            val c = i % cols
            val r = i / cols
            fill.color = when (g[i]) {
                ScanEngine.GOOD -> Color.argb(230, 60, 200, 90)
                ScanEngine.BLANK -> Color.argb(150, 40, 110, 70)
                ScanEngine.LOW -> Color.argb(230, 230, 170, 30)
                else -> Color.argb(170, 120, 120, 120)
            }
            canvas.drawRect(left + c * cwid + 1, top + r * chgt + 1, left + (c + 1) * cwid - 1, top + (r + 1) * chgt - 1, fill)
        }
        footprint?.let { f ->
            val p = Path()
            for (k in 0 until 4) {
                val x = left + f[2 * k].coerceIn(-0.2f, 1.2f) * mw
                val y = top + f[2 * k + 1].coerceIn(-0.2f, 1.2f) * mh
                if (k == 0) p.moveTo(x, y) else p.lineTo(x, y)
            }
            p.close()
            line.color = Color.WHITE
            line.strokeWidth = 3f
            canvas.drawPath(p, line)
            line.strokeWidth = 4f
        }
    }
}
