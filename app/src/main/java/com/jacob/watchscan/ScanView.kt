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

/** Viewfinder, plus a live preview of the page being built up while sweeping. */
class ScanView(context: Context) : View(context) {
    private var frame: Bitmap? = null
    private var still: Bitmap? = null
    private var mosaic: Bitmap? = null
    private var footprint: FloatArray? = null
    private var lostFlag = false

    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val fill = Paint()
    private val r = RectF()

    fun setFrame(bmp: Bitmap) { frame = bmp; invalidate() }
    fun setStill(bmp: Bitmap?) { still = bmp; invalidate() }

    fun setMosaic(bmp: Bitmap?) { if (bmp != null) mosaic = bmp; invalidate() }

    fun setFootprint(f: FloatArray?, lost: Boolean) { footprint = f; lostFlag = lost; invalidate() }

    fun clearOverlays() { mosaic = null; footprint = null; lostFlag = false; invalidate() }

    private fun fit(bw: Int, bh: Int, maxW: Float, maxH: Float, cx: Float, cy: Float, out: RectF) {
        val s = min(maxW / bw, maxH / bh)
        out.set(cx - bw * s / 2, cy - bh * s / 2, cx + bw * s / 2, cy + bh * s / 2)
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.BLACK)
        val st = still
        if (st != null) {
            fit(st.width, st.height, width.toFloat(), height.toFloat(), width / 2f, height / 2f, r)
            canvas.drawBitmap(st, null, r, bmpPaint)
            return
        }
        val fr = frame ?: return
        fit(fr.width, fr.height, width.toFloat(), height.toFloat(), width / 2f, height / 2f, r)
        canvas.drawBitmap(fr, null, r, bmpPaint)
        if (lostFlag) {
            line.color = Color.RED
            canvas.drawRect(r.left + 3, r.top + 3, r.right - 3, r.bottom - 3, line)
        }

        val m = mosaic ?: return
        // Page-so-far preview on the right, vertically centred (safe on round screens).
        fit(m.width, m.height, width * 0.36f, height * 0.42f, width * 0.93f - width * 0.18f, height / 2f, r)
        fill.color = Color.argb(170, 0, 0, 0)
        canvas.drawRect(r.left - 4, r.top - 4, r.right + 4, r.bottom + 4, fill)
        canvas.drawBitmap(m, null, r, bmpPaint)
        footprint?.let { f ->
            val p = Path()
            for (k in 0 until 4) {
                val x = r.left + f[2 * k].coerceIn(-0.3f, 1.3f) * r.width()
                val y = r.top + f[2 * k + 1].coerceIn(-0.3f, 1.3f) * r.height()
                if (k == 0) p.moveTo(x, y) else p.lineTo(x, y)
            }
            p.close()
            line.color = if (lostFlag) Color.RED else Color.rgb(80, 220, 120)
            line.strokeWidth = 3f
            canvas.drawPath(p, line)
            line.strokeWidth = 4f
        }
    }
}
