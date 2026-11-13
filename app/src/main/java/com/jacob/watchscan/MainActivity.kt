package com.jacob.watchscan

import android.Manifest
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.util.Size
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.util.Range
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size as CvSize
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.QRCodeDetector
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

class MainActivity : ComponentActivity() {

    enum class Mode { KEY_QR, OVERVIEW, STARTING, SWEEP, FINISHING, ANSWER }

    @Volatile private var mode = Mode.OVERVIEW

    private lateinit var prefs: android.content.SharedPreferences
    private lateinit var scanView: ScanView
    private lateinit var statusText: TextView
    private lateinit var mainButton: Button
    private lateinit var leftButton: Button
    private lateinit var rightButton: Button
    private lateinit var bottomBar: LinearLayout
    private lateinit var progress: ProgressBar
    private lateinit var answerPanel: ScrollView
    private lateinit var answerText: TextView
    private lateinit var settingsPanel: ScrollView
    private lateinit var settingsBox: LinearLayout

    private val analyzerExec = Executors.newSingleThreadExecutor()
    private val workExec = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var camera: Camera? = null
    @Volatile private var hasAf = false
    @Volatile private var continuousAf = false
    @Volatile private var camDesc = ""
    @Volatile private var lastFocusMs = 0L

    private val lock = Any()
    private var lastFrame: Mat? = null
    private var lastQuad: Array<Point>? = null

    @Volatile private var engine: ScanEngine? = null
    private var yBuf: ByteArray? = null
    private var qrDetector: QRCodeDetector? = null
    private var qrCounter = 0
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var lastAnswer = ""
    private var cvOk = false

    private val askPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else setStatus("Camera permission is needed. Allow it in system settings.")
    }

    // ---------------------------------------------------------------- settings

    private val apiKey get() = prefs.getString("api_key", "") ?: ""
    private val model get() = prefs.getString("model", ClaudeClient.MODELS[0].first) ?: ClaudeClient.MODELS[0].first
    private val useFront get() = prefs.getBoolean("front", false)
    private val highRes get() = prefs.getBoolean("high_res", true)
    private val torch get() = prefs.getBoolean("torch", false)
    private val saveScans get() = prefs.getBoolean("save", true)

    // ---------------------------------------------------------------- lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        prefs = getSharedPreferences("watchscan", Context.MODE_PRIVATE)
        buildUi()

        cvOk = try { OpenCVLoader.initLocal() } catch (e: Throwable) { false }
        if (!cvOk) {
            setStatus("Could not load OpenCV on this device.")
            return
        }
        qrDetector = QRCodeDetector()
        tts = TextToSpeech(this) { st -> ttsReady = st == TextToSpeech.SUCCESS }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = onBack()
        })

        if (apiKey.isEmpty()) enterMode(Mode.KEY_QR) else enterMode(Mode.OVERVIEW)

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else askPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        analyzerExec.shutdown()
        workExec.shutdown()
    }

    // ---------------------------------------------------------------- UI

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    private fun makeButton(text: String, onClick: () -> Unit): Button = Button(this).apply {
        this.text = text
        isAllCaps = false
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        setTextColor(Color.WHITE)
        background = GradientDrawable().apply { cornerRadius = dp(22f).toFloat(); setColor(Color.argb(220, 40, 90, 200)) }
        setPadding(dp(10f), dp(4f), dp(10f), dp(4f))
        minHeight = dp(44f)
        minimumHeight = dp(44f)
        setOnClickListener { onClick() }
    }

    private fun buildUi() {
        val root = FrameLayout(this)
        root.setBackgroundColor(Color.BLACK)
        scanView = ScanView(this)
        scanView.setOnClickListener { focusCentre() }
        root.addView(scanView, FrameLayout.LayoutParams(-1, -1))

        statusText = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            gravity = Gravity.CENTER
            setBackgroundColor(Color.argb(140, 0, 0, 0))
            setPadding(dp(28f), dp(14f), dp(28f), dp(6f))
        }
        root.addView(statusText, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        progress = ProgressBar(this).apply { isIndeterminate = true; visibility = View.GONE }
        root.addView(progress, FrameLayout.LayoutParams(dp(56f), dp(56f), Gravity.CENTER))

        bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(18f), 0, dp(18f), dp(14f))
        }
        leftButton = makeButton("⚙") { toggleSettings() }
        mainButton = makeButton("Capture page") { onMain() }
        rightButton = makeButton("↺") { restart() }
        bottomBar.addView(leftButton, LinearLayout.LayoutParams(dp(48f), dp(48f)).apply { marginEnd = dp(6f) })
        bottomBar.addView(mainButton, LinearLayout.LayoutParams(0, dp(48f), 1f))
        bottomBar.addView(rightButton, LinearLayout.LayoutParams(dp(48f), dp(48f)).apply { marginStart = dp(6f) })
        root.addView(bottomBar, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        // Answer panel
        answerPanel = ScrollView(this).apply { setBackgroundColor(Color.BLACK); visibility = View.GONE }
        val ansBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(26f), dp(26f), dp(26f), dp(30f))
        }
        answerText = TextView(this).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTextIsSelectable(false)
        }
        ansBox.addView(answerText)
        val ansButtons = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(14f), 0, 0) }
        ansButtons.addView(makeButton("🔊 Read aloud") { speak() }, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8f) })
        ansButtons.addView(makeButton("Show scan") { showScanFromAnswer() }, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8f) })
        ansButtons.addView(makeButton("New scan") { restart() }, LinearLayout.LayoutParams(-1, -2))
        ansBox.addView(ansButtons)
        answerPanel.addView(ansBox)
        root.addView(answerPanel, FrameLayout.LayoutParams(-1, -1))

        // Settings panel
        settingsPanel = ScrollView(this).apply { setBackgroundColor(Color.argb(245, 15, 15, 20)); visibility = View.GONE }
        settingsBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(26f), dp(24f), dp(26f), dp(30f))
        }
        settingsPanel.addView(settingsBox)
        root.addView(settingsPanel, FrameLayout.LayoutParams(-1, -1))

        setContentView(root)
    }

    private fun rebuildSettings() {
        settingsBox.removeAllViews()
        val lp = { LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8f) } }
        settingsBox.addView(TextView(this).apply {
            text = "Settings\nAPI key: " + if (apiKey.isEmpty()) "not set" else "set (…${apiKey.takeLast(4)})"
            setTextColor(Color.WHITE)
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, dp(10f))
        })
        settingsBox.addView(makeButton("Scan API key QR") { toggleSettings(); enterMode(Mode.KEY_QR) }, lp())
        settingsBox.addView(makeButton("Type API key") { typeKey() }, lp())
        val modelName = ClaudeClient.MODELS.firstOrNull { it.first == model }?.second ?: model
        settingsBox.addView(makeButton("Model: $modelName") {
            val i = ClaudeClient.MODELS.indexOfFirst { it.first == model }
            prefs.edit().putString("model", ClaudeClient.MODELS[(i + 1) % ClaudeClient.MODELS.size].first).apply()
            rebuildSettings()
        }, lp())
        settingsBox.addView(makeButton("Camera: " + if (useFront) "front" else "back") {
            prefs.edit().putBoolean("front", !useFront).apply(); rebuildSettings(); startCamera()
        }, lp())
        settingsBox.addView(makeButton("Detail: " + if (highRes) "high (slower)" else "normal") {
            prefs.edit().putBoolean("high_res", !highRes).apply(); rebuildSettings(); startCamera()
        }, lp())
        settingsBox.addView(makeButton("Flashlight: " + if (torch) "on" else "off") {
            prefs.edit().putBoolean("torch", !torch).apply(); rebuildSettings()
            val c = camera
            if (c != null && c.cameraInfo.hasFlashUnit()) c.cameraControl.enableTorch(torch)
            else if (torch) toast("This camera has no flashlight.")
        }, lp())
        settingsBox.addView(makeButton("Save scans: " + if (saveScans) "on" else "off") {
            prefs.edit().putBoolean("save", !saveScans).apply(); rebuildSettings()
        }, lp())
        settingsBox.addView(makeButton("Close") { toggleSettings() }, lp())
    }

    private fun toggleSettings() {
        if (settingsPanel.visibility == View.VISIBLE) {
            settingsPanel.visibility = View.GONE
        } else {
            rebuildSettings()
            settingsPanel.visibility = View.VISIBLE
        }
    }

    private fun typeKey() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
            hint = "sk-ant-…"
            setText(apiKey)
        }
        AlertDialog.Builder(this)
            .setTitle("Claude API key")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                prefs.edit().putString("api_key", input.text.toString().trim()).apply()
                rebuildSettings()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setStatus(s: String) = runOnUiThread { statusText.text = s }

    private fun toast(s: String) = runOnUiThread {
        android.widget.Toast.makeText(this, s, android.widget.Toast.LENGTH_SHORT).show()
    }

    /** Ask the camera to focus (and meter) on the centre of the view. */
    private fun focusCentre() {
        val c = camera ?: return
        if (!hasAf) return
        lastFocusMs = System.currentTimeMillis()
        try {
            val pt = SurfaceOrientedMeteringPointFactory(1f, 1f).createPoint(0.5f, 0.5f, 0.3f)
            val action = FocusMeteringAction.Builder(pt, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(3, TimeUnit.SECONDS)
                .build()
            c.cameraControl.startFocusAndMetering(action)
        } catch (_: Throwable) {
        }
    }

    private fun vibrate(ms: Long) {
        try {
            @Suppress("DEPRECATION")
            val v = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            if (Build.VERSION.SDK_INT >= 26) {
                v?.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                v?.vibrate(ms)
            }
        } catch (_: Exception) {
        }
    }

    // ---------------------------------------------------------------- modes

    private fun enterMode(m: Mode) {
        mode = m
        runOnUiThread {
            progress.visibility = if (m == Mode.STARTING || m == Mode.FINISHING) View.VISIBLE else View.GONE
            answerPanel.visibility = if (m == Mode.ANSWER) View.VISIBLE else View.GONE
            bottomBar.visibility = if (m == Mode.ANSWER) View.GONE else View.VISIBLE
            statusText.visibility = if (m == Mode.ANSWER) View.GONE else View.VISIBLE
            mainButton.isEnabled = m == Mode.OVERVIEW || m == Mode.SWEEP || m == Mode.KEY_QR
            when (m) {
                Mode.KEY_QR -> {
                    scanView.setStill(null); scanView.clearOverlays()
                    mainButton.text = "Skip"
                    statusText.text = "Point the camera at your API-key QR code"
                }
                Mode.OVERVIEW -> {
                    scanView.setStill(null); scanView.clearOverlays()
                    mainButton.text = "Capture page"
                    statusText.text = "Step 1: fit the whole page in view, then tap Capture"
                }
                Mode.STARTING -> statusText.text = "Reading the page…"
                Mode.SWEEP -> {
                    mainButton.text = "Done"
                    statusText.text = "Step 2: move closer and sweep slowly over the page"
                }
                Mode.FINISHING -> mainButton.text = "…"
                Mode.ANSWER -> {}
            }
        }
    }

    private fun onMain() {
        when (mode) {
            Mode.KEY_QR -> enterMode(Mode.OVERVIEW)
            Mode.OVERVIEW -> capturePage()
            Mode.SWEEP -> finishScan()
            else -> {}
        }
    }

    private fun onBack() {
        when {
            settingsPanel.visibility == View.VISIBLE -> toggleSettings()
            mode == Mode.ANSWER || mode == Mode.SWEEP || mode == Mode.KEY_QR -> restart()
            mode == Mode.FINISHING && answerPanel.visibility == View.GONE && lastAnswer.isNotEmpty() -> enterMode(Mode.ANSWER)
            else -> finish()
        }
    }

    private fun restart() {
        tts?.stop()
        val e = engine
        engine = null
        workExec.execute { e?.release() }
        lastAnswer = ""
        enterMode(Mode.OVERVIEW)
    }

    private fun capturePage() {
        val grabbed: Pair<Mat, Array<Point>>? = synchronized(lock) {
            val f = lastFrame
            if (f == null) {
                null
            } else {
                val w = f.cols().toDouble()
                val h = f.rows().toDouble()
                Pair(
                    f.clone(),
                    lastQuad ?: arrayOf(
                        Point(w * 0.03, h * 0.03), Point(w * 0.97, h * 0.03),
                        Point(w * 0.97, h * 0.97), Point(w * 0.03, h * 0.97)
                    )
                )
            }
        }
        if (grabbed == null) return
        val (frame, quad) = grabbed
        vibrate(40)
        enterMode(Mode.STARTING)
        workExec.execute {
            try {
                val e = ScanEngine()
                e.start(frame, quad)
                engine = e
                runOnUiThread {
                    scanView.setOutline(null, Color.CYAN)
                    scanView.setGrid(IntArray(e.cols * e.rows), e.cols, e.rows, e.rh.toFloat() / e.rw, null)
                }
                enterMode(Mode.SWEEP)
            } catch (t: Throwable) {
                setStatus("Could not start: ${t.message}")
                enterMode(Mode.OVERVIEW)
            } finally {
                frame.release()
            }
        }
    }

    private fun finishScan() {
        val e = engine ?: return
        if (mode != Mode.SWEEP) return
        enterMode(Mode.FINISHING)
        setStatus("Building the scan…")
        workExec.execute {
            try {
                val result = e.result()
                // Preview of the finished scan
                val ps = min(1.0, 900.0 / max(result.cols(), result.rows()))
                val prev = Mat()
                Imgproc.resize(result, prev, CvSize(result.cols() * ps, result.rows() * ps), 0.0, 0.0, Imgproc.INTER_AREA)
                val bmp = Bitmap.createBitmap(prev.cols(), prev.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(prev, bmp)
                prev.release()
                runOnUiThread { scanView.clearOverlays(); scanView.setStill(bmp) }

                if (saveScans) {
                    try { saveJpeg(encodeJpeg(result, 92)) } catch (t: Throwable) { /* not fatal */ }
                }

                val key = apiKey
                if (key.isEmpty()) {
                    result.release()
                    showAnswer(
                        "Scan finished" + (if (saveScans) " and saved to Pictures/WatchScan." else ".") +
                            "\n\nTo get answers from Claude, add your API key in settings (⚙)."
                    )
                    return@execute
                }
                val closeUps = e.shots.sortedBy { it.cy * 10 + it.cx }.map { it.jpeg }
                val (tw, th) = sizeForClaude(result.cols(), result.rows())
                val small = Mat()
                Imgproc.resize(result, small, CvSize(tw.toDouble(), th.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                result.release()
                val jpeg = encodeJpeg(small, 88)
                small.release()
                val modelName = ClaudeClient.MODELS.firstOrNull { it.first == model }?.second ?: model
                setStatus("Asking Claude ($modelName)…")
                val answer = try {
                    ClaudeClient.ask(key, model, jpeg, closeUps)
                } catch (t: Throwable) {
                    "Couldn't get an answer.\n\n${t.message ?: t.javaClass.simpleName}\n\nCheck the watch's internet connection, then tap New scan to try again."
                }
                vibrate(120)
                showAnswer(answer)
            } catch (t: Throwable) {
                showAnswer("Something went wrong while building the scan: ${t.message}")
            }
        }
    }

    private fun showAnswer(text: String) {
        lastAnswer = text
        runOnUiThread { answerText.text = text; answerPanel.scrollTo(0, 0) }
        enterMode(Mode.ANSWER)
    }

    private fun showScanFromAnswer() {
        // Hide the answer to look at the finished scan; back returns to the answer.
        mode = Mode.FINISHING
        answerPanel.visibility = View.GONE
        progress.visibility = View.GONE
        statusText.visibility = View.VISIBLE
        statusText.text = "Your scan. Press back to return to the answer."
        bottomBar.visibility = View.GONE
    }

    private fun speak() {
        if (!ttsReady) { answerText.append("\n\n(Text-to-speech isn't available on this watch.)"); return }
        tts?.speak(lastAnswer, TextToSpeech.QUEUE_FLUSH, null, "answer")
    }

    private fun saveJpeg(bytes: ByteArray) {
        val name = "scan_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
        if (Build.VERSION.SDK_INT < 29) {
            // Older Android: keep scans in the app's own Pictures folder (no storage permission needed).
            val dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: return
            java.io.File(dir, name).writeBytes(bytes)
            return
        }
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/WatchScan")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
    }

    // ---------------------------------------------------------------- camera

    private fun startCamera() {
        if (!cvOk) return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                bindCamera(provider)
            } catch (t: Throwable) {
                setStatus("Camera error: ${t.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun bindCamera(provider: ProcessCameraProvider) {
        val target = if (highRes) Size(2560, 1920) else Size(1920, 1440)
        val wanted = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
        val other = if (useFront) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
        val camSel = when {
            safeHas(provider, wanted) -> wanted
            safeHas(provider, other) -> other
            else -> { setStatus("No camera found."); return }
        }
        // Read what this camera can do: autofocus, nearest focus distance, frame-rate ranges.
        val c2 = try {
            camSel.filter(provider.availableCameraInfos).firstOrNull()?.let { Camera2CameraInfo.from(it) }
        } catch (_: Throwable) { null }
        val afModes = try { c2?.getCameraCharacteristic(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) } catch (_: Throwable) { null } ?: IntArray(0)
        val minFocus = try { c2?.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) } catch (_: Throwable) { null } ?: 0f
        val fpsRanges = try { c2?.getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) } catch (_: Throwable) { null }
        hasAf = minFocus > 0f && (afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) || afModes.contains(CaptureRequest.CONTROL_AF_MODE_AUTO))
        continuousAf = afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        camDesc = if (hasAf) "autofocus, focuses from ${(100f / minFocus).toInt()} cm" else "fixed focus"

        val selector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(target, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
            .build()
        val builder = ImageAnalysis.Builder()
            .setResolutionSelector(selector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        val ext = Camera2Interop.Extender(builder)
        if (afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)) {
            ext.setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        }
        // Prefer a fast, fixed frame rate: it caps the exposure time, which cuts motion blur.
        val fast: Range<Int>? = fpsRanges?.filter { it.lower >= 20 }?.maxWithOrNull(compareBy<Range<Int>>({ it.lower }, { -it.upper }))
        if (fast != null) ext.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fast)
        val analysis = builder.build()
        analysis.setAnalyzer(analyzerExec) { img ->
            try { analyze(img) } catch (t: Throwable) { setStatus("Error: ${t.message}") } finally { img.close() }
        }
        provider.unbindAll()
        frameCount = 0
        val cam = provider.bindToLifecycle(this, camSel, analysis)
        camera = cam
        if (cam.cameraInfo.hasFlashUnit()) cam.cameraControl.enableTorch(torch)
    }

    private fun safeHas(p: ProcessCameraProvider, s: CameraSelector) = try { p.hasCamera(s) } catch (_: Exception) { false }

    private fun yPlane(img: ImageProxy): Mat {
        val plane = img.planes[0]
        val w = img.width
        val h = img.height
        val buf = plane.buffer
        val rs = plane.rowStride
        val need = w * h
        var arr = yBuf
        if (arr == null || arr.size != need) { arr = ByteArray(need); yBuf = arr }
        buf.rewind()
        if (rs == w && buf.remaining() >= need) {
            buf.get(arr, 0, need)
        } else {
            for (r in 0 until h) {
                buf.position(r * rs)
                buf.get(arr, r * w, w)
            }
        }
        val m = Mat(h, w, CvType.CV_8UC1)
        m.put(0, 0, arr)
        return m
    }

    private var frameCount = 0

    private fun analyze(img: ImageProxy) {
        val raw = yPlane(img)
        val up: Mat
        when (img.imageInfo.rotationDegrees) {
            90 -> { up = Mat(); Core.rotate(raw, up, Core.ROTATE_90_CLOCKWISE); raw.release() }
            180 -> { up = Mat(); Core.rotate(raw, up, Core.ROTATE_180); raw.release() }
            270 -> { up = Mat(); Core.rotate(raw, up, Core.ROTATE_90_COUNTERCLOCKWISE); raw.release() }
            else -> up = raw
        }
        frameCount++
        if (frameCount == 1) {
            camDesc = "${up.cols()}x${up.rows()}, " + camDesc
            if (mode == Mode.OVERVIEW) setStatus("Camera $camDesc\nFit the whole page in view, then tap Capture")
        }
        // Cameras without continuous autofocus need a nudge every few seconds.
        if (!continuousAf && (mode == Mode.SWEEP || mode == Mode.OVERVIEW) &&
            System.currentTimeMillis() - lastFocusMs > 2500) focusCentre()

        // Viewfinder
        val vw = max(160, min(480, scanView.width))
        val s = vw.toDouble() / max(up.cols(), up.rows())
        val small = Mat()
        Imgproc.resize(up, small, CvSize(up.cols() * s, up.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        val bmp = Bitmap.createBitmap(small.cols(), small.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(small, bmp)
        small.release()
        scanView.post { scanView.setFrame(bmp) }

        when (mode) {
            Mode.OVERVIEW -> {
                val q = PageDetector.detect(up)
                synchronized(lock) {
                    lastFrame?.release()
                    lastFrame = up
                    lastQuad = q
                }
                val norm = q?.let { qq ->
                    FloatArray(8) { i -> if (i % 2 == 0) (qq[i / 2].x / up.cols()).toFloat() else (qq[i / 2].y / up.rows()).toFloat() }
                }
                scanView.post { if (mode == Mode.OVERVIEW) scanView.setOutline(norm, Color.GREEN) }
                return // `up` is kept as lastFrame
            }
            Mode.SWEEP -> {
                val e = engine
                if (e != null && !busy.getAndSet(true)) {
                    val frame = up
                    workExec.execute {
                        try {
                            if (mode == Mode.SWEEP && engine === e) {
                                val info = e.process(frame)
                                onInfo(info)
                            }
                        } catch (t: Throwable) {
                            setStatus("Tracking error: ${t.message}")
                        } finally {
                            frame.release()
                            busy.set(false)
                        }
                    }
                    return // released by the worker
                }
            }
            Mode.KEY_QR -> {
                if (qrCounter++ % 3 == 0) readQr(up)
            }
            else -> {}
        }
        up.release()
    }

    private fun readQr(up: Mat) {
        val det = qrDetector ?: return
        val s = min(1.0, 1280.0 / max(up.cols(), up.rows()))
        val m = Mat()
        Imgproc.resize(up, m, CvSize(up.cols() * s, up.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        val text = try { det.detectAndDecode(m) ?: "" } catch (_: Throwable) { "" }
        m.release()
        if (text.isEmpty()) return
        val key = text.trim()
        if (key.startsWith("sk-ant-")) {
            prefs.edit().putString("api_key", key).apply()
            vibrate(150)
            enterMode(Mode.OVERVIEW)
            setStatus("API key saved ✓  Now fit a page in view and tap Capture")
        } else {
            setStatus("That QR code isn't a Claude API key (should start with sk-ant-)")
        }
    }

    private fun onInfo(info: ScanEngine.Info) {
        val pct = 100 * info.good / max(1, info.total)
        val hint = when {
            !info.ok -> "Lost the page — move back a little so more text is visible"
            info.f > ScanEngine.GOOD_F && hasAf -> "Move closer to the paper"
            info.f > ScanEngine.GOOD_F -> "Move a bit closer, but stop before the view gets blurry"
            !info.accepted -> "Hold steady…"
            else -> "Sweep over the yellow/grey areas"
        }
        val e = engine ?: return
        runOnUiThread {
            if (mode != Mode.SWEEP) return@runOnUiThread
            statusText.text = "$pct% captured\n$hint"
            scanView.setOutline(info.pageInFrame, Color.CYAN)
            scanView.setGrid(info.status, info.cols, info.rows, e.rh.toFloat() / e.rw, info.footprint)
            if (info.done) {
                vibrate(200)
                finishScan()
            }
        }
    }
}
