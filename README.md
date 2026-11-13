# WatchScan

Android app for a camera smartwatch (built for the VP39, Android 11). It builds a sharp scan of a paper page from many close-up camera frames, then sends the page to Claude and shows the answer to the question on it.

## Using it

1. **First launch:** point the camera at a QR code of your Claude API key (or tap **Skip** and add it later under ⚙ → *Type API key*).
2. **Step 1:** hold the watch back so the whole page fits. A green outline means the page was found. Tap **Capture page**.
3. **Step 2:** move closer (about 15–20 cm) and sweep slowly over the page. The mini-map on the right fills in:
   - green = captured sharply, dark green = blank paper (nothing to capture)
   - yellow = captured but too far away, grey = not yet captured
4. When the whole map is green it finishes on its own, or tap **Done** at any time.
5. The scan is saved to *Pictures/WatchScan* and sent to Claude. The answer appears on screen; **Read aloud** speaks it.

Tips: good light helps most. If it says *Lost the page*, back off a little so more text is in view.

## Settings (⚙)

- **Model:** Sonnet 5.5 (fast, default) or Opus 5.5 (stronger on hard questions).
- **Camera:** back/front, for watches with a rotating camera.
- **Detail:** normal (1920×1440 frames) or high (2560×1920, slower).
- **Save scans:** keep a copy of each scan in the gallery.

## Building

Every push to `main` builds the APK on GitHub Actions and publishes it as the **latest** release (`WatchScan.apk`). The repo includes a fixed debug signing key so new builds install over old ones.
