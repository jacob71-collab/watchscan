# WatchScan

Android app for a camera smartwatch (built for the VP39, Android 11). It builds a sharp scan of a paper page from many close-up camera frames, then sends the page to Claude and shows the answer to the question on it.

## Using it

1. **First launch:** point the camera at a QR code of your Claude API key (or tap **Skip** and add it later under ⚙ → *Type API key*).
2. Hold the watch close over one corner of the page and tap **Start**.
3. Sweep slowly across the page in an S pattern. The preview on the right shows the page building up; the watch ticks each time a new piece is added. A red outline means it lost its place: move back over a part that is already in the preview.
4. When the whole page is in the preview, tap **Done**.
5. The paper is found and straightened, saved to *Pictures/WatchScan*, and sent to Claude (with a few of the sharpest close-ups). The answer appears on screen; **Read aloud** speaks it.

Tips: good light helps most. Keep roughly the same distance the whole time, and overlap each pass of the S with the previous one.

## Settings (⚙)

- **Model:** Sonnet 5.5 (fast, default) or Opus 5.5 (stronger on hard questions).
- **Camera:** back/front, for watches with a rotating camera.
- **Detail:** normal (1920×1440 frames) or high (2560×1920, slower).
- **Flashlight:** keeps the camera light on while scanning, if there is one.
- **Save test frames:** saves a zip of frames + a log to *Download/WatchScan* for troubleshooting.
- **Save scans:** keep a copy of each scan in the gallery.

## Building

Every push to `main` builds the APK on GitHub Actions and publishes it as the **latest** release (`WatchScan.apk`). The repo includes a fixed debug signing key so new builds install over old ones.
