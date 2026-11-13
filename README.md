# WatchScan

Android app for a camera smartwatch (built for the VP39, Android 11). It takes sharp close-up photos as you sweep over a paper page, joins them into one straightened scan on the watch, sends the scan to Claude, and shows the answer to the question on it.

## Using it

1. **First launch:** point the camera at a QR code of your Claude API key (or tap **Skip** and add it later under ⚙ → *Type API key*).
2. Hold the watch close over the top-left of the page and tap **Start**.
3. Sweep slowly across the page in an S (across, down a bit, back across), going a little past the paper's edges. About once a second the watch keeps the sharpest view as a photo and ticks. The preview on the right shows the photos being joined into one page; the status shows how many are joined. A red outline means the view is too blurry: slow down.
4. When the preview shows the whole page, tap **Done**.
5. The watch joins the photos into one sheet, finds the paper's edges, straightens it and evens the lighting. The sheet is saved to *Pictures/WatchScan* and sent to Claude (plus any photos that couldn't be joined). The answer appears on screen; **Read aloud** speaks it.

Tips: good light helps most. Keep the text crisp on screen, and overlap each pass of the S with the previous one.

## How the joining works

Each photo is matched against recent and nearby photos. A link between two photos is kept only if many matching points agree, they're spread across the photo, and the geometry is plausible for two handheld shots of a flat page. All links are then solved together, so a single bad link can't throw a photo off; links that disagree with the joint solution are dropped. Photos that can't be joined are left out rather than pasted in the wrong place.

## Settings (⚙)

- **Model:** Sonnet 5.5 (fast, default) or Opus 5.5 (stronger on hard questions).
- **Camera:** back/front, for watches with a rotating camera.
- **Detail:** normal (1920×1440 frames) or high (2560×1920, slower).
- **Flashlight:** keeps the camera light on while scanning, if there is one.
- **Save test frames:** saves a zip of the photos + a log to *Download/WatchScan* for troubleshooting.
- **Save scans:** keep a copy of each scan in the gallery.

## Building

Every push to `main` builds the APK on GitHub Actions and publishes it as the **latest** release (`WatchScan.apk`). The repo includes a fixed debug signing key so new builds install over old ones.
