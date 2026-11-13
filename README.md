# WatchScan

Android app for a camera smartwatch (built for the VP39, Android 11). It takes a series of sharp close-up photos as you sweep over a paper page, sends them to Claude, and shows the answer to the question on it.

## Using it

1. **First launch:** point the camera at a QR code of your Claude API key (or tap **Skip** and add it later under ⚙ → *Type API key*).
2. Hold the watch close over the top-left of the page and tap **Start**.
3. Sweep slowly across the page in an S (across, down a bit, back across). About once a second the watch keeps the sharpest view as a photo and ticks; the latest photo shows on the right. A red outline means the view is too blurry: slow down.
4. When you have covered the whole page, tap **Done**.
5. Up to 20 photos are sent to Claude, which pieces the page together and answers the question. **Read aloud** speaks the answer. A sheet of all the photos is saved to *Pictures/WatchScan*.

Cost: with Sonnet 5.5, 20 photos is roughly 50k input tokens, about $0.10 per question (about $0.20 with Opus 5.5).

Tips: good light helps most. Keep the text crisp on screen, and overlap each pass of the S with the previous one.

## Settings (⚙)

- **Model:** Sonnet 5.5 (fast, default) or Opus 5.5 (stronger on hard questions).
- **Camera:** back/front, for watches with a rotating camera.
- **Detail:** normal (1920×1440 frames) or high (2560×1920, slower).
- **Flashlight:** keeps the camera light on while scanning, if there is one.
- **Save test frames:** saves a zip of the photos + a log to *Download/WatchScan* for troubleshooting.
- **Save scans:** keep a copy of each scan in the gallery.

## Building

Every push to `main` builds the APK on GitHub Actions and publishes it as the **latest** release (`WatchScan.apk`). The repo includes a fixed debug signing key so new builds install over old ones.
