# Quick Start — no coding experience required

This guide gets you from zero to a working analysis using only your mouse.
(Everything here is also covered, with screenshots, in the step-by-step
protocol: https://dx.doi.org/10.17504/protocols.io.n92ld16znl5b)

## What this tool does (in one paragraph)

XRCT scans of rocks contain thousands of tiny segmented objects that are not
real grains — they are few-voxel artifacts that distort any fabric (SPO)
measurement. This tool uses a pre-trained machine-learning classifier to find
an **objective minimum grain-volume threshold (Vmin)** for each sample, filters
the artifacts out, and then computes the mean fabric tensor and the fabric
parameters **P′** (anisotropy) and **T** (shape), with bootstrap confidence
intervals.

## Step 1 — Install Python (once, ~5 minutes)

1. Go to https://www.python.org/downloads/ and download Python 3.8 or newer.
2. Run the installer. **IMPORTANT: tick the box "Add python.exe to PATH"**
   on the first installer screen. Everything else: just click "Install".

## Step 2 — Download this repository (once)

- On the GitHub page, click the green **`<> Code`** button → **Download ZIP**.
- Unzip it anywhere (e.g. your Desktop).

## Step 3 — Launch the app (every time)

- **Windows**: open the unzipped folder and **double-click `run_app.bat`**.
  The first run installs the required libraries automatically (needs internet);
  after that it starts in seconds. A window titled
  **"ML Threshold Selection System"** will open.
- **macOS**: open the unzipped folder and **double-click `run_app.command`**.
  (The first time, macOS may block it: right-click `run_app.command` →
  **Open** → **Open**, once. After that a normal double-click works.)
  The first run installs the required libraries automatically (needs internet).
- **Linux / manual**: open a terminal in the unzipped folder and run:
  ```bash
  pip install -r requirements.txt && python3 main.py
  ```

## Step 4 — Analyse a sample (mouse only)

The buttons are numbered in the order you use them. For a typical analysis
with our pre-trained model you only need four clicks:

1. **Load Last Model** — loads the pre-trained classifier shipped in the
   `trained model` folder (no training needed).
2. **6a. Load Single Test Data** (or **6b. Load Multi Test Data**) — select
   your sample table(s) (`total<Sample>.xlsx`; see Step 5 below). You will be
   asked for the voxel size of each sample in mm (e.g. 0.03).
3. **7. Predict Analysis** — computes the objective loose/strict thresholds.
4. **Mean Fabric** and **Fabric Boxplots** — compute the fabric tensor, P′ and
   T with bootstrap confidence intervals, and save publication-quality figures
   and `.txt` results into the `outputs` folder.

(To retrain the classifier on your own samples instead, follow buttons
1 → 2 → 3 → 5 as described in the protocol, Steps 52–55.)

## Step 5 — Prepare your own data (from Avizo)

Your Avizo *Label-Analysis* export cannot be used directly — two double-click
tools convert it:

1. **`tools/run_BatchFile`** (`.bat` on Windows, `.command` on macOS) — select
   your raw Avizo CSV export(s); it produces `total<Sample>.xlsx` (the app's
   input). Enter a volume threshold of 0 when asked, unless you already know one.
2. **`tools/run_To_tomofab`** (`.bat` on Windows, `.command` on macOS) — only
   needed for training samples that go to TomoFab; produces `TT_total<Sample>.xls`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Double-clicking does nothing / window flashes | Install Python from python.org and tick **"Add python.exe to PATH"**, then retry |
| "Python was not found" | Same as above |
| Window does not open, Tkinter error | Your Python lacks Tkinter. Use the python.org installer (includes it), or in Anaconda: `conda install tk` |
| Library installation fails | Check your internet connection / proxy, then double-click again |
| "Missing required columns" | Your table must contain `Volume3d (mm^3) `, `EigenVal1-3`, `EigenVec1-3 X/Y/Z` (note the trailing space in the volume header — keep the original Avizo headers, or use `tools/run_BatchFile.bat`) |

Still stuck? Open an issue on GitHub, or check `docs/user_guide.md`.
