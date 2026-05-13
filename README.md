# Ruff Tracking in Multi-Camera Systems Using YOLO and SAM2

Automatic detection, tracking, and zone classification of ruffs (a shorebird species) from three-camera video recordings, using YOLOv9 and SAM2. Built for biological research at the Max Planck Institute for Biological Intelligence.

---

## What This Project Does

This system processes video footage of captive male ruffs recorded simultaneously from **three angles** — left side, right side, and top — and answers two questions:

1. **How many birds are in each zone at any given moment?**
2. **Which bird is in which zone?**

The pipeline has three stages:

```
Training data generation  →  YOLO model training  →  Position detection & zone classification
```

Each stage has its own graphical interface (built with Streamlit) and can be run independently.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Training Data](#1-generate-training-data)
  - [2. Train the YOLO Model](#2-train-the-yolo-model)
  - [3. Run Position Detection](#3-run-position-detection)
- [Output](#output)
- [Remote Access](#remote-access)
- [Technical Overview](#technical-overview)
- [Project Structure](#project-structure)

---

## Requirements

### Hardware
- A dedicated **GPU** is strongly recommended (24 GB VRAM or more for SAM2)
- Sufficient **SSD storage**: model files and dependencies take ~3.25 GB; each set of three 40-minute videos generates ~2.2 GB of output

### Software
- Ubuntu 22.04 LTS (server-side)
- [Apptainer](https://apptainer.org/) (for containerized execution)
- Google Chrome (for accessing the web UI)

---

## Installation

### 1. Extract the project

Download and extract the archive to any directory.

### 2. Start the Apptainer environment

Open **two terminals**.

**Terminal 1** — start the persistent Apptainer instance:

```bash
./launch_port8888.sh
```

Wait until you see output like:
```
Jupyter Server 2.14.2 is running at:
http://127.0.0.1:8888/tree?token=...
```

**Terminal 2** — enter the environment:

```bash
apptainer shell instance://sam2eszter
source ~/.envs/sam2eszterdevel2/bin/activate
```

Your prompt will change to `(sam2eszterdevel2) Apptainer>` when you're inside.

### 3. Install Decord (GPU-accelerated video decoder)

From the project root directory:

```bash
cd decord
cd build
cmake .. -DCUDA_ARCHITECTURES="75" -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make
cd ../python
pwd=$PWD
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All three modules are launched from the **project root directory**, with the Apptainer environment already running.

---

### 1. Generate Training Data

This step uses SAM2 to semi-automatically annotate birds in a video and export labeled frames for YOLO training.

**Start the UI:**

```bash
streamlit run train_gen/ui.py
```

Open `localhost:8501` in Chrome.

**Fill in the fields:**

| Field | Description |
|---|---|
| Video file path | Path to the input video |
| Start Time (HH:MM:SS) | Where to start processing |
| End Time (HH:MM:SS) | Where to stop processing |
| Max number of ruffs | Number of birds visible (1–7) |
| Project Directory | Where to save the generated data |

Click **OK**, then **click on each bird** in the preview image to annotate them. Select a label from the dropdown before each click. Place points close to the center of each bird's body.

Once all birds are marked, optionally check **"Create YAML file for training?"**, then click **Generate Data**. A progress bar tracks the frames being processed.

> SAM2 processes videos in chunks of 300 frames. Leave the browser tab open while it runs — the bar may pause between chunks while the next batch loads.

**Output directories shown on completion:**
- `labels/` — YOLO-format annotation `.txt` files
- `images/` — extracted video frames
- `overlay/` — visualizations of the generated masks (for quality checking)

---

### 2. Train the YOLO Model

Fine-tunes a YOLOv9 segmentation model on the data generated in step 1.

**Start the UI:**

```bash
streamlit run yolo_train/ui.py
```

Open `localhost:8501` in Chrome.

**Fill in the fields:**

| Field | Description |
|---|---|
| YAML path | Path to the `.yaml` file created in step 1 |
| Project Directory | Where to save the trained model |
| YOLO Model | Choose `yolov9c-seg.pt` (faster, less memory) or `yolov9e-seg.pt` (larger) |

Click **Start Training**. Training runs for 100 epochs and takes approximately 8 hours for ~4500 samples.

**Logs are saved to:**
- `output.txt` — training progress messages
- `errors.txt` — warnings and errors (note: most informational messages also go here due to how the trainer works)

**The trained model weights are saved at:**
```
<Project Directory>/train*/weights/best.pt
```

> Two separate models should be trained: one on side-view footage and one on top-view footage, since the birds look different from each angle.

---

### 3. Run Position Detection

Runs both YOLO models on all three camera feeds, synchronizes detections across views, computes real-world positions, and assigns birds to zones.

**Start the UI:**

```bash
streamlit run yolo_positions/ui.py
```

Open `localhost:8501` in Chrome.

**Fill in the fields (in the sidebar):**

| Field | Description |
|---|---|
| Right side video path | `.mp4` file for right-side camera |
| Left side video path | `.mp4` file for left-side camera |
| Top video path | `.mp4` file for top-view camera |
| Project Directory | Where results will be saved |
| YOLO side model path | `.pt` model trained on side-view data |
| YOLO top model path | `.pt` model trained on top-view data |

Click **Process Videos**. The system will generate background images and attempt to automatically detect the corner points of the lek (the birds' area).

**Corner point review:**

The detected corner points are shown for all three views. If detection failed for any view, an **Add points** button appears. You can also edit auto-detected points with **Edit points**. Coordinates are entered manually via number inputs that update a preview image.

Once you're happy with the points, click **Submit Points**. The system then runs automatically through:

- Loading and synchronizing the three video streams
- Running YOLO detection on each view
- Calculating real-world positions using homography (top view) and inverse projection (side views)
- Matching detections across the three views using the Hungarian algorithm
- Assigning each bird to a zone

A status list shows completed steps in real time. No further input is needed.

---

## Output

The final result is saved to:

```
<Project Directory>/coordinates/zones.txt
```

This file contains, **per video frame**, how many birds were found in each of the four zones.

The four zones are defined by distance from the females' enclosure window:
- **Zone 1**: 0–40 cm
- **Zone 2**: 40–140 cm
- **Zone 3**: 140–175 cm
- **Zone 4**: everything beyond

Additional files saved during processing:

```
<Project Directory>/
├── backgrounds/     # Background images and undistorted versions
├── camera/          # Extrinsic camera parameters (JSON)
├── coordinates/     # Position and zone text files
├── output/          # Annotated output videos (YOLO masks + bounding boxes)
├── sync/            # Frame synchronization results (JSON)
└── yolo_results/    # Raw YOLO predictions (Pickle files)
```

---

## Stopping the Application

1. Stop the Streamlit UI: press `Ctrl+C` in the terminal running it
2. Stop the Apptainer environment: press `Ctrl+C` twice in Terminal 1
3. Stop the Apptainer instance:

```bash
apptainer instance stop sam2eszter
```

---

## Remote Access

If the program runs on a remote server, forward the port to your local machine in a **third terminal**:

```bash
ssh -NTL 8500:127.0.0.1:8501 username@server-address
```

Keep this terminal open while using the app. Then open `localhost:8500` (or whichever local port you chose) in Chrome.

The port used by the Streamlit app can be changed by editing the launch script.

---

## Technical Overview

| Component | Technology |
|---|---|
| Bird detection & segmentation | YOLOv9c-seg (Ultralytics) |
| Training data generation | SAM2 (Meta AI) — modified for chunked, memory-efficient processing |
| 2D→3D position mapping | Homography (top view) + inverse projection with camera calibration (side views) |
| Multi-view matching | Hungarian algorithm (scipy) on Euclidean distances |
| Video decoding | Decord (GPU-accelerated) |
| UI | Streamlit |
| Container runtime | Apptainer |

**Performance on reference hardware:**
- SAM2 annotation: ~8.5 minutes per 300 frames (~12 seconds of video)
- YOLO training: ~8 hours for 100 epochs on ~4500 samples
- Full position detection pipeline: ~10 hours per 40-minute session (3 hours for YOLO inference, rest for matching and saving)

**Accuracy on a 40-minute test session (sampled every 30 seconds):**
- Precision (correct zone given detected): **95%**
- Recall (birds found and correctly placed): **82%**

---

## Common Error Messages

| Message | Cause | Fix |
|---|---|---|
| `Invalid time format` | Time not in HH:MM:SS format | Re-enter in correct format |
| `There are less frames than the specified end index` | End time exceeds video length | Use a shorter end time |
| `Start index is greater than end index` | Start time is after end time | Swap or correct the timestamps |
| `Start and end indices are the same` | Zero-length interval | Provide at least 1 second of range |
| `[...] path does not point to a file` | File not found at given path | Check the path and try again |
| `[...] path must be an .mp4 file` | Wrong video format | Convert or use an `.mp4` file |
| `[...] path must be a .pt file` | Wrong model format | Provide a valid PyTorch `.pt` model |

---

## Credits

Developed by **Molnár Eszter** as a BSc thesis at Eötvös Loránd University, Faculty of Informatics, Department of Artificial Intelligence, 2025.

Supervisor: Dr. Gelencsér-Horváth Anna

Research context: Max Planck Institute for Biological Intelligence — Behavioural Genetics and Evolutionary Ecology group

Parts of the camera synchronization and extrinsic parameter estimation code were adapted from Gergely Dinya's 2024 thesis. Background estimation code was adapted from Zsombor Fülöp's 2022 thesis.
