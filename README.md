# AI-Powered PPE & Restricted-Zone Worker Safety Monitoring

## Problem
Automated PPE-compliance and restricted-zone monitoring for construction sites using computer vision.

## Architecture
Custom PPE Detection → PPE-to-Worker Association → Worker Tracking → Temporal PPE State → Pose-Assisted Hazard-Zone Monitoring → Alert Logic → Annotated Video Output

## Dataset
Roboflow **Construction Site Safety** (`abdullah-alsehli/construction-site-safety-f0uho`, v1), with **717 source images**.

After filtering to the five project classes, the dataset was rebuilt with a leakage-aware, group-aware split:

- Train: **573 images**
- Validation: **72 images**
- Test: **72 images**

## Classes

| ID | Class |
|---:|---|
| 0 | Person |
| 1 | Hardhat |
| 2 | NO-Hardhat |
| 3 | Safety Vest |
| 4 | NO-Safety Vest |

## Training

- Base model: **YOLO11n (`yolo11n.pt`)**
- Epochs: **100**
- Image size: **640**
- Patience: **20**
- Seed: **42**
- Framework: **Ultralytics**

The training curves showed late-stage overfitting: training loss continued to decrease while validation loss plateaued. For this reason, the project uses the best validation checkpoint (`best.pt`) rather than the final-epoch weights.

## Validation Results

| Metric | Value |
|---|---:|
| Precision | **0.8445** |
| Recall | **0.5868** |
| mAP50 | **0.6619** |
| mAP50-95 | **0.4035** |

Per-class validation results showed the strongest mAP50 for **NO-Safety Vest (0.731)** and **Person (0.709)**, while **Safety Vest (0.570)** was the most challenging class and had the highest false-negative tendency.

## Final Test Results

The final test split was kept separate from validation/tuning.

| Metric | Value |
|---|---:|
| Precision | **0.783** |
| Recall | **0.525** |
| mAP50 | **0.499** |
| mAP50-95 | **0.289** |

Locked thresholds used for final evaluation:

- Confidence threshold: **0.30**
- IoU threshold: **0.70**

## Tracking & PPE Association

`model.track(..., persist=True)` with ByteTrack is used for persistent worker tracking. PPE detections are associated with workers using simple geometric head/torso regions rather than appearance-based Re-ID.

Worker PPE state is represented as:

- **Compliant**
- **Violation**
- **Unknown**

A violation requires explicit `NO-Hardhat` or `NO-Safety Vest` evidence; missing PPE detections alone are not treated as violations.

## Pose Usage

A pretrained Ultralytics pose model provides ankle keypoints to estimate each worker's ground-contact point. When ankle keypoints are unreliable, the system falls back to the person's bounding-box bottom-center.

Pose is used as the project's additional computer-vision task beyond object detection. Fall detection is intentionally out of scope.

## Hazard Zone

A manually defined polygon represents the restricted area. An alert is triggered only when both conditions are true:

1. the worker is inside the restricted zone, and
2. the worker has a confirmed PPE violation.

## Video Analytics

The final OpenCV pipeline performs:

`capture → detect/track → associate PPE → determine PPE state → pose-assisted ground point → zone check → annotate → write video`

The notebook generates a browser-playable H.264 annotated output video (`ppe_safety_output.mp4`).

## Deployment

The trained detector is exported to **ONNX** using Ultralytics `model.export(format="onnx")`. ONNX was selected as a portable optimized deployment format that can be used across multiple runtimes and devices.

Generated weights and ONNX artifacts are intentionally excluded from Git through `.gitignore`.

## How to Run

1. Open `ppe_safety_monitoring_capstone.ipynb` in Google Colab.
2. Select a GPU runtime when available.
3. Add `ROBOFLOW_API_KEY` to **Colab Secrets**. Never hard-code the key in the notebook.
4. Run the notebook cells in order.
5. Upload a demo video to `/content/demo_video.mp4` before the video-analytics section.
6. Run the tracking/video cells to generate the final annotated MP4.

Main dependencies include Ultralytics, Roboflow, OpenCV, Shapely, ImageHash, NumPy, pandas, and Matplotlib.

## Repository Contents

- `ppe_safety_monitoring_capstone.ipynb` — executed project notebook and captured evidence
- `README.md` — project documentation
- `.gitignore` — excludes secrets, datasets, runs, weights, and generated artifacts

## Limitations

- Tracker ID switches may occur during heavy occlusion.
- PPE annotation coverage in the source data is not exhaustive for every person.
- Safety Vest showed lower recall than the other classes on validation.
- The hazard-zone polygon is manually defined for the selected camera/video rather than automatically detected.
- The dataset is relatively small, so generalization to very different construction environments may require additional data.

## Training Program Attribution

Completed as the capstone project for the **Computer Vision for Developers with Ultralytics** program, **SDAIA Academy — August 2026 session**.

SDAIA Academy: https://github.com/SDAIAAcademy
