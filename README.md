# AI-Powered PPE & Restricted-Zone Worker Safety Monitoring

## Problem
Automated PPE-compliance and hazard-zone monitoring for construction sites via computer vision.

## Architecture
Custom PPE Detection -> PPE-to-Worker Association -> Worker Tracking -> Temporal PPE State -> Pose-Assisted Hazard-Zone Monitoring -> Alert Logic -> Annotated Video Output

## Dataset
Roboflow "Construction Site Safety" (abdullah-alsehli/construction-site-safety-f0uho, v1), 717 source images, group-aware leakage-checked ~80/10/10 split (train 573 / valid 72 / test 72).

## Classes
0 Person, 1 Hardhat, 2 NO-Hardhat, 3 Safety Vest, 4 NO-Safety Vest

## Training
Base model: yolo26n.pt | epochs (max): 100 | imgsz: 640 | patience: 20 | seed: 42

## Evaluation (test set, untouched until locked)
Precision 0.783 | Recall 0.525 | mAP50 0.499 | mAP50-95 0.289
Confidence threshold: 0.3 | IoU threshold: 0.7

## Tracking & PPE Association
`model.track(persist=True)` for persistent worker IDs; PPE boxes associated to workers via head/torso zone containment (no Re-ID). Compliant/Violation/Unknown state uses temporal smoothing; violation requires explicit NO-Hardhat/NO-Safety Vest evidence, never inferred from silence.

## Pose Usage
Pretrained pose model refines each worker's ground-contact point via ankle keypoints when confident, falling back to person-box bottom-center otherwise. No fall detection.

## Hazard Zone
One manually defined polygon; alert fires only when a confirmed Violation worker's ground-contact point falls inside it.

## Output
Annotated video saved and converted to browser-playable H.264 format. ONNX export at runs/detect/ppe_training/baseline/weights/best.onnx.

## How to Run
Open the notebook in Colab (GPU runtime), add ROBOFLOW_API_KEY as a Colab Secret, run top to bottom. Upload a demo clip to /content/demo_video.mp4 before the tracking section.

## Limitations
Tracker ID switches possible under heavy occlusion; PPE label coverage in the source data is not exhaustive per-person; hazard-zone polygon is fixed per video, not auto-detected.

## Attribution
Completed as the capstone project for the Computer Vision for Developers with Ultralytics program, SDAIA Academy. Reference: https://github.com/SDAIAAcademy
