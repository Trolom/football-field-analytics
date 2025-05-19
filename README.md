# Football Field Analytics

An ongoing project for end-to-end football video analysis combining player and pitch tracking, team assignment, and event annotation.

## Features

Player Detection & TrackingUses Ultralytics YOLO + ByteTrack to detect players, goalkeepers, referees, and the ball across video frames.

Team AssignmentClusters player appearances (e.g. shirt colors) via CLIP embeddings to assign each player a stable team ID and color.

Pitch Reference PointsDetects key field landmarks (e.g., center circle, penalty spots) via a custom pitch-keypoint model (Roboflow/inference).

Annotation & VisualizationDraws ellipses and labels for tracked objects, triangles for ball control, and dots for pitch keypoints with OpenCV.

## Installation

1. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the player detection model

After installing the requirements, download the pretrained player detection model from Google Drive:
[player_detection.pt Download Link](https://drive.google.com/file/d/1zxooXx3Re91XmcZG8jX2Lj1h7Mkua077/view?usp=sharing)

Then create a folder named models in the project root and place best.pt inside:

```
mkdir models
mv /path/to/downloaded/player_detection.pt models/
```

4. Model training & preprocessing notebooks

The notebook used to train the player detection model is located in the development_and_analysis folder.
It was trained on Kaggle using their available GPUs.

Another notebook in the same folder is used for extracting player crops to assist in team assignment using appearance-based features.
Goalkeepers, however, are assigned using spatial centroids instead of visual features.

5. Using your own video
If you want to try the system on a different video:

Delete stub files from the stubs folder, these were used during development to speed up processing in notebook cells.

You are encouraged to use videos from this Kaggle dataset for better compatibility with the pipeline:
[⚽ DFL Bundesliga - 30s Clips Dataset](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv)