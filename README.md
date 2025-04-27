# Football Field Analytics

An ongoing project for end-to-end football video analysis combining player and pitch tracking, team assignment, and event annotation.

## Features

Player Detection & TrackingUses Ultralytics YOLO + ByteTrack to detect players, goalkeepers, referees, and the ball across video frames.

Team AssignmentClusters player appearances (e.g. shirt colors) via CLIP embeddings to assign each player a stable team ID and color.

Pitch Reference PointsDetects key field landmarks (e.g., center circle, penalty spots) via a custom pitch-keypoint model (Roboflow/inference).

Annotation & VisualizationDraws ellipses and labels for tracked objects, triangles for ball control, and dots for pitch keypoints with OpenCV.

## Installation

1. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

2. Install dependencies:

pip install -r requirements.txt

Will be completed in the following weeks!!
