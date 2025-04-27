import cv2
import numpy as np
import supervision as sv
from inference import get_model


class PitchAnnotator:
    def __init__(self, model_id, api_key, conf=0.3, color=(147, 20, 255), radius=6):
        self.model   = get_model(model_id=model_id, api_key=api_key)
        self.conf    = conf
        self.color   = color  # BGR
        self.radius  = radius

    def detect_keypoints_frame(self, frame, conf=None):
        """
        Run a single-frame inference via .infer(); return (M,2) array of points.
        """
        c = conf if conf is not None else self.conf

        # high-level .infer handles resizing, dtype, batch dim, etc.
        result = self.model.infer(frame, confidence=c)[0]

        # convert to Supervision KeyPoints & filter by confidence
        kp   = sv.KeyPoints.from_inference(result)
        mask = kp.confidence[0] > c
        pts  = kp.xy[0][mask]     # shape (M,2)

        return pts
        return pts

    def draw_points(self, frame, pts):
        # pts is an (M,2) numpy array of (x,y) coords
        for x,y in pts:
            cv2.circle(frame, (int(x),int(y)), self.radius, self.color, -1)
        return frame
