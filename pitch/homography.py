import cv2
import numpy as np

class ViewTransformer:
	def __init__(self, source: np.ndarray, target: np.ndarray):
		source = source.astype(np.float32)
		target = target.astype(np.float32)
		self.m, _ = cv2.findHomography(source, target)
		
	def transform_points(self, points: np.ndarray):
		points = points.reshape(-1, 1, 2).astype(np.float32)
		points = cv2.perspectiveTransform(points, self.m)
		return points.reshape(-1, 2).astype(np.float32)