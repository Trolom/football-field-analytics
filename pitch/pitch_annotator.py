import cv2
import numpy as np
import supervision as sv
from inference import get_model
from .homography import ViewTransformer  # wherever you defined it
from pitch import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram_2

class PitchAnnotator:
    def __init__(
        self,
        api_key: str,
        model_id: str,
        vertices: np.ndarray,
        edges: list[tuple[int, int]],
        conf: float = 0.3,
    ):
        # load the Roboflow/inference pitch model
        self.model  = get_model(model_id=model_id, api_key=api_key)
        self.conf   = conf

        # static pitch schema (in model coordinates)
        self.vertices = np.array(vertices, dtype=np.float32)  # shape (V,2)
        self.edges    = edges

        # Supervision annotators for vertices and edges
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=6
        )
        self.edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#000FFF'),
            thickness=2,
            edges=self.edges
        )

    def annotate_frames(self, frame) -> list[np.ndarray]:
        canvas = frame.copy()

        # 1) run inference via .infer()
        result = self.model.infer(frame, confidence=self.conf)[0]

        # 2) pull out keypoints
        kp = sv.KeyPoints.from_inference(result)
        if kp.xy is None or kp.confidence is None:
            return canvas

        all_pts     = kp.xy[0]         # (N,2)
        confidences = kp.confidence[0] # (N,)

        # 3) select only those points above your threshold
        mask   = confidences > 0.5
        src_pts = self.vertices[mask]  # model-space coords of detected vertices
        dst_pts = all_pts[mask]        # image-space coords of those same vertices

        # 4) if we have at least four, build & apply the ViewTransformer
        if src_pts.shape[0] >= 4:
            transformer = ViewTransformer(source=src_pts, target=dst_pts)

            # warp every vertex in your full CONFIG.vertices list
            frame_all_points = transformer.transform_points(self.vertices)

            # wrap as a Supervision KeyPoints to draw edges + dots
            kp_all = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

            canvas = self.edge_annotator.annotate(
                scene=canvas,
                key_points=kp_all
            )
            canvas = self.vertex_annotator.annotate(
                scene=canvas,
                key_points=kp_all
            )

        return canvas


    def annotate_tactical_board(
            self,
            frame,
            tracks: dict,
            frame_idx: int,
            CONFIG,
        ) -> np.ndarray:
        
        # 1) run inference via .infer()
        result = self.model.infer(frame, confidence=self.conf)[0]

        # 2) pull out keypoints
        key_points = sv.KeyPoints.from_inference(result)

        # 1) Filter homography reference points
        mask   = key_points.confidence[0] > 0.5
        src_pts = key_points.xy[0][mask]
        dst_pts = np.array(CONFIG.vertices)[mask]

        # 2) Build the homography transformer
        transformer = ViewTransformer(source=src_pts, target=dst_pts)

        # 3) Generic position transformer
        def transform_positions(track_dict):
            # track_dict: {id: {'position': (x,y), ...}, ...}
            pts = np.array([info['position'] for info in track_dict.values()])
            if pts.size:
                return transformer.transform_points(points=pts)
            return np.empty((0, 2))

        # 4) Transform each group
        ball_dict    = tracks['ball'][frame_idx]      # this is a dict
        player_dict  = tracks['players'][frame_idx]
        referee_dict = tracks['referees'][frame_idx]

        # 3) transform them into pitch‐space:
        pitch_ball    = transform_positions(ball_dict)
        pitch_players = transform_positions(player_dict)
        pitch_refs    = transform_positions(referee_dict)
        # 5) Draw the pitch
        board = draw_pitch(CONFIG)

        # --- Ball (always white) ---
        board = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=board
        )
        
        # --- Players (team colors) ---
        player_colors = [info['team_color'] for info in player_dict.values()]
        # Unique colors preserving order
        seen = {}
        for idx, color in enumerate(player_colors):
            seen.setdefault(color, []).append(idx)

        for team_color, indices in seen.items():
            pts = pitch_players[indices]
            b, g, r = team_color
            my_color = sv.Color(r=r, g=g, b=b)
            board = draw_points_on_pitch(
                config=CONFIG,
                xy=pts,
                face_color=my_color,
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=board
            )

        # --- Referees (static gold) ---
        board = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_refs,
            face_color=sv.Color.from_hex("FFD700"),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=board
        )

        return board
    
    def annotate_voronoi(
            self,
            frame,
            tracks: dict,
            frame_idx: int,
            CONFIG,
        ) -> np.ndarray:
        
        # 1) run inference via .infer()
        result = self.model.infer(frame, confidence=self.conf)[0]

        # 2) pull out keypoints
        key_points = sv.KeyPoints.from_inference(result)

        # 1) Filter homography reference points
        mask   = key_points.confidence[0] > 0.5
        src_pts = key_points.xy[0][mask]
        dst_pts = np.array(CONFIG.vertices)[mask]

        # 2) Build the homography transformer
        transformer = ViewTransformer(source=src_pts, target=dst_pts)

        # 3) Generic position transformer
        def transform_positions(track_dict):
            # track_dict: {id: {'position': (x,y), ...}, ...}
            pts = np.array([info['position'] for info in track_dict.values()])
            if pts.size:
                return transformer.transform_points(points=pts)
            return np.empty((0, 2))

        # 4) Transform each group
        ball_dict    = tracks['ball'][frame_idx]      # this is a dict
        player_dict  = tracks['players'][frame_idx]

        # 3) transform them into pitch‐space:
        pitch_ball    = transform_positions(ball_dict)
        pitch_players = transform_positions(player_dict)
        
        teams = np.array([info['team'] for info in player_dict.values()], dtype=int)
        team1_xy = pitch_players[teams == 0]
        team2_xy = pitch_players[teams == 1]

        board = draw_pitch_voronoi_diagram_2(
            config       = CONFIG,
            team_1_xy    = team1_xy,
            team_2_xy    = team2_xy,
            # you can override the below if you like:
            team_1_color = sv.Color.from_hex('00BFFF'),
            team_2_color = sv.Color.from_hex('FF1493'),
            opacity      = 0.5,
            padding      = 50,
            scale        = 0.1,
            pitch        = None  # let it call draw_pitch internally
        )

        return board