import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
from .team import TeamClassifier
from collections import deque, Counter


class TeamAssigner:
    def __init__(self, device='cpu', batch_size=32):
        self.team_classifier = TeamClassifier(device=device, batch_size=batch_size)


    def collect_crops_from_tracks(self, tracks, video_frames, read_from_stub, stub_path):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading crops and player info from {stub_path}...")
            with open(stub_path, 'rb') as f:
                fitting_crops, all_crops, player_info = pickle.load(f)
            return fitting_crops, all_crops, player_info

        # Otherwise, compute crops fresh
        all_crops = []
        fitting_crops = []
        player_info = []

        for frame_num, player_track in enumerate(tracks['players']):
            frame = video_frames[frame_num]
            for player_id, track in player_track.items():
                bbox = track['bbox']
                crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                all_crops.append(crop)
                player_info.append((frame_num, player_id))

                if frame_num % 30 == 0:  # fitting crop sampling
                    fitting_crops.append(crop)

        # Save the results for next time
        if stub_path is not None:
            print(f"Saving crops and player info to {stub_path}...")
            with open(stub_path, 'wb') as f:
                pickle.dump((fitting_crops, all_crops, player_info), f)

        return fitting_crops, all_crops, player_info


    def assign_teams(self, tracks, video_frames, read_from_stub=False, stub_path=None):
        # 1. Collect crops
        fitting_crops, all_crops, player_info = self.collect_crops_from_tracks(
            tracks, video_frames, read_from_stub, stub_path
        )

        # 2. Fit the team classifier using only fitting_crops
        self.team_classifier.fit(fitting_crops)

        # 3. Predict on all crops
        team_ids = self.team_classifier.predict(all_crops)

        team_colors = {
            0: (0, 191, 255),  # DeepSkyBlue
            1: (255, 20, 147)  # DeepPink
        }

        history = {} #amazing idea have to write ab it

        # 4. Assign predicted teams back to all players
        for (frame_num, player_id), raw_team in zip(player_info, team_ids):
            buf = history.setdefault(player_id, deque(maxlen=3))
            buf.append(raw_team)

            # if we have at least 3 entries, majority vote; else just use raw_team
            if len(buf) == 3:
                team = Counter(buf).most_common(1)[0][0]
            else:
                team = raw_team

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_colors[team]
        
        num_frames = len(tracks['players'])
        for frame_num in range(num_frames):
            # this is a dict: { player_id: info_dict, … }
            players_dict = tracks['players'][frame_num]
            goalkeepers_dict = tracks['goalkeepers'][frame_num]

            # 1) Extract bottom-center anchors and group players by their assigned team
            team_anchors = {0: [], 1: []}
            for pid, info in players_dict.items():
                x1, y1, x2, y2 = info['bbox']
                xc = (x1 + x2) / 2
                yb = y2
                team = info.get('team', None)
                if team in (0, 1):
                    team_anchors[team].append((xc, yb))

            # If we don’t have at least one anchor in each team, skip keeper assignment
            if not team_anchors[0] or not team_anchors[1]:
                continue

            # 2) Compute each team’s centroid
            cent0 = np.mean(team_anchors[0], axis=0)
            cent1 = np.mean(team_anchors[1], axis=0)

            # 3) Assign each goalkeeper to the nearest centroid
            for gid, ginfo in goalkeepers_dict.items():
                x1, y1, x2, y2 = ginfo['bbox']
                xc = (x1 + x2) / 2
                yb = y2
                dist0 = np.hypot(xc - cent0[0], yb - cent0[1])
                dist1 = np.hypot(xc - cent1[0], yb - cent1[1])
                team_id = 0 if dist0 < dist1 else 1

                # Write back into your tracks
                ginfo['team'] = team_id
                ginfo['team_color'] = team_colors[team_id]

        

    
    