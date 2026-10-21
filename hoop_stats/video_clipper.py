"""
video_clipper.py
================
Stat-Linked Video Clip Generator & Registry.

Replicates and extends commercial features found in SportsVisio, Hudl Assist, and Veo:
- Links every box score statistic (FGM, 3PM, FGA, Rebound, Assist, Steal, Block,
  Traveling, Double Dribble, 3-Seconds, Backcourt Violation) to exact video timestamps.
- Exports a machine-readable Clip Manifest (JSON) and interactive HTML player showcase.
- Provides automated sub-clip extraction via OpenCV VideoWriter / FFmpeg.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import cv2


@dataclass
class StatClip:
    """Represents a discrete stat-linked video clip."""
    clip_id: str
    event_type: str
    player_id: Optional[int]
    jersey: Optional[str]
    team: str
    event_frame: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    description: str
    points: int = 0
    zone: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    clip_filename: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class StatClipRegistry:
    """
    Maintains the catalog of stat-linked video clips across a basketball game.
    Enables instant video review for any stat line in the box score.
    """

    def __init__(self, fps: float = 30.0, pre_pad_s: float = 2.5, post_pad_s: float = 2.0):
        self.fps = fps
        self.pre_pad_s = pre_pad_s
        self.post_pad_s = post_pad_s
        self.pre_pad_frames = int(fps * pre_pad_s)
        self.post_pad_frames = int(fps * post_pad_s)
        self.clips: List[StatClip] = []
        self._clip_counter: int = 1

    def add_event(self, event: dict, player_registry=None, total_frames: int = 99999999) -> StatClip:
        """
        Converts a detected event (shot, turnover, steal, assist, block) into a stat-linked clip.
        """
        event_frame = event.get("frame", 0)
        start_frame = max(0, event_frame - self.pre_pad_frames)
        end_frame = min(total_frames, event_frame + self.post_pad_frames)

        start_time_s = round(start_frame / self.fps, 2)
        end_time_s = round(end_frame / self.fps, 2)
        duration_s = round((end_frame - start_frame) / self.fps, 2)

        evt_type = event.get("type", "UNKNOWN")
        pid = event.get("player_id") or event.get("shooter_id") or event.get("passer_id") or event.get("blocker_id")
        team = event.get("team") or event.get("passer_team") or event.get("blocker_team") or "unknown"
        pts = event.get("points", 0)
        zone = event.get("zone")

        jersey = "Unknown"
        if player_registry and pid is not None:
            if pid in player_registry.track_to_jersey:
                jersey = player_registry.track_to_jersey[pid]

        clip_id = f"clip_{self._clip_counter:04d}_{evt_type.lower()}"
        self._clip_counter += 1

        clip_filename = f"{clip_id}_j{jersey}_{start_frame}_{end_frame}.mp4"

        desc = event.get("description", f"{evt_type} at {start_time_s}s")

        clip = StatClip(
            clip_id=clip_id,
            event_type=evt_type,
            player_id=pid,
            jersey=jersey,
            team=team,
            event_frame=event_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            duration_s=duration_s,
            description=desc,
            points=pts,
            zone=zone,
            metadata=event,
            clip_filename=clip_filename,
        )
        self.clips.append(clip)
        return clip

    def get_clips_for_player(self, jersey: str) -> List[StatClip]:
        """Returns all clips associated with a specific jersey number."""
        return [c for c in self.clips if str(c.jersey) == str(jersey)]

    def get_clips_by_event(self, event_type: str) -> List[StatClip]:
        """Returns all clips matching an event type (e.g. 'SHOT_MAKE', 'TRAVELING')."""
        return [c for c in self.clips if c.event_type.upper() == event_type.upper()]

    def to_manifest(self) -> List[dict]:
        return [c.to_dict() for c in self.clips]

    def export_manifest(self, output_path: str):
        """Exports JSON clip manifest."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"total_clips": len(self.clips), "clips": self.to_manifest()}, f, indent=2)

    def extract_clips(self, video_path: str, output_dir: str, max_clips: Optional[int] = None) -> List[str]:
        """
        Extracts individual MP4 video clips from the source game video.
        """
        if not os.path.exists(video_path):
            return []

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        exported_paths = []
        target_clips = self.clips[:max_clips] if max_clips else self.clips

        for clip in target_clips:
            out_file = os.path.join(output_dir, clip.clip_filename or f"{clip.clip_id}.mp4")
            out_writer = cv2.VideoWriter(out_file, fourcc, fps, (fw, fh))

            cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
            current_f = clip.start_frame

            while current_f <= clip.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                # Overlay brief indicator banner
                cv2.rectangle(frame, (10, 10), (520, 50), (0, 0, 0), -1)
                label = f"{clip.event_type} | #{clip.jersey} ({clip.team}) | {clip.duration_s}s"
                cv2.putText(frame, label, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
                out_writer.write(frame)
                current_f += 1

            out_writer.release()
            exported_paths.append(out_file)

        cap.release()
        return exported_paths
