import json
import os.path as osp
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from dataclasses import dataclass


from data.ego4d.Ego4dDataset import BaseDataset, Features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Sample:
    video_start_frame: int
    video_end_frame: int
    
    video_start_sec: float
    video_end_sec: float
    
    label: int
    video_pnr_frame: int


@dataclass
class Clip:
    uid: str
    video_uid: str
    clip_uid: str

    video_start_frame: int
    video_end_frame: int
    
    video_start_sec: float
    video_end_sec: float
    
    samples: List[Sample]
    
    narrations: List[Tuple[float, str]]


class PNRDataset(BaseDataset):
    """PNR dataset for the Ego4D dataset."""

    def __init__(self, pad_to: Optional[int] = 1024, *args, **kwargs):
        
        # Initialize the dataset
        super().__init__('ego4d/pnr', *args, **kwargs)
        
        logger.info(f"Initializing Ego4d PNR dataset with split v{self.version}/{self.split} using {self.features.name} features.")
        
        # Pad graphs to a fixed size
        self.pad_to = pad_to
        
        self.segments: List[Clip] = self._load_annotations()
        self.segments = [segment for segment in self.segments if osp.exists(osp.join(self._features_path, f"{segment.video_uid}.pt"))]
        
    def _load_annotations(self) -> List[Clip]:
        annotations_path = osp.join(self.raw_dir, f"annotations/v{self.version}", f"fho_oscc-pnr_{self.split}.json")
        annotations = json.load(open(annotations_path, 'r'))

        self.clips: Dict[str, Clip] = {}
        for i, sample in enumerate(annotations['clips']):
            uid = sample['unique_id']
            video_uid = sample['video_uid']
            
            if ('state_change' not in sample) or (sample['state_change'] == 0):
                # Skip samples without state changes
                continue
            
            # clip_uid = sample.get('clip_uid', uid) or uid  # use the clip_uid if both present and not None
            clip_uid = sample['clip_id'] + f"_{i}"
            
            clip = self.clips.get(clip_uid, Clip(uid, video_uid, clip_uid, 0, 0, 0, 0, [], []))
            self.clips[clip_uid] = clip
            
            clip.samples.append(Sample(
                sample['parent_start_frame'], sample['parent_end_frame'],
                sample['parent_start_sec'], sample['parent_end_sec'],
                int(sample['state_change']) if 'state_change' in sample else 0,
                sample['parent_pnr_frame']
            ))

        # Ignore clips without state changes
        self.clips = {clip_uid: clip for clip_uid, clip in self.clips.items() if len(clip.samples) > 0}

        narrations: Dict[str, List[Tuple[float, str]]] = self._load_narrations()

        for clip_uid, clip in self.clips.items():
            # Compute start and end frames for the clip based on the available annotations
            clip.video_start_frame = min([action.video_start_frame for action in clip.samples])
            clip.video_end_frame = max([action.video_end_frame for action in clip.samples])
            
            clip.video_start_sec = min([action.video_start_sec for action in clip.samples])
            clip.video_end_sec = max([action.video_end_sec for action in clip.samples])
            
            clip.narrations += [
                (ts, text) for (ts, text) in narrations.get(clip.video_uid, []) 
                if clip.video_start_frame <= ts <= clip.video_end_frame
            ]
            
            clip.samples = list(sorted(clip.samples, key=lambda x: x.video_start_frame))
            clip.narrations = list(sorted(clip.narrations, key=lambda x: x[0]))

        return list(self.clips.values())

    def len(self) -> int:
        return len(self.segments)

    def get(self, idx: int):
        clip: Clip = self.segments[idx]

        # load features
        video_features = torch.load(osp.join(self._features_path, f"{clip.video_uid}.pt"))#.numpy()
        start_frame, end_frame = clip.video_start_frame, clip.video_end_frame
        
        if self.split == "train":
            random_length_seconds = np.random.uniform(5, 8)
            random_start_seconds = clip.video_start_sec + np.random.uniform(8 - random_length_seconds)
            start_frame = np.floor(random_start_seconds * 30).astype(np.int32)
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > clip.video_end_sec:
                random_end_seconds = clip.video_end_sec
            end_frame = np.floor(random_end_seconds * 30).astype(np.int32)
            if clip.samples[0].video_pnr_frame > end_frame:
                end_frame = clip.video_end_frame
            if clip.samples[0].video_pnr_frame < start_frame:
                start_frame = clip.video_start_frame
        
        candidate_frames = np.linspace(start_frame, end_frame, num=16, dtype=int, endpoint=False)
        candidate_frames = np.clip(candidate_frames, start_frame, end_frame)
        
        feat_start, feat_end = start_frame // self.features.stride, end_frame // self.features.stride
        res_frame_start = start_frame % self.features.stride
        feat_start = max(0, feat_start)
        feat_end = max(feat_start, feat_end)
        
        feats = video_features[torch.linspace(feat_start, feat_end - 1, steps=16, dtype=int)]
        
        # feats = torch.from_numpy(feats).float()
        mask = torch.ones(feats.shape[0], dtype=torch.bool)
        
        labels = torch.zeros((feats.shape[0])).float()
        pnr_idx = None
        if len(clip.samples):
            distances = torch.from_numpy(np.abs(candidate_frames - clip.samples[0].video_pnr_frame)).long()
            labels[distances.argmin()] = 1.0
            pnr_idx = distances.argmin()
        
        positions = ((torch.arange(len(feats))) * self.features.stride) / self.features.fps #/ 2.0
        
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if len(clip.samples):
            start = torch.Tensor([segment.video_start_frame for segment in clip.samples])
            start = (start - start_frame)
            
            end = torch.Tensor([segment.video_end_frame for segment in clip.samples])
            end = (end - start_frame)
            
            segments = torch.stack([start, end], dim=-1)
            segments = segments + res_frame_start
            segments = segments / self.features.fps
            
            pnr_frames = torch.Tensor([segment.video_pnr_frame for segment in clip.samples])
            pnr_secs = (pnr_frames - start_frame) / self.features.fps
                
            # raw narrations timestamps
            narrations = [narration for _, narration in clip.narrations]
            narration_timestamps = torch.tensor([(timestamp - start_frame + res_frame_start) / self.features.fps for timestamp, _ in clip.narrations])
        else:
            segments, labels, narrations, narration_timestamps = None, None, None, None
            pnr_secs = None

        assert len(feats) == 16, f"Expected 16 frames, got {len(feats)}"

        return Data(
            uid=clip.uid,
            video_uid=clip.video_uid,
            clip_uid=clip.clip_uid,
            x=feats.unsqueeze(1),
            mask=mask,
            indices=torch.arange(len(feats)),
            # index of the nodes in the graph
            pos=positions,
            distances_from_pnr=distances,
            pnr_idx=pnr_idx,
            candidate_frames=torch.from_numpy(candidate_frames).unsqueeze(0),
            # action labels, start and end timestamps
            labels=labels,
            segments=segments,
            pnr_secs=pnr_secs,
            # Stride between feature vectors
            feat_stride=self.features.stride,
            # Number of frames / feature vector
            feat_num_frames=self.features.window,
            # Video info (FPS and duration in seconds)
            fps=self.features.fps, duration=(len(feats) * self.features.stride) / self.features.fps,
            # Narrations
            narrations=narrations,
            narration_timestamps=narration_timestamps,
            start_frame=clip.video_start_frame,
            end_frame=clip.video_end_frame,
            pnr_frame=clip.samples[0].video_pnr_frame
        )


if __name__ == "__main__":
    features: Features = Features(name='egovlp', size=256, stride=16, window=16, fps=30)
    dset = PNRDataset(root="data/ego4d", split="train", features=features, pad_to=None)
    dset[42]
    breakpoint()
    pass
