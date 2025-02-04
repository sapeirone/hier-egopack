import json
import os.path as osp
import logging
from typing import List, Optional, Tuple, Dict

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
    
    label: int


@dataclass
class Clip:
    uid: str
    video_uid: str
    clip_uid: str

    video_start_frame: int
    video_end_frame: int
    
    samples: List[Sample]
    
    narrations: List[Tuple[float, str]]


class OSCCDataset(BaseDataset):
    """OSCC dataset for the Ego4D dataset."""

    def __init__(self, pad_to: Optional[int] = 1024, *args, **kwargs):
        
        # Initialize the dataset
        super().__init__('ego4d/oscc', *args, **kwargs)
        
        logger.info(f"Initializing Ego4d OSCC dataset with split v{self.version}/{self.split} using {self.features.name} features.")
        
        # Pad graphs to a fixed size
        self.pad_to = pad_to
        
        self.segments: List[Clip] = self._load_annotations()
        self.segments = [segment for segment in self.segments if osp.exists(osp.join(self._features_path, f"{segment.video_uid}.pt"))]
        
    def _load_annotations(self) -> List[Clip]:
        annotations_path = osp.join(self.raw_dir, f"annotations/v{self.version}", f"fho_oscc-pnr_{self.split}.json")
        annotations = json.load(open(annotations_path, 'r'))

        self.clips: Dict[str, Clip] = {}
        for sample in annotations['clips']:
            uid = sample['unique_id']
            video_uid = sample['video_uid']
            
            # clip_uid = sample.get('clip_uid', uid) or uid  # use the clip_uid if both present and not None
            clip_uid = sample['clip_id']
            
            clip = self.clips.get(clip_uid, Clip(uid, video_uid, clip_uid, 0, 0, [], []))
            self.clips[clip_uid] = clip
            
            clip.samples.append(Sample(
                sample['parent_start_frame'], sample['parent_end_frame'],
                int(sample['state_change']) if 'state_change' in sample else 0
            ))

        narrations: Dict[str, List[Tuple[float, str]]] = self._load_narrations()

        for clip_uid, clip in self.clips.items():
            # Compute start and end frames for the clip based on the available annotations
            clip.video_start_frame = min([action.video_start_frame for action in clip.samples])
            clip.video_end_frame = max([action.video_end_frame for action in clip.samples])
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
        video_features = torch.load(osp.join(self._features_path, f"{clip.video_uid}.pt"))

        # take only the features that are part of the clip
        feat_start, feat_end = clip.video_start_frame // self.features.stride, clip.video_end_frame // self.features.stride
        res_frame_start = clip.video_start_frame % self.features.stride
        feat_start = max(0, feat_start)
        feat_end = max(feat_start, feat_end)
        feats = video_features[feat_start:feat_end, :]

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if len(clip.samples):
            start = torch.Tensor([segment.video_start_frame for segment in clip.samples])
            start = (start - clip.video_start_frame)
            
            end = torch.Tensor([segment.video_end_frame for segment in clip.samples])
            end = (end - clip.video_start_frame)
            
            segments = torch.stack([start, end], dim=-1)
            segments = segments + res_frame_start
            segments = segments / self.features.fps
                
            # raw narrations timestamps
            narrations = [narration for _, narration in clip.narrations]
            narration_timestamps = torch.tensor([(timestamp - clip.video_start_frame + res_frame_start) / self.features.fps for timestamp, _ in clip.narrations])

            labels = torch.tensor([segment.label for segment in clip.samples]).long()
        else:
            segments, labels, narrations, narration_timestamps = None, None, None, None
            
        # pad features to a fixed size length
        if self.pad_to is not None:
            mask = torch.cat((torch.ones(len(feats), dtype=torch.bool), torch.zeros(self.pad_to - len(feats), dtype=torch.bool)))
            feats = torch.cat((feats, torch.zeros(1024 - feats.shape[0], feats.shape[1])))
        else:
            mask = torch.ones(len(feats), dtype=torch.bool)

        return Data(
            uid=clip.uid,
            video_uid=clip.video_uid,
            clip_uid=clip.clip_uid,
            x=feats.unsqueeze(1),
            mask=mask,
            indices=torch.arange(len(feats)),
            # index of the nodes in the graph
            pos=(0.5 + torch.arange(len(feats))) * self.features.stride / self.features.fps,
            # action labels, start and end timestamps
            labels=labels,
            segments=segments,
            # Stride between feature vectors
            feat_stride=self.features.stride,
            # Number of frames / feature vector
            feat_num_frames=self.features.window,
            # Video info (FPS and duration in seconds)
            fps=self.features.fps, duration=(len(feats) * self.features.stride) / self.features.fps,
            # Narrations
            narrations=narrations,
            narration_timestamps=narration_timestamps
        )


if __name__ == "__main__":
    features: Features = Features(name='egovlp', size=256, stride=16, window=16, fps=30)
    dset = OSCCDataset(root="data/ego4d", split="train", features=features, pad_to=None)
    breakpoint()
    pass
