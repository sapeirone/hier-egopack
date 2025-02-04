import json
import os.path as osp
import logging
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

from dataclasses import dataclass

from typing import Dict


from data.ego4d.Ego4dDataset import BaseDataset, Features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Action:
    video_start_frame: int
    video_end_frame: int
    
    verb_label: str
    verb_label_id: int
    
    noun_label: str
    noun_label_id: int


@dataclass
class Clip:
    video_uid: str
    clip_uid: str

    video_start_frame: int
    video_end_frame: int
    
    actions: List[Action]
    
    narrations: List[Tuple[float, str]]


class ARDataset(BaseDataset):
    """Action Recognition (using LTA annotations) dataset for the Ego4D dataset."""
    
    def __init__(self, pad_to: Optional[int] = 1024,  num_verbs: int = 115, num_nouns: int = 478, *args, **kwargs):
        # Initialize the dataset
        super().__init__('ego4d/ar', *args, **kwargs)
        
        logger.info(f"Initializing Ego4d AR dataset with split v{self.version}/{self.split} using {self.features.name} features.")
        
        # Pad graphs to a fixed size
        self.pad_to = pad_to
        
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        
        # Load the verbs and nouns taxonomy
        self.verb_labels, self.noun_labels = self._load_fho_taxonomy()
        
        assert len(self.verb_labels) == self.num_verbs, "mismatch in number of verb labels and expected number of verbs"
        assert len(self.noun_labels) == self.num_nouns, "mismatch in number of noun labels and expected number of nouns"
        
        self.segments: List[Clip] = self._load_annotations()
        self.segments = [segment for segment in self.segments if osp.exists(osp.join(self._features_path, f"{segment.video_uid}.pt"))]
        
    def _load_annotations(self) -> List[Clip]:
        # Load FHO annotations into a list of Ego4dFHOEntry objects
        annotations_path = osp.join(self.raw_dir, f"annotations/v{self.version}", f"fho_lta_{self.split}.json")
        annotations = json.load(open(annotations_path, 'r'))

        self.clips: Dict[str, Clip] = {}
        for action in annotations['clips']:
            video_uid = action['video_uid']
            clip_uid = action['clip_uid']
            video_start_frame = action['clip_parent_start_frame']
            video_end_frame = action['clip_parent_end_frame']
            
            clip = self.clips.get(clip_uid, Clip(video_uid, clip_uid, video_start_frame, video_end_frame, [], []))
            self.clips[clip_uid] = clip
            
            clip.actions.append(Action(
                action['clip_parent_start_frame'] + action['action_clip_start_frame'],
                action['clip_parent_start_frame'] + action['action_clip_end_frame'],
                action['verb'], action['verb_label'], 
                action['noun'], action['noun_label']
            ))

        narrations: Dict[str, List[Tuple[float, str]]] = self._load_narrations()

        for clip_uid, clip in self.clips.items():
            clip.narrations += [
                (ts, text) for (ts, text) in narrations.get(clip.video_uid, []) 
                if clip.video_start_frame <= ts <= clip.video_end_frame
            ]
            
            clip.actions = list(sorted(clip.actions, key=lambda x: x.video_start_frame))
            clip.narrations = list(sorted(clip.narrations, key=lambda x: x[0]))
        
        return list(self.clips.values())

    @property
    def class_labels(self) -> Tuple[List[str], List[str]]:
        return (self.verb_labels, self.noun_labels)

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
        if len(clip.actions):
            start = torch.Tensor([segment.video_start_frame for segment in clip.actions])
            start = (start - clip.video_start_frame)
            
            end = torch.Tensor([segment.video_end_frame for segment in clip.actions])
            end = (end - clip.video_start_frame)
            
            segments = torch.stack([start, end], dim=-1)
            segments = segments + res_frame_start
            segments = segments / self.features.fps
            
            verb_labels = torch.Tensor([segment.verb_label_id for segment in clip.actions]).long()
            noun_labels = torch.Tensor([segment.noun_label_id for segment in clip.actions]).long()
                
            # raw narrations timestamps
            narrations = [narration for _, narration in clip.narrations]
            narration_timestamps = torch.tensor([(timestamp - clip.video_start_frame + res_frame_start) / self.features.fps for timestamp, _ in clip.narrations])

        else:
            segments, verb_labels, noun_labels, narrations, narration_timestamps = None, None, None, None, None
            
        # pad features to a fixed size length
        if self.pad_to is not None:
            mask = torch.cat((torch.ones(len(feats), dtype=torch.bool), torch.zeros(self.pad_to - len(feats), dtype=torch.bool)))
            feats = torch.cat((feats, torch.zeros(1024 - feats.shape[0], feats.shape[1])))
        else:
            mask = torch.ones(len(feats), dtype=torch.bool)

        return Data(
            video_uid=clip.video_uid,
            clip_uid=clip.clip_uid,
            x=feats.unsqueeze(1),
            mask=mask,
            indices=torch.arange(len(feats)),
            # index of the nodes in the graph
            pos=(0.5 + torch.arange(len(feats))) * self.features.stride / self.features.fps,
            # action labels, start and end timestamps
            verb_labels=verb_labels,
            noun_labels=noun_labels,
            labels=(verb_labels, noun_labels),
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
        
    def get_by_clip_uid(self, uid: str) -> Optional[Data]:
        i = next((i for i, ann in enumerate(self.segments) if ann.clip_uid == uid), -1)
        if i > -1:
            return self.get(i)
        return None


if __name__ == "__main__":
    features: Features = Features(name='egovlp', size=256, stride=16, window=16, fps=30)
    dset = ARDataset(root="data/ego4d", split="train", features=features, pad_to=None)
    pass
