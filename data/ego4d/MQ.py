import json
import os.path as osp
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from dataclasses import dataclass

import pandas as pd

from data.ego4d.Ego4dDataset import BaseDataset, Features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Segment:
    video_start_frame: int
    video_end_frame: int
    
    label: str
    label_id: int

@dataclass
class Ego4dMQEntry:
    video_uid: str
    clip_uid: str

    video_start_frame: int
    video_end_frame: int
    
    segments: List[Segment]
    
    narrations: List[Tuple[float, str]]


class MQDataset(BaseDataset):
    """Moment Query dataset for Ego4D.

    NOTE: In ActionFormer they do NOT use the raw annotations. Instead, they convert the train and val json files
    using the tools/convert_ego4d_trainval.py script.
    """
    

    def __init__(self, pad_to: Optional[int] = 1024,
                 *args, **kwargs):
        # Initialize the dataset
        super().__init__('ego4d/mq', *args, **kwargs)
        logger.info("Initializing Ego4d MQ dataset with split v%d/%s using %s features.", self.version, self.split, self.features.name)
        
        # Pad graphs to a fixed size
        self.pad_to = pad_to

        labels_path = osp.join(self.raw_dir, "annotations", "mq_labels.tsv")
        labels = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])
        labels = labels.sort_values(by="id", ascending=True)
        self.labels = [x.strip() for x in labels.label.values]
        
        self.segments: List[Ego4dMQEntry] = self._load_annotations()
        self.segments = [segment for segment in self.segments if osp.exists(osp.join(self._features_path, f"{segment.video_uid}.pt"))]

        # Load the list of unique video ids
        self.video_uids = list(set((entry.video_uid for entry in self.segments)))
        self.clip_uids = list(set((entry.clip_uid for entry in self.segments)))
        
    def _load_annotations(self):
        # Load FHO annotations into a list of Ego4dFHOEntry objects
        annotations_path = osp.join(self.raw_dir, f"annotations/v{self.version}", f"moments_{self.split}.json")
        if not osp.exists(annotations_path):
            raise FileNotFoundError(f"Could not find the MQ annotations file {annotations_path}.")

        annotations = json.load(open(annotations_path, 'r'))

        path = osp.join(self.raw_dir, f"annotations/v{self.version}", "narration.json")
        if not osp.exists(path):
            raise FileNotFoundError(f"Could not find the narrations file {path}.")
        narrations = json.load(open(path, 'r'))

        mq_segments = []
        
        # for each video
        for video in annotations['videos']:
            
            # load all narrations corresponding to this video
            video_narrations = []
            video_narrations_timestamps = []
            for key, narration_pass in narrations[video['video_uid']].items():
                if not key.startswith('narration_pass'):
                    continue
                
                for narration in narration_pass['narrations']:
                    video_narrations.append(narration['narration_text'])
                    video_narrations_timestamps.append(narration['timestamp_frame'])
            video_narrations_timestamps = np.array(video_narrations_timestamps)
            
            # for each clip inside the video
            for clip in video['clips']:
                
                clip_segments: List[Segment] = []
                
                if 'test' not in self.split:
            
                    # for each annotator
                    for annotator in clip['annotations']:
                        
                        for label in annotator['labels']:
                            if not label['primary']:
                                continue
                            
                            if any(
                                s.video_start_frame == label['video_start_frame'] 
                                and s.video_end_frame == label['video_end_frame'] 
                                and s.label == label['label']
                                for s in clip_segments
                            ):
                                # duplicate segment
                                continue
                            
                            label['label'] = label['label'].lstrip('"').rstrip('"')
                            segment = Segment(
                                label['video_start_frame'], label['video_end_frame'],
                                label['label'], self.labels.index(label['label'])
                            )
                            clip_segments.append(segment)
                
                    if len(clip_segments) == 0:
                        logger.debug("No segments found for clip %s. Skipping.", clip['clip_uid'])
                        continue
                
                mq_entry = Ego4dMQEntry(
                    video['video_uid'], clip['clip_uid'], 
                    clip['video_start_frame'], clip['video_end_frame'], 
                    clip_segments,
                    narrations=[
                        (timestamp, narration) 
                        for timestamp, narration in zip(video_narrations_timestamps, video_narrations)
                        if clip['video_start_frame'] <= timestamp <= clip['video_end_frame']
                    ]
                )
                mq_segments.append(mq_entry)
                
        return mq_segments

    @property
    def class_labels(self) -> List[str]:
        return self.labels

    def len(self) -> int:
        return len(self.clip_uids)

    def get(self, idx):
        clip: Ego4dMQEntry = self.segments[idx]

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
        if len(clip.segments):
            start = torch.Tensor([segment.video_start_frame for segment in clip.segments])
            start = (start - clip.video_start_frame)
            
            end = torch.Tensor([segment.video_end_frame for segment in clip.segments])
            end = (end - clip.video_start_frame)
            
            segments = torch.stack([start, end], dim=-1)
            # Add a residual with respect to the features grid
            segments = segments + res_frame_start
            segments = segments / self.features.fps
            assert torch.all(segments[:, 1] > segments[:, 0]), "AFAIK time has only one direction"
            
            labels = torch.Tensor([segment.label_id for segment in clip.segments]).long()
    
            # raw narrations timestamps
            narrations = [narration for _, narration in clip.narrations]
            narration_timestamps = torch.tensor([(timestamp - clip.video_start_frame + res_frame_start) / self.features.fps for timestamp, _ in clip.narrations])

        else:
            segments, labels, narrations, narration_timestamps = None, None, None, None
            
        # pad features to a fixed size length
        if self.pad_to is not None:
            mask = torch.cat((torch.ones(len(feats), dtype=torch.bool), torch.zeros(self.pad_to - len(feats), dtype=torch.bool)))
            feats = torch.cat((feats, torch.zeros(1024 - feats.shape[0], feats.shape[1])))
        else:
            mask = torch.ones(len(feats), dtype=torch.bool)
            
        clip_length = (clip.video_end_frame - clip.video_start_frame) / self.features.fps

        return Data(
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
            narration_timestamps=narration_timestamps,
            # coverage with respect to the video
            coverage=((segments[:, 1] - segments[:, 0]) / clip_length) if segments is not None else None,
            length=segments[:, 1] - segments[:, 0] if segments is not None else None,
            num_instances=len(segments) if segments is not None else None
        )
        
    def get_by_clip_uid(self, uid: str) -> Optional[Data]:
        i = next((i for i, ann in enumerate(self.annotations) if ann.clip_uid == uid), -1)
        if i > -1:
            return self.get(i)
        return None


def iou_1d(intervals1, intervals2):
    """
    Computes Intersection over Union for multiple pairs of 1D intervals.

    Args:
        intervals1 (torch.Tensor): A tensor of shape (N, 2) representing the first set of intervals.
        intervals2 (torch.Tensor): A tensor of shape (N, 2) representing the second set of intervals.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the IoU for each pair of intervals.
    """

    start_intersections = torch.max(intervals1[:, 0], intervals2[:, 0])
    end_intersections = torch.min(intervals1[:, 1], intervals2[:, 1])
    intersection_lengths = torch.clamp(end_intersections - start_intersections, min=0)

    union_lengths = (intervals1[:, 1] - intervals1[:, 0]) + (intervals2[:, 1] - intervals2[:, 0]) - intersection_lengths
    ious = intersection_lengths / (union_lengths + 1e-7)
    return ious


if __name__ == "__main__":
    features: Features = Features(name='egovlp', size=256, stride=16, window=16, fps=30)
    dset = MQDataset(root="data/ego4d", split="test_unannotated", features=features, pad_to=None)
    pass
