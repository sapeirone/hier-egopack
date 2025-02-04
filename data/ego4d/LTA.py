import json
import os.path as osp
import logging
from typing import List, Tuple

import torch
from torch_geometric.data import Data

from dataclasses import dataclass

from typing import Dict, Optional


from data.ego4d.Ego4dDataset import BaseDataset, Features

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class LTASegment:
    """An LTA segment, with an associated seen video segment and a list of actions to forecast."""
    video_uid: str
    clip_uid: str

    video_start_frame: int
    video_end_frame: int
    
    verb_labels: List[int]
    noun_labels: List[int]
    
    narrations: List[Tuple[float, str]]
    
    last_idx: int


@dataclass
class Action:
    """An action in the LTA dataset."""
    idx: int # action_idx
    
    video_start_frame: int
    video_end_frame: int
    
    verb_labels: Optional[int]
    noun_labels: Optional[int]


@dataclass
class Clip:
    """A clip in the LTA dataset, with a list of actions and textual narrations."""
    video_uid: str
    clip_uid: str
    
    samples: List[Action]
    
    narrations: List[Tuple[float, str]]


class LTADataset(BaseDataset):
    """LTA dataset for the Ego4d dataset."""
    
    def __init__(self, num_input_clips: int = 2, Z: int = 20, 
                 num_verbs: int = 115, num_nouns: int = 478, *args, **kwargs):
        """Initialize the Ego4d LTA dataset.

        Parameters
        ----------
        num_input_clips : int, optional
            number of input clips for LTA, by default 2
        Z : int, optional
            number of actions to forecast, by default 20
        """
        # Initialize the dataset
        super().__init__('ego4d/lta', *args, **kwargs)
        
        logger.info(f"Initializing Ego4d LTA dataset with split v{self.version}/{self.split} using {self.features.name} features.")
        
        self.num_input_clips = num_input_clips
        self.Z = Z
        
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        
        # Load LTA segments
        self.segments: List[LTASegment] = self._load_segments()
        # and filter segments that correspond to non-existing features
        self.segments = [segment for segment in self.segments if osp.exists(osp.join(self._features_path, f"{segment.video_uid}.pt"))]
            
        # Load the verbs and nouns taxonomy
        self.verb_labels, self.noun_labels = self._load_fho_taxonomy()
        
        assert len(self.verb_labels) == self.num_verbs, "mismatch in number of verb labels and expected number of verbs"
        assert len(self.noun_labels) == self.num_nouns, "mismatch in number of noun labels and expected number of nouns"
        
    def _load_segments(self) -> List[LTASegment]:
        """Load LTA segments.

        Returns
        -------
        List[LTASegment]
            LTA segments
        """
        annotations_path = osp.join(self.raw_dir, f"annotations/v{self.version}", f"fho_lta_{self.split}.json")
        annotations = json.load(open(annotations_path, 'r'))

        self.clips: Dict[str, Clip] = {}
        for sample in annotations['clips']:
            video_uid = sample['video_uid']
            
            # clip_uid = sample.get('clip_uid', uid) or uid  # use the clip_uid if both present and not None
            clip_uid = sample['clip_uid']
            
            clip = self.clips.get(clip_uid, Clip(video_uid, clip_uid, [], []))
            self.clips[clip_uid] = clip
            
            clip_start_frame = sample['clip_parent_start_frame']
            clip.samples.append(Action(
                sample['action_idx'], 
                clip_start_frame + sample['action_clip_start_frame'], 
                clip_start_frame + sample['action_clip_end_frame'],
                sample['verb_label'] if 'verb_label' in sample else None, sample['noun_label'] if 'noun_label' in sample else None
            ))
        
        narrations: Dict[str, List[Tuple[float, str]]] = self._load_narrations()
        
        # Collect all LTA segments that have at least (self.num_input_clips actions + self.Z) actions
        segments = []

        for clip_uid, clip in self.clips.items():
            clip.samples = list(sorted(clip.samples, key=lambda x: x.idx))
            
            if 'test' in self.split:
                for i in range(len(clip.samples) - self.num_input_clips):
                    input_clips = clip.samples[i:i + self.num_input_clips]
                    segments.append(LTASegment(clip.video_uid, clip.clip_uid, input_clips[0].video_start_frame, input_clips[-1].video_end_frame, [], [], [], input_clips[-1].idx))
            else:
                for i in range(len(clip.samples) - self.num_input_clips - self.Z):
                    idx_seen_start, idx_seen_end, idx_unseen_end = i, i + self.num_input_clips, i + self.num_input_clips + self.Z
                    
                    input_clips = clip.samples[idx_seen_start: idx_seen_end]
                    forecast_clips = clip.samples[idx_seen_end:idx_unseen_end]
                    
                    verb_labels = [action.verb_labels for action in forecast_clips]
                    noun_labels = [action.noun_labels for action in forecast_clips]
                    
                    segment = LTASegment(clip.video_uid, clip.clip_uid, input_clips[0].video_start_frame, input_clips[-1].video_end_frame, verb_labels, noun_labels, [], input_clips[-1].idx)  # type: ignore
            
                    segment.narrations += [
                        (ts, text.strip()) for (ts, text) in narrations.get(clip.video_uid, []) 
                        if segment.video_start_frame <= ts <= segment.video_end_frame
                    ]
                    
                    segments.append(segment)

        return segments

    def len(self) -> int:
        """Return the number of segments in the dataset.

        Returns
        -------
        int
            number of LTA segments in the dataset
        """
        return len(self.segments)

    def get(self, idx: int) -> Data:
        """Get the idx-th segment of the dataset.

        Parameters
        ----------
        idx : int
            index of the segment to retrieve

        Returns
        -------
        torch_geometric.data.Data
            the graph representation of the LTA segment
        """
        segment: LTASegment = self.segments[idx]

        # Load the raw features as a torch tensor
        video_features = torch.load(osp.join(self._features_path, f"{segment.video_uid}.pt"))

        # Cut the portion of the features that correspond to the input (seen) clips
        feat_start, feat_end = segment.video_start_frame // self.features.stride, segment.video_end_frame // self.features.stride
        # Residual with respect to the features grid (e.g. start_frame = 19, stride = 16 -> res_frame_start = 1)
        res_frame_start = segment.video_start_frame % self.features.stride
        
        # Make sure that the feature start and end are within the feature tensor
        feat_start = max(0, feat_start)
        feat_end = max(feat_start, feat_end)
        
        # Get the features, possibly with downsampling
        feats = video_features[feat_start:feat_end, :]
            
        # Extract the corresponding narrations
        narration_texts = [narration for _, narration in segment.narrations]
        narration_timestamps = torch.tensor([(timestamp - segment.video_start_frame + res_frame_start) / self.features.fps for timestamp, _ in segment.narrations])

        # Extract the verb and noun labels corresponding to the clips to forecast
        verb_labels = torch.tensor([label for label in segment.verb_labels]).long()
        noun_labels = torch.tensor([label for label in segment.noun_labels]).long()
            
        # No masking is needed for this task, so we set all the nodes as valid
        mask = torch.ones(len(feats), dtype=torch.bool)
        
        segments = torch.Tensor([segment.video_start_frame, segment.video_end_frame]).unsqueeze(0)

        return Data(
            video_uid=segment.video_uid,
            clip_uid=segment.clip_uid,
            x=feats.unsqueeze(1),
            mask=mask,
            indices=torch.arange(len(feats)),
            # index of the nodes in the graph
            pos=torch.round((0.5 + torch.arange(len(feats))) * self.features.stride / self.features.fps, decimals=3),
            # action labels, start and end timestamps
            verb_labels=verb_labels,
            noun_labels=noun_labels,
            labels=(verb_labels, noun_labels),
            # Stride between feature vectors
            feat_stride=self.features.stride,
            # Number of frames / feature vector
            feat_num_frames=self.features.window,
            # Video info (FPS and duration in seconds)
            fps=self.features.fps, 
            duration=round(len(feats) * self.features.stride / self.features.fps, ndigits=3),
            # Narrations
            narrations=narration_texts,
            narration_timestamps=narration_timestamps,
            # unused argument
            segments=segments,
            last_idx=segment.last_idx
        )


if __name__ == "__main__":
    features: Features = Features(name='egovlp', size=256, stride=16, window=16, fps=30)
    dset = LTADataset(root="data/ego4d", split="test_unannotated", features=features)
    breakpoint()
    pass
