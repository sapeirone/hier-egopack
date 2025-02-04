from typing import Dict, Literal, List, Tuple

from torch_geometric.data import Dataset

import json
import os.path as osp

from data.features import Features


Ego4dSplit = Literal['train', 'val', 'test_unannotated']


class BaseDataset(Dataset):
    
    def __init__(self, name, root: str, split: Ego4dSplit, features: Features, version: int = 1):
        """Ego4d dataset base class.

        Parameters
        ----------
        root : str
            root directory
        version : int, optional
            dataset version, by default 1
        """
        super().__init__(root)
        self.name = name
        self.split = split
        self.version = version
        self.features = features
    
    def _load_narrations(self) -> Dict[str, List[Tuple[float, str]]]:
        """Read all textual narrations.

        Parameters
        ----------
        root : str
            root directory of the narrations
        version : int
            version of the annotations to use

        Returns
        -------
        Dict[str, List[Tuple[float, str]]]
            return the textual narrations for each video, as (timestamp_frame, narration_text) tuples
        """
        path = osp.join(self.raw_dir, f"annotations/v{self.version}", "narration.json")
        narrations = json.load(open(path, 'r'))

        return {
            video_uid: [
                (narration['timestamp_frame'], narration['narration_text']) 
                for pass_key, narration_pass in narration_passes.items() if pass_key.startswith('narration')
                for narration in narration_pass['narrations']
            ]
            for video_uid, narration_passes in narrations.items()
        }
        
    def len(self) -> int:
        """Returne the size of the dataset

        Returns
        -------
        int
            size of the dataset
        """
        raise NotImplementedError()
        
    @property
    def _features_path(self) -> str:
        """Get the path to the features directory.
        
        TODO: at the moment there is no distinction here between v1 and v2 features, we should fix this.
        TODO: this should be move to a shared BaseDataset class.

        Returns
        -------
        str
            path to the features directory
        """
        return osp.join(self.raw_dir, 'features', self.features.name)
    
    def _load_fho_taxonomy(self) -> Tuple[List[str], List[str]]:
        """Load the FHO verbs and nouns taxonomies.

        Returns
        -------
        Tuple[List[str], List[str]]
            the verbs and nouns taxonomies

        Raises
        ------
        FileNotFoundError
            if the taxonomy file does not exist
        """
        path = osp.join(self.raw_dir, f"annotations/v{self.version}", "fho_lta_taxonomy.json")
        
        if not osp.exists(path):
            raise FileNotFoundError(f"Could not find the FHO taxonomy at {path}")

        labels = json.load(open(path, 'r'))  # {'verbs': [...], 'nouns': [...]}
        return [x.strip() for x in labels['verbs']], [x.strip() for x in labels['nouns']]
