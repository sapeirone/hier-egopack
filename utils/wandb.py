import omegaconf
from typing import Optional, Tuple

from datetime import datetime


def flatten_cfg(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        k = k.replace("/", sep)  # workaround for omegaconf override notation
        if isinstance(v, str):
            v = v.replace("/", "_")
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        elif isinstance(v, omegaconf.listconfig.ListConfig) or isinstance(v, list):
            items.append((new_key, '-'.join([str(x).replace("/", "_") for x in sorted(v)])))
        else:
            items.append((new_key, v))
    return dict(items)


def format_run_name(pattern: Optional[str], cfg: omegaconf.OmegaConf) -> Tuple[Optional[str], Optional[str]]:
    if pattern is None:
        return (None, None)

    name: str = pattern.format(**flatten_cfg(cfg))
    name: str = name.replace("/", "-")
    
    now = datetime.now()
    name_with_date = name + "_" + now.strftime("%Y-%m-%d-%H-%M")
    
    return name_with_date, name 
