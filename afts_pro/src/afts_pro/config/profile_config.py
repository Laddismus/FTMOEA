from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from afts_pro.config.loader import load_yaml

logger = logging.getLogger(__name__)


class ProfileIncludes(BaseModel):
    environment: str
    execution: str
    assets: str
    strategy: str
    risk: str
    behaviour: str
    features: str
    extras: str
    runlogger: str

    model_config = {"populate_by_name": True}


class ProfileConfig(BaseModel):
    name: str
    description: Optional[str] = None
    includes: ProfileIncludes

    model_config = {"populate_by_name": True}


def load_profile(path: str) -> ProfileConfig:
    data = load_yaml(path)
    profile_data = data.get("profile", data)
    profile = ProfileConfig(**profile_data)
    logger.info("PROFILE | name=%s | path=%s | includes=%s", profile.name, path, profile.includes.model_dump())
    return profile


def list_profile_paths(base_dir: str = "configs/profiles") -> Dict[str, str]:
    base_path = Path(base_dir).resolve()
    profiles: Dict[str, str] = {}
    if not base_path.exists():
        return profiles

    for file in base_path.glob("*.yaml"):
        try:
            profile = load_profile(str(file))
        except Exception:
            logger.warning("Skipping invalid profile file: %s", file)
            continue
        profiles[profile.name] = str(file)
    return profiles


def get_profile_include_paths(profile_path: str) -> List[Path]:
    profile = load_profile(profile_path)
    base = Path(profile_path).resolve()
    project_root = Path(__file__).resolve().parents[3]
    includes = [
        profile.includes.environment,
        profile.includes.execution,
        profile.includes.assets,
        profile.includes.strategy,
        profile.includes.risk,
        profile.includes.behaviour,
        profile.includes.features,
        profile.includes.extras,
        profile.includes.runlogger,
    ]
    resolved_paths: List[Path] = []
    for inc in includes:
        path_obj = Path(inc)
        if not path_obj.is_absolute():
            path_obj = (base.parent / path_obj).resolve()
        if not path_obj.exists():
            alt = (project_root / inc).resolve()
            path_obj = alt
        resolved_paths.append(path_obj)
    resolved_paths.append(base)
    return resolved_paths
