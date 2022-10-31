from typing import List

from dataclasses import dataclass


@dataclass()
class DownloadParams:
    paths: List[str]
    output_folder: str
