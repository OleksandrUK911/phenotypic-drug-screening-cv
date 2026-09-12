from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WellMetadata:
    plate: str
    well: str
    compound: str
    concentration_um: float
    moa: str | None
    replicate: int

    @property
    def is_control(self) -> bool:
        return self.compound.strip().upper() in {"DMSO", "MOCK", "CONTROL"}


@dataclass(frozen=True)
class ImageRecord:
    plate: str
    well: str
    site: int
    channel_paths: dict[str, str]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted(self.channel_paths))
