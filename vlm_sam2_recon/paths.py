"""Central path conventions for the reconstruction project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = Path("outputs/project_manifest.json")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = DEFAULT_PROJECT_ROOT

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        return cls(root=Path(root).resolve() if root else DEFAULT_PROJECT_ROOT.resolve())

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def manifest(self) -> Path:
        return self.root / DEFAULT_MANIFEST_PATH

    @property
    def sam2_masks(self) -> Path:
        return self.outputs / "sam2_masks"

    @property
    def trellis2_interface(self) -> Path:
        return self.outputs / "trellis2_interface"

    @property
    def physxomni(self) -> Path:
        return self.outputs / "physxomni"

    @property
    def particulate(self) -> Path:
        return self.outputs / "particulate"

    @property
    def egoforce_rgb_right(self) -> Path:
        return self.outputs / "egoforce_rgb_right"

    @property
    def hunyuan3d(self) -> Path:
        return self.outputs / "hunyuan3d"

    @property
    def trellis2_mesh_uploads(self) -> Path:
        return self.inputs / "trellis2_meshes"

    @property
    def hunyuan3d_mesh_uploads(self) -> Path:
        return self.inputs / "hunyuan3d_meshes"

    @property
    def rigid_mesh_uploads(self) -> Path:
        return self.inputs / "rigid_meshes"

    def ensure_base_dirs(self) -> None:
        for path in (self.configs, self.docs, self.inputs, self.outputs):
            path.mkdir(parents=True, exist_ok=True)
