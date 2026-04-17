from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def define_new_path_column(df: pd.DataFrame, output_dir: str, tag: str, col:str='img', ext:str='.nii.gz') -> pd.DataFrame:
    """
    Define a new column in the dataframe for the downsampled image paths based on the raw file paths.

    :param df: input dataframe containing a 'raw' column with the paths to the raw files
    :param output_dir: base output directory where the downsampled files will be saved
    :param tag: tag to add to the filename to indicate the type of downsampled file (e.g., 'downsampled')
    :param col: name of the new column to be added to the dataframe (default='img')
    :return: updated dataframe with the new column containing the paths to the new file names
    """

    # check that the dataframe has the required columns
    required_cols = ['sub', 'hemisphere', 'chunk', 'raw']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain the following columns: {required_cols}. Missing column: {c}")

    df[col] = [""] * len(df)

    # group by sub, hemisphere, chunk to define the new path column for each group
    for (sub, hemi, chunk), df_sub_hemi_chunk in df.groupby(
        ["sub", "hemisphere", "chunk"]
    ):
        curr_dir = f"{output_dir}/sub-{sub}/hemi-{hemi}/chunk-{chunk}/"

        # ensure the directory exists
        os.makedirs(curr_dir, exist_ok=True)

        for idx, row in df_sub_hemi_chunk.iterrows():
            raw_file = os.path.basename(row["raw"])

            if ".nii.gz" not in raw_file:
                # If the raw file does not have the .nii.gz extension, add the tag before the extension
                out_file = os.path.splitext(raw_file)[0] + f"_{tag}{ext}"
            else:
                # If the raw file has the .nii.gz extension, replace it with _{tag}.nii.gz
                out_file = re.sub(".nii.gz", f"_{tag}{ext}", raw_file)

            downsample_file = f"{curr_dir}/{out_file}"

            df.at[idx, col] = downsample_file

    return df



def _multires_root_dir(output_dir):
    return f"{output_dir}/3_multires_align/"


def _init_align_filename(output_dir, sub, hemi, chunk) -> Path:
    init_align_dir = _init_align_dir(output_dir)
    init_align_chunk_dir = _init_align_chunk_dir(init_align_dir, sub, hemi, chunk)

    os.makedirs(init_align_chunk_dir, exist_ok=True)

    return (
        f"{init_align_chunk_dir}/sub-{sub}_hemi-{hemi}_chunk-{chunk}_init_align.nii.gz"
    )


def _init_align_chunk_dir(output_dir, sub, hemi, chunk) -> Path:
    return f"{output_dir}/sub-{sub}/hemi-{hemi}/chunk-{chunk}/"


def _init_align_dir(output_dir) -> Path:
    return f"{output_dir}/2_init_align/"


class Stage(Enum):
    INIT_ALIGN = "init_align"  # 2_init_align
    LANDMARK_ALIGN = "landmark_align"  # landmark_align
    INTERMEDIATE_VOLUME = "intermediate_volume"  # 3.1_intermediate_volume
    ALIGN_3D = "align_3d"  # 3.2_align_3d
    ALIGN_2D = "align_2d"  # 3.3_align_2d


@dataclass()
class MultiResPaths:
    """Path schema for BrainBuilder multiresolution stages at a given
    (sub, hemisphere, chunk, resolution, pass_step).

    This replaces get_multiresolution_filenames() with a single object
    that computes paths lazily and can export a pandas-friendly record.
    """

    # identity
    sub: str
    hemisphere: str
    chunk: int

    # resolution context
    resolution: float
    resolution_3d: float
    pass_step: int

    # global settings
    output_dir: Path
    use_3d_syn_cc: bool = True
    stage_tag: str = ""

    moving_landmark: str = None
    fixed_landmark: str = None

    ref_landmark_volume: Optional[str] = None
    moving_landmark_volume: Optional[str] = None
    fixed_landmark_volume: Optional[str] = None

    moving: Optional[str] = None
    fixed: Optional[str] = None

    section_thickness: Optional[float] = 0.02  # assume default 20um section thickness

    # ---------- derived naming primitives ----------

    @property
    def multires_out_dir(self) -> Path:
        root_dir = _multires_root_dir(self.output_dir)
        return f"{root_dir}/sub-{self.sub}/hemi-{self.hemisphere}/chunk-{self.chunk}/{self.resolution}mm{self.stage_tag}/pass_{self.pass_step}/"

    @property
    def prefix(self) -> str:
        return f"sub-{self.sub}_hemi-{self.hemisphere}_chunk-{self.chunk}_{self.resolution}mm"

    @property
    def prefix_3d(self) -> str:
        return f"sub-{self.sub}_hemi-{self.hemisphere}_chunk-{self.chunk}_itr-{self.resolution}_{self.resolution_3d}mm"

    @property
    def init_volume(self) -> Path:
        return _init_align_filename(
            self.output_dir, self.sub, self.hemisphere, self.chunk
        )

    @property
    def nl_method(self) -> str:
        return "CC" if self.use_3d_syn_cc else "Mattes"

    # ---------- stage directories ----------

    @property
    def init_align_dir(self) -> Path:
        return _init_align_dir(self.output_dir)

    @property
    def intermediate_volume_dir(self) -> Path:
        return Path(f"{self.multires_out_dir}/3.1_intermediate_volume/")

    @property
    def align_3d_dir(self) -> Path:
        return Path(f"{self.multires_out_dir}/3.2_align_3d/")

    @property
    def align_2d_dir(self) -> Path:
        return f"{self.multires_out_dir}/3.3_align_2d/"

    @property
    def landmark_dir(self) -> Path:
        return f"{self.multires_out_dir}/landmarks/"

    # ---------- files (keep your existing key names for compatibility) ----------

    @property
    def acq_rsl_fn(self) -> Path:
        return f"{self.intermediate_volume_dir}/{self.prefix_3d}_acq.nii.gz"

    @property
    def acq_pad_fn(self) -> Path:
        return f"{self.intermediate_volume_dir}/{self.prefix_3d}_acq_pad.nii.gz"

    @property
    def rec_3d_rsl_fn(self) -> Path:
        return f"{self.align_3d_dir}/{self.prefix}_rec_space-mri.nii.gz"

    @property
    def ref_3d_rsl_fn(self) -> Path:
        return f"{self.align_3d_dir}/{self.prefix_3d}_mri_gm_space-rec.nii.gz"

    @property
    def nl_3d_tfm_fn(self) -> Path:
        return f"{self.align_3d_dir}/{self.prefix}_rec_to_mri_SyN_{self.nl_method}_Composite.h5"

    @property
    def nl_3d_tfm_inv_fn(self) -> Path:
        return f"{self.align_3d_dir}/{self.prefix}_rec_to_mri_SyN_{self.nl_method}_InverseComposite.h5"

    @property
    def nl_2d_vol_fn(self) -> Path:
        return f"{self.align_2d_dir}/{self.prefix}_nl_2d.nii.gz"

    def space_ref_nat_fn(self) -> Path:
        return f"{self.align_2d_dir}/{self.prefix}_ref_space-nat.nii.gz"

    @property
    def nl_2d_vol_cls_fn(self) -> Path:
        return f"{self.align_2d_dir}/{self.prefix}_nl_2d_cls.nii.gz"

    @property
    def ref_rsl_fn(self) -> Path:
        return f"{self.align_3d_dir}/sub-{self.sub}_hemi-{self.hemisphere}_chunk-{self.chunk}_{self.resolution_3d}mm_ref_vol.nii.gz"

    @property
    def acq_landmark_volume(self) -> Path:
        return f"{self.landmark_dir}/sub-{self.sub}_hemi-{self.hemisphere}_chunk-{self.chunk}_acq_landmarks_itr-{self.resolution}mm.nii.gz"

    # ---------- convenience ----------

    def ensure_stage_dirs(self) -> None:
        Path(self.intermediate_volume_dir).mkdir(parents=True, exist_ok=True)
        Path(self.align_3d_dir).mkdir(parents=True, exist_ok=True)
        Path(self.align_2d_dir).mkdir(parents=True, exist_ok=True)
        Path(self.landmark_dir).mkdir(parents=True, exist_ok=True)

    def as_record(self) -> Dict[str, Any]:
        """Pandas/CSV boundary: return the same “row keys” you currently use.
        Convert Paths -> str here.
        """
        return {
            "sub": self.sub,
            "hemisphere": self.hemisphere,
            "chunk": self.chunk,
            "resolution": self.resolution,
            "multires_out_dir": str(self.multires_out_dir) + "/",
            "acq_dir": str(self.intermediate_volume_dir) + "/",
            "align_3d_dir": str(self.align_3d_dir) + "/",
            "nl_2d_dir": str(self.align_2d_dir) + "/",
            "section_thickness": self.section_thickness,
            "acq_rsl_fn": str(self.acq_rsl_fn),
            "acq_pad_fn": str(self.acq_pad_fn),
            "rec_3d_rsl_fn": str(self.rec_3d_rsl_fn),
            "ref_3d_rsl_fn": str(self.ref_3d_rsl_fn),
            "nl_3d_tfm_fn": str(self.nl_3d_tfm_fn),
            "nl_2d_vol_fn": str(self.nl_2d_vol_fn),
            "nl_2d_vol_cls_fn": str(self.nl_2d_vol_cls_fn),
            "ref_rsl_fn": str(self.ref_rsl_fn),
            "init_volume": str(self.init_volume),
            "ref_landmark_volume": str(self.ref_landmark_volume)
            if self.ref_landmark_volume is not None
            else None,
            "acq_landmark_volume": str(self.acq_landmark_volume)
            if self.acq_landmark_volume is not None
            else None,
            "moving_landmark_volume": str(self.moving_landmark_volume),
            "fixed_landmark_volume": str(self.fixed_landmark_volume),
            "nl_method": self.nl_method,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame for easy CSV export."""
        rec = self.as_record()
        return pd.DataFrame([rec])

    def select_volumes(
        self,
        moving: str,
        fixed: str,
        moving_landmark: Optional[str] = None,
        fixed_landmark: Optional[str] = None,
        overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Resolve moving/fixed volume paths by name with optional overrides."""
        rec = self.as_record()
        if overrides:
            rec.update(overrides)

        # check that keys exist in aliases
        for key in [moving, fixed, moving_landmark, fixed_landmark]:
            if key is not None:
                if key not in rec and key not in {
                    "acq_vol",
                    "ref_vol",
                    "acq_landmark",
                    "ref_landmark",
                }:
                    raise KeyError(
                        f"Unknown key '{key}'. Known keys: {sorted(rec.keys())}"
                    )

        # Optional stable aliases so callers don't need to know internal keys.

        # the logic for setting the fixed and moving landmarks is a little bit complicated because one of the file paths
        # will be passed directly, because the reference landmark path is generated by the user prior to launching the pipeline.
        # hence it is fixed and not part of the path schema.
        # the acquisition landmark path is generated by the pipeline and is part of the path schema. the 2D landmark files are provided by the user
        # but are spatially transformed using the current, best 2D transformation files and combiend together into a single acq_landmark_volume volume.

        # check that either the moving_landmark_volume or fixed_landmark_volume are provided directly
        error_message = f"Either moving_landmark_volume or fixed_landmark_volume must be provided and valid. Got:\n{self.moving_landmark_volume}\n{self.fixed_landmark_volume}"
        moving_landmark_volume_flag = (
            Path(self.moving_landmark_volume).exists()
            if self.moving_landmark_volume is not None
            else False
        )
        fixed_landmark_volume_flag = (
            Path(self.fixed_landmark_volume).exists()
            if self.fixed_landmark_volume is not None
            else False
        )
        assert moving_landmark_volume_flag ^ fixed_landmark_volume_flag, error_message

        # check if moving_landmark_volume and fixed_landmark_volume are provided directly, otherwise use the path schema to get them.
        if self.moving_landmark_volume is None:
            moving_landmark_volume = rec[self.moving_landmark]
        elif Path(self.moving_landmark_volume).exists():
            moving_landmark_volume = self.moving_landmark_volume
        else:
            raise FileNotFoundError(
                f"Moving landmark path {self.moving_landmark_volume} is invalid."
            )

        if self.fixed_landmark_volume is None:
            fixed_landmark_volume = rec[self.fixed_landmark]
        elif Path(self.fixed_landmark_volume).exists():
            fixed_landmark_volume = self.fixed_landmark_volume
        else:
            raise FileNotFoundError(
                f"Fixed landmark path {self.fixed_landmark_volume} is invalid."
            )

        return {
            "moving_volume": rec[self.moving],
            "fixed_volume": rec[self.fixed],
            "moving_landmark_volume": moving_landmark_volume,
            "fixed_landmark_volume": fixed_landmark_volume,
        }

    def __post_init__(self):
        self.ensure_stage_dirs()

        if self.moving is not None and self.fixed is not None:
            sel0 = self.select_volumes(
                self.moving,
                self.fixed,
                moving_landmark=self.moving,
                fixed_landmark=self.fixed,
            )
            self.fixed_volume = sel0["fixed_volume"]
            self.moving_volume = sel0["moving_volume"]

        if (self.moving_landmark is not None or self.moving_landmark_volume) and (
            self.fixed_landmark is not None or self.fixed_landmark_volume
        ):
            sel1 = self.select_volumes(
                self.moving,
                self.fixed,
                moving_landmark=self.moving_landmark,
                fixed_landmark=self.fixed_landmark,
            )
            self.fixed_landmark_volume = sel1["fixed_landmark_volume"]
            self.moving_landmark_volume = sel1["moving_landmark_volume"]


@dataclass()
class PathBundle:
    """Immutable attribute-accessible mapping.

    bundle.foo -> value for "foo"
    bundle["foo"] -> same
    bundle.as_dict() -> underlying data (strings by default)
    """

    _data: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(
                f"{name!r} not found. Available keys: {sorted(self._data.keys())}"
            ) from e

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def require(self, *names: str) -> "PathBundle":
        missing = [
            n for n in names if n not in self._data or self._data[n] in (None, "")
        ]
        if missing:
            raise KeyError(
                f"Missing required keys: {missing}. Available: {sorted(self._data.keys())}"
            )
        return self
