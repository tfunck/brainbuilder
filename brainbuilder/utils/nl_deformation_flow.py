import os
import subprocess

import ants
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import brainbuilder.utils.ants_nibabel as nib


def scale_displacement_field(
    transform_path, scale_factor, output_filename, clobber=False
):
    """Loads an ANTs composite displacement field (.h5), scales it by the given factor,
    and saves the scaled displacement field to a NIfTI file.

    Parameters:
    -----------
    transform_path : str
        Path to the input ANTs composite transform (.h5) or displacement field (.nii.gz).
    scale_factor : float
        Scale to apply to the deformation (e.g., 0.5 for halfway interpolation).
    output_filename : str
        Path to save the scaled displacement field as a NIfTI file (.nii.gz).
    """
    if (
        os.path.exists(transform_path)
        and not os.path.exists(output_filename)
        or clobber
    ):
        # Step 1: Read the transform as an image
        try:
            transform_img = ants.image_read(transform_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read transform file {transform_path}. Ensure it's a NIfTI (.nii.gz) image."
            ) from e

        # Step 2: Check if it's a displacement field (must be a vector field)
        if transform_img.components <= 1:
            raise ValueError(
                "The input transform does not appear to be a displacement field (expected multi-component vector image)."
            )

        # Step 3: Scale the displacement field
        # x0 = np.mean(np.abs(transform_img.numpy()) )
        scaled_transform = transform_img * scale_factor

        # Step 4: Write the result to a NIfTI file
        ants.image_write(scaled_transform, output_filename)

    return output_filename


def nl_deformation_flow(
    sec0_path: str,
    sec1_path: str,
    ymin: int,
    ymax: int,
    output_dir: str,
    mode: int = -1,
    clobber: bool = False,
):
    """Use ANTs to calculate SyN alignment between two sections. Let the deformation field = D.
    Then let s = i/max(steps) where i is an integer from 0 to max(steps).
    Then the flow field is given by D_s(X0) = D * s, where X0 is the original section, sec0


    For each step, calculate output images: X_i = D(X0)*s + D^-1(X1)*(1-s)

    and save them to the output directory.

    Args:
        sec0_path (str): path to the first section
        sec1_path (str): path to the second section
        ymin (int): minimum y-coordinate of the section
        ymax (int): maximum y-coordinate of the section
        output_dir (str): directory to save the output images

    Returns:
        None
    """
    qc_dir = f"{output_dir}/qc"
    prefixdir = f"{output_dir}/tfm_{ymin}_{ymax}"

    os.makedirs(prefixdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    outprefix = f"{prefixdir}/deformation_field_{ymin}_{ymax}"

    write_composite_transform = 0

    if write_composite_transform:
        fwd_tfm_path = f"{outprefix}_Composite.h5"
        inv_tfm_path = f"{outprefix}_InverseComposite.h5"
    else:
        fwd_tfm_path = f"{outprefix}_0Warp.nii.gz"
        inv_tfm_path = f"{outprefix}_0InverseWarp.nii.gz"

    mv_rsl_fn = f"{outprefix}_SyN_GC_cls_rsl.nii.gz"

    steps = ymax - ymin

    if not os.path.exists(fwd_tfm_path) or os.path.exists(inv_tfm_path) or not clobber:
        # Load the sections
        try:
            cmd = "antsRegistration --verbose 0 --dimensionality 2 --float 0 --collapse-output-transforms 1"
            cmd += f" --output [ {outprefix}_,{mv_rsl_fn},/tmp/tmp.nii.gz ] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ]"
            cmd += f" --transform SyN[ 0.1,3,0 ] --metric CC[ {sec1_path},{sec0_path},1,4 ]"
            cmd += " --convergence [ 300x200x150x100x50,1e-6,10 ] --shrink-factors 8x4x3x2x1 --smoothing-sigmas 4x2x1.5x1x0vox"

            subprocess.run(cmd, shell=True, executable="/bin/bash")

        except RuntimeError as e:
            print("Error in registration:", e)

    assert os.path.exists(fwd_tfm_path), f"Error: output does not exist {fwd_tfm_path}"
    assert os.path.exists(inv_tfm_path), f"Error: output does not exist {inv_tfm_path}"

    output_image_list = []

    y_list = np.arange(ymin + 1, ymax).astype(np.uint)

    for i, y in zip(range(steps), y_list):
        s = i / steps

        output_image_path = f"{prefixdir}/flow_{y}.nii.gz"

        output_image_list.append(output_image_path)

        sec0 = ants.image_read(sec0_path)
        sec1 = ants.image_read(sec1_path)

        if not os.path.exists(output_image_path) or clobber:
            scaled_fwd_tfm_path = f"{output_dir}/scaled_fwd_tfm_{y}.nii.gz"
            scale_displacement_field(
                fwd_tfm_path, s, scaled_fwd_tfm_path, clobber=clobber
            )

            scaled_inv_tfm_path = f"{output_dir}/scaled_inv_tfm_{y}.nii.gz"
            scale_displacement_field(
                inv_tfm_path, 1 - s, scaled_inv_tfm_path, clobber=clobber
            )

            if os.path.exists(fwd_tfm_path) and os.path.exists(inv_tfm_path) or clobber:
                # Calculate the flow field for s
                # D_s(X0) = D * s
                sec0_fwd = ants.apply_transforms(
                    sec1,
                    sec0,
                    interpolator="linear",
                    transformlist=[scaled_fwd_tfm_path],
                    verbose=False,
                )

                # Calculate the inverse flow field for s
                # D_s(X1) = D^-1 * (1-s)
                sec1_inv = ants.apply_transforms(
                    sec0,
                    sec1,
                    interpolator="linear",
                    transformlist=[scaled_inv_tfm_path],
                    verbose=False,
                )

                # Combine the two sections
                if mode == 0:
                    output_image = sec0
                elif mode == 1:
                    output_image = sec1
                elif mode == 2:
                    output_image = sec0_fwd
                elif mode == 3:
                    output_image = sec1_inv
                elif mode == 4:
                    output_image = ants.apply_transforms(
                        sec1,
                        sec0,
                        interpolator="linear",
                        transformlist=[fwd_tfm_path],
                        verbose=False,
                    )
                elif mode == 5:
                    output_image = ants.apply_transforms(
                        sec0,
                        sec1,
                        interpolator="linear",
                        transformlist=[scaled_inv_tfm_path],
                        verbose=False,
                    )
                else:
                    output_image = (sec0_fwd + sec1_inv) / 2.0
            else:
                # apply linear interpolation
                sec0 = ants.image_read(sec0_path)
                sec1 = ants.image_read(sec1_path)

                sec0_fwd = sec0 * (1 - s)
                sec1_inv = sec1 * s
                output_image = sec0_fwd + sec1_inv

            # Save the output image
            ants.image_write(output_image, output_image_path)

        qc_png = f"{qc_dir}/qc_flow_{y}.png"

        if not os.path.exists(qc_png):
            # create qc image
            img = nib.load(output_image_path).get_fdata()
            sec1 = nib.load(sec1_path).get_fdata()

            plt.figure(figsize=(15, 10))
            grad_sec0 = np.array(np.gradient(img))
            grad_sec0 = np.linalg.norm(grad_sec0, axis=0)

            plt.subplot(1, 2, 1)
            plt.imshow(sec0.numpy(), cmap="gray")
            plt.title(f"{ymin} -> {y} -> {ymax}")
            plt.imshow(grad_sec0, cmap="Reds", alpha=0.65)

            plt.subplot(1, 2, 2)
            plt.imshow(sec1, cmap="gray")
            plt.title(f"{ymin} -> {y} -> {ymax}")
            plt.imshow(grad_sec0, cmap="Reds", alpha=0.65)

            plt.tight_layout()
            plt.savefig(qc_png)
            plt.close()

    return y_list, output_image_list


def nl_deformation_flow_3d(
    vol: np.array,
    output_dir: str,
    origin: tuple = None,
    spacing: tuple = None,
    mode: int = -1,
    clobber: bool = False,
):
    """Apply  nl intersection_flow to a volume where there are missing sections along axis=1"""
    os.makedirs("2d", exist_ok=True)

    valid_idx = np.where(np.max(vol, axis=(0, 2)) > 0)[0]

    assert (
        len(valid_idx) > 0
    ), "No valid sections found in the volume. Please check the input volume."

    if len(valid_idx) == 0:
        return vol

    out_vol = vol.copy()

    def process_section(y0, y1, output_dir, clobber):
        y0_ants = ants.from_numpy(vol[:, y0, :], origin=origin, spacing=spacing)
        y1_ants = ants.from_numpy(vol[:, y1, :], origin=origin, spacing=spacing)

        orig_dir = f"{output_dir}/orig"

        os.makedirs(orig_dir, exist_ok=True)

        y0_ants_path = f"{orig_dir}/flow_{y0}.nii.gz"
        if not os.path.exists(y0_ants_path) or clobber:
            y0_ants.to_filename(y0_ants_path)

        y1_ants_path = f"{orig_dir}/flow_{y1}.nii.gz"
        if not os.path.exists(y1_ants_path) or clobber:
            y1_ants.to_filename(y1_ants_path)

        print("\t", y0, y1)

        return nl_deformation_flow(
            y0_ants_path, y1_ants_path, y0, y1, output_dir, mode=mode, clobber=clobber
        )

    results = Parallel(n_jobs=-1)(
        delayed(process_section)(y0, y1, output_dir, clobber)
        for y0, y1 in zip(valid_idx[:-1], valid_idx[1:])
        if y1 > y0 + 1
    )

    for y_list, inter_images in results:
        for y, image_path in zip(y_list, inter_images):
            out_vol[:, y, :] = ants.image_read(image_path).numpy()

    return out_vol
