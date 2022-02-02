import os
import numpy as np
from tempfile import mktemp
from ants.core import ants_image_io as iio
from ants.core import ants_transform_io as txio


def fit_transform_to_paired_points(
    moving_points, fixed_points, transform_type="Affine", regularization=1e-4
, centerX=[], centerY=[]):
    """
    Estimate an optimal matrix transformation from paired points, potentially landmarks

    ANTsR (actually patchMatchR) function: fitTransformToPairedPoints

    Arguments
    ---------
    moving_points : array
        points in the moving image domain defined in physical space,
        number of points by dimensionality

    fixed_points : array
        points in the fixed image domain defined in physical space,
        number of points by dimensionality

    transform_type : character
        affine, rigid or similarity

    regularization : scalar
        regularization parameter

    Returns
    -------
    ANTs transform

    Example
    -------
    >>> import ants
    >>> fixed_points = np.array([[1,2],[4,5],[6,7],[8,9]])
    >>> moving_points = np.array([[1.1,2.3],[4.1,5.4],[6.1,7],[8,9]])
    >>> tx = ants.fit_transform_to_paired_points( mpts, fpts )
    """
    n = fixed_points.shape[0]
    idim = fixed_points.shape[1]
    if centerX == [] : centerX = fixed_points.mean(axis=0)
    if centerY == [] : centerY = moving_points.mean(axis=0)
    print('Center X:', centerX, 'Y:', centerY)

    x = fixed_points - centerX
    y = moving_points - centerY
    myones = np.ones(n)
    x11 = np.c_[x, myones]  # or np.concatenate( (x, myones.reshape(4,1) ),axis=1 )
    temp = np.linalg.lstsq(x11, y, rcond=None)
    A = temp[0].transpose()[:idim, :idim]
    trans = temp[0][idim, :] + centerY - centerX
    if transform_type == "Rigid" or transform_type == "Similarity":
        covmat = np.dot(y.transpose(), x)
        scaleDiag = np.zeros((idim, idim))
        np.fill_diagonal(scaleDiag, regularization)
        x_svd = np.linalg.svd(covmat + scaleDiag)
        myd = np.linalg.det(np.dot(x_svd[0].T, x_svd[2].T))
        if myd < 0:
            x_svd[2][idim - 1, :] *= -1
        A = np.dot(x_svd[0], x_svd[2].T)
        if transform_type == "Similarity":
            scaling = np.math.sqrt(
                (np.power(y, 2).sum(axis=1) / n).mean()
            ) / np.math.sqrt((np.power(x, 2).sum(axis=1) / n).mean())
            scaleDiag = np.zeros((idim, idim))
            np.fill_diagonal(scaleDiag, scaling)
            A = np.dot(A, scaleDiag)

    aff = txio.create_ants_transform(
        matrix=A, translation=trans, dimension=idim, center=centerX
    )

    return aff

