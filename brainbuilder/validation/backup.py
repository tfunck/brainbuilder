from skimage.segmentation import slic
from skimage.transform import resize


def classify_section(crop, seg, max_roi=7):
    n_roi = np.random.randint(5, 10)
    # print(n_roi, len(np.unique(crop)))
    # print(len(np.unique(seg)), np.max(seg), seg.dtype )
    # print('n_roi', n_roi)
    im_cls = slic(crop, n_segments=n_roi, compactness=1000, mask=seg.astype(bool))
    # assert len(np.unique(im_cls)) > 2, 'Error, only one label created in psuedo-cls'

    return im_cls


def classify_autoradiograph(autoradiograph_fn, mask_fn, out_fn, y, chunk, resolution):
    # load autoradiograph
    img = nib.load(autoradiograph_fn)
    vol = img.get_fdata()

    original_shape = vol.shape
    new_shape = np.rint(np.array(vol.shape) / 10)
    np.product(img.affine[[0, 1, 2], [0, 1, 2]])

    # load mask
    mask_img = nib.load(mask_fn)
    mask_vol = mask_img.get_fdata()
    mask_vol_rsl = resize(mask_vol, new_shape, order=0)
    vol_rsl = resize(vol, new_shape, order=5)
    out_rsl = classify_section(vol_rsl, mask_vol_rsl)

    print(f"/tmp/tmp_{os.path.basename(out_fn)}.png")
    plt.imshow(out_rsl)
    plt.savefig(f"/tmp/tmp_{os.path.basename(out_fn)}.png")
    plt.clf()
    plt.cla()

    out_unique = np.unique(out_rsl)[1:]
    # out_label_sizes = np.bincount(out_rsl.reshape(-1,))[1:].astype(float)
    # out_label_sizes = out_label_sizes[ out_label_sizes > 0 ]
    # out_label_sizes *= voxel_size

    for l in out_unique:
        index = np.core.defchararray.add(str(chunk) + str(y), str(l)).astype(int)
        print(l, index)
        out_rsl[out_rsl == l] = index

        # out[ out > 0 ] = index
    if np.sum(out_rsl) == 0:
        out_rsl = mask_vol_rsl

    out = resize(out_rsl.astype(float), original_shape, order=0)

    # plt.subplot(2,1,1)
    # plt.imshow(out_rsl);
    # plt.subplot(2,1,2)
    # plt.imshow(out)
    # plt.savefig('/tmp/test.png')

    if np.sum(out) == 0:
        print("Error empty pseudo cls")
        exit(1)
    out = np.ceil(out).astype(np.uint32)
    if np.sum(out) == 0:
        print("Error empty pseudo cls after conversion to int")
        exit(1)

    # save classified image as nifti
    print("Writing", out_fn)
    nib.Nifti1Image(out, img.affine).to_filename(out_fn)


if create_pseudo_cls:
    print("create_pseudo_cls")
    cls_check = df["pseudo_cls_fn"].apply(file_check).values
else:
    cls_check = np.zeros_like(crop_check)


def create_pseudo_classifications(df, crop_str, resolution):
    to_do = []
    for i, row in df.iterrows():
        pseudo_cls_fn = row["pseudo_cls_fn"]
        crop_fn = row[crop_str]
        seg_fn = row["seg_fn"]
        chunk_order = int(row["chunk_order"])
        chunk = int(row["chunk"])
        if not os.path.exists(pseudo_cls_fn):
            to_do.append([crop_fn, seg_fn, pseudo_cls_fn, chunk_order, chunk])

    os_info = os.uname()
    if os_info[1] == "imenb079":
        num_cores = 4
    else:
        num_cores = min(14, multiprocessing.cpu_count())

    Parallel(n_jobs=num_cores)(
        delayed(pseudo_classify_autoradiograph)(
            crop_fn, seg_fn, pseudo_cls_fn, chunk_order, chunk, resolution
        )
        for crop_fn, seg_fn, pseudo_cls_fn, chunk_order, chunk in to_do
    )

    # if create_pseudo_cls :
    #    create_pseudo_classifications(df, crop_str, resolution)
