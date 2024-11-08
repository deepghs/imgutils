"""
Overview:
    Detect human keypoints in anime images.

    The model is from `https://github.com/IDEA-Research/DWPose <https://github.com/IDEA-Research/DWPose>`_.

    .. image:: dwpose_demo.plot.py.svg
        :align: center

    This is an overall benchmark of all the keypoint detect models:

    .. image:: dwpose_benchmark.plot.py.svg
        :align: center

"""
import warnings
from typing import Tuple, List

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

from .format import OP18KeyPointSet
from ..data import ImageTyping, load_image
from ..detect import detect_person
from ..utils import open_onnx_model, ts_lru_cache


def _dwpose_preprocess(img: np.ndarray, out_bbox=None, input_size: Tuple[int, int] = (288, 384)) \
        -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Do preprocessing for RTMPose model inference.

    Args:
        img (np.ndarray): Input image in shape.
        out_bbox (List[Tuple[int, int, int, int]]): Bounding box of each person.
        input_size (tuple): Input image size in shape (w, h).

    Returns:
        tuple:
        - resized_img (np.ndarray): Preprocessed image.
        - center (np.ndarray): Center of image.
        - scale (np.ndarray): Scale of image.
    """
    # get shape of image
    out_img, out_center, out_scale = [], [], []
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        # get center and scale
        center, scale = _bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = _top_down_affine(input_size, scale, center, img)

        # normalize image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def _dwpose_inference(session, img: List[np.ndarray]) -> List[np.ndarray]:
    """
    Inference RTMPose model.

    Args:
        session (ort.InferenceSession): ONNXRuntime session.
        img (np.ndarray): Input image in shape.

    Returns:
        outputs (np.ndarray): Output of RTMPose model.
    """
    all_out = []
    for i in range(len(img)):
        # build output
        input_values = {session.get_inputs()[0].name: img[i].transpose(2, 0, 1)[None, ...].astype(np.float32)}
        output_names = [out.name for out in session.get_outputs()]

        # run model
        outputs = session.run(output_names, input_values)
        all_out.append(outputs)

    return all_out


def _dwpose_postprocess(outputs: List[np.ndarray], model_input_size: Tuple[int, int],
                        center: List[np.ndarray], scale: List[np.ndarray], simcc_split_ratio: float = 2.0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess for RTMPose model output.

    Args:
        outputs (np.ndarray): Output of RTMPose model.
        model_input_size (tuple): RTMPose model Input image size.
        center (tuple): Center of bbox in shape (x, y).
        scale (tuple): Scale of bbox in shape (w, h).
        simcc_split_ratio (float): Split ratio of simcc.

    Returns:
        tuple:
        - keypoints (np.ndarray): Rescaled keypoints.
        - scores (np.ndarray): Model predict scores.
    """
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        # use simcc to decode
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = _output_decode(simcc_x, simcc_y, simcc_split_ratio)

        # rescale keypoints
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)


def _bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # get bbox center and scale
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio.

    Args:
        bbox_scale (np.ndarray): The image scale (w, h) in shape (2, )
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.ndarray: The reshaped image scale in (2, )
    """
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float,
                     output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.),
                     inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # compute transformation matrix
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # get four corners of the src rectangle in the original image
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # get four corners of the dst rectangle in the input image
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    src, dst = src.astype(np.float32), dst.astype(np.float32)
    if inv:
        src, dst = dst, src  # pragma: no cover
    warp_mat = cv2.getAffineTransform(src, dst)

    return warp_mat


def _top_down_affine(input_size: Tuple[int, int], bbox_scale: np.ndarray, bbox_center: np.ndarray,
                     img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform.

    Args:
        input_size (dict): The input size of the model.
        bbox_scale (dict): The bbox scale of the img.
        bbox_center (dict): The bbox center of the img.
        img (np.ndarray): The original image.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: img after affine transform.
        - np.ndarray[float32]: bbox scale after affine transform.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # reshape bbox to fixed aspect ratio
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # get the affine matrix
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = _get_warp_matrix(center, scale, rot, output_size=(w, h))

    # do affine transform
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def _get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    n, k, wx = simcc_x.shape
    simcc_x = simcc_x.reshape(n * k, -1)
    simcc_y = simcc_y.reshape(n * k, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(n, k, 2)
    vals = vals.reshape(n, k)

    return locs, vals


def _output_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian.

    Args:
        simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
        simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
        simcc_split_ratio (float): The split ratio of simcc.

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: keypoints in shape (K, 2) or (n, K, 2)
        - np.ndarray[float32]: scores in shape (K,) or (n, K)
    """
    keypoints, scores = _get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores


mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


def _dwpose_reorder_body_points(keypoints, scores):
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    # neck score when visualizing pred
    neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info
    keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

    return keypoints, scores


def _split_data(keypoints, scores) -> List[OP18KeyPointSet]:
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    return [OP18KeyPointSet(info) for info in keypoints_info]


@ts_lru_cache()
def _open_dwpose_model():
    return open_onnx_model(hf_hub_download(
        repo_id='yzd-v/DWPose',
        filename='dw-ll_ucoco_384.onnx',
    ))


def dwpose_estimate(image: ImageTyping, auto_detect: bool = True,
                    out_bboxes=None, person_detect_cfgs=None) -> List[OP18KeyPointSet]:
    """
    Performs inference on the RTMPose model and returns keypoints and scores.

    :param image: Input image.
    :type image: ImageTyping
    :param auto_detect: Auto detect person with :func:`imgutils.detect.person.detect_person`.
    :type auto_detect: bool
    :param out_bboxes: Bounding boxes.
    :type out_bboxes: Optional[List[Tuple[int, int, int, int]]]
    :param person_detect_cfgs: Config arguments for :func:`imgutils.detect.person.detect_person`.
    :type person_detect_cfgs: Optional[Dict]

    :return: List of mapping of different parts, including ``all``, ``head``, ``body``, ``foot``, ``hand1`` and ``hand2``.
    :rtype: List[OP18KeyPointSet]

    Examples:
        >>> from imgutils.data import load_image
        >>> from imgutils.pose import dwpose_estimate, op18_visualize
        >>>
        >>> image = load_image('dwpose/squat.jpg')
        >>> keypoints = dwpose_estimate(image)
        >>> keypoints
        [<imgutils.pose.format.OP18KeyPointSet object at 0x7f5ca933f3d0>]
        >>>
        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(op18_visualize(image, keypoints))
        <matplotlib.image.AxesImage object at 0x7f5c98069790>
        >>> plt.show()

        .. note::
            Function :func:`imgutils.pose.visual.op18_visualize` can be used to visualize this result.
    """
    session = _open_dwpose_model()
    h, w = session.get_inputs()[0].shape[-2:]
    model_input_size = (w, h)

    image = load_image(image, mode='RGB')
    np_image = np.array(image)
    if auto_detect:
        if out_bboxes is not None:
            warnings.warn('Out bboxes provided, auto detection will be disabled.')
        else:
            out_bboxes = [
                (x0, y0, x1, y1) for (x0, y0, x1, y1), _, _ in
                detect_person(image, **(person_detect_cfgs or {}))
            ]
    elif out_bboxes is None:
        out_bboxes = [(0, 0, image.width, image.height)]

    resized_img, center, scale = _dwpose_preprocess(np_image, out_bboxes, model_input_size)
    outputs = _dwpose_inference(session, resized_img)
    keypoints, scores = _dwpose_postprocess(outputs, model_input_size, center, scale)
    if keypoints.shape[0] > 0:
        keypoints, scores = _dwpose_reorder_body_points(keypoints, scores)
        return _split_data(keypoints, scores)
    else:
        return []
