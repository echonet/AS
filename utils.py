import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
import pydicom
import cv2
from typing import Tuple, Union, List

import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydicom.pixel_data_handlers.util import convert_color_space, apply_voi_lut
from numpy.typing import ArrayLike



from scipy.signal import savgol_filter, find_peaks


def load_model(device, weights_path, num_classes):
    weights = torch.load(weights_path, map_location=device)
    weights = {key[2:]: val for key, val in weights.items()}
    model = torchvision.models.video.r2plus1d_18(num_classes=num_classes)
    model.load_state_dict(weights)
    model.to(device)
    return model.eval()


def get_ybr_to_rgb_lut(filepath, save_lut=False):
    global _ybr_to_rgb_lut
    _ybr_to_rgb_lut = None
    # return lut if already exists
    if _ybr_to_rgb_lut is not None:
        return _ybr_to_rgb_lut
    # try loading from file
    lut_path = Path(filepath).parent / "ybr_to_rgb_lut.npy"
    if lut_path.is_file():
        _ybr_to_rgb_lut = np.load(lut_path)
        return _ybr_to_rgb_lut
    # else generate lut
    a = np.arange(2**8, dtype=np.uint8)
    ybr = np.concatenate(
        np.broadcast_arrays(
            a[:, None, None, None], a[None, :, None, None], a[None, None, :, None]
        ),
        axis=-1,
    )
    _ybr_to_rgb_lut = pydicom.pixel_data_handlers.util.convert_color_space(
        ybr, "YBR_FULL", "RGB"
    )
    if save_lut:
        np.save(lut_path, _ybr_to_rgb_lut)
    return _ybr_to_rgb_lut


def ybr_to_rgb(filepath, pixels: np.array):
    lut = get_ybr_to_rgb_lut(filepath)
    return lut[pixels[..., 0], pixels[..., 1], pixels[..., 2]]


# for measurement 
ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)

def segmentation_to_coordinates(
    logits: torch.Tensor, normalize: bool = True, order="YX"
):
    """
    Expects logits with shape (..., n_points, h, w). 
    Returns (..., n_points, 2) coordinate pairs in YX ordering. uses weighted averaging to compute centroid of logits. 
    It is recommended to first sigmoid your logits if they are coming right out of a model. You may want to cast the return to int if you want to use it for indexing.
    """

    predictions_rows, predictions_cols = torch.meshgrid(
        torch.arange(logits.shape[-2], device=logits.device),
        torch.arange(logits.shape[-1], device=logits.device),
        indexing="ij",
    )  # (h, w)
    predictions_rows = predictions_rows * logits  # (..., h, w)
    predictions_cols = predictions_cols * logits  # (..., h, w)
    # weighted average of y values
    predictions_rows = predictions_rows.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )
    # weighted average of x values
    predictions_cols = predictions_cols.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )
    if normalize:
        predictions_rows = predictions_rows / (logits.shape[-2])
        predictions_cols = predictions_cols / (logits.shape[-1])
    if order == "YX":
        predictions = torch.stack([predictions_rows, predictions_cols], dim=-1)
    elif order == "XY":
        predictions = torch.stack([predictions_cols, predictions_rows], dim=-1)
    else:
        raise ValueError(f"Invalid order: {order}")
    return predictions


def get_coordinates_from_dicom(
    dicom: pydicom.Dataset,
) -> tuple[tuple[int, int, int, int], tuple]:
    """
    Looks through ultrasound region tags in the DICOM file. Usually, 
    there are two regions, and the doppler image is the lower one. 
    Returns the coordinates of this region's bounding box.
    """

    REGION_COORD_SUBTAGS = [
        REGION_X0_SUBTAG,
        REGION_Y0_SUBTAG,
        REGION_X1_SUBTAG,
        REGION_Y1_SUBTAG,
    ]

    if ULTRASOUND_REGIONS_TAG in dicom:
        all_regions = dicom[ULTRASOUND_REGIONS_TAG].value
        regions_with_coords = []
        for region in all_regions:
            region_coords = []
            for coord_subtag in REGION_COORD_SUBTAGS:
                if coord_subtag in region:
                    region_coords.append(region[coord_subtag].value)
                else:
                    region_coords.append(None)

            # Keep only regions that have a full set of 4 coordinates
            if all([c is not None for c in region_coords]):
                regions_with_coords.append((region, region_coords))

        # We sort regions by their y0 coordinate, as the Doppler region we want should be the lowest region
        regions_with_coords = list(
            sorted(regions_with_coords, key=lambda x: x[1][1], reverse=True)
        )

        return regions_with_coords[0]

    else:
        print("No ultrasound regions found in DICOM file.")
        return None, None
    
def find_horizontal_line(
    image: np.ndarray,
    angle_threshold: float = np.pi / 180,
    line_threshold: float = 100,
) -> int:
    """
    Horizontal line detection for Doppler images.
    
    Uses Canny edge detection and the Hough Transform to find the most prominent horizontal line in the image. 
    Returns the y-coordinate of this line.
    """

    if len(image.shape) == 2: #Already gray image
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            if (
                abs(theta - np.pi / 2) < angle_threshold
                or abs(theta - 3 * np.pi / 2) < angle_threshold
            ):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                y = int(y0)
                return y
    return None

#Convert YBR_FULL_422 to RGB
lut=np.load("./ybr_to_rgb_lut.npy")

# def ybr_to_rgb(x):
#     return lut[x[..., 0], x[..., 1], x[..., 2]]

def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
    """
    Masks all pixels outside the ultrasound region in a video.

    Args:
    vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

    Returns:
    np.ndarray: A numpy array with pixels outside the ultrasound region masked.
    """
    try:
        testarray=np.copy(original_pixels)
        vid=np.copy(original_pixels)
        ##################### CREATE MASK #####################
        # Sum all the frames
        frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
        frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
        frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
        frames = testarray.shape[0]
        for i in range(frames): # Go through every frame
            frame = testarray[i, :, :, :].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.where(frame>0,1,0) # make all non-zero values 1
            frame_sum = np.add(frame_sum,frame)

        # Erode to get rid of the EKG tracing
        kernel = np.ones((3,3), np.uint8)
        frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

        # Make binary
        frame_sum = np.where(frame_sum > 0, 1, 0)

        # Make the difference frame fr difference between 1st and last frame
        # This gets rid of static elements
        frame0 = testarray[0].astype(np.uint8)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
        frame_diff = abs(np.subtract(frame0, frame_last))
        frame_diff = np.where(frame_diff > 0, 1, 0)

        # Ensure the upper left hand corner 20x20 box all 0s.
        # There is a weird dot that appears here some frames on Stanford echoes
        frame_diff[0:20, 0:20] = np.zeros([20, 20])

        # Take the overlap of the sum frame and the difference frame
        frame_overlap = np.add(frame_sum,frame_diff)
        frame_overlap = np.where(frame_overlap > 1, 1, 0)

        # Dilate
        kernel = np.ones((3,3), np.uint8)
        frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

        # Fill everything that's outside the mask sector with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
        # Find the convex hull
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
        frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
        # Fill everything that's outside hull with some other number like 100
        cv2.floodFill(frame_overlap, None, (0,0), 100)
        # make all non-100 values 255. The rest are 0
        frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
        ################## Create your .avi file and apply mask ##################
        # Store the dimension values

        # Apply the mask to every frame and channel (changing in place)
        for i in range(len(vid)):
            frame = vid[i, :, :, :].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
            vid[i,:,:,:]=frame
        return vid
    except Exception as e:
        print("Error masking returned as is.")
        return vid

def read_video(
    path: Union[str, Path],
    n_frames: int = None,
    sample_period: int = 1,
    out_fps: float = None,  # Output fps
    fps: float = None,  # input fps of video (default to avi metadata)
    frame_interpolation: bool = True,
    random_start: bool = False,
    res: Tuple[int] = None,  # (width, height)
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0):

    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    if res is None:
        out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
    else:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps

def crop_and_scale(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.0
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Apply zoom
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img

def write_to_avi(frames: np.ndarray, out_file, fps=30):
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'MJPG'), fps, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()