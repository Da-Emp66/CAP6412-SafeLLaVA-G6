from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple
from PIL import Image
import cv2
import numpy as np
import requests

class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    IMAGE_OR_VIDEO = "image or video"

def open_images(media, verbose=False):
    """Open image, 1-D array of images, or 2-D array of images.
    Returns the media in the same format they were originally.
    """
    def _open_image(image):
        """Open image of any format using Pillow."""

        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, str):
            if image.startswith("http") or image.startswith("https"):
                return Image.open(requests.get(image, stream=True).raw)
            elif image.startswith("blob"):
                return Image.open(image[5:])
            return Image.open(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    if isinstance(media, list):
        if len(media) > 0:
            if isinstance(media[0], str):
                media = [_open_image(image) for image in media]
            elif isinstance(media[0], list):
                if isinstance(media[0][0], str):
                    media = [[_open_image(image) for image in image_list] for image_list in media]
                elif verbose:
                    print(f"{media} is not in a supported format. You must have a preprocessing function defined or a non-standard model for this to work.")
    elif isinstance(media, str):
        media = _open_image(media)
    elif verbose:
        print(f"{media} is not in a supported format. You must have a preprocessing function defined or a non-standard model for this to work.")

    return media

def get_video_length_seconds(video_path: str) -> float:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return "Error: Could not open video."
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    video.release()
    return duration

def sample_video(video: Any, sample_rate: int = 3) -> Tuple[List[Image.Image], int]:
    video = cv2.VideoCapture(video)

    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    samples = int((total_num_frames / fps) * sample_rate)
    interval = total_num_frames // samples

    frames = []
    for i in range(total_num_frames):
        ret, frame = video.read()
        if not ret:
            continue
        if i % interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
    
    video.release()

    if len(frames) == samples + 1:
        samples += 1

    return (frames[:samples], samples)

def load_image(image: Any):
    return open_images(image)

def load_video(video: Any, sample_rate: int = 3) -> Tuple[List[Image.Image], int]:
    if type(video) == str:
        return sample_video(video, sample_rate=sample_rate)
    elif isinstance(video, list):
        raise NotImplementedError()
    
    raise TypeError(f"`{video}` is not of an acceptable type. It should be a `str` or a `PIL.Image`.")

def load_media(media_filepath: str, video_sample_rate: int) -> Tuple[MediaType, List[Image.Image], int]:
    if media_filepath.endswith(tuple(['.jpg', '.jpeg', '.png', '.webp'])):
        return (MediaType.IMAGE, [load_image(media_filepath)], 1)
    elif media_filepath.endswith(tuple(['.mp4'])):
        return (MediaType.VIDEO, *load_video(media_filepath, sample_rate=video_sample_rate))
    else:
        raise NotImplementedError(f"`{Path(media_filepath).suffix}` not supported.")

