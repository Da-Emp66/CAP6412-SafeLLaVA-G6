from enum import Enum
from io import BytesIO, StringIO
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import Any, Dict, List, Tuple, TypeAlias, Union
from urllib.parse import urlparse
import urllib.request
import uuid
from PIL import Image
import cv2
import numpy as np
import pytube
import requests

#####################################################
# Basic File I/O
#####################################################

class FileType(Enum):
    STRINGIFIED_CONTENT = [str]
    IN_MEMORY_FILE = [BytesIO, StringIO]
    TEMPORARY_FILE = [TemporaryFile, NamedTemporaryFile]
    REAL_FILE = [str, Path]

SomeFileType: TypeAlias = Union[
    str,
    BytesIO,
    StringIO,
    TemporaryFile,
    NamedTemporaryFile,
    Path,
]

def convert_string_to_file(content: str, target: FileType, **kwargs: Dict[str, Any]) -> SomeFileType:
    """Convert the string to a given file type.

    Args:
        content (str): Stringified file contents
        target (FileType): Target file type to convert contents to

    Returns:
        SomeFileType: The file with the specified contents written.
    """

    mode = kwargs.get("mode", "w+")
    filename = kwargs.get("filename", str(uuid.uuid4()))

    if target == FileType.STRINGIFIED_CONTENT:
        return content
    elif target == FileType.IN_MEMORY_FILE:
        return StringIO(content)
    elif target == FileType.TEMPORARY_FILE:
        file = NamedTemporaryFile(mode)
        file.write(content)
        return file
    elif target == FileType.REAL_FILE:
        with open(filename, mode) as file:
            file.write(content)
            file.close()
        return file

def load_online_files(
        urls: List[str],
        target: FileType = FileType.REAL_FILE,
        downloads_dir: str = "./data_downloads",
        skip_if_exists: bool = True,
    ) -> List[SomeFileType]:
    """Load online files to strings, in-memory files, temporary files, or real files.

    Args:
        urls (List[str]): _description_
        target (FileType, optional): _description_. Defaults to FileType.REAL_FILE.
        downloads_dir (str, optional): _description_. Defaults to "./data_downloads".
        skip_if_exists (bool, optional): _description_. Defaults to True.

    Returns:
        List[SomeFileType]: _description_
    """

    files = []

    for idx, url in enumerate(urls):
        parsed_url = urlparse(url)
        future_filename = os.path.join(downloads_dir, os.path.basename(parsed_url.path))

        if not os.path.exists(future_filename) or (os.path.exists(future_filename) and not skip_if_exists):
            if target == FileType.REAL_FILE:
                os.makedirs(downloads_dir, exist_ok=True)

            if parsed_url.scheme == "http" or parsed_url.scheme == "https":
                response = requests.get(url)
                
                if response.ok:
                    files.append(
                        convert_string_to_file(
                            response.text,
                            target=target,
                            filename=future_filename,
                        )
                    )
                else:
                    print(f"Failed to get `{url}`: {response.status_code}", flush=True)
            elif parsed_url.scheme == "ftp":
                urllib.request.urlretrieve(url, future_filename)
            else:
                print(f"Scheme `{parsed_url.scheme}` not supported. Please use `http`, `https`, or `ftp`.")
        else:
            print(f"Skipping `{url}` download as it already exists at `{future_filename}`")

    return files

def download_youtube_video(video_id: str, download_folder: str = ".") -> str:
    """Download a Youtube video based on its video ID

    Args:
        video_id (str): Youtube video ID
        
    Returns:
        str: _description_
    """
    yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
    stream = yt.streams.get_highest_resolution()
    filename = f"{stream.default_filename.replace('.')[:-1]}.mp4"
    return stream.download(
        output_path=download_folder,
        filename=filename,
    )

#####################################################
# Images, Videos, and Other Media
#####################################################

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
