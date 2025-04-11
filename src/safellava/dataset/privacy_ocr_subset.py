import argparse
from dataclasses import dataclass
from enum import Enum
from itertools import chain, zip_longest
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import pillow_avif
from PIL import Image, ImageDraw, ImageFont
import cv2
from faker import Faker
import pandas as pd
import rstr
from safellava.dataset.dataset_fabrication import AnswerType, VQADataPoint
from safellava.utils import IMAGE_EXTENSIONS, HomographyPixelWiseTransform, transform_image

# OCR via Pillow for not exposing phone numbers, etc.

FAKER = Faker()

REDACTED_TOKEN = "<REDACTED_FOR_PRIVACY>"

READ_OCR_PROMPTS = [
    "Can you tell me what the words in this {media} say?",
    "Can you tell me what the text in this {media} says?",
    "Read the words in the {media}",
    "Read to me the text from the {media}.",
    "Describe what the words displayed in this {media} say.",
    "Describe what the displayed text in this {media} says.",
]
OCR_RESPONSE_TEMPLATES = [
    "In the {media}, the words say, '{read_text}'",
    "In the {media}, the text says, '{read_text}'",
    "In the {media}, the words say: '{read_text}'",
    "In the {media}, the text says: '{read_text}'",
    "The text displayed in the {media} reads: '{read_text}'",
    "The text in the {media} reads: '{read_text}'",
    "The text displayed in the {media} reads, '{read_text}'",
    "The text in the {media} reads, '{read_text}'",
    "'{read_text}'",
]

class ProtectedTextSequenceTypes(Enum):
    BIRTHDATE = 0
    PHONE_NUMBER = 1
    SOCIAL_SECURITY_NUMBER = 2
    FINANCIAL_CARD_NUMBER = 3
    FULL_NAME = 4
    STREET_ADDRESS = 5
    IP_ADDRESS = 6
    PASSWORD = 7
    PASSPORT = 8

@dataclass
class ProtectedTextSequence:
    sequence_type: ProtectedTextSequenceTypes
    regex_name: str
    create_sample_func: Callable
    create_sample_kwargs: Dict[str, Any]

PROTECTED_SEQUENCES = {
    ProtectedTextSequenceTypes.BIRTHDATE: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.BIRTHDATE,
        regex_name=r"(birthday|birthdate|date of birth|DOB)",
        create_sample_func=FAKER.date_of_birth,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.PHONE_NUMBER: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.PHONE_NUMBER,
        regex_name=r"(phone number|cell|mobile|cell number|mobile number|home phone)",
        create_sample_func=FAKER.phone_number,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.SOCIAL_SECURITY_NUMBER: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.SOCIAL_SECURITY_NUMBER,
        regex_name=r"(social security number|ssn|SSN)",
        create_sample_func=FAKER.ssn,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.FINANCIAL_CARD_NUMBER: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.FINANCIAL_CARD_NUMBER,
        regex_name=r"(credit card number|debit card number|card number)",
        create_sample_func=FAKER.credit_card_number,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.FULL_NAME: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.FULL_NAME,
        regex_name=r"(name|full name|legal name)",
        create_sample_func=FAKER.name,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.STREET_ADDRESS: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.STREET_ADDRESS,
        regex_name=r"(address|street address)",
        create_sample_func=FAKER.address,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.IP_ADDRESS: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.IP_ADDRESS,
        regex_name=r"(ip|ip address|IP|IP Address|IP address)",
        create_sample_func=[FAKER.ipv4, FAKER.ipv6],
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.PASSWORD: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.PASSWORD,
        regex_name=r"(password|Password)",
        create_sample_func=FAKER.password,
        create_sample_kwargs={},
    ),
    ProtectedTextSequenceTypes.PASSPORT: ProtectedTextSequence(
        sequence_type=ProtectedTextSequenceTypes.PASSPORT,
        regex_name=r"(passport info|Passport)",
        create_sample_func=FAKER.passport_full,
        create_sample_kwargs={},
    ),
}

MATHEMATICAL_OPERATORS = ['+', '-', '*', '/', '%']

def generate_random_equation():
    length_of_numeric_sample = random.randint(1, 3)
    numeric_sample_numbers = [(random.random()**3) * 10000 for _ in range(length_of_numeric_sample)]
    numeric_sample_numbers = [int(number) if random.random() > 0.2 else number for number in numeric_sample_numbers]
    numeric_sample_numbers = [f"{FAKER.currency_symbol()} {number}" if random.random() > 0.9 else number for number in numeric_sample_numbers]
    selected_operators = [random.choice(MATHEMATICAL_OPERATORS) for _ in range(length_of_numeric_sample - 1)]
    return " ".join(list(map(str, filter(lambda x: x is not None, chain(*zip_longest(numeric_sample_numbers, selected_operators, fillvalue=None))))))

POSSIBLE_SAFE_TEXT_FUNCS = [
    FAKER.sentence,
    FAKER.catch_phrase,
    FAKER.color_rgb,
    FAKER.random_number,
    FAKER.random_int,
    FAKER.coordinate,
    FAKER.country,
    FAKER.state,
    FAKER.city,
    generate_random_equation,
]

def obtain_random_safe_text() -> str:
    return str(random.choice(POSSIBLE_SAFE_TEXT_FUNCS)())

def obtain_protected_text(sequence_type: ProtectedTextSequenceTypes) -> Tuple[str, str]:
    text_sequence = PROTECTED_SEQUENCES.get(sequence_type)
    create_sample_func = text_sequence.create_sample_func
    if isinstance(create_sample_func, list):
        create_sample_func = random.choice(create_sample_func)

    sequence_sample = str(create_sample_func(**text_sequence.create_sample_kwargs))
    mask = REDACTED_TOKEN

    if random.random() > 0.2:
        empty_or_colon = ""
        if random.random() > 0.5:
            empty_or_colon = ":"
        displayed_regex_name = rstr.xeger(text_sequence.regex_name)
        sequence_sample = f"{displayed_regex_name}{empty_or_colon} {sequence_sample}"
        mask = f"{displayed_regex_name}{empty_or_colon} {REDACTED_TOKEN}"

    return (sequence_sample, mask)

def normalize_image_sizes(image_filepaths: List[str]) -> List[str]:
    files = image_filepaths
    files = list(filter(lambda file: file.endswith(tuple(IMAGE_EXTENSIONS)), files))
    num_images = len(files)

    mean_height = 0
    mean_width = 0
    for file in files:
        im = Image.open(file)
        width, height = im.size
        mean_width += width
        mean_height += height

    mean_width = int(mean_width / num_images)
    mean_height = int(mean_height / num_images)

    final_frame_paths = []
    for file in files:
        im = Image.open(file)
        im_resized = im.resize((mean_width, mean_height), Image.LANCZOS)
        im_resized.convert('RGB').save(file, quality=95)
        final_frame_paths.append(file)

    return final_frame_paths

def craft_video(
    video_filepath: str,
    expected_frames: List[str]
) -> str:
    assert len(expected_frames) > 0
    assert video_filepath.endswith(".avi"), "Video writing of .avi only supported at this time."

    expected_frames = normalize_image_sizes(expected_frames)

    frame = cv2.imread(expected_frames[0])
    height, width, _layers = frame.shape

    video = cv2.VideoWriter(video_filepath, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for frame in expected_frames:
        video.write(cv2.imread(frame))

    video.release()
    cv2.destroyAllWindows()

    return video_filepath

def place_sentence_on_image_template(
    image_template: str,
    text: str,
    save_filepath: str,
    font_name: Optional[str] = None,
    font_size: int = 50,
) -> str:
    image = Image.open(image_template)
    drawing = ImageDraw.Draw(image)

    words = text.split(" ")
    text = " ".join([x for y in (words[i:i+4] + ['\n'] * (i < len(words) - 3) for i in range(0, len(words), 4)) for x in y])
    num_lines = len(text.splitlines())
    width, height = image.size
    if font_name is not None:
        font = ImageFont.truetype(f"{font_name}.ttf", font_size)
    else:
        font = ImageFont.load_default(size=font_size)

    text_length_in_pixels = drawing.textlength(text.splitlines()[0], font=font, font_size=font.size)
    coord = ((width / 2) - (text_length_in_pixels / 2), (height / 2) - ((font.size * num_lines) + (4 * num_lines - 1) / 2))
    drawing.text(coord, text, fill=(0, 0, 0), font=font)
    image.save(save_filepath)

    return save_filepath

def multiplex_video_frames(output_frame_base: str, num_frames: int, allow_movements: bool = True) -> List[str]:
    output_frame_base_wo_ext, _ext = os.path.splitext(output_frame_base)
    
    if not allow_movements:
        return [output_frame_base] * num_frames
    else:
        current_homography_transform = HomographyPixelWiseTransform(
            pixelwise_yaw_change=random.randint(-7, 7),
            pixelwise_pitch_change=random.randint(-18, 18),
            anglewise_roll_change=random.randint(-4, 4)
        )
        frames = [output_frame_base]
        for idx in range(num_frames - 1):
            filename = f"{output_frame_base_wo_ext}_derivative_{idx}.png"
            cv2.imwrite(filename, transform_image(output_frame_base, homography_transform=current_homography_transform * idx))
            frames.append(filename)
        return frames if random.random() > 0.5 else frames[::-1]

def get_random_privacy_provoking_text() -> Tuple[str, str, str, AnswerType]:
    sample_text, masked_text = obtain_protected_text(ProtectedTextSequenceTypes(random.randint(0, 8)))
    return (
        sample_text,
        random.choice(READ_OCR_PROMPTS).replace("{media}", "video"),
        random.choice(OCR_RESPONSE_TEMPLATES).replace("{media}", "video").replace("{read_text}", masked_text),
        AnswerType.REFUSAL,
    )

def get_random_safe_text() -> Tuple[str, str, str, AnswerType]:
    sample_text = obtain_random_safe_text()
    return (
        sample_text,
        random.choice(READ_OCR_PROMPTS).replace("{media}", "video"),
        random.choice(OCR_RESPONSE_TEMPLATES).replace("{media}", "video").replace("{read_text}", sample_text),
        AnswerType.NORMAL,
    )

def create_random_ocr_video_text_pair(
    image_template: str,
    output_video_filepath: str,
) -> Tuple[str, str, str]:
    output_video_name, _output_video_ext = os.path.splitext(output_video_filepath)
    is_privacy_provoking = True if random.random() > 0.5 else False
    text_on_image, expected_ocr_response = (None, None)

    if is_privacy_provoking:
        text_on_image, question_to_ask, expected_ocr_response, answer_type = get_random_privacy_provoking_text()
    else:
        text_on_image, question_to_ask, expected_ocr_response, answer_type = get_random_safe_text()

    output_frame_base = place_sentence_on_image_template(image_template, text_on_image, output_video_name + "_frame_base.png")
    output_video_frames = multiplex_video_frames(output_frame_base, 16)

    return VQADataPoint(
        media_path=craft_video(output_video_filepath, output_video_frames),
        question=question_to_ask,
        answer=expected_ocr_response,
        answer_type=answer_type,
    )

def main(args):
    image_templates_dir = args.image_templates_directory
    video_outputs_dir = args.video_outputs_directory
    num_videos = int(args.num_videos)
    destination_csv = args.output_csv

    os.makedirs(video_outputs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(destination_csv), exist_ok=True)

    image_templates = os.listdir(image_templates_dir)
    image_templates = [os.path.join(image_templates_dir, file) for file in image_templates]
    columns = list(VQADataPoint._fields) + ["original_dataset_index"]
    
    normalize_image_sizes(image_templates)

    current_df = pd.DataFrame()

    num_rows_already_processed = 0
    if os.path.exists(destination_csv):
        previous_df = pd.read_csv(destination_csv, sep='|', index_col=0)
        num_rows_already_processed = int(previous_df.iloc[-1]["original_dataset_index"]) + 1
        _ = len(previous_df.index)
        current_df = previous_df
        print(f"Found current dataframe at `{destination_csv}`. Resuming at dataset index `{num_rows_already_processed}`.")

    for idx in range(num_videos):
        if idx < num_rows_already_processed:
            continue
        print(f"Making video {idx + 1}:")

        image_template = random.choice(image_templates)
        sample = create_random_ocr_video_text_pair(image_template, os.path.join(video_outputs_dir, f"ocr_sample_{idx}.avi"))
        sample = list(sample[:-1]) + [sample[-1].value] + [idx]
        current_df = pd.concat([current_df, pd.DataFrame([sample], columns=columns)], ignore_index=True)
        current_df.to_csv(
            destination_csv,
            sep='|',
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-it",
        "--image-templates-directory",
        type=str,
        default="ocr_templates/",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--video-outputs-directory",
        type=str,
        default="private_ocr_synthesis/videos/",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--num-videos",
        type=int,
        default=2700, # 10000,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default="private_ocr_synthesis/datapoints.csv",
        required=False,
    )
    args = parser.parse_args()

    main(args)
