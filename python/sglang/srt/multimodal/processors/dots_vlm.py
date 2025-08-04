from typing import List, Union, Dict
import re
import math
import asyncio
from PIL import Image
from sglang.srt.models.dots_vlm import DotsVLMForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

class DotsVLMImageProcessor(BaseMultimodalProcessor):
    models = [DotsVLMForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        # The single, pre-expanded image token.
        self.IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"
        # The regex that matches expanded image tokens.
        self.IMAGE_TOKEN_REGEX = re.compile(r"<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>")

        assert len(_processor.tokenizer.encode("<|img|>"))==1
        self.im_start_id = _processor.tokenizer.encode("<|img|>")[0]
        self.im_end_id = _processor.tokenizer.encode("<|endofimg|>")[0]
        self.im_token_id = _processor.tokenizer.encode("<|imgpad|>")[0]
        self.IM_TOKEN_ID = self.im_token_id
        self.IM_START_ID = self.im_start_id
        self.IM_END_ID = self.im_end_id


        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        self.IMAGE_FACTOR = patch_size * merge_size
        self.MIN_PIXELS = _processor.image_processor.min_pixels
        self.MAX_PIXELS = _processor.image_processor.max_pixels
        self.MAX_RATIO = 200

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if isinstance(image_data, str):
            image_data = [image_data]

        if isinstance(image_data, list) and image_data and isinstance(image_data[0], list):
            image_data = sum(image_data, [])

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN,
                image_token_regex=self.IMAGE_TOKEN_REGEX,
            ),
            max_req_input_len=max_req_input_len,
        )

        def smart_resize(
            height: int,
            width: int,
            factor: int = self.IMAGE_FACTOR,
            min_pixels: int = self.MIN_PIXELS,
            max_pixels: int = self.MAX_PIXELS,
        ) -> tuple[int, int]:
            """Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > self.MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
                )
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
            return h_bar, w_bar

        def resize_image(image, size_factor: int = self.IMAGE_FACTOR) -> Image.Image:
            width, height = image.size
            min_pixels = self.MIN_PIXELS
            max_pixels = self.MAX_PIXELS
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            return image

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by
            'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number'
            that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that
            is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        async def resize_image_async(image):
            return resize_image(image)

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [resize_image_async(image) for image in base_output.images]
            base_output.images = await asyncio.gather(*resize_tasks)

        combined_mm_item, input_ids = self.process_and_combine_mm_data(base_output)

        if combined_mm_item is None:
            return None

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": combined_mm_item,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "im_token_id": self.im_token_id,
        }
