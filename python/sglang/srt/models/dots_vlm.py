from typing import Iterable, List, Tuple, Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

from .dots_vlm_vit import DotsVisionTransformer
from sglang.srt.configs.dots_vlm import DotsVLMConfig



class DotsVLMForCausalLM(nn.Module):
    """DotsVLM model for sglang inference"""
    
    def __init__(self, config: DotsVLMConfig, quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        
        self.image_token_id = config.im_span_id
        self.video_token_id = config.video_span_id

        self.language_model = DeepseekV2ForCausalLM(config.language_config, quant_config)
    
        # Initialize vision tower (matching transformers naming for weight compatibility)
        self.vision_tower = DotsVisionTransformer(config.vision_config)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""
        weights = list(weights)
        
        # Separate vision tower weights and language model weights
        vision_weights = []
        language_weights = []
        
        for name, loaded_weight in weights:
            if name.startswith("vision_tower."):
                # Remove "vision_tower." prefix for vision tower weights
                vision_name = name[len("vision_tower."):]
                vision_weights.append((vision_name, loaded_weight))
            else:
                # All other weights go to language model
                language_weights.append((name, loaded_weight))
        
        # Load vision tower weights
        vision_state_dict = dict(vision_weights)
        self.vision_tower.load_state_dict(vision_state_dict, strict=True)
    
        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return DeepseekV2ForCausalLM.get_model_config_for_expert_location(config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input_ids with multimodal tokens"""
        # Get image token ID for padding pattern
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        mm_inputs.im_token_id = self.image_token_id
        padded_input_ids = pattern.pad_input_tokens(input_ids, mm_inputs)
        return padded_input_ids

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features from multimodal data items"""
        if any(item.precomputed_features is not None for item in items):
            # If features are precomputed, just concatenate them
            if not all(item.precomputed_features is not None for item in items):
                raise NotImplementedError("MM inputs where only some items are precomputed.")
            return torch.concat([item.precomputed_features for item in items])
        
        # Extract pixel values and grid information (following reference pattern)
        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).type(self.vision_tower.dtype)
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0).to(self.vision_tower.device)
        
        # Add dimension checks like in reference code
        assert pixel_values.dim() == 2, f"{pixel_values.dim()=}"
        assert image_grid_thw.dim() == 2, f"{image_grid_thw.dim()=}"
        
        # Process through vision tower
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)
        
        # Ensure consistent dtype for FlashInfer compatibility
        # Force bfloat16 to match model's expected dtype
        if image_embeds.dtype != torch.bfloat16 and hasattr(self.language_model.model, 'embed_tokens'):
            target_dtype = self.language_model.model.embed_tokens.weight.dtype
            image_embeds = image_embeds.to(target_dtype)
        
        return image_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            image_data_embedding_func=self.get_image_feature,
            language_model=self.language_model,
        )
        return hidden_states


EntryClass = [DotsVLMForCausalLM]