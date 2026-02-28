from typing import Optional, List, Dict

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.utils.generic import ModelOutput, TransformersKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class Qwen3VLCoTRewardModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    score: torch.FloatTensor | None = None
    

class Qwen3VLCoTRewardModelConfig(Qwen3VLConfig):
    def __init__(
        self, 
        special_token_ids, 
        score_levels,
        score_weight,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.score_levels = score_levels
        self.score_weight = score_weight
        self.special_token_ids = special_token_ids


class Qwen3VLCoTRewardModel(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.score_levels = config.score_levels
        self.score_weight = config.score_weight
        self.special_token_ids = list(config.special_token_ids)
        
        self.score_head = nn.Linear(config.text_config.hidden_size, config.score_levels, bias=False)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: Dict[str, torch.LongTensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: TransformersKwargs,
    ):
        batch_size = input_ids.shape[0]

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0]  # [B, L, D]
        
        logits = self.lm_head(hidden_states)

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == special_token_id)
        assert all(special_token_mask.sum(dim=-1) == len(self.special_token_ids)), f"Number of special tokens in each sample must be {len(self.special_token_ids)}"
        
        socre_hidden_states = hidden_states[special_token_mask]
        score_logits = self.score_head(socre_hidden_states)
        score_logits = score_logits.view(batch_size, len(self.special_token_ids), -1)   # [B, N_special, -1]
        
        loss = None
        if labels is not None:
            lm_labels = labels["lm"]    # [B, L]
            score_labels = labels["score"]  # [B, N_special]
            loss = self.loss_function(logits=logits, labels=lm_labels, vocab_size=self.config.text_config.vocab_size)
            score_loss = nn.CrossEntropyLoss(reduction="none")(score_logits, score_labels)
            loss = loss + self.score_weight * score_loss.mean()
        
        return Qwen3VLCoTRewardModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            score=score_logits,
        )