
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.cache_utils import StaticCache
from transformers.utils import ModelOutput
import copy

class Qwen2VLForInterCoT(Qwen2VLForConditionalGeneration):
    num_line_break = 0
    num_sub_imgs = 0
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        judge = input_ids[:, -1] in [198]
        if judge:
            self.num_line_break += 1

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                self.reasoning_img_embeds = image_embeds[-(image_grid_thw[:, 1] * image_grid_thw[:, 2] // 4)[-1]:, ...]
                self.query_image_start = (input_ids == 151652).nonzero(as_tuple=True)[1][-1] + 1
                self.query_image_end = (input_ids == 151653).nonzero(as_tuple=True)[1][-1]
                self.query_image_mask = torch.zeros_like(input_ids, device=input_ids.device, dtype=torch.bool)
                self.query_image_mask[:, self.query_image_start: self.query_image_end] = True

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                delta += self.num_sub_imgs*18
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        if self.num_line_break and self.num_line_break % 2 == 0 and self.num_sub_imgs < 3:
            
            tmp_copy_pkv = copy.deepcopy(past_key_values)
            with torch.no_grad():
                outputs = self.model(
                                input_ids=None,
                                past_key_values=tmp_copy_pkv,
                                inputs_embeds=inputs_embeds,
                                use_cache=False,
                                output_attentions=True,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                            )
            del tmp_copy_pkv
            image_attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]
            if self.query_image_mask.shape[-1] != image_attentions.shape[-1]:
                self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                      image_attentions.shape[-1] - self.query_image_mask.shape[-1],
                                                                                      device=self.query_image_mask.device).bool()],
                                                                                      dim=1)
            image_attentions = image_attentions[self.query_image_mask]
            indices = image_attentions.topk(16)[1].sort()[0]
            sampled_reasoning_embeds = self.reasoning_img_embeds[indices]
            x_ids = torch.tensor([151652] + [151655]*16 + [151653], device=input_ids.device).unsqueeze(0)
            x_embeds = self.model.embed_tokens(x_ids)
            x_embeds = x_embeds.masked_scatter((x_ids == 151655).unsqueeze(-1).expand_as(x_embeds), sampled_reasoning_embeds.unsqueeze(0))
            inputs_embeds = torch.cat([inputs_embeds, x_embeds], dim=1)
            interleaved_position_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5], 
                                                    [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5], 
                                                    [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]], device=input_ids.device).unsqueeze(1)
            position_ids = torch.cat([position_ids, interleaved_position_ids + position_ids.max() + 1], dim=-1)
            self.rope_deltas += -12
            self.num_sub_imgs += 1
            cache_position += 18*(self.num_sub_imgs-1)
            cache_position = torch.cat([cache_position, torch.arange(18, device=cache_position.device) + cache_position[-1]+1], dim=0)
        
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], self.num_sub_imgs*18), device=attention_mask.device)], dim=-1)
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
