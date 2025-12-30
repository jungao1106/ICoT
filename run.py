import os
import argparse
from ruamel.yaml import YAML
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/config.yaml', help='global environment configs')
args = parser.parse_args()
yaml = YAML()

# Reading a YAML file
with open(args.config, 'r') as file:
    config = yaml.load(file)
    print(config)
    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.environ.get('CUDA_VISIBLE_DEVICES', -1) == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import json
import copy
import random
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

IMG_FOLDER = './data/m3cot/images/'
EVAL_FILE = './data/m3cot/test.jsonl'
DATA_NAME = 'm3cot'


@dataclass
class CausalLMOutputWithPastForInterCoT(CausalLMOutputWithPast):
    selected_vokens: torch.LongTensor = None 


class ChameleonForInterCoT(ChameleonForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        sub_image_masks = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # inilize for kv cot
        if output_attentions and past_key_values.key_cache == [] and pixel_values is not None:
            self.new_tokens = 0
            self.num_selected_patches = num_selected_patches
            self.query_image_start = (input_ids == 8197).nonzero(as_tuple=True)[1][-1] + 1
            self.boi_token, self.eoi_token = input_ids[:, self.query_image_start-1].unsqueeze(0), input_ids[:, self.query_image_start+1024].unsqueeze(0)
            with torch.no_grad():
                self.query_vokens = self.model.get_image_tokens(pixel_values[-1].unsqueeze(0))
            self.query_image_mask = torch.zeros_like(input_ids, device=input_ids.device).bool()
            self.query_image_mask[:, self.query_image_start: self.query_image_start+1024] = True

            self.num_line_break = 0

        elif output_attentions:
            self.new_tokens += 1
            false_tensor = torch.tensor([[False]], device=self.query_image_mask.device)
            self.query_image_mask = torch.cat([self.query_image_mask, false_tensor], dim=-1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            sub_image_masks=sub_image_masks,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        logits = logits.float()

        # Disallow image tokens which does not include special begin-image and end-image tokens
        image_tokens = self.model.vocabulary_mapping.image_tokens

        selected_vokens = None
        if output_attentions and input_ids[:, -1] == 16397:
            self.num_line_break += 1
        
        if output_attentions and self.num_line_break % 2 == 0 and (input_ids[:, -1] == 16397 or pixel_values is not None) :
            image_attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]
            image_attentions = image_attentions[self.query_image_mask]
            selected_patches = image_attentions.topk(self.num_selected_patches)[1]
            selected_patches = sorted((selected_patches).tolist())
            selected_vokens = torch.cat([self.boi_token,
                                         self.query_vokens[:, selected_patches],
                                         self.eoi_token], dim=-1)
            selected_patches = torch.tensor(selected_patches, device=input_ids.device) + self.query_image_start
            self.query_image_mask[:, selected_patches] = False
            self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                  self.num_selected_patches+2,
                                                                                  device=self.query_image_mask.device).bool()],
                                                                                  dim=1)
        logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

        loss = None
        if labels is not None:
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

        return CausalLMOutputWithPastForInterCoT(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            selected_vokens=selected_vokens,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                if 'selected_vokens' in kwargs and kwargs['selected_vokens'] is not None:
                    input_ids = input_ids[:, cache_position[-1]-kwargs['selected_vokens'].shape[-1]: ]
                else:
                    input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            cache_position = position_ids[0] #TODO: There might be a better way :)
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be `None` because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if 'sub_image_masks' in kwargs:
            model_inputs.update(
                    {
                        "sub_image_masks": kwargs['sub_image_masks'],
                    }
                )


        return model_inputs


    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            cache_name, cache = self._extract_past_from_model_output(outputs)
            model_kwargs[cache_name] = cache


            if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
                model_kwargs['selected_vokens'] = outputs['selected_vokens']
            elif 'selected_vokens' in model_kwargs:
                model_kwargs.pop('selected_vokens')

            if getattr(outputs, "state", None) is not None:
                model_kwargs["state"] = outputs.state

            # update token_type_ids with last value
            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
                
            if not is_encoder_decoder:
                # update attention mask
                if "attention_mask" in model_kwargs:
                    attention_mask = model_kwargs["attention_mask"]
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
                        model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones_like(outputs['selected_vokens'])], dim=-1)
            else:
                # update decoder attention mask
                if "decoder_attention_mask" in model_kwargs:
                    decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                    model_kwargs["decoder_attention_mask"] = torch.cat(
                        [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                        dim=-1,
                    )
            #TODO: cache_position is not applied to the prefix vokens
            if model_kwargs.get("use_cache", True):
                if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
                    num_new_tokens += outputs['selected_vokens'].shape[-1]
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            else:
                past_positions = model_kwargs.pop("cache_position")
                new_positions = torch.arange(
                    past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
                ).to(past_positions.device)
                model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
            return model_kwargs

num_selected_patches = 64
MCOT = config['MCOT']


TRAING_CASE_1 = {'id': 'physical-commonsense-1426',
 'category': 'Status',
 'image_id': 'commonsense-physical-commonsense-49',
 'question': 'What general conclusion can you draw about this kitchen?',
 'choices': ['This is the kitchen of a restaurant',
  'The equipment in front has not been cleaned for a long time',
  'Someone searched in this kitchen',
  'All options are correct'],
 'context': '',
 'answer': 'D',
 'rationale': 'First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant.\nTherefore, option A is correct.\nSecond, there are grease stains on the front of appliances which are indicative of not being cleaned in a while.\nSo option B is correct answer.\nThird, cabinet doors are opened up throughout the kitchen which shows someone was searching for something.\nSo option C is correct.\nTherefore, we can infer that option A, B and C are all correct.\nSo, option "(D) All options are correct" is correct answer.',
 'split': 'train',
 'image': 'data\\images\\physical-commonsense-1426.png',
 'domain': 'commonsense',
 'topic': 'physical-commonsense'}


dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
dataset = [x for x in dataset if x['image'] is not None ]

model_path = './models/chameleon-7b'
processor = ChameleonProcessor.from_pretrained(model_path)
model = ChameleonForInterCoT.from_pretrained(model_path, attn_implementation=config['attn']).to(device='cuda', dtype=torch.bfloat16)

generation_config = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'min_new_tokens': 32,
    'max_new_tokens': 512
}


def calculate_generated_text(prompt, vision_x, return_image_masks=False):
    """
    Calculate generated text given a prompt and vision data.

    Parameters:
    - prompt (str): The input prompt.
    - vision_x (list[PIL Images]): List of PIL Images containing vision data.

    Returns:
    Tuple[str, str]: Tuple containing the raw and salt answer text.
    """

    """
    Example Prompt:
    In zero-shot: "<image> <Question> <Options> Answer: "
    In few-shot: "<image> <Question> <Options> Answer: <Answer> <image> <Question> <Options> Answer: "
    """
    
    inputs, sub_image_masks = processor(text = prompt, images=vision_x, padding=True, return_tensors="pt", return_for_text_completion=False, return_image_masks=True)

    inputs = inputs.to(device='cuda', dtype=torch.bfloat16)
    sub_image_masks = sub_image_masks.view(1, -1) if return_image_masks else None

    if return_image_masks:
        padding = torch.full((1, 2048 - sub_image_masks.shape[-1]), True, dtype=torch.bool)
        sub_image_masks = torch.cat([sub_image_masks, padding], dim=1).to(device='cuda').view(2, -1)
        if prompt.count('<image>') == 2: # two global images are provided
            sub_image_masks= torch.cat([torch.ones(1, 1024).bool().to(device='cuda'), sub_image_masks], dim=0).to(device='cuda')
        inputs['sub_image_masks'] = sub_image_masks
      
    inputs['output_attentions'] = MCOT
    
    out = model.generate(**inputs,  **generation_config)
    
    out = out[0][inputs['input_ids'].shape[1]: ]
    
    generated_text = processor.decode(out, skip_special_tokens=True)
    
    return generated_text

zero_shot_prompt_template = '''<image>Question: {}
Options:
'''

mcot_induct = '''<image>Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
<image11-21-0-11>First, the image shows large ovens in a kitchen area that indicates it is a kitchen of a restaurant. Therefore, option A is correct. <image4-10-23-29>Second, there are grease stains on the front of appliances which are indicative of not being cleaned in a while. So option B is correct answer. <image21-32-1-9>Third, cabinet doors are opened up throughout the kitchen which shows someone was searching for something. So option C is correct. Therefore, we can infer that option A, B and C are all correct.
Answer: {}'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'], TRAING_CASE_1['answer'])


def main():
    
    mcot_one_fh = open('./results/chameleon/{}/chameleon_mcot_one.json'.format(DATA_NAME), 'a')
    mcot_zero_fh = open('./results/chameleon/{}/chameleon_mcot_zero.json'.format(DATA_NAME), 'a')
    for data in tqdm(dataset):
        mcot_input_str = zero_shot_prompt_template.format(data['question'])
        for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
            mcot_input_str += '{}. {}\n'.format(i, c)

        one_shot_vision = [Image.open(os.path.join('./data/m3cot/images', TRAING_CASE_1['id']+'.png')),
                    Image.open(os.path.join('./data/m3cot/images', TRAING_CASE_1['id']+'.png')),
                    Image.open(os.path.join(IMG_FOLDER, data['id']+'.png' if DATA_NAME == 'm3cot' else data['image'] ))]
        
        zero_shot_vision = [Image.open(os.path.join(IMG_FOLDER, data['id']+'.png' if DATA_NAME == 'm3cot' else data['image'] ))]
        

        one_shot_mcot_input_str = mcot_induct+'\n'+mcot_input_str

        zero_shot_mcot_input_str = mcot_input_str


        one_shot = calculate_generated_text(one_shot_mcot_input_str, one_shot_vision, return_image_masks= True)
        zero_shot = calculate_generated_text(zero_shot_mcot_input_str, zero_shot_vision, return_image_masks= False)

        
        oneshot_mcot_output = copy.deepcopy(data)
        oneshot_mcot_output['pred'] = one_shot
        
        
        zeroshot_mcot_output = copy.deepcopy(data)
        zeroshot_mcot_output['pred'] = zero_shot
        
        mcot_one_fh.write(json.dumps(oneshot_mcot_output) + '\n')
        mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + '\n')
        
if __name__ == '__main__':
    main()
