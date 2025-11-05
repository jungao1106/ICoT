
import os
import argparse
import random
import torch
import json
import copy
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, set_seed, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from icot_qwen_model import *
from icot_qwen_utils import *


IMG_FOLDER = './data/m3cot/images/'
EVAL_FILE = './data/m3cot/test.jsonl'
DATA_NAME = 'm3cot'

dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
dataset = [x for x in dataset if x['image'] is not None ]

model_path = './models/Qwen2-VL-7B-Instruct'
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2VLForInterCoT.from_pretrained(model_path, attn_implementation="eager").to(device='cuda', dtype=torch.float16)

def calculate_generated_text(messages):
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                    )
    inputs = inputs.to(device='cuda', dtype=torch.bfloat16)
    inputs['output_attentions'] = True
    out = model.generate(**inputs,  **generation_config)
    
    out = out[0][inputs['input_ids'].shape[1]: ]
    
    generated_text = processor.decode(out, skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    mcot_one_fh = open('./results/qwen2-vl/{}/chameleon_mcot_one.json'.format(DATA_NAME), 'a')
    mcot_zero_fh = open('./results/qwen2-vl/{}/chameleon_mcot_zero.json'.format(DATA_NAME), 'a')
    for data in tqdm(dataset):
        mcot_input_str = zero_shot_prompt_template.format(data['question'])
        for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
            mcot_input_str += '{}. {}\n'.format(i, c)
        mcot_input_str += '''Let's think step by step.\n'''
        vision_x = [os.path.join('./data/m3cot/images', TRAING_CASE_1['id']+'.png'),
                    os.path.join(IMG_FOLDER, data['id']+'.png' if DATA_NAME == 'm3cot' else data['image'])]
    

        zero_shot_messages_mcot = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": vision_x[-1],
                    },
                    {"type": "text", "text": mcot_input_str},
                ],
            }
        ]
        one_shot_messages_mcot = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": vision_x[0],
                        },
                        {"type": "text", "text": mcot_induct_0},
                        {
                            "type": "image",
                            "image": './path/to/icot_image_0.png',
                        },
                        {"type": "text", "text": mcot_induct_1},
                        {
                            "type": "image",
                            "image": './path/to/icot_image_1.png',
                        },
                        {"type": "text", "text": mcot_induct_2},
                        {
                            "type": "image",
                            "image": './path/to/icot_image_2.png',
                        },
                        {"type": "text", "text": mcot_induct_3},
                        {"type": "text", "text": mcot_induct_4 + '\n'},

                        {
                            "type": "image",
                            "image": vision_x[-1],
                        },
                        {"type": "text", "text": mcot_input_str},
                    ],
                }
            ]
        
        zero_shot_mcot_ans = calculate_generated_text(zero_shot_messages_mcot)
        one_shot_mcot_ans = calculate_generated_text(one_shot_messages_mcot)

        zero_shot_output = copy.deepcopy(data)
        zero_shot_output['pred'] = zero_shot_mcot_ans
        
        
        one_shot_output = copy.deepcopy(data)
        one_shot_output['pred'] = one_shot_mcot_ans
        
        mcot_zero_fh.write(json.dumps(zero_shot_output) + '\n')
        mcot_one_fh.write(json.dumps(one_shot_output) + '\n')
