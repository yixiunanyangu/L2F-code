# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from venv import logger

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
import numpy as np

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func  #9.1修改
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  #9.1修改
#     Qwen2_5_VLVisionAttention,  # 使用这个类
#     apply_rotary_pos_emb_flashatt,
#     flash_attn_varlen_func
# )
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward    #9.1修改
# Qwen2_5_VLVisionAttention.forward = custom_forward #9.1修改


# ----------------------- Main Script -----------------------
#设置最大像素和最小像素
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


#系统提示词
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        self.list_data_dict = json.load(open(data_path, "r"))
        # for file in os.listdir(data_path):
        #     print(f"Processing {file}...")
        #     ds_path = os.path.join(data_path, file)
        #     sub_data = json.load(open(ds_path, "r"))
        #     for x in sub_data:
        #         image_size = x['img_size']
        #         if image_size[0] * image_size[1] > 6000000: 
        #         # if image_size[0]>=3840 and image_size[1] >= 2160: 
        #             continue
        #         else:
        #             self.list_data_dict.append(x)
            # self.list_data_dict.extend(sub_data)

    def __len__(self):
        return len(self.list_data_dict)

    #修改提示词：先进行区域框的预测，打一个区域框的标签<region> </region>。然后再在这个区域框里面进行预测答案，<answer> </answer>则是最终point
    def __getitem__(self, i):
        def make_conversation_image(example):
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": (
                                f"Given a GUI screenshot and an icon-finding instruction: {example['element'][0]['instruction']}, "
                                "identify a **meaningful sub-region** within the image that contains the icon and provides helpful visual context, "
                                "such as a window, panel, or functional section where the icon is located. "
                                "The selected region should help users or models narrow down the icon's location more easily. "
                                "Ensure the region contains the icon, fits logically as a sub-area of the interface, and is not excessively large. "
                                "The bounding box must stay within the image and have a minimum size of 600*600 pixels. "
                                "First, output the reasoning process in <think> </think> tags, explaining why the selected region is appropriate. "
                                "Then, output the final result in <answer> </answer> tags as a JSON: {\"box_2d\": [x1, y1, x2, y2]}."
                                )
                            },
                        ],
                    },
                ],

                #对比random
                # "prompt": [
                #     {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "image"},
                #             {"type": "text", "text": (
                #                 f"<image>Given a GUI screenshot and an icon-finding instruction: {example['element'][0]['instruction']}, "
                #                 " identify a sub-region containing the icon. "
                #                 "The bounding box must stay within the image and have a minimum size of 600*600 pixels. "
                #                 "First, output the reasoning process in <think> </think> tags, explaining why the selected region is appropriate. "
                #                 "Then, output the final result in <answer> </answer> tags as a JSON: {\"box_2d\": [x1, y1, x2, y2]}."
                #                 )
                #             },
                #         ],
                #     },
                # ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        
        # image_path = os.path.join(image_root, example['img_url'])
        # image = Image.open(image_path).convert("RGB")

        #7.29修改
        image_path = os.path.join(image_root, example['img_url'])
        if not os.path.exists(image_path):
            # 如果图片不存在，跳过此样本
            return self.__getitem__((i + 1) % len(self))  # 尝试下一个样本，避免索引越界
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((i + 1) % len(self))

        #改成输入的是region_bbox
        #7.29到此
        
        point = example['element'][0]['point']
        region_bbox = example['element'][0]['region_bbox']  #7.29更改
        # image_size = example['img_size']
        # point = [int(point[0]*image_size[0]), int(point[1]*image_size[1])]
        # region_bbox = [int(region_bbox[0]*image_size[0]), int(region_bbox[1]*image_size[1]), int(region_bbox[2]*image_size[0]), int(region_bbox[3]*image_size[1])]
        
        return {
            'image': image,
            'problem': example['element'][0]['instruction'],
            'solution': {"point": point, "region_bbox": region_bbox}, #7.29更改
            # 'solution': {"point": point},  #7.29更改
            'prompt': make_conversation_image(example)['prompt'],
        }

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''


#计算交并比
def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union



def result_reward(completions, solution, **kwargs):
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        model_output = content
        point = sol['point']
        region_bbox = sol['region_bbox']

        pattern = r'"box_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        match = re.search(pattern, model_output)
        if match:
            coordinate = list(map(int, match.groups()))
            x1, y1, x2, y2 = coordinate
        else:
            coordinate = [0,0,0,0]
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        mianji = (y2-y1) * (x2-x1)

        # Count correct answers
        # print(f"point: {point}, pred_bbox: {coordinate}")
        in_reward = 0
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            in_reward = 1.0
        
        iou_reward = iou(coordinate, region_bbox)
        
        
        #修改奖励函数：改成预测框和最终任务点得分的总和，reward = region_reward + in_reward * area_reward
        ideal_area = 614656
        area_reward = np.exp(-((mianji - ideal_area) ** 2) / (2 * (0.25 * ideal_area) ** 2))
        reward = in_reward * iou_reward + area_reward*0.2
        
        # reward = in_reward * area_reward #7.31改

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards



#修改格式得分，格式为：region+answer
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    # completion_contents = [completion[0]["content"].split('assistant\n')[1] for completion in completions]
    # matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    # 正则模式匹配结构化输出：<think>...</think> <answer> {...[x1, y1, x2, y2]...}</answer>
    pattern = re.compile(
        r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?\}.*?</answer>",
        re.DOTALL
    )

    format_rewards = []
    for completion in completions:
        # 提取 assistant 的输出内容
        content = completion[0]["content"]
        # 判断是否完全匹配所要求的格式
        is_valid = bool(pattern.fullmatch(content))
        format_rewards.append(1.0 if is_valid else 0.0)

    return format_rewards


reward_funcs_registry = {
    "accuracy": result_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args), #model_args = GRPOModelConfig(model_name_or_path='/workspace/VLM-R1/Qwen2.5-VL-3B-Instruct'
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
