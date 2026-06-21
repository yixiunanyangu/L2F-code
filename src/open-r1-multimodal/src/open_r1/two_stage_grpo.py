# end_to_end_two_stage.py
# 只需修改 import 和添加几个新参数

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
# ===== 修改：导入两阶段 Trainer =====
from trainer.insert_trainer import TwoStageGRPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.trainer.grpo_config import GRPOConfig
import json
import numpy as np

# Flash attention fix（保持原有）
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
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

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ===== 参数定义（保持原有 + 新增两阶段参数）=====
@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    image_root: Optional[str] = field(default=None)
    
    # ===== 新增：两阶段参数 =====
    region_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained region prediction model"},
    )
    use_gt_region: bool = field(
        default=False,
        metadata={"help": "Use ground truth region from dataset instead of prediction"},
    )
    crop_min_size: int = field(
        default=224,
        metadata={"help": "Minimum size for cropped region"},
    )
    crop_padding_ratio: float = field(
        default=0.1,
        metadata={"help": "Padding ratio when cropping"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


# 系统提示词（保持原有）
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class LazySupervisedDataset(Dataset):
    """保持原有的数据加载逻辑"""
    
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = json.load(open(data_path, "r"))

    def __len__(self):
        return len(self.list_data_dict)

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
                                f"Given a GUI screenshot and an instruction: {example['element'][0]['instruction']}, "
                                "your task is to find the exact point (x, y coordinate) where the user should click or tap to perform the requested action. "
                                "Analyze the interface carefully and identify the specific UI element that corresponds to the instruction. "
                                "First, think through your reasoning in <think> </think> tags, explaining why you selected this specific point. "
                                "Then, output only the final coordinate in <answer> </answer> tags as a JSON object: {\"point\": [x, y]}. "
                                "Make sure the point is within the image boundaries and precisely targets the relevant UI element."
                                "Then, output the final result in <answer> </answer> tags as a JSON: {\"point\": [x,y]}."
                            )},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        
        image_path = os.path.join(image_root, example['img_url'])
        if not os.path.exists(image_path):
            return self.__getitem__((i + 1) % len(self))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((i + 1) % len(self))

        point = example['element'][0]['point']
        bbox = example['element'][0]['bbox']
        region_bbox = example['element'][0].get('region_bbox', None)  # 获取 region
        
        return {
            'image': image,
            'problem': example['element'][0]['instruction'],
            'solution': {
                "point": point,
                "bbox": bbox,
                "region_bbox": region_bbox,  # 传递 region
            },
            'prompt': make_conversation_image(example)['prompt'],
        }


# ===== Reward 函数（需要修改以支持坐标转换）=====

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
    return float(inter)/union if union > 0 else 0


def result_reward(completions, solution, converted_points=None, **kwargs):
    """
    准确性奖励 - 修改以支持坐标转换
    优先使用 converted_points（从裁剪图坐标转换到原图坐标）
    """
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        tar_bbox = sol['bbox']
        
        # 优先使用转换后的坐标
        pred_point = None
        if converted_points and idx < len(converted_points):
            pred_point = converted_points[idx]
        
        # 如果没有转换坐标，从输出中解析
        if pred_point is None:
            pattern_point = r'"point"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]'
            match_point = re.search(pattern_point, content)
            if match_point:
                pred_point = [int(match_point.group(1)), int(match_point.group(2))]
        
        # 计算奖励
        if pred_point:
            x_point, y_point = pred_point
            x1, y1, x2, y2 = tar_bbox
            if x1 <= x_point <= x2 and y1 <= y_point <= y2:
                reward = 1.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Pred point: {pred_point}, Target bbox: {tar_bbox}\n")
                f.write(f"Content: {content[:200]}...\n\n")
    
    return rewards


def format_reward(completions, **kwargs):
    """格式奖励（保持原有）"""
    pattern = re.compile(
        r"<think>.*?</think>\s*<answer>\s*\{\"point\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\}\s*</answer>",
        re.DOTALL
    )
    
    format_rewards = []
    for completion in completions:
        content = completion[0]["content"]
        is_valid = bool(pattern.fullmatch(content))
        format_rewards.append(1.0 if is_valid else 0.0)
    
    return format_rewards


reward_funcs_registry = {
    "accuracy": result_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("=" * 60)
    print("Two-Stage GRPO Training")
    print("=" * 60)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Region Model: {script_args.region_model_path}")
    print(f"Use GT Region: {script_args.use_gt_region}")
    print(f"Reward functions: {[f.__name__ for f in reward_funcs]}")
    print("=" * 60)

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    # ===== 使用两阶段 Trainer =====
    trainer = TwoStageGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        # ===== 新增两阶段参数 =====
        region_model_path=script_args.region_model_path,
        use_gt_region=script_args.use_gt_region,
        crop_min_size=script_args.crop_min_size,
        crop_padding_ratio=script_args.crop_padding_ratio,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
