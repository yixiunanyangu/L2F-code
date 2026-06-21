from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from peft import PeftModel
import pickle

MODEL_PATH="/internfs/Djinhan/VLM-R1/Qwen2.5-VL-3B-Instruct" 
OUTPUT_PATH=f"/internfs/Djinhan/VLM-R1/output/logs/inf_0.json"
BSZ=4
random.seed(42)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

# lora_path = '/internfs/Djinhan/VLM-R1/output/Qwen2.5-VL-7B-GRPO-REC-lora/checkpoint-1800'
# model = PeftModel.from_pretrained(model, lora_path, adapter_name="refta_1800")
# model = model.merge_and_unload()
# default processer

processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = 'left'
def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    # if content_answer_match:
    #     content_answer = content_answer_match.group(1).strip()
    #     bbox_match = re.search(bbox_pattern, content_answer)
    #     if bbox_match:
    #         bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
    #         x1, y1, x2, y2 = bbox
    #         return bbox, False
    # else:
    bbox_pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(bbox_pattern, content)
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        x1, y1, x2, y2 = bbox
        return bbox, False
    return [0, 0, 0, 0], False


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

sample_num = 200

data_path = '/internfs/Djinhan/VLM-R1/mind2web'
category = ['Mind2Web_domain']
episodes = []


for cate in category:
    with open(f"{data_path}/{cate}.obj", "rb") as rp:
        data = pickle.load(rp)
    for episode_id, episode in data.items():
        for cur_episode in episode:
            if cur_episode['target_action'] == 'CLICK':
                episodes.append(cur_episode)

    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format." 
    data = episodes[:sample_num]
    messages = []

    for x in data:
        image_path = x['image_path'].replace('/workspace/GUI/Mind2Web/data', '/internfs/Djinhan/VLM-R1/mind2web')
        question = x['question'].replace('Please generate the next move', 'Please provide the bounding box that requires interaction')
        message = [
            # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question)
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data, all_outputs):
        original_output = model_output
        ground_truth = input_example['bbox']
        bbox = [x * 1000 for x in ground_truth]

        coordinate_pattern = r'"coordinate": \[(\d+), (\d+)\]'
        match = re.search(coordinate_pattern, original_output)
        if match:
            coordinate = [int(match.group(1)), int(match.group(2))]
            x, y = coordinate
        else:
            coordinate = [0,0]
            x, y = 0, 0
        
        # Count correct answers
        correct_number = 0
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            correct_number += 1
        
        # Create a result dictionary for this example
        result = {
            'question': question,
            'bbox': bbox,
            'coordinate': coordinate,
            'correct': correct_number
        }
        final_output.append(result)

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy of {cate}: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print("-"*100)





