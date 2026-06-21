import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
from pprint import pprint
import random
from peft import PeftModel
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device


MODEL_PATH="/workspace/experiment/grpo_join/Qwen2.5-VL-new_7.29" 
OUTPUT_PATH=f"/workspace/experiment/grpo_log/grpo_new_test_7.29"


BSZ=2


# DATA_ROOT = "/workspace/ShowUI-main/Value_datasets/metadata/hf_test_full.json"
DATA_ROOT = "/workspace/ShowUI-main/Value_datasets/metadata/hf_test_full.json"

# IMAGE_ROOT = "/workspace/ShowUI-main/Value_datasets/images"
IMAGE_ROOT = "/workspace/ShowUI-main/Value_datasets/images"

random.seed(42)

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# lora_path = '/internfs/Djinhan/VLM-R1/Qwen2.5-VL-3B-Instruct'
# model = PeftModel.from_pretrained(model, lora_path, adapter_name="refta_1800")
# model = model.merge_and_unload()
# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# model, processor = accelerator.prepare(model, processor)

processor.tokenizer.padding_side = 'left'


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

sample_num = 4
data = []
data = json.load(open(DATA_ROOT, "r"))
# random.shuffle(data)
# data = data[:sample_num]



messages = []

for x in data:
    image_path = os.path.join(IMAGE_ROOT, x['img_url'])
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
                "text": (
                f"Given a GUI screenshot and an icon-finding instruction: {x['element'][0]['instruction']}, "
                "identify a **meaningful sub-region** within the image that contains the icon and provides helpful visual context, "
                "such as a window, panel, or functional section where the icon is located. "
                "The selected region should help users or models narrow down the icon's location more easily. "
                "Ensure the region contains the icon, fits logically as a sub-area of the interface, and is not excessively large. "
                "The bounding box must stay within the image and have a minimum size of 600*600 pixels. "
                "First, output the reasoning process in <think> </think> tags, explaining why the selected region is appropriate. "
                "Then, output the final result in <answer> </answer> tags as a JSON: {\"box_2d\": [x1, y1, x2, y2]}."
            )
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
    inputs = inputs.to(model.device)

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
mianji_list = []
for input_example, model_output in zip(data, all_outputs):
    element = input_example['element'][0]
    image_size = input_example['img_size']
    point = element['point']
    
    # point = [point[0]*image_size[0], point[1]*image_size[1]]

    pattern = r'"box_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    match = re.search(pattern, model_output)
    if match:
        coordinate = list(map(int, match.groups()))
        x1, y1, x2, y2 = coordinate
    else:
        coordinate = [0,0,0,0]
        x1, y1, x2, y2 = 0, 0, 0, 0
    
    mianji = (y2-y1) * (x2-x1)

    mianji_list.append(mianji)
    # Count correct answers
    correct = 0
    print(f"point: {point}, pred_bbox: {coordinate}")
    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
        correct = 1
        correct_number += 1
    
    # Create a result dictionary for this example
    result = {
        'img_filename': input_example['img_url'],
        'question': input_example['element'][0]['instruction'],
        'point': point,
        'model_output': model_output,
        'extracted_answer': coordinate,
        'correct': correct,
        'mianji': mianji
    }
    final_output.append(result)

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
avg_mianji = sum(mianji_list)/len(mianji_list)
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'avg_mianji': avg_mianji,
        'results': final_output
    }, f, indent=2)

print(f"Results saved to {output_path}")
print("-"*100)





