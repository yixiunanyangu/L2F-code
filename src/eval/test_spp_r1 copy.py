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

# MODEL_PATH=f"path/to/Qwen2.5-VL-3B-GRPO-REC/checkpoint-{steps}" 
MODEL_PATH="/workspace/VLM-R1/Qwen2.5-VL-3B-Instruct" 
OUTPUT_PATH=f"/workspace/VLM-R1/output/logs/inf_0.json"


BSZ=1


DATA_ROOT = "/workspace/ShowUI-main/Value_datasets/metadata/hf_train.json"

# TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
# IMAGE_ROOT = "/data/shz/dataset/coco"

TEST_DATASETS = ['android_studio_macos']
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

sample_num = 12
data = []
for ds in os.listdir(DATA_ROOT):
    print(f"Processing {ds}...")
    # ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    ds_path = os.path.join(DATA_ROOT, ds)
    sub_data = json.load(open(ds_path, "r"))
    data.extend(sub_data)
    # random.shuffle(data)
    # data = data[:sample_num]


QUESTION_TEMPLATE = "Predict the relative position that should be clicked based on the instructions:{Question}. First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
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
                "text": QUESTION_TEMPLATE.format(Question=x['instruction'])
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

for input_example, model_output in zip(data, all_outputs):
    bbox = input_example['bbox']

    coordinate_pattern = r'"coordinate": \[(\d+), (\d+)\]'
    match = re.search(coordinate_pattern, model_output)
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
        'question': input_example['instruction'],
        'ground_truth': bbox,
        'model_output': model_output,
        'extracted_answer': coordinate,
        'correct': correct_number
    }
    final_output.append(result)

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2)

print(f"Results saved to {output_path}")
print("-"*100)





