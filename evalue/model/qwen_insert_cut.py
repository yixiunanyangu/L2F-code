import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import base64
import re
import os
from io import BytesIO
from PIL import Image

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_qwen2_5vl_prompt_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                        "<think> reasoning process here </think><answer> answer here </answer>"
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"Given a GUI screenshot and an instruction: '{instruction}', "
                        "your task is to find the exact point (x, y coordinate) where the user should click or tap to perform the requested action. "
                        "Analyze the interface carefully and identify the specific UI element that corresponds to the instruction. "
                        "First, think through your reasoning in <think> </think> tags, explaining why you selected this specific point. "
                        "Then, output only the final coordinate in <answer> </answer> tags as a JSON object: {\"point\": [x, y]}. "
                        "Make sure the point is within the image boundaries and precisely targets the relevant UI element."
                    )
                }
            ]
        }
    ]



GUIDED_PROMPT = """<|im_start|>system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
f"Given a GUI screenshot and an instruction: {{instruction}}, "
"your task is to find the exact point (x, y coordinate) where the user should click or tap to perform the requested action. "
"Analyze the interface carefully and identify the specific UI element that corresponds to the instruction. "
"First, think through your reasoning in <think> </think> tags, explaining why you selected this specific point. "
"Then, output only the final coordinate in <answer> </answer> tags as a JSON object: {\"point\": [x, y]}. "
"Make sure the point is within the image boundaries and precisely targets the relevant UI element."
<|im_start|>assistant
<think>"""


class Qwen2_5VLModel_insert():
    # def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    def load_model(self, model_name_or_path="/workspace/ShowUI-main/Test_model/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        # self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.generation_config = GenerationConfig.from_pretrained("/workspace/ShowUI-main/Test_model/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=99999999,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        messages = get_qwen2_5vl_prompt_msg(image, instruction, resized_width, resized_height)

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # guide_text = "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": ["
        # # guide_text = "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"mouse_move\", \"coordinate\": ["
        # text_input = text_input + guide_text
        
        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        print("Len: ", len(inputs.input_ids[0]))
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        # response = guide_text + response
        # cut_index = response.rfind('}')
        # if cut_index != -1:
        #     response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        # try:
        #     action = json.loads(response.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        #     coordinates = action['arguments']['coordinate']
        #     if len(coordinates) == 2:
        #         point_x, point_y = coordinates
        #     elif len(coordinates) == 4:
        #         x1, y1, x2, y2 = coordinates
        #         point_x = (x1 + x2) / 2
        #         point_y = (y1 + y2) / 2
        #     else:
        #         raise ValueError("Wrong output format")
        #     print(point_x, point_y)
        #     result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        # except (IndexError, KeyError, TypeError, ValueError) as e:
        #     pass

        try:
        # Extract reasoning and answer
            reasoning = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            
            if answer:
                
                answer_text = answer.group(1).strip()
                # 移除所有空白字符，包括换行符
                answer_text = re.sub(r'\s+', '', answer_text)
                answer_text = answer_text.replace('\\n', '').replace('\\t', '').replace('\\r', '')

                while answer_text.endswith("}}"):
                    answer_text = answer_text[:-1]

                # answer_json = json.loads(answer.group(1).strip())
                answer_json = json.loads(answer_text)
                if "point" in answer_json:
                    point_x, point_y = answer_json["point"]
                    result_dict["point"] = [point_x / resized_width, point_y / resized_height]
                elif "point_2d" in answer_json:
                    point_x, point_y = answer_json["point_2d"]
                    result_dict["point"] = [point_x / resized_width, point_y / resized_height]

            json_point_pattern = r'\{\s*"point"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\}'
            
            json_match = re.search(json_point_pattern, response)
            if json_match:
                try:
                    point_x, point_y = int(json_match.group(1)), int(json_match.group(2))
                    result_dict["point"] = [point_x / resized_width, point_y / resized_height]
                    print(f"✅ 方法2成功解析坐标: ({point_x}, {point_y})")
                    return result_dict
                except (ValueError, IndexError) as e:
                    print(f"方法2解析失败: {e}")
            
            # 方法3: 尝试匹配简单的 [x, y] 数组格式（在</think>标签之后）
            # 匹配 </think>\n[1487, 69] 这种格式
            array_patterns = [
                r'</think>\s*\[(\d+)\s*,\s*(\d+)\]',  # </think>后面的数组
                r'(\d+)\s*,\s*(\d+)',  # 任意位置的数字对
            ]
            
            for pattern in array_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    if len(match) == 2:
                        try:
                            point_x, point_y = int(match[0]), int(match[1])
                            # 验证坐标合理性
                            if 0 <= point_x <= resized_width and 0 <= point_y <= resized_height:
                                result_dict["point"] = [point_x / resized_width, point_y / resized_height]
                                print(f"✅ 方法3成功解析坐标: ({point_x}, {point_y})")
                                return result_dict
                        except (ValueError, IndexError):
                            continue
                
            numbers = re.findall(r'\b(\d+)\b', response)
            if len(numbers) >= 2:
                # 取最后两个数字作为坐标
                try:
                    point_x, point_y = int(numbers[-2]), int(numbers[-1])
                    if 0 <= point_x <= resized_width and 0 <= point_y <= resized_height:
                        result_dict["point"] = [point_x / resized_width, point_y / resized_height]
                        print(f"✅ 方法4成功解析坐标: ({point_x}, {point_y})")
                        return result_dict
                except (ValueError, IndexError):
                    pass
                    
            # Also try to extract region if available
            region = re.search(r'<region>(.*?)</region>', response, re.DOTALL)
            if region:
                region_json = json.loads(region.group(1).strip())
                if "box_2d" in region_json:
                    result_dict["bbox"] = region_json["box_2d"]
                    
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            print(f"Error parsing response: {e}")
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


class CustomQwen2_5VL_VLLM_Model():
    def __init__(self):
        # Check if the current process is daemonic.
        from multiprocessing import current_process
        process = current_process()
        if process.daemon:
            print("Latest vllm versions spawns children processes, therefore can not be started in a daemon process. Are you using multiprocess.Pool? Try multiprocess.Process instead.")

    # def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=99999999):  #2007040
    def load_model(self, model_name_or_path="/workspace/ShowUI-main/Test_model/Qwen2.5-VL-7B-Instruct", max_pixels=99999999):  #2007040
        from vllm import LLM
        self.max_pixels = max_pixels
        self.model = LLM(
            model_name_or_path,
            gpu_memory_utilization=0.99,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": self.max_pixels,
            },
        )

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):
        from vllm import SamplingParams
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=14 * 2,
            min_pixels=28 * 28,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=self.max_pixels,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        inputs = {
            "prompt": GUIDED_PROMPT.replace("{{screen_width}}", str(resized_width)).replace("{{screen_height}}", str(resized_height)).replace("{{instruction}}", instruction),
            "multi_modal_data": {"image": resized_image}
        }

        generated = self.model.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=100))

        response = generated[0].outputs[0].text.strip()
        print(response)
        response = """<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [""" + response

        cut_index = response.rfind('}')
        if cut_index != -1:
            response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
            coordinates = action['arguments']['coordinate']
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()
