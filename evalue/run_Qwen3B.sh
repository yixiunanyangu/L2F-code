python eval_screenspot_cut.py  \
    --model_type "../Qwen2.5-VL-3B-Instruct"  \
    --model_name_or_path "/workspace/VLM-R1/Qwen2.5-VL-3B-Instruct"  \
    --screenspot_imgs "../ScreenSpot-Pro/images"  \
    --screenspot_test "../ScreenSpot-Pro/annotations"  \
    --task "all" \
    --language "en" \
    --gt_type "positive" \
    --log_path "../output_log/Qwen2.5-VL-3B-Instruct.json" \
    --inst_style "instruction"