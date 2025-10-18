import torch
import os
from huggingface_hub import login

# # Get the base path that works in both Docker and local environments
# def get_base_path():
#     # First check if we're in a Docker container at /app
#     if os.path.exists("/app/src"):
#         return "/app"
#     # Otherwise use current working directory
#     elif os.path.exists("/scratch"):
#         return "/scratch/general/vast/USER/Agent-Benchmark/"
#     else:
#         return os.getcwd()

# # Set HF cache to base directory
# BASE_PATH = get_base_path()
# print(f"Using base path: {BASE_PATH}")
# os.environ["HF_HOME"] = os.path.join(BASE_PATH, ".huggingface")

# Handle Hugging Face authentication
hf_key = os.environ.get("HF_KEY")
if hf_key:
    login(token=hf_key)

def gemma_3_27b_it():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    model_id = "google/gemma-3-27b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    def query(messages):
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded
    
    return query

def gemma_3_27b_it_quantized():
    from llama_cpp import Llama
    model = Llama.from_pretrained(
        repo_id="google/gemma-3-27b-it-qat-q4_0-gguf",
        filename="gemma-3-27b-it-q4_0.gguf",
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=2048       # Context length
    )

    def query(messages):
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=10)
        return response.choices[0].message['content']
    return query

def gemma_3_12b_it_quantized():
    from llama_cpp import Llama

    model = Llama.from_pretrained(
        repo_id="google/gemma-3-12b-it-qat-q4_0-gguf",
	    filename="gemma-3-12b-it-q4_0.gguf",
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=2048       # Context length
    )

    def query(messages):
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=10)
        return response.choices[0].message['content']
        


    return query
    
def qwen2_5_vl_72b_instruct():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    def query(messages):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
    return query

_models = {
    "gemma-3-27b-it": gemma_3_27b_it,
    "gemma-3-12b-it-qat-q4_0-gguf": gemma_3_12b_it_quantized,
    "gemma-3-27b-it-qat-q4_0-gguf": gemma_3_27b_it_quantized,
    "qwen2-5-vl-72b-instruct": qwen2_5_vl_72b_instruct,

}

def load_model(model_name, *args, **kwargs):
    return _models[model_name](*args, **kwargs)
