import os
import requests
import subprocess
import json
import time
from huggingface_hub import login
from google import genai
from google.genai import types
from PIL import Image
import io

# # Get the base path that works in both Docker and local environments
# def get_base_path():
#     # First check if we're in a Docker container at /app
#     if os.path.exists("/app/src"):
#         return "/app"
#     # Otherwise use current working directory
#     elif os.path.exists("/scratch"):
#         return "."
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

def gemma3_12b_cloud():
    """Factory function for Gemma 3 12B cloud service."""
    return CloudRunModel(CLOUD_SERVICES["gemma3-12b-cloud"])

def gemma3_27b_cloud():
    """Factory function for Gemma 3 27B cloud service."""
    return CloudRunModel(CLOUD_SERVICES["gemma3-27b-cloud"])

def gemma_google_ai(model_id="gemma-3-27b-it"):
    """Factory function for Gemini model using Google GenAI client with vision support."""
    # Prefer an API key when provided, otherwise use ADC/service-account auth via Vertex AI.
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        client_kwargs = {"vertexai": True, "location": location}
        if project:
            client_kwargs["project"] = project
        client = genai.Client(**client_kwargs)

    def query(messages):
        """
        Query the Google GenAI model with structured messages.

        Args:
            messages: Structured messages in the format used by binary_judge_image

        Returns:
            str: The model's response text
        """
        try:
            # Convert structured messages to Google GenAI format
            contents = []

            for message in messages:
                if message.get("role") == "system":
                    # Add system instruction as text
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                contents.append(item.get("text", ""))
                    elif isinstance(content, str):
                        contents.append(content)

                elif message.get("role") == "user":
                    # Add user content (text and images)
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                contents.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                image_path = item.get("image")
                                if str(image_path).endswith('.png'):
                                    mimetype = "image/png"
                                elif str(image_path).endswith('.jpg') or str(image_path).endswith('.jpeg'):
                                    mimetype = "image/jpeg"
                                elif str(image_path).endswith('.jpg'):
                                    mimetype = "image/jpg"
                                else:
                                    raise ValueError(f"Unsupported image format: {image_path}")
                                with open(image_path, "rb") as img_file:
                                    image_bytes = img_file.read()
                                contents.append(
                                    types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type=mimetype
                                    )
                                )

            response = client.models.generate_content(
                model=model_id,
                contents=contents
            )
            return response.text

        except Exception as e:
            print(f"Google GenAI API error: {e}")
            return "Error: Google GenAI API unavailable"

    return query

_models = {
    # Local models
    "gemma-3-27b-it": gemma_3_27b_it,
    "gemma-3-12b-it-qat-q4_0-gguf": gemma_3_12b_it_quantized,
    "gemma-3-27b-it-qat-q4_0-gguf": gemma_3_27b_it_quantized,
    "qwen2-5-vl-72b-instruct": qwen2_5_vl_72b_instruct,

    # Google AI API models
    "gemma-google-ai": lambda: gemma_google_ai(model_id="gemma-3-27b-it"),
    "gemini-2.5-flash-google-ai": lambda: gemma_google_ai(model_id="gemini-2.5-flash"),
    "gemini-3-flash-google-ai": lambda: gemma_google_ai(model_id="gemini-3-flash-preview"),

    # Cloud Run services - same interface as local models
    "gemma3-12b-cloud": gemma3_12b_cloud,
    "gemma3-27b-cloud": gemma3_27b_cloud,
}

# Cloud Run service configurations
CLOUD_SERVICES = {
    "gemma3-12b-cloud": {
        "url": "https://YOUR-CLOUD-RUN-SERVICE.us-central1.run.app",
        "model": "gemma3:12b",
        "endpoint": "/api/generate"
    },
    "gemma3-27b-cloud": {
        "url": "https://YOUR-CLOUD-RUN-27B-SERVICE.us-central1.run.app",
        "model": "gemma3:27b",
        "endpoint": "/api/generate"
    }
    # Easy to add more cloud services here
}

class CloudRunModel:
    """
    Generic Cloud Run model client for any deployed model service.
    Provides the same interface as local models for seamless integration.
    """

    def __init__(self, service_config):
        self.url = service_config["url"]
        self.model_name = service_config["model"]
        self.endpoint = service_config["endpoint"]
        self.token = None
        self._refresh_token()

    def _refresh_token(self):
        """Get Google Cloud identity token for authentication."""
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-identity-token'],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            self.token = result.stdout.strip()
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Warning: Failed to get auth token: {e}")
            self.token = None
            return False

    def _make_request(self, payload, max_retries=3, timeout=120):
        """Make HTTP request to the cloud service with retry logic."""
        headers = {
            "Content-Type": "application/json"
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        url = f"{self.url}{self.endpoint}"

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 401 and attempt < max_retries - 1:
                    # Token might be expired, refresh and retry
                    print("Token expired, refreshing...")
                    if self._refresh_token():
                        headers["Authorization"] = f"Bearer {self.token}"
                        continue

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == max_retries - 1:
                        raise Exception(f"Cloud service request failed: {error_msg}")
                    else:
                        print(f"Request failed (attempt {attempt + 1}), retrying: {error_msg}")
                        time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Network error calling cloud service: {e}")
                else:
                    print(f"Network error (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(2 ** attempt)

        raise Exception("All retry attempts failed")

    def query(self, messages_or_prompt):
        """
        Query the cloud model with structured messages or simple prompt.
        Handles both structured messages and simple string prompts.

        Args:
            messages_or_prompt: Either structured messages or a simple text prompt

        Returns:
            str: The model's response text
        """
        # Convert structured messages to simple text prompt for cloud service
        if isinstance(messages_or_prompt, list):
            # Extract text from structured messages
            system_text = ""
            user_text_parts = []

            for message in messages_or_prompt:
                if message.get("role") == "system":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                system_text += item.get("text", "") + " "
                    elif isinstance(content, str):
                        system_text += content + " "

                elif message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                user_text_parts.append(item.get("text", ""))
                            # Images are ignored in cloud model (text-only)

            prompt = f"{system_text.strip()} Question: {' '.join(user_text_parts)}"
        else:
            # Simple string prompt
            prompt = str(messages_or_prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response_data = self._make_request(payload)
            return response_data.get('response', '')
        except Exception as e:
            print(f"Cloud model query failed: {e}")
            # Return a default response to maintain evaluation flow
            return "Error: Cloud service unavailable"

    def is_healthy(self):
        """Check if the cloud service is healthy and accessible."""
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            # Try a simple health check or minimal request
            response = requests.get(f"{self.url}/health", headers=headers, timeout=10)
            return response.status_code in [200, 404]  # 404 is OK if no health endpoint
        except:
            return False

def load_model(model_name, *args, **kwargs):
    return _models[model_name](*args, **kwargs)
