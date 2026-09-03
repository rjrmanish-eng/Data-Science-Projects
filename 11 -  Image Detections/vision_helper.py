# vision_helper.py - WORKING MODELS FEB 2026
import base64
import io
import requests
from PIL import Image

class OpenRouterVision:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # ✅ Currently working free vision models (tested Feb 2026)
        self.working_models = [
            "meta-llama/llama-4-scout:free",        # Llama 4 Scout
            "google/gemma-3-27b-it:free",           # Google Gemma 3
            "qwen/qwen-2.5-vl-72b-instruct:free",   # Qwen VL (naya version)
            "openrouter/free"                         # Auto fallback
        ]
    
    def encode_image(self, image_input):
        """Image ko base64 mein convert karo"""
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            buffered = io.BytesIO()
            image_input.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_image_complete(self, image_input):
        """Ek call me object + description"""
        base64_image = self.encode_image(image_input)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "CIFAR-10 Vision"
        }
        
        # Pehle prompt - clear format
        prompt = """Analyze this image and respond with:
OBJECT: [one word main object]
DESCRIPTION: [one short sentence describing the image]

Example:
OBJECT: cat
DESCRIPTION: A brown cat sitting on a sofa.
"""
        
        for model in self.working_models:
            try:
                print(f"Trying model: {model}")
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 200
                }
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Parse OBJECT and DESCRIPTION
                    obj = "unknown"
                    desc = "No description"
                    
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith("OBJECT:"):
                            obj = line.replace("OBJECT:", "").strip()
                        elif line.startswith("DESCRIPTION:"):
                            desc = line.replace("DESCRIPTION:", "").strip()
                    
                    # Agar format alag ho to fallback
                    if obj == "unknown" and desc == "No description":
                        # Saara content description maan lo
                        desc = content[:200]
                        # Pehla word object maan lo
                        words = content.split()
                        if words:
                            obj = words[0]
                    
                    return obj, desc
                    
                elif response.status_code == 404:
                    print(f"Model {model} not available, trying next...")
                    continue
                else:
                    print(f"Error {response.status_code}: {response.text[:100]}")
                    
            except Exception as e:
                print(f"Exception with {model}: {e}")
                continue
        
        return "Error", "No working vision model available"