from pydantic import BaseModel
from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn


class RequestModel(BaseModel):
    prompt: str
    max_new_token: int = 32


model_id = "Qwen/Qwen2.1-5B"
output_dir = "./qwen2_peft_result"

class QwenDeployer:
    def __init__(self, model_id, output_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(self.base_model, output_dir)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int=32):
        inputs = self.tokenizer(prompt, return_tensor="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

api = FastAPI()
inference = QwenDeployer(model_id=model_id, output_dir=output_dir)


@api.get("/")
async def root():
    return {"The API is running! Enter your prompt in /generate to get results."}


@api.post("/generate")
async def generate(request: RequestModel):
    result = inference.generate(request.prompt, request.max_new_token)
    
    return {"Generated text": result}


if __name__ == "__main__":
    uvicorn app:app --host 0.0.0.0 --port 8001 --reload