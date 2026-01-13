import os
import sys

# Use local source for llmcompressor (gpt_oss module may not be in installed version)
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
os.environ["HF_HOME"] = "/home/beomseok.kwon/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.gpt_oss import convert_model_for_quantization_gptoss
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# MODEL_ID = "openai/gpt-oss-20b"
MODEL_ID = "/mnt/nas/group/edgefm/hp_project/checkpoints/aman_gpt_22experts_lr4e-05_epoch5.0_30ksamples/checkpoint-705"
# MODEL_ID = "AmanPriyanshu/gpt-oss-16.1b-specialized-all-pruned-moe-only-24-experts"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Convert fused MoE experts (GptOssExperts) to LinearExperts for quantization
# This unfuses gate_up_proj into separate gate_proj, up_proj Linear layers
print("[GPT-OSS] Converting fused MoE experts to LinearExperts for quantization...")
convert_model_for_quantization_gptoss(model)
print("[GPT-OSS] Conversion completed.")

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 32 via ptq
#   * only target MoE expert weights (gate_proj, up_proj, down_proj in experts)
recipe = QuantizationModifier(
    targets="re:.*mlp\\.experts\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$",
    scheme="MXFP4",
    ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# print("\n\n")
# print("========== SAMPLE GENERATION ==============")
# dispatch_for_generation(model)
# input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
#     model.device
# )
# output = model.generate(input_ids, max_new_tokens=100)
# print(tokenizer.decode(output[0]))
# print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
