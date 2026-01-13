# MXFP4 Quantization for GPT-OSS Models

GPT-OSS MoE(Mixture of Experts) 모델을 MXFP4(Microscaling FP4)로 양자화하고, OpenAI 호환 포맷으로 변환하는 도구입니다.

## 파일 설명

### 1. `gpt_oss_mxfp4.py`

GPT-OSS 모델의 MoE expert 가중치를 MXFP4로 양자화하는 스크립트입니다.

**주요 기능:**
- GPT-OSS 모델 로드
- Fused MoE experts (`GptOssExperts`)를 개별 `LinearExperts`로 변환 (양자화 호환성을 위해)
- MoE expert 가중치만 선택적으로 MXFP4 양자화 (`gate_proj`, `up_proj`, `down_proj`)
- 양자화된 모델을 compressed-tensors 포맷으로 저장

**양자화 대상:**
- `model.layers.*.mlp.experts.experts.*.gate_proj`
- `model.layers.*.mlp.experts.experts.*.up_proj`
- `model.layers.*.mlp.experts.experts.*.down_proj`

**제외 대상:**
- Attention 레이어
- Router
- Embedding (`embed_tokens`)
- LM Head (`lm_head`)

### 2. `convert_to_openai_format.py`

llm-compressor로 양자화된 MXFP4 모델을 OpenAI GPT-OSS 호환 포맷으로 변환합니다.

**주요 기능:**
- llm-compressor 포맷의 packed weights를 OpenAI 포맷으로 reshape
- `gate_proj`와 `up_proj`를 interleave하여 `gate_up_proj`로 병합
- 모든 expert를 하나의 텐서로 스택
- OpenAI 호환 config.json 생성

**포맷 변환:**
```
llm-compressor 포맷:
  model.layers.{L}.mlp.experts.experts.{E}.gate_proj.weight_packed
  model.layers.{L}.mlp.experts.experts.{E}.up_proj.weight_packed
  model.layers.{L}.mlp.experts.experts.{E}.down_proj.weight_packed

OpenAI 포맷:
  model.layers.{L}.mlp.experts.gate_up_proj_blocks  # [num_experts, out*2, groups, 16]
  model.layers.{L}.mlp.experts.gate_up_proj_scales  # [num_experts, out*2, groups]
  model.layers.{L}.mlp.experts.down_proj_blocks     # [num_experts, out, groups, 16]
  model.layers.{L}.mlp.experts.down_proj_scales     # [num_experts, out, groups]
```

## 사용법

### Step 1: 모델 양자화

`gpt_oss_mxfp4.py`를 수정하여 모델 경로를 설정합니다:

```python
# 모델 경로 설정
MODEL_ID = "openai/gpt-oss-20b"  # HuggingFace 모델
# 또는
MODEL_ID = "/path/to/local/model"  # 로컬 모델

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

양자화 실행:

```bash
cd experimental/mxfp4
uv run python gpt_oss_mxfp4.py
```

출력 디렉토리: `{model_name}-MXFP4/`

### Step 2: OpenAI 포맷으로 변환

```bash
uv run python convert_to_openai_format.py <input_dir> <output_dir>
```

**예시:**

```bash
# gpt-oss-20b 모델 변환
uv run python convert_to_openai_format.py \
    gpt-oss-20b-MXFP4 \
    gpt-oss-20b-MXFP4-openai

# checkpoint 모델 변환
uv run python convert_to_openai_format.py \
    checkpoint-705-MXFP4 \
    checkpoint-705-MXFP4-openai
```

## 전체 파이프라인 예시

```bash
cd experimental/mxfp4

# 1. 양자화 (gpt_oss_mxfp4.py에서 MODEL_ID 수정 후)
uv run python gpt_oss_mxfp4.py

# 2. OpenAI 포맷 변환
uv run python convert_to_openai_format.py \
    gpt-oss-20b-MXFP4 \
    gpt-oss-20b-MXFP4-openai

# 3. (선택) llama.cpp GGUF 변환
# python convert_hf_to_gguf.py gpt-oss-20b-MXFP4-openai
```

## 요구사항

- Python 3.10+
- llm-compressor
- transformers
- safetensors
- torch

## 참고사항

- MXFP4 양자화는 group_size=32를 사용합니다
- 양자화 후 모델 크기가 약 1/4로 감소합니다
- Expert 가중치만 양자화되므로 attention 품질은 유지됩니다
