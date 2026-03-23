# Torch-Pruning + Expert_Sparsity for Hugging Face Model Pruning

이 저장소는 **Torch-Pruning**과 **Expert_Sparsity**를 함께 정리한 공개용 레포입니다. 목적은 두 프로젝트를 소개만 하는 것이 아니라, **한 저장소 안에서 Hugging Face 모델 pruning 코드를 dense/CNN 경로와 MoE 경로로 나누어 함께 관리**하는 것입니다.

사용 목적은 아래와 같습니다.

- **Torch-Pruning 경로**: CNN 및 일반 dense Hugging Face 모델 pruning
- **Expert_Sparsity 경로**: MoE Hugging Face 모델 pruning

즉, 이 레포는 “dense는 Torch-Pruning으로, MoE는 Expert_Sparsity로” 처리하는 **통합 정리 레포**입니다.

## Acknowledgements

- **Torch-Pruning**: 구조적 pruning, dependency graph, high-level pruner 설계의 기반을 제공합니다.
- **Expert_Sparsity**: expert pruning / dynamic skipping 기반 MoE pruning workflow의 기반을 제공합니다.

Expert_Sparsity 원본 저장소는 expert pruning, dynamic skipping, Mixtral 계열 수정 모델 파일, 그리고 `main.py --method {layerwise_pruning,progressive_pruning,dynamic_skipping}` 형태의 실행 구조를 제공합니다. 이 레포에서는 그 아이디어를 MoE 전용 경로로 통합했습니다. [Source: Expert_Sparsity GitHub README](https://github.com/Lucky-Lance/Expert_Sparsity). 

## Design Rules

1. **Dense/CNN 모델은 Torch-Pruning 경로만 사용합니다.**
2. **MoE 모델은 Expert_Sparsity 경로만 사용합니다.**
3. 두 경로는 레포 안에 함께 존재하지만, 목적과 실행 스크립트는 분리합니다.
4. 모든 공개 entry point 상단에는 acknowledgement 주석을 둡니다.
5. Hugging Face 모델 구조에 맞춰 pruning helper를 별도 패키지로 제공합니다.

## Repository Structure

```text
torch_pruning/
└── hf/
    ├── __init__.py
    ├── dense.py   # Torch-Pruning 기반 dense/CNN helper
    └── moe.py     # Expert_Sparsity 스타일 MoE helper

examples/hf_models/
├── README.md
├── dense_pruning.py
└── moe_pruning.py

tests/
└── test_hf_examples.py
```

## What Is Implemented

### Dense / CNN Path

`torch_pruning/hf/dense.py`와 `examples/hf_models/dense_pruning.py`는 다음을 담당합니다.

- Hugging Face dense/CNN 모델에 Torch-Pruning 적용
- classifier / lm_head 등 task head 기본 제외
- `BasePruner` 기반 구조적 pruning 수행

예시 실행:

```bash
python examples/hf_models/dense_pruning.py \
  --model google/vit-base-patch16-224 \
  --pruning-ratio 0.5
```

### MoE Path

`torch_pruning/hf/moe.py`와 `examples/hf_models/moe_pruning.py`는 다음을 담당합니다.

- Hugging Face MoE block 탐색 (`experts`, `gate` / `router`)
- Expert_Sparsity 스타일 expert scoring 및 top-r expert 보존
- router linear layer를 keep index에 맞춰 함께 축소
- optional dynamic skipping annotation 지원

예시 실행:

```bash
python examples/hf_models/moe_pruning.py \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --method layerwise_pruning \
  --r 6
```

또는 dynamic skipping 스타일 설정:

```bash
python examples/hf_models/moe_pruning.py \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --method dynamic_skipping \
  --r 6 \
  --beta 0.2
```

## Why This Matches My Chat Request

이 README는 내가 요청한 아래 조건을 기준으로 다시 정리했습니다.

- Torch-Pruning과 Expert_Sparsity를 **한 레포에 함께 정리할 것**
- 다만 역할은 분리할 것
- Torch-Pruning은 CNN / dense 모델 전용
- Expert_Sparsity는 MoE 모델 전용
- 상단 acknowledgment를 넣을 것
- Hugging Face 모델 구조에 맞는 pruning 코드를 둘 것
- 간단한 테스트 스크립트를 넣고 공개 가능한 구조로 정리할 것

## Tests

`tests/test_hf_examples.py`는 아래를 검증합니다.

- dense helper가 classifier를 ignore하는지
- dense pruning 이후 출력 shape이 유지되는지
- MoE helper가 expert/router block을 찾고 expert 수를 줄이는지
- dynamic skipping annotation이 붙는지

## Notes for Local Integration

두 원본 레포에서 이미 별도 수정이 있었다면, 아래 파일들이 **로컬 수정 코드의 주 진입점**이 됩니다.

- Dense 쪽 추가 작업: `torch_pruning/hf/dense.py`
- MoE 쪽 추가 작업: `torch_pruning/hf/moe.py`
- 공개 실행 스크립트: `examples/hf_models/dense_pruning.py`, `examples/hf_models/moe_pruning.py`

즉, 이번 구조는 Expert_Sparsity를 더 이상 placeholder로 두지 않고, **MoE 경로 자체를 레포 안에 포함하는 방향**으로 바꾼 것입니다.
