# Unified Hugging Face Pruning Repository

이 저장소는 **Torch-Pruning**과 **Expert_Sparsity**의 핵심 코드를 함께 포함하는 통합 레포입니다. 목적은 단순히 두 프로젝트를 나란히 소개하는 것이 아니라, **Hugging Face 모델이 주어졌을 때 모델 속성을 보고 Dense 모델이면 Torch-Pruning으로, MoE 모델이면 Expert_Sparsity로 자동 프루닝하는 공개 레포**를 제공하는 것입니다.

## What lives in this repository

### 1. Torch-Pruning path for dense / CNN Hugging Face models
- 구조적 pruning
- dependency graph 기반 pruning
- dense/CNN Hugging Face helper

### 2. Expert_Sparsity path for MoE Hugging Face models
- calibration dataset loader
- method registry
- expert pruning
- dynamic skipping
- Mixtral-like patch helper
- Expert_Sparsity-style CLI entry point

## Repository layout

```text
torch_pruning/
└── hf/
    ├── dense.py
    ├── moe.py
    └── router.py

expert_sparsity/
├── data/
│   ├── __init__.py
│   ├── build.py
│   └── dataset.py
├── method/
│   ├── __init__.py
│   ├── dynamic_skipping.py
│   └── expert_pruning.py
├── model/
│   ├── __init__.py
│   └── modeling_mixtral.py
├── __init__.py
└── main.py

examples/hf_models/
├── dense_pruning.py
├── moe_pruning.py
└── prune_hf_model.py
```

## Routing rule

이 레포의 핵심 규칙은 단순합니다.

- **Dense / CNN 모델**: `torch_pruning.hf.prune_dense_hf_model(...)`
- **MoE 모델**: `expert_sparsity.method.METHODS[...]` 또는 `torch_pruning.hf.prune_moe_hf_model(...)`
- **자동 분기**: `torch_pruning.hf.prune_hf_model(...)`

`torch_pruning/hf/router.py`는 `experts` ModuleList, `router/gate` 모듈, 그리고 일부 알려진 HF config `model_type`을 사용해 MoE 여부를 판단합니다.

## Original user request reflected here

이 구조는 다음 요구사항을 그대로 반영합니다.

- Expert_Sparsity와 Torch-Pruning의 핵심 코드는 지금 레포 안에 있어야 한다.
- HF 모델 속성을 보고 Dense면 Torch-Pruning, MoE면 Expert_Sparsity를 쓰는 통합 레포여야 한다.
- acknowledgment를 포함해야 한다.
- 공개 가능한 README / 구조 / 테스트를 가져야 한다.

## Implemented entry points

### Dense only
```bash
python examples/hf_models/dense_pruning.py --model google/vit-base-patch16-224 --pruning-ratio 0.5
```

### MoE only
```bash
python examples/hf_models/moe_pruning.py --model mistralai/Mixtral-8x7B-v0.1 --method layerwise_pruning --r 6
```

### Unified auto-routing
```bash
python examples/hf_models/prune_hf_model.py --model mistralai/Mixtral-8x7B-v0.1 --moe-method dynamic_skipping --r 6 --beta 0.2
```

## Tests

`tests/test_hf_examples.py`는 다음을 검증합니다.

- dense helper pruning
- Expert_Sparsity method registry
- calibration loader construction
- MoE helper pruning and dynamic skipping
- unified HF router dispatch

## Source acknowledgement

- Torch-Pruning original repository: https://github.com/VainF/Torch-Pruning
- Expert_Sparsity original repository: https://github.com/Lucky-Lance/Expert_Sparsity
