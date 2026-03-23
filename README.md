# Pruning Sandbox for Dense and MoE Hugging Face Models

이 저장소는 **Torch-Pruning 경로**와 **Expert_Sparsity 경로**를 한 레포 안에서 함께 관리하기 위한 공개용 정리본입니다. 다만 두 구현을 하나의 공통 엔진으로 강제로 합치지는 않습니다. 이 레포의 목표는 다음 두 작업을 **명확히 분리해서** 운영하는 것입니다.

- **Dense / CNN Hugging Face 모델 pruning**: Torch-Pruning 기반으로 수행
- **MoE Hugging Face 모델 pruning**: Expert_Sparsity 기반으로 별도 수행

즉, 이 저장소는 “한 코드베이스로 merge된 pruning 라이브러리”가 아니라, **dense 경로와 moe 경로를 나란히 유지하는 작업 레포**입니다.

## Acknowledgements

- **Torch-Pruning**: dense/CNN pruning workflow의 기본 구조와 의존성 그래프 기반 structural pruning 아이디어를 제공합니다.
- **Expert_Sparsity**: MoE 모델 pruning 및 expert 단위 sparsity workflow를 구성할 때 참고하는 별도 경로입니다.

이 저장소의 공개 버전에서는 두 프로젝트의 출처를 명확히 남기고, 실제 구현은 **dense와 moe를 독립적인 실행 경로**로 유지하는 것을 원칙으로 합니다.

## Repository Policy

1. **Dense 경로와 MoE 경로는 분리합니다.**
2. **Expert_Sparsity 코드는 Torch-Pruning 내부에 병합하지 않습니다.**
3. 공용 README와 예제는 “어떤 모델이 어느 경로로 가는지”를 먼저 설명해야 합니다.
4. Hugging Face 모델 기준으로:
   - CNN / 일반 dense 모델은 `dense_pruning.py` 경로를 사용합니다.
   - MoE 모델은 `moe_pruning.py` 경로를 사용합니다.

## Current Public Structure

```text
examples/hf_models/
├── README.md                # HF 모델 pruning 사용 가이드
├── dense_pruning.py         # Torch-Pruning 기반 dense/CNN pruning entry point
└── moe_pruning.py           # Expert_Sparsity 기반 MoE pruning entry point placeholder

tests/
└── test_hf_examples.py      # public 구조 회귀 테스트
```

## Installation

기본 설치:

```bash
pip install -e .
```

Hugging Face 예제를 직접 실행하려면 필요에 따라 아래 패키지를 추가 설치하세요.

```bash
pip install transformers
```

## Usage

### 1) Dense / CNN Hugging Face 모델 pruning

Torch-Pruning 경로는 `examples/hf_models/dense_pruning.py`를 기준으로 사용합니다.

```bash
python examples/hf_models/dense_pruning.py \
  --model google/vit-base-patch16-224 \
  --pruning-ratio 0.5
```

이 경로는 다음을 목표로 합니다.

- Hugging Face 모델을 로드
- 예시 입력 생성
- classifier / lm_head 등 task head는 기본적으로 제외
- Torch-Pruning으로 구조적 pruning 수행

### 2) MoE Hugging Face 모델 pruning

MoE 경로는 `examples/hf_models/moe_pruning.py`를 기준으로 사용합니다.

```bash
python examples/hf_models/moe_pruning.py \
  --model mistralai/Mixtral-8x7B-v0.1 \
  --output-dir outputs/mixtral
```

이 파일은 공개 레포에서 **MoE 작업 경로의 인터페이스와 문서 구조를 고정하는 역할**을 합니다. 실제 Expert_Sparsity 수정 구현은 여기에 사용자가 직접 넣는 전제를 둡니다.

## What Should Be Added Next

사용자가 로컬 수정 코드를 넣을 때는 아래 순서로 채우면 됩니다.

1. `examples/hf_models/dense_pruning.py`
   - 실제 대상 HF dense/CNN 모델별 ignore rule 보강
   - 모델별 example input 생성 로직 추가
   - pruning 후 저장 / 평가 코드 추가
2. `examples/hf_models/moe_pruning.py`
   - Expert_Sparsity 쪽 실행 함수 연결
   - router / experts / shared experts pruning 정책 반영
   - pruning 후 평가 및 checkpoint 저장 추가
3. 테스트 확장
   - dense 실제 HF 모델 smoke test
   - moe local workflow smoke test

## Notes for Public Release

- README는 이제 Torch-Pruning 단독 레포 설명이 아니라, **dense vs moe 두 경로를 가진 새 레포 설명**이어야 합니다.
- 이 저장소는 두 경로를 함께 정리하지만, **Expert_Sparsity 구현 자체를 Torch-Pruning 패키지 안으로 병합하지 않습니다.**
- 상단 acknowledgment 주석은 각 실행 파일에 따로 둡니다.
