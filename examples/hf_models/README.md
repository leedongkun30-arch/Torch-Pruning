# Hugging Face Dense + MoE Pruning Workflows

이 디렉터리는 **합쳐진 레포** 안에서 Hugging Face pruning 경로를 두 가지로 제공합니다.

- `dense_pruning.py`: Torch-Pruning 기반 CNN / dense 모델 pruning
- `moe_pruning.py`: Expert_Sparsity 스타일 MoE 모델 pruning

구현 위치는 패키지 기준으로 아래와 같습니다.

- `torch_pruning/hf/dense.py`
- `torch_pruning/hf/moe.py`

즉, 이번 구조는 dense와 moe를 분리해서 설명하되, **두 경로를 하나의 공개 레포 안에 함께 포함하는 형태**입니다.
