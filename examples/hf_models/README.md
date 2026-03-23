# Hugging Face Dense + MoE Pruning Workflows

이 디렉터리는 이제 세 가지 entry point를 제공합니다.

- `dense_pruning.py`: Torch-Pruning 기반 dense/CNN 전용 실행기
- `moe_pruning.py`: Expert_Sparsity 기반 MoE 전용 실행기
- `prune_hf_model.py`: HF 모델 속성을 보고 dense vs MoE 경로를 자동 선택하는 통합 실행기

패키지 구조는 다음 두 축으로 나뉩니다.

- `torch_pruning/hf/`: dense helper + 통합 router
- `expert_sparsity/`: 원본 Expert_Sparsity 구조를 반영한 data / method / model / main 패키지
