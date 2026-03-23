# Hugging Face Pruning Workflows

이 디렉터리는 Hugging Face 모델 pruning 경로를 두 개로 분리해서 관리합니다.

- `dense_pruning.py`: Torch-Pruning 기반 dense/CNN 모델 pruning
- `moe_pruning.py`: Expert_Sparsity 기반 MoE 모델 pruning 자리

원칙은 단순합니다.

1. **Dense와 MoE를 한 함수로 자동 통합하지 않습니다.**
2. **실행 파일을 분리해서 유지합니다.**
3. **acknowledgement는 각 파일 상단에 명시합니다.**

공개 레포에서는 이 구조를 유지하고, 실제 로컬 수정 구현은 각 파일 안에 채워 넣으면 됩니다.
