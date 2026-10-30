# MLOps CI/CD 실습

학습 코드의 검사, 컨테이너 빌드, 클라우드 학습 작업을 연결하는 흐름을 이해하기 위한 교육 실습 저장소입니다.

**원본:** [KennethanCeyer/mlops-quicklab-cicd](https://github.com/KennethanCeyer/mlops-quicklab-cicd)
이 저장소는 원본을 바탕으로 한 fork입니다. 원본 구성과 현재 코드의 학습·변경 범위를 구분해 기록합니다.

## 현재 코드에서 확인할 수 있는 내용

| 파일 | 내용 |
| --- | --- |
| [trainer.py](trainer.py) | `klue/bert-base`와 APEACH 데이터를 사용하는 이진 분류 학습·평가·저장 코드 |
| [tests/test_trainer.py](tests/test_trainer.py) | 모델 설정과 주요 함수의 존재를 확인하는 기본 검사 |
| [Dockerfile](Dockerfile) | PyTorch 기반 학습 환경을 구성하고 trainer를 실행하는 컨테이너 정의 |
| [.github/workflows/cicd.yml](.github/workflows/cicd.yml) | 코드 검사·학습·pytest → Artifact Registry → Vertex AI 작업 제출 구성 |
| [requirements.txt](requirements.txt) | 학습 코드의 패키지 목록 |

현재 `trainer.py`에는 학습·평가 각각 최대 100개 표본과 1에폭 설정이 들어 있습니다.
기본 검사는 모델 품질이나 전체 배포의 정상 동작을 증명하지 않습니다.

## 실습에서 살펴보는 흐름

```text
학습 코드와 설정 확인
  → 코드 검사·기본 테스트
  → 학습용 컨테이너 빌드
  → 이미지 저장소 등록
  → 클라우드 학습 작업 제출
```

GitHub Actions 파일은 이 과정을 자동화하기 위한 구성입니다.
실제 클라우드 실행에는 프로젝트·인증·이미지 저장소 설정이 필요하며 과금이 발생할 수 있습니다.
워크플로 파일의 존재와 현재 배포 성공 여부는 구분합니다.

## 로컬 환경 준비

```bash
git clone https://github.com/juyoo0729/mlops-quicklab-cicd.git
cd mlops-quicklab-cicd
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.venv\Scripts\python.exe -m pytest tests/ -q
```

macOS·Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt pytest
.venv/bin/python -m pytest tests/ -q
```

학습 실행은 환경 준비 후 `python trainer.py`로 진행합니다.
모델과 데이터 다운로드, 사용 가능한 메모리·장치를 먼저 확인해야 합니다.
이번 README 정리에서는 학습과 클라우드 작업을 실행하지 않았습니다.

## 함께 볼 학습 기록

아래는 AIFFEL의 별도 실습 문서입니다. 이 저장소에서 생성한 결과와 구분합니다.

- [MLflow 실험 추적 결과](https://github.com/juyoo0729/AIFFEL_quest_eng/blob/main/MLOps/MLOps06/results_summary.md): 강의 실습 재현, 5개 실행의 설정·지표와 모델 등록 기록.
- [Vertex AI 서빙 기록](https://github.com/juyoo0729/AIFFEL_quest_eng/blob/main/MLOps/MLOps03/03-gcp-serving.md): 별도 추천모델의 학습·서빙·예측 응답 기록. 해당 엔드포인트는 검증 후 삭제됨.
- [AIFFEL 학습 목록](https://github.com/juyoo0729/AIFFEL_quest_eng) · [프로필](https://github.com/juyoo0729)

## 추가로 정리할 내용

- 원본 대비 본인 변경 사항을 파일·커밋 단위로 연결하기.
- 실행 환경과 실패 원인, 설정 변경의 근거를 남기기.
- 기본 함수 검사에 더해 입력·출력 계약과 저장 모델의 재로딩을 검증하기.
- 실제 클라우드 작업을 수행할 때 실행 결과와 사용 종료 처리를 함께 기록하기.

원본의 [라이선스](LICENSE)를 유지합니다.
