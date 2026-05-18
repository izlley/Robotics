# <Model / Paper Name>

> **출처**: <arXiv ID / Conference / 저자 + 연도>
> **읽은 일자**: <YYYY-MM-DD>
> **PDF**: [`papers/<category>/<filename>.pdf`](../../papers/<category>/<filename>.pdf)

## 한 줄 요약

<논문 한 문장 요약. 메인 contribution.>

---

## 1. Motivation & 문제 정의

- 어떤 문제를 풀려는가
- 기존 방법의 한계는 무엇이었나
- 본 논문이 attack하는 specific gap

## 2. 핵심 아이디어

- 한두 문장으로 압축한 contribution
- 가장 새롭거나 직관과 어긋나는 부분

## 3. 아키텍처

### 3.1 입력 / 출력
- Vision: <카메라 수, 해상도, depth 여부, multi-view 여부>
- Language: <natural language? task token? embedding?>
- State / proprioception: <어떤 형식>
- Output: <action 형식 — joint position, end-effector pose, action chunk, ...>

### 3.2 Backbone
- VLM/LLM/encoder 종류, 크기, pretrained source

### 3.3 Action head
- discrete token / continuous regression / flow matching / diffusion / ACT chunking 중 어떤 방식
- chunk size, horizon, frequency

### 3.4 (옵션) 다이어그램
```
<텍스트 다이어그램>
```

## 4. 데이터

- 학습 데이터셋 (이름, 크기, 소스)
- robot embodiment 구성 (어떤 로봇, 얼마나 다양)
- 전처리·augmentation
- web/cross-modal co-training 사용 여부

## 5. 학습

- Loss: <action reconstruction / NLL / flow matching MSE / diffusion score / ...>
- Optimization: <optimizer, lr schedule, batch, steps, hardware>
- Scaling 관련 관찰
- Stage: pretraining → fine-tuning → (RL)?

## 6. 평가

| Benchmark | Setting | Metric | 본 모델 | 비교 baseline | 비고 |
|---|---|---|---|---|---|
| <name> | <sim/real, task suite> | <success rate / score> | x% | x% | <조건> |

- 핵심 ablation 결과
- 어떤 baseline을 깼는가, 어떤 task에서 안 깨졌는가

## 7. 강점 / 한계

### 강점
- 

### 한계 (저자가 인정 / 추론)
- 

## 8. 다른 모델과의 관계

- 어떤 선행 연구를 직접 이어받았는가
- 어떤 후속 모델이 이 논문을 인용·확장했는가
- 본 프로젝트 비교 8편 중 어느 위치에 놓이는가 (architecture-evolution 관점)

## 9. 우리 스터디에서 재현·실험 가능한 포인트

- 오픈 weight·코드·데이터 존재 여부
- 재현 난이도 (compute·데이터 측면)
- 흥미로운 ablation 또는 new idea 후보
- LeRobot / openpi / OpenVLA 등 어느 stack과 호환되는가
- 본 프로젝트 학습 절차 (b)·(c)에서 활용할 수 있는 방향

---

## 부록: 인용 / 추가 자료

- 함께 읽기: [[other-paper-slug]] — 왜 함께 읽는 게 유익한가
- 공식 blog / repo URL (있다면)
- 본 요약 작성 중 추가로 발견한 외부 자료 → `../external/`
