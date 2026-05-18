# <Model / Paper Name>

> **출처**: <arXiv ID / Conference / 저자 + 연도>
> **읽은 일자**: <YYYY-MM-DD>
> **PDF**: [`papers/<category>/<filename>.pdf`](../../papers/<category>/<filename>.pdf)
> **분량**: <원문 N 페이지>

## 작성 원칙 (작성자용 reminder, 발행 시 삭제 가능)

이 요약은 **사용자의 학습 교재**다. 사용자는 논문 원문을 거의 보지 않는다. 따라서:

- **Self-contained**: 이 문서만 읽고도 핵심 80% 이해 가능해야 함. "논문 참조" 같은 회피 금지.
- **수식·notation 적극**: 논문 수식은 반드시 포함하고 기호 한 줄씩 정의. 수식 없이 글로만 설명된 메커니즘은 직접 수식화하여 보충.
- **구체 예시 필수**: 추상 설명 뒤 즉시 구체 수치·token string·input/output 실제 예시.
- **Analogy 풍부히**: LLM/VLM 엔지니어 사용자에게 익숙한 개념과 1:1 대응 표.
- **메커니즘 설명**: "X가 Y한다"만이 아니라 "왜 그게 작동하는가 / 구조적 원인은 무엇인가".
- **한계는 mechanism 단위**: "X가 안 됨" → "왜 안 됨 (구조적 원인)" → "후속 모델이 어떻게 해결?"
- **분량**: 400~800 줄 권장 (분량보다 완결성).
- **시각화**: ASCII 다이어그램, 풍부한 표.

자세한 규칙: [`../../memory/preferences/summary-style.md`](../../memory/preferences/summary-style.md)

---

## 한 줄 요약

<논문의 메인 contribution 한 문장.>

## TL;DR (3~5 bullet)

- <핵심 발견 1>
- <핵심 발견 2>
- <핵심 발견 3>
- <후속 영향>

---

## 1. Motivation & 문제 정의

### 1.1 풀려는 문제
<구체적·일상적 묘사>

### 1.2 기존 방법의 한계
<어떤 시도들이 있었고 왜 부족한가>

### 1.3 본 논문의 가설
<인용 형식으로 한 문장>

## 2. 핵심 아이디어

### 2.1 한 줄
<contribution의 압축>

### 2.2 왜 직관과 어긋나는가 / 왜 새로운가
<엄밀히 무엇이 새로운지>

### 2.3 LLM/VLM 도구와의 analogy
| LLM 개념 | 본 논문 대응 |
|---|---|
| ... | ... |

## 3. 아키텍처 (상세)

### 3.1 입력 / 출력
| 항목 | 형식 | 예시 |
|---|---|---|
| Vision | ... | ... |
| Language | ... | ... |
| ... | ... | ... |

### 3.2 Backbone
<크기, layer 수, 어떤 모델을 baseline으로 했는가, parameter 수>

### 3.3 핵심 모듈 (action head / encoder / decoder 등)
<수식 또는 수도코드. 입력 차원 → 출력 차원 명확히>

```
# pseudocode 예시
def forward(image, text):
    visual_tokens = ViT(image)              # [n, d]
    text_tokens   = tokenize(text)          # [m]
    seq           = concat(visual, text)    # [n+m, d]
    output        = VLM(seq)                # [k, vocab_size]
    return output
```

### 3.4 다이어그램
```
<ASCII 다이어그램>
```

### 3.5 구체 예시 (token string·tensor shape)
<실제 입력 string과 출력 string을 예로 보여주기>

## 4. 데이터 (상세)

### 4.1 학습 데이터 구성
<출처, 크기, 다양성, 수집 방식, embodiment>

### 4.2 전처리·tokenization
<구체적으로 어떻게>

### 4.3 데이터 mixture·sampling 전략
<수식 또는 비율 표>

### 4.4 비교 — 기존 LLM/VLM 학습 규모 대비
<예: 1B image-text pair vs LLaMA-2의 2T token>

## 5. 학습 (상세)

### 5.1 Loss
<수식 + 각 기호 정의>

예시:
$$
\mathcal{L} = \mathbb{E}_{(x, y) \sim D} \left[ - \sum_{t} \log p_\theta(y_t \mid y_{<t}, x) \right]
$$

- $x$: input (image + text prompt)
- $y$: target token sequence
- $\theta$: VLM parameter
- $D$: 학습 데이터 분포

### 5.2 Optimization
| Hyperparameter | 값 | 비고 |
|---|---|---|
| Learning rate | ... | ... |
| Batch size | ... | ... |
| Steps | ... | ... |

### 5.3 학습 단계 (pretraining → fine-tune → ...)
<stage별 무엇이 다른지>

### 5.4 Scaling 관찰
<ablation 표 + 해석>

## 6. 평가 (상세)

### 6.1 Setup
<robot, 환경, trial 수, simulation vs real>

### 6.2 Baselines
| Baseline | 특징 | 비교 의미 |
|---|---|---|
| ... | ... | ... |

### 6.3 핵심 결과 표
<원논문 표 형태로, 단위·조건 명시>

### 6.4 결과 해석 — 수치가 의미하는 바
<단순 수치 나열이 아니라 "62%가 RT-1의 32% 대비 2x인 것이 어떤 의미인가">

### 6.5 Ablation
<무엇을 켜고/끄고 얼마나 변하는가>

### 6.6 정성적 발견 (qualitative)
<emergent capability, failure case 등>

## 7. 강점 / 한계

### 7.1 강점
- 항목: 왜 강점인가, 구조적 원인

### 7.2 한계 (mechanism 분석)
| 한계 | 구조적 원인 | 후속 모델이 어떻게 해결? |
|---|---|---|
| ... | ... | ... |

## 8. 다른 모델과의 관계

### 8.1 직접적 선행 연구
### 8.2 후속 연구 (본 프로젝트 8편과 어떻게 연결되는가)
### 8.3 Architecture-evolution tree에서의 위치
```
<위치 표시>
```

## 9. 우리 스터디에서 재현·실험 가능한 포인트

### 9.1 재현 가능성
<오픈 weight·code·data 여부, 재현 난이도>

### 9.2 흥미로운 ablation·new idea 후보
<Track c에서 시도해볼 만한 것들>

### 9.3 LeRobot / openpi / OpenVLA 등 stack 호환성
### 9.4 LLM 엔지니어 관점 핵심 takeaway
<표 형태로 LLM 도구와의 매핑 다시 한 번>

---

## 부록: 인용 / 추가 자료

- **함께 읽기**: [[other-paper-slug]] — 왜 함께 읽으면 유익한가
- **공식 자료**: blog, repo, project page URL
- **본 요약 작성 중 추가로 발견한 외부 자료** → `../external/` 또는 [[reference-slug]]
