---
name: diagram-alignment
description: ASCII diagram 작성 시 monospace alignment를 엄격히 유지. CJK/Unicode 문자 혼용으로 한 칸씩 밀리는 현상 방지
metadata:
  type: preference
  created_at: 2026-05-23
  updated_at: 2026-05-23
---

# 핵심 규칙

ASCII art diagram을 작성할 때 **각 column이 monospace로 정확히 정렬되어야 한다**. 한 칸이라도 밀리면 가독성 크게 손상.

# Why (계기)

2026-05-23 session-009 (π0 정독) 후 사용자가 명시:
> "다이어그램을 작성하실때는 그림의 alignment가 잘 맞도록 작성해주세요. 가끔씩 한칸씩 밀리거나 그런 현상이 보이네요"

# 원인 분석 — 왜 alignment가 깨지는가

Monospace font에서:
- **ASCII 문자** (a-z, 0-9, 기호): **1 column 폭**
- **한글/CJK 문자** (안, 국, 중, 日): **2 column 폭**
- **Box-drawing 문자** (─, │, ┌, ┐ 등): **1 column 폭** (보통)
- **Unicode 첨자/수식** (₁, ², ℓ, τ): **불확실** (font/터미널마다 다름)
- **이모지** (🎯, ✅): **2 column 폭** (보통)

따라서 다음 같은 mixing이 정렬 깨뜨림:
```
│ Image (이미지)    │  ← "이미지" 6 chars지만 12 columns 차지
│ Image (img)       │  ← "img" 3 chars = 3 columns
```

# How to apply

## 규칙 1: Box 내부에는 ASCII만 사용 (가장 안전)

❌ Bad:
```
┌──────────────┐
│ 이미지 입력  │   ← "이미지 입력" = 5 한글 + 1 ASCII = 11 cols, 시각적 폭 불확정
└──────────────┘
```

✅ Good:
```
┌──────────────┐
│ Image input  │   ← 모두 ASCII, 12 cols 정확
└──────────────┘
```

또는 한글이 꼭 필요하면 박스 밖에 라벨:
```
┌──────────────┐
│ image_input  │  이미지 입력 (label)
└──────────────┘
```

## 규칙 2: Unicode subscript/그리스 문자 신중

❌ Risky: `I₁, I_t, τ, ℓ` 같은 문자를 박스 내 사용
- 일부 터미널에서 1 column, 다른 곳에서 2 column

✅ Safer:
- 박스 내: `I_1`, `tau`, `l_t` (ASCII로 표기)
- 박스 외 본문: `I₁`, `τ`, `ℓ` (수식 표현 OK)

## 규칙 3: 모든 row의 box 너비 정확히 일치

작성 후 각 column을 **세로로 시각 검사**. 예:
```
┌─────────┐  ┌─────────┐    ← 두 박스 시작/끝 정확히 정렬
│ width=9 │  │ width=9 │
└─────────┘  └─────────┘
```

내부 텍스트 padding도 신중:
```
│ short   │   ← 5글자 + 3 space padding
│ longer  │   ← 6글자 + 2 space padding (각 row 우변 정렬)
```

## 규칙 4: 복잡한 다이어그램은 indented text/arrow로

ASCII box로 정렬 보장이 어려우면 **들여쓰기 + arrow notation** 권장:

✅ Alternative style (alignment 부담 적음):
```
Input
  ├─ Image (camera frames)
  ├─ Language: "fold the shirt"
  └─ Robot state
        ↓
  VLM (PaliGemma)
        ↓
  Action Expert (flow matching)
        ↓
  Velocity prediction → ODE 10 step → Action chunk
```

또는 단순 sequential flow:
```
Image → ViT → tokens ─┐
Language → Tokenizer ─┼→ VLM → features → Action Expert → velocity
State → Linear ────────┘
```

이 형태는 box drawing 부담 없이 흐름 표현 가능.

## 규칙 5: 표(table)를 적극 활용

박스 다이어그램 대신 **markdown table**이 더 안정적 (렌더링이 alignment 보장):

❌ Box diagram (정렬 깨질 위험):
```
┌──────────┬──────────────┐
│ Param    │ Value        │
├──────────┼──────────────┤
│ width    │ 1024         │
│ depth    │ 18           │
└──────────┴──────────────┘
```

✅ Markdown table:
```markdown
| Param | Value |
|---|---|
| width | 1024 |
| depth | 18 |
```

## 규칙 6: ASCII art는 code block 내부에만

````markdown
```
ASCII diagram here
```
````

이렇게 code block 안에 두면 monospace 보장. 본문 안에 직접 두면 폰트가 바뀌어 정렬 깨질 가능성.

## 규칙 7: 작성 후 자가 검증

각 row 마지막 column이 모두 같은 위치인지 시각적으로 확인. 의심되면 더 simple한 형태로 재작성.

# 회고: 본 프로젝트에서 발견된 문제 예시

- `reports/papers/pi0.md` § 3.2 다이어그램에 `I_1, ..., In` (Unicode subscripts) + 한글 라벨 + 박스 → 일부 row 한 칸 밀림 가능성
- `reports/papers/SmolVLA.md` § 3.4 다이어그램에 `[B, N_patches, d]` 같은 텐서 shape이 box 안에 들어가 폭 계산 어려움
- 다음 정독부터는 위 규칙 엄격 적용

# 회고: 향후 작성 default 전략

| 상황 | 권장 형태 |
|---|---|
| 모듈 간 데이터 흐름 (3-5 노드) | **Indent + arrow** 또는 단일 row sequential |
| 비교 (값/parameter 표) | **Markdown table** |
| 복잡한 attention mask, block 구조 | **단순 ASCII (모두 1-column 폭)** + label은 박스 밖 |
| 수식 포함 | LaTeX `$...$` 사용, ASCII art에 끼우지 않음 |
| 사용자에게 의미 전달이 핵심 (정확한 정렬보다) | **indent tree** 형태 (alignment 부담 최소) |

# 관련 메모리

- [[summary-style]] — 학습 자료 작성 일반 규칙 (이 메모리가 그 보완)
- [[user-role]] — LLM 엔지니어. 가독성 높은 다이어그램이 학습 효율에 직결
