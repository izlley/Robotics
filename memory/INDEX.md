# Memory Index

> 이 프로젝트의 모든 영구 메모리 파일 카탈로그. 새 메모리 추가 시 해당 섹션에 한 줄 entry로 등록.
>
> 형식: `- [Title](path) — one-line hook`

## Decisions

- [Track 선택: 이론 심화](decisions/track-selection.md) — 학습 절차 (a) Track A를 첫 트랙으로 선택, baseline 4편 + 확장 자료 수집
- [Baseline 논문 스코프](decisions/baseline-papers-scope.md) — 정독 8편 + 2nd tier/foundations/data/surveys 보강 정책
- [영구 메모리 정책](decisions/persistent-memory-policy.md) — project-dir 자체에 memory/sessions 보존, GitHub 동기화
- [확장 커리큘럼 — 옵션 3 (RFM R&D)](decisions/extended-curriculum-for-rfm-research.md) — 원 8편은 책무의 50-60%만 cover. Track A 완료 후 World Model 3편 + LAPA + Q-Transformer 추가 정독 → Track B

## Learnings

- [RT-2: action-as-token 패러다임](learnings/rt2-action-as-token-paradigm.md) — 256 bin × 8 차원, VLM vocab의 token slot 빌려 action ID로 재해석
- [RT-2: co-fine-tuning trick](learnings/rt2-co-fine-tuning-trick.md) — web data를 50~66% 비중으로 robot fine-tune에 함께 흘리면 generalization +10pp
- [OpenVLA: DINOv2+SigLIP 비전 인코더 fusion](learnings/openvla-dual-vision-encoder-fusion.md) — 두 인코더 patch feature를 channel-wise concat. spatial + semantic 양쪽 강화. vision encoder는 freeze 금지
- [OpenVLA: LoRA r=32 fine-tune recipe](learnings/openvla-lora-recipe.md) — VLA fine-tune에서 LoRA r=32가 full FT와 동등, 1.4% params, single A100, 10-15h
- [OpenVLA: quantile-based discretization](learnings/openvla-quantile-discretization.md) — min-max 대신 1-99 percentile로 bin width 산정. outlier 제거, granularity ~100x
- [SmolVLA: VLM layer skipping](learnings/smolvla-vlm-layer-skipping.md) — VLM의 앞 L/2층만 perception에 사용. 마지막 N층은 generation 특화라 불필요. "큰 VLM 절반 ≈ 작은 VLM 통째"
- [SmolVLA: Async inference (model-agnostic)](learnings/smolvla-async-inference.md) — Action 실행/예측을 별 process로. idle 0, latency 30%↓, 처리량 2x. 모든 chunk-출력 모델에 적용 가능
- [SmolVLA vs OpenVLA: 두 패러다임 비교 (token vs flow-matching)](learnings/smolvla-flow-matching-vs-token.md) — VLA의 본질적 두 분기점. 정밀·multimodal action엔 flow-matching, 단순·CoT 결합엔 token-based

## Preferences

- [사용자 역할·배경](preferences/user-role.md) — dnotitia LLM 엔지니어, LLM/VLM 배경, 로보틱스 신규 진입
- [Pod 환경 제약](preferences/pod-environment.md) — CephFS만 영구, H200 kubectl-only, offload 금지 등
- [Summary 작성 스타일](preferences/summary-style.md) — md 요약이 학습 교재. self-contained, 수식·예시·analogy·메커니즘 포함, 400~800줄 권장
- [Light 코드 체크인 정책](preferences/light-code-checkpoints.md) — 이론 정독 중간에 5~10분 가벼운 inference 체크인. 환경 setup 없이 from_pretrained + 1회 forward만. 본격 hands-on은 Track A 완료 후

## References

_(아직 없음 — 외부 자료 수집 시 추가)_
