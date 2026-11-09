# 보유 MOSS-Image LoRA 카탈로그 (바탕화면 압축본 7종 연동)

바탕화면 압축 아카이브에서 추출하여 **LoRA Trainer 로컬 보관함(`loras/`)** 및 **ComfyUI LoRA 폴더(`D:\AI\ComfyUI_windows_portable\ComfyUI\models\loras/`)** 양쪽에 동기화 배치된 7종의 MOSS-Image LoRA 목록입니다.

> 대용량 가중치 원본(`.safetensors`)은 `.gitignore`에 의해 Git 커밋에서 제외되며, 로컬 학습기(이어학습 `init_weights` 및 `LoRA점검.bat`)와 ComfyUI에서 즉시 사용할 수 있습니다.

| 번호 | 파일명 | 베이스 모델 | 해상도 | Rank / Alpha | 학습량 | 모듈 수 | 평균 변화량 ($\|\Delta W\|_F$) | 상태 |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `moss_UD_큰눈이_v3_1024_R16_1.008k.safetensors` | MOSS-Image | 1024×1024 | 16 / 16.0 | 56 epoch (1,008 step) | 246 | `0.760` | 정상 학습 완료 |
| 2 | `moss_UD_믹스걸02_2010.safetensors` | MOSS-Image | 1024×1024 | 16 / 16.0 | 67 epoch (2,010 step) | 252 | `1.110` | 정상 학습 완료 |
| 3 | `moss_UD_LW가비친구셋_V3_0.317k.safetensors` | MOSS-Image Base | 512×512 | 32 / 32.0 | 317 step (EMA 0.98) | 252 | `0.206` | 정상 학습 완료 |
| 4 | `moss_UD_LW가비친구셋_V3_0.396k.safetensors` | MOSS-Image Base | 512×512 | 32 / 32.0 | 396 step (EMA 0.98) | 252 | `0.224` | 정상 학습 완료 |
| 5 | `moss_UD_가비셋_V3_0.198k.safetensors` | MOSS-Image Base | 512×512 | 32 / 32.0 | 198 step (EMA 0.98) | 252 | `0.183` | 정상 학습 완료 |
| 6 | `moss_UD_그록범용_V3_0.15k.safetensors` | MOSS-Image Base | 1024×1024 | 32 / 32.0 | 150 step (EMA 0.98) | 252 | `0.137` | 정상 학습 완료 |
| 7 | `moss_UD_지피동그리라스트_V3_0.1k.safetensors` | MOSS-Image Base | 512×512 | 32 / 32.0 | 100 step (EMA 0.98) | 252 | `0.159` | 정상 학습 완료 |

## 이어학습(`init_weights`) 호환 안내

- 위 7종의 LoRA는 어텐션 가중치가 `layers.N.proj.q`, `layers.N.proj.k`, `layers.N.proj.v`로 분리된 레이아웃입니다.
- 본 `LoRA_Trainer`의 `LoRANetwork.load_weights()`는 분리된 `proj.q/k/v` LoRA 가중치를 결합된 `proj.qkv` LoRA 레이어로 **자동 SVD 병합 변환**하여 252개 전체 어텐션·FFN 모듈을 누락 없이 이어학습할 수 있습니다.
