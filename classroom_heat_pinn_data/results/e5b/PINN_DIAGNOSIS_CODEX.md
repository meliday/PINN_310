# E5b 구현 검증과 PINN 실패 진단 — Codex

> **후속 실험 (Claude 추가, LOG-071·072, 미커밋):** §6-3·4의 후보 일부를 사전 등록해 5000-step으로 대조했다 — f 직접 학습(linear f), data400+sink sampling, 둘의 결합. 정답 미사용 승격 기준(추정 계수를 FD에 재대입한 센서 RMS < baseline의 0.5배)을 셋 다 넘지 못해 전체 실행 없이 **E5b PINN 강도 추정을 사전 등록 범위 내 실패로 종결**했다. f 직접 학습도 1000 step 안에 f≈0으로 가 'exp 매개변수화의 gradient 소멸' 가설은 기각됐다. 결과: `results/e5b/claude_param_short_seed0/`.
>
> **후속 검증 갱신 (LOG-065·067–069):** 아래 본문은 LOG-063 당시 진단과 발견 이력이다. §5의 두 결함은 이후 Claude가 수정했고 Codex가 독립 검증했다. 추가로 FD 전체 실패 시 빈 그림 생성 예외도 Codex가 보완했다. 현재 전체 회귀 87개·기존 독립 결함 재현 검사 2개·진단 self-test 3개 통과. 따라서 본문의 “미수정/승인 보류”는 당시 상태이며 현재 두 결함은 해결됐다. 기존 과학 결과는 보존했고 PINN 계수 추정 실패는 여전히 남는다. 최종 commit hash와 실제 터미널 증거는 공동 기록 LOG-069를 따른다. §1 hash는 진단 당시 snapshot이며 runner/test의 후속 hash는 LOG-068에 있다.

작성: Codex, 2026-11-07. 사용자 요청에 따라 **구현 검증 → 실패 재현 → 원인 분리 실험 → 개선안** 순서로 수행했다. 공동 기록 LOG-057–063과 함께 읽는다.

**현재 결론: FD는 냉각 강도를 복원하지만 현재 PINN 역추정은 실패한다.** 데이터 손실 가중치를 바꾸면 온도는 개선되지만 강도는 약 90% 과소 추정된다. 반면 정답 계수를 고정한 같은 구조의 PINN은 센서 없이 냉각 온도장을 학습한다. 따라서 이 사례에서는 **온도장과 물리 계수를 함께 학습하는 최적화·손실 결합**이 핵심 병목이라는 해석이 강하게 지지된다. PINN의 실용적 우위는 입증되지 않았다.

생산 코드의 버그 수정이나 기본 설정 교체는 수행하지 않았다. 진단 도구·새 결과·이 보고서와 계획/공동 기록을 추가·갱신했다. 새로 발견한 실행 경로 결함 2건은 별도 수정이 필요하다.

## 1. 검증한 snapshot과 범위

기준 HEAD는 `76edcd0051166bb01fd06984bd2c48cf9a73a46b`다. 이 commit에는 E5b 데이터·loader·FD 순방향·국소 식별성 및 LOG-049의 기존 결함 수정이 들어 있다. 그 뒤 Claude LOG-053–056의 FD/PINN 역추정 구현과 이번 진단 도구·결과는 **미커밋**이다. 아래 hash가 그 작업 트리를 식별하며 HEAD만으로 새 구현까지 확정했다고 표현하지 않는다. commit/push는 하지 않았다.

| 파일 | SHA-256 |
|---|---|
| `src/baselines.py` | `1847930fe4852bf08289928c302c0f6e6b044730701a49606a51007c69fe7506` |
| `src/pinn.py` | `ddbe8057527423d146cfb5d3f69fdeb9aab5f4407a94e72e9c37990a681e2f54` |
| `src/run_experiments.py` | `81f4882231641f7271f6931a4bbf4ec416827972a9319e66c6cb9680e7a50d52` |
| `diagnostics/pinn_sink_diagnosis.py` 현재본 | `c35a63a1b1f22da6eac530c4c25b3fe5eaa831a9fb614cf704c1d815c24e95b9` |
| `diagnostics/pinn_oracle_forward.py` | `baf145c7ca216d90e7bc40f5c7e8c92b1dc54b012fb87cea6bdddc94c679b144` |

표의 경로는 `classroom_heat_pinn_data/` 기준이다. baseline 실행 당시 진단 도구 hash는 `6c2f3c57984dd62e540464ce6c0e9b944d1371910615dc14185d10da8a720e9b`였고, 이후 데이터 비교 guard와 미사용 변수만 정리했다. 학습·측정 함수는 바꾸지 않았다. 각 실행의 `protocol.json`에 당시 hash를 보존했다.

환경: `/Users/meliday/miniforge3/envs/DeepStudy/bin/python`, Python 3.12.13, PyTorch 2.13.0, NumPy 2.5.1, SciPy 1.18.0, CPU 8 threads, float64. 패키지 설치는 하지 않았다.

| 검증 대상 | 이번에 실제 수행한 확인 | 판정과 한계 |
|---|---|---|
| 누적 생산 코드·테스트 | 기존 unittest 전체 83개 재실행 | 통과. 모든 입력 경로의 무결함을 뜻하지 않음 |
| LOG-049의 이전 P2 3건 수정 | 데이터 routing, ill-conditioned SVD covariance, rank 부족 그림 등을 별도 6개 검사 | 수정 수용. 새로운 별도 입력 결함은 아래에 기재 |
| E0/E1/E2/E3/E5 저장 결과 | 6개 결과 폴더, PINN checkpoint 5개 재예측·Bi/벽 flux/이력 대조, RBF 3개 재계산, FD 3개 전체 multi-start 재적합 | 11개 저장 field의 핵심 수치 재현. 과거 PINN 전체 학습을 모두 다시 한 것은 아님 |
| E5b 데이터·순방향·식별성 | 생성 데이터 재현, 독립 FV 행렬·해석적 민감도, 격자 세분화, noise seed·열수지 | 통과. 국소 분석이며 전역 유일성 증명은 아님 |
| E5b FD | clean/noisy × 냉각 포함/제외를 모든 시작점에서 재적합 | 저장 수치 일치 |
| E5b PINN | clean/noisy checkpoint와 CSV 재계산; clean C1은 전체 재학습 | clean 최종 파라미터가 원 checkpoint와 정확히 일치 |
| Truth 격리·보존 | 실제 FD fitting 및 짧은 PINN runner와 진단 전체 학습의 forbidden-read 감시, 기존 실행 거부 확인 | 검사한 경로 위반 0. oracle은 정답 입력을 명시한 별도 실험 |
| 추가 예외 입력·실패 처리 | A_mismatch의 E0/E2 경로와 E5b FD 부분 실패 주입 | **추가 assertion 2개 실패, 신규 P2 2건** |

E4·E6–E8과 내부 벽 X1·결합 X2는 미구현/미실행이므로 검증 완료로 표시하지 않는다. 실측 전처리의 회귀 검사는 수행했지만 Medellín Excel 시간 파일을 포함한 전체 재전처리는 openpyxl 부재로 이번에도 하지 않았다.

실제 실행 로그의 핵심 발췌다. 개별 test 행·긴 JSON은 생략했고 원문 위치는 공동 기록에 있다.

```text
$ /Users/meliday/miniforge3/envs/DeepStudy/bin/python -B -m unittest discover -s tests -v
Ran 83 tests in 23.420s
OK
Exit Code: 0

$ /Users/meliday/miniforge3/envs/DeepStudy/bin/python -B /private/tmp/310estimate_check_legacy_saved.py
legacy_max_field_metric_difference_C 6.938893903907228e-18
Ran 1 test in 6.679s
OK
Exit Code: 0
```

## 2. 기존 결과를 독립 재계산한 값

E1은 전체 격자 RMSE, 나머지는 학습 센서 격자를 제외한 Q1 RMSE다. 서로 다른 데이터 조건의 차이를 방법 성능 차이로 해석하지 않는다.

| 기존 실험 | PINN RMSE °C | FD RMSE °C | RBF RMSE °C | 해석 |
|---|---:|---:|---:|---|
| E0 A_clean | 해당 없음 | 0.00045934 | 0.51260826 | baseline/tool 검사 |
| E1 true Bi 고정 | 0.04236531 | 해당 없음 | 해당 없음 | oracle forward diagnostic |
| E2 A_clean 기본 budget | 0.07917905 | 별도 E0 결과 | 해당 없음 | Bi_N·flux 목표 미달 이력 보존 |
| E2 A_clean C1 | 0.00684300 | 별도 E0 결과 | 해당 없음 | 단일 seed 개발 결과 |
| E3 A_noisy C1 | 0.04191679 | 0.04499725 | 0.50583067 | 단일 잡음 실현 |
| E5 A_mismatch C1 | 0.35354196 | 0.36438666 | 0.69797553 | 여러 mismatch가 섞인 한 조건, 약 3% 차이를 일반 우위로 확대하지 않음 |

E5b는 위치·폭이 알려진 Gaussian 냉각원과 Bi 3개, 냉각 강도 `f`를 다룬다. `f=0.1`은 난방 열량의 10% 제거를 뜻한다. 위치/폭 추정은 이번 범위에 없다.

| E5b 원 결과 | f 추정 | f 상대 오차 | Q1 RMSE °C | fitting 시간 s |
|---|---:|---:|---:|---:|
| FD clean, 냉각 포함 | 0.09996425 | −0.03575% | 0.00047649 | 2.781 |
| PINN clean C1 | 0.00046379 | −99.53621% | 0.18134861 | 334.070 |
| FD noisy, 냉각 포함 | 0.09109329 | −8.90671% | 0.07573117 | 3.429 |
| PINN noisy C1 | 0.00047817 | −99.52183% | 0.19237720 | 332.539 |
| FD clean, 냉각 제외 대조 | 추정하지 않음 | 해당 없음 | 0.18870897 | 원 summary 참조 |
| FD noisy, 냉각 제외 대조 | 추정하지 않음 | 해당 없음 | 0.20082651 | 원 summary 참조 |

시간은 Claude가 저장한 원 실행의 multi-start 전체 fitting/학습 시간이며 이번 진단 오버헤드를 포함한 재실행 시간과 구분한다. FD와 PINN은 최적화 알고리즘·예산이 다르다. FD는 f bounds [0.001, 0.5], Bi bounds [0.01, 50]이고 PINN은 exp parameter로 양수만 보장한다. 이 차이는 원 실행에 사전 등록돼 있었으며 공정성의 한계로 남긴다. bounds를 맞추기만 하면 해결된다는 근거는 없다.

### 데이터 자체나 source 구현이 원인인지

- 독립 조립 FV 해와 reference의 최대 차이는 `1.11e-14°C`, 101/201 격자 차이는 `0.003808°C`였다. 생성 데이터·관측 noise seed 2027·열수지를 재현했다.
- 독립 해석적 민감도의 4미지수 Jacobian은 rank 4, 조건수 `38.0834`다. σ=0.1°C에서 log f의 국소 표준편차는 약 `13.5564%`, f–Bi_S 상관은 `−0.74481`이다. 민감도 중앙차분과 해석식도 일치했다.
- PINN source의 부호·gamma autograd·optimizer 등록을 확인했다. 해석식 대비 gamma 미분 최대 차이는 `8.88e-16`이다. detach나 누락된 parameter 때문이라는 근거는 없다.
- PINN 연속 source 정규화와 erf 정확 적분의 상대 차이는 약 `0.004625%`다. f의 99.5% 과소 추정을 설명하는 크기가 아니다.

따라서 현재 사례는 국소적으로 정보가 전혀 없는 문제나 잘못된 source 부호로 설명되지 않는다. 국소 rank와 여러 시작점의 FD 성공만으로 전역 식별 가능성을 증명한 것도 아니다.

## 3. 실패 재현과 단일 변수 대조

원 clean C1을 fresh seed0에서 **Adam 20000 + L-BFGS 100**으로 그대로 재학습했다. 최종 파라미터와 Claude 원 checkpoint의 최대 차이가 **0.0**이었다. 온도 RMSE와 f도 같았다. 진단을 포함한 시간은 354.941s다.

대조 설계는 실행 전 LOG-058에 고정했다. 모든 짧은 대조는 **Adam 5000, L-BFGS 0**, 같은 초기값·모델·관측이다. baseline의 Adam5000 checkpoint와 비교한다. 전체 학습의 baseline과 짧은 후보를 같은 예산으로 비교하지 않는다.

후보 선별은 정답 f/reference를 사용하지 않고 고정 probe에서 `400 × sensor_theta_MSE + uniform_PDE_MSE + BC_MSE`가 baseline5000보다 낮은 후보 중 최소인 것으로 정했다. 이 score는 이번 개발 선별 기준이며 역추정 성공 판정기가 아니다. reference는 fitting 후 평가에만 사용했다.

| 5000-step 조건 | 바꾼 변수 1개 | f 추정 | 센서 RMS °C | Q1 RMSE °C | 고정 probe score |
|---|---|---:|---:|---:|---:|
| baseline Adam5000 | 없음 | 0.00471086 | 0.28876267 | 0.23264038 | 0.08428226 |
| data400 | λ_data: 1 → 400 | 0.00585155 | 0.03943323 | 0.10980889 | 0.01043745 |
| sink_sampling | 내부 점: uniform40% + heater30% + sink30% | 0.00812950 | 0.27274472 | 0.22187643 | 0.07565512 |
| freeze2000 | 처음 2000회 gamma 고정 후 해제 | 0.00564774 | 0.27788037 | 0.22413257 | 0.07812090 |

400은 θ를 °C로 환산하는 ΔT²=20²에서 택한 한 대조값이며, 최적 가중치나 잡음 likelihood 보정을 의미하지 않는다. sampling 대조는 별도 RNG로 sink 점을 만들어 기존 wall sampling RNG를 보존했다. gamma 고정도 한 변수만 바꿨다. 진단이 optimizer `.grad`를 바꾸지 않는지와 원 sampler 보존은 self-test로 확인했다. 갱신된 self-test 3개가 통과했다.

![동일 5000 Adam 예산의 대조](/Users/meliday/Personal/310estimate/classroom_heat_pinn_data/results/e5b/codex_diagnosis_ablations_seed0/ablations.png)

왼쪽은 f 궤적, 가운데는 센서 RMS, 오른쪽은 Q1 오차다. 점선 f=0.1이 정답이며 모든 대조가 강도 복원에 실패했다. 냉각 영역 점을 늘리는 것과 초기 gamma 고정 각각만으로는 충분하지 않았다.

선별된 **data400 한 후보만** fresh seed0에서 원 C1 전체 예산으로 확인했다. 추가 후보를 계속 돌려 성공 사례를 고르는 방식은 사용하지 않았다.

실제 전체 확인 `metrics.csv`의 주요 행이다.

```text
variant,data,adam_steps,f_hat,f_relerr,q1_rmse_C,sensor_rms_C,uniform_pde_mse,selection_score
data400,A_sink_clean,20000,0.010052909964028734,-0.8994709003597127,0.06382408042698902,0.02643027889897278,0.0014614044094940415,0.005756242905151317
Exit Code: 0
```

Exit 0은 진단 실행 종료를 뜻한다. **f 오차 −89.95%이므로 역추정 성공은 아니다.** Bi_S도 +16.73% 편향됐다. Q1은 0.18135 → 0.06382°C로 약 64.8% 감소했지만 FD clean 0.000476°C보다 훨씬 크다. clean 강도 회복에 실패했으므로 개선 설정의 noisy 추가 실행은 보류했다.

![동일 C1 전체 예산 확인](/Users/meliday/Personal/310estimate/classroom_heat_pinn_data/results/e5b/codex_diagnosis_data400_full_seed0/full_confirmation.png)

왼쪽은 전체 f 궤적(마지막 점에 L-BFGS 포함), 가운데는 Q1 RMSE의 로그 축, 오른쪽은 추정 강도다. 온도 개선과 계수 실패가 동시에 드러난다.

## 4. 무엇이 실패를 만드는가

### 4.1 측정된 gradient 불균형과 계수 감소 경로

초기 고정 probe의 net gradient norm은 data `0.81825`, PDE `40.13631`으로 약 **49.05배** 차이가 났다. 단순 loss 크기 비교를 넘어 실제 gradient를 측정한 값이다. net data/PDE gradient cosine은 step100에서 `−0.5044`, Adam5000에서 `−0.6403`이었다.

이 구현에서 센서 data loss는 net에 직접 연결되고 gamma에는 직접 연결되지 않는다. gamma는 PDE를 통해 간접적으로 관측과 결합한다. `f=exp(gamma)`, `Q=Q_heater−f A(x,y)`이면 `r=−Δθ−Q=r0+f A`이므로 `∂r/∂gamma=f A`다. 잘못된 초기 온도장을 고정했을 때 PDE를 줄이는 방향이 참 f를 향할 필요는 없다.

실제로 초기 gamma PDE gradient는 `+0.0978445`였으므로 gradient descent가 f를 낮추는 방향이다. 초기 net에서 조건부 PDE 최적 f도 약 `0.01204`였고, step100에는 비음수 범위의 조건부 최적값이 0이었다. 이후 f가 전체적으로 감소하면 gamma의 PDE 민감도 역시 f에 비례해 작아져 되돌아오기 어려워질 수 있다. 이 마지막 부분은 수식과 궤적이 뒷받침하는 기전 해석이며 모든 실패 경로의 수학적 증명은 아니다.

### 4.2 국소 sampling은 기여 후보지만 단독 해결책이 아니다

원 sampler의 냉각 중심 1σ 이내 점은 평균 `5.575/1000`, 2σ 이내는 `18.700/1000`이었다. focus 대조에서는 1σ 점이 `143/1000`으로 늘었다. 그래도 5000-step f=0.00813에 머물렀다. 따라서 국소 냉각점 확대는 후속 실험 가치가 있지만 이것만으로 해결됐다고 할 수 없다. gamma를 2000회 고정하는 간단한 warm-up도 실패했다.

### 4.3 관측을 잘 맞추면서 틀린 물리 계수를 낼 수 있다

PINN이 추정한 Bi·f를 별도 FD 201 격자에 넣어 같은 관측을 평가했다. 이는 평가 후 물리 일관성 점검이며 fitting이나 후보 선별에 사용하지 않았다.

| 전체 예산 모델 | NN 센서 RMS °C | 같은 추정 계수의 FD 센서 RMS °C | NN ↔ 해당 FD 전체장 RMS °C |
|---|---:|---:|---:|
| 원 C1 | 0.22134741 | 0.22822209 | 0.02022873 |
| data400 | 0.02643028 | 0.25369802 | 0.28026621 |

data400은 더 좋은 온도 fitting을 만들었지만 그 계수의 물리 해가 관측을 재현하지 못한다. 전체장 차이도 커졌다. 이때 Robin energy error는 약 0.304%로 작다. 따라서 작은 센서 loss·평균 PDE residual·총 에너지 오차 중 어느 하나만으로 계수 회복을 승인할 수 없다. 이것은 `physical_consistency.json`의 실제 결과다.

### 4.4 정답 계수 고정 forward는 성공했다

계수 동시 학습과 온도장 학습 자체를 분리하려고 마지막 대조를 사전 등록했다. true Bi와 f=0.1을 **oracle 입력**으로 제공하고 고정했다. 센서 없이 fresh seed0, 같은 4×64 tanh/float64·원 sampling·Adam20000 + L-BFGS100으로 학습했다. optimizer에 net만 들어가는지와 계수의 전후 동일성을 assert했다. 상대 L2≤1%, energy error≤1%를 실행 전에 정했고 둘 다 통과했다.

```text
$ /Users/meliday/miniforge3/envs/DeepStudy/bin/python -u -B classroom_heat_pinn_data/diagnostics/pinn_oracle_forward.py --output-dir classroom_heat_pinn_data/results/e5b/codex_oracle_forward_seed0
oracle_f,relative_L2,temperature_rmse_C,energy_error,targets_met
0.1,0.0004841396906189796,0.008796667814704819,0.00022652108630260976,True
Exit Code: 0
```

상대 L2 **0.0484%**, 전체장 RMSE **0.008797°C**, energy error **0.02265%**다. 이는 같은 신경망이 냉각 온도장을 학습할 수 있다는 관찰 근거다. 초기 손실 불균형과 data400의 계수–장 불일치까지 합치면, 현재 핵심 문제는 공동 역추정의 최적화/soft constraint 결합 쪽이라는 진단이 강해진다. 모든 seed에서의 성공이나 PINN 일반의 표현 능력을 증명하는 것은 아니다.

이 oracle은 **역추정 결과가 아니다.** 계수 정답을 알고 푼 진단이며 checkpoint를 역추정 초기값으로 재사용하지 않았다. oracle RMSE는 센서를 제외하지 않은 전체장 값이므로 Q1과 정의도 구분한다.

## 5. 신규 코드 결함 2건 — 미수정

### BASE-MISMATCH-REFERENCE-01 · P2

`src/run_experiments.py:168`의 BASE_DATA에 A_mismatch가 포함되고 E0/E2가 이를 허용한다. E2의 `load_reference`는 기본 reference를 고른다. 실제 임시 2-step `run_e2('A_mismatch', 0, 'C1')`가 completed로 끝났으나 사용한 reference kind는 `reference`뿐이었다. 이 데이터의 평가 정답은 mismatch_reference이므로 잘못된 평가 결과를 만들 수 있다. 현재 저장된 정상 E0/E2나 올바른 E5/E5b 결과를 무효로 만드는 발견은 아니다.

제안 수정: 실험별 허용 데이터를 나눠 E0/E2에서는 A_mismatch를 학습·출력 전에 거부하고 E5로 안내한다. E3도 계약을 명시적으로 제한한다. 다른 방법으로 지원하려면 평가 reference 선택과 물리 의미까지 함께 맞춰야 한다. 최소 회귀는 해당 CLI/API 호출이 출력 폴더 생성 전에 거부되는지 검사한다.

### E5B-FD-EXIT-01 · P2

`src/run_experiments.py:1087`의 FD summary에는 전체 status가 없고, `:1259`의 E5b_fd dispatch는 실패 집계 없이 return한다. 냉각 FD fitting 두 건에 nonconvergence를 주입하고 무냉각 대조는 실제 적합했더니 metrics에는 failed가 있어도 CLI가 exit 0이었다.

제안 수정: condition별 status와 전체 status를 집계하고 하나라도 실패하면 CLI exit 2를 반환한다. 실패 시 결과/그림 경로도 안전하게 처리한다. 최소 회귀는 한 조건 실패·한 조건 성공을 주입했을 때 실패 행 보존과 전체 status/exit를 함께 검사한다. 현재 저장 FD는 모두 수렴했으므로 기존 수치에는 영향이 없다.

실제 추가 검사 출력이다. 생산 코드를 고치지 않고 임시 폴더에서 검사했다.

```text
$ /Users/meliday/miniforge3/envs/DeepStudy/bin/python -B /var/folders/s6/sysz200n1lddbdd5634fnbd80000gn/T/310estimate-turn057-79xa_feo/check_remaining_contracts.py
Ran 2 tests in 6.677s
FAILED (failures=2)
E2_MISMATCH_ROUTE {"data": "sensors_A_mismatch.csv", "reference_kinds": ["reference"], "status": "completed"}
FD_FAILURE_EXIT {"exit_code": 0, "top_status": null, "metric_statuses": ["failed", "completed", "failed", "completed"]}
Exit Code: 1
```

**83개 회귀 통과와 이 2개 실패를 함께 보고한다. 구현 전체의 무조건 승인은 보류한다.** 최초 지적 2건이며 동일 쟁점 상호 거절·재수정 2회가 아니므로 Freeze 조건은 발생하지 않았다.

## 6. Codex의 개선 제안과 다음 순서

아래는 이번 측정을 바탕으로 한 **Codex 제안**이며 아직 실행하지 않은 항목이다. data400이나 focus를 성공한 기본 설정으로 채택하지 않는다. 후속 성능 기준·선별·예산은 실행 전에 별도로 고정한다.

1. **새 P2 두 건부터 원자 단위로 수정하고 교차 검증한다.** runner + 테스트 + 공동 기록의 최대 3파일 규칙을 지킨다. 과학적 f 추정 실패와 실행 계약 결함을 같은 문제로 혼동하지 않는다. 미커밋 구현은 검토 후 실제 commit에 연결해야 한다.
2. **성공 기준에 계수–장 물리 일관성을 넣는다.** 관측 밖 Q1, f/Bi 오차, uniform/냉각 주변 residual, Robin 및 gradient flux, 추정 계수 FD 재대입을 같이 보고한다. 제안하는 다음 clean 개발 gate는 f·Bi 각 상대 오차≤5%, Q1≤0.02°C, NN↔추정 계수 FD 장 RMS≤0.02°C, energy error≤1%다. 이는 *다음 실험의 새 제안 기준*이며 이번 결과에 사후 적용해 통과를 만드는 기준이 아니다. 잡음 조건은 noise·식별성에 맞는 별도 허용 범위가 필요하다.
3. **손실과 국소 sampling의 상호작용을 한 단계 확인한다.** λ_data=400 조건을 기준으로 sink_sampling만 추가한 아직 없는 조합을 같은 5000-step 예산에서 비교한다. 지금까지는 각각만 시험했으므로 결합 효과는 모른다. 다른 optimizer·architecture·initial f는 동시에 바꾸지 않는다. 사전 고정 probe 기준과 물리 일관성 gate를 모두 만족한 후보에만 전체 예산을 준다.
4. **그 다음 계수 학습을 따로 조건화한다.** gamma/Bi와 net의 gradient 크기를 별도로 보고, parameter scaling/학습률 또는 교대 최적화를 한 변수씩 검토한다. 단순 gamma 2000-step 잠금은 이미 실패했으므로 같은 제안을 해결책으로 반복하지 않는다. net을 각 추정 계수에서 충분히 물리적으로 수렴시킨 뒤 계수를 갱신하는 방법은 후보지만 비용과 inner solve 허용 오차를 같이 보고해야 한다. oracle 정답이나 그 가중치를 넘겨주는 초기화는 허용하지 않는다.
5. **상수 가중치의 한계가 남으면 gradient 기반 적응 균형을 별도 대조한다.** data400은 단위 스케일 대조일 뿐 해결책이 아니다. adaptive weight 역시 관측 fitting만 개선하고 물리를 느슨하게 만들 수 있으므로 위 gate를 유지한다. 같은 관측·알려진 물리·가능한 한 같은 bounds와 전체 fitting 비용을 FD와 비교한다.
6. **개발 설정을 잠근 뒤 독립 평가한다.** 예: training seeds 1·2·3과 별도 noisy 관측 실현을 미리 고정하고, 재튜닝 없이 실패율·f/Bi·온도·비용을 모두 보고한다. 먼저 clean 계수 복원 gate를 충족해야 한다. 여러 noise/seed에서의 안정성과 실용적 이득은 현재 미확인이다.
7. **사용자가 요청한 내부 벽·결합 문제는 유지한다.** E5b 강도 복원/검토 → 필요 시 위치 추정 → X1 벽 순방향 검증 → X2 기본/벽만/냉각원만/결합 네 조건 비교 순서다. 기존 실패를 숨기기 위해 형상을 복잡하게 만들거나 PINN 우위를 미리 결론짓지 않는다. 현 단계의 빠르고 정확한 기준선은 FD다. FD를 초기화나 inner solver로 쓰는 변형은 hybrid로 명명하고 순수 PINN의 성과로 합치지 않는다.

복합 손실의 gradient 불균형과 이를 gradient 통계로 조정하는 접근은 [Wang·Teng·Perdikaris 원 논문](https://arxiv.org/abs/2001.04536)에 제시돼 있다. 여기서는 적응 가중치 후보의 근거로만 참고하며 그 논문의 성능 개선 수치를 이 프로젝트의 결과로 가져오지 않는다. PINN soft constraint의 최적화 난점과 curriculum 접근도 [Krishnapriyan 등 원 논문](https://arxiv.org/abs/2109.01050)에 보고돼 있다. 해당 논문과 우리 문제의 PDE/실험은 다르며, 이번 원인 진단의 직접 근거는 위 독립 재현·gradient·단일 변수 대조·oracle 결과다.

## 7. 재현 경로·한계·보존

생산 runner는 현재 `--exp E5b_ident`, `E5b_fd`, `E5b_pinn`을 지원한다. 아래 진단 파일은 별도 실행이며 새 exp ID가 아니다. 각 `--output-dir`은 **빈 새 경로**여야 한다. 저장된 결과를 읽기만 할 때는 재실행할 필요가 없다.

```bash
# 저장소 루트; python은 위 검증 interpreter를 사용한다.
python -B classroom_heat_pinn_data/diagnostics/pinn_sink_diagnosis.py --mode self-test
python -u -B classroom_heat_pinn_data/diagnostics/pinn_sink_diagnosis.py --mode baseline --output-dir /tmp/e5b_new_baseline
python -u -B classroom_heat_pinn_data/diagnostics/pinn_sink_diagnosis.py --mode ablations --baseline-dir /tmp/e5b_new_baseline --output-dir /tmp/e5b_new_ablations
python -u -B classroom_heat_pinn_data/diagnostics/pinn_sink_diagnosis.py --mode promote --variant data400 --output-dir /tmp/e5b_new_data400_full
python -u -B classroom_heat_pinn_data/diagnostics/pinn_oracle_forward.py --output-dir /tmp/e5b_new_oracle
```

결과 폴더의 `metrics.csv`, `summary.json`, `result_*.json`, `protocol.json`, `probes_*.json`, `history_*.json`, `progress.jsonl`, checkpoint, truth access log가 수치와 학습 경로의 근거다. 물리 재대입 값은 `codex_diagnosis_data400_full_seed0/physical_consistency.json`에 있다. 새 진단 CSV는 각 폴더 안에 두었고 기존 전역 metrics의 원 결과를 바꾸지 않았다.

추가 검증 원문은 `/var/folders/s6/sysz200n1lddbdd5634fnbd80000gn/T/310estimate-turn057-79xa_feo/`의 `regression.log`, `previous_findings.log`, `saved_estimates.log`, `data_ident_numerics.log`, `fitting_isolation_corrected.log`, `remaining_contracts.log`, `legacy_saved.log` 등이다. 임시 파일은 장기 보존이 보장되지 않으므로 핵심 명령·실제 출력·exit code를 공동 기록과 이 보고서에 함께 남겼다. 격리 검사 첫 시도는 검증 스크립트의 .05 정확 동등 비교 때문에 실패했고, 허용 오차 1e-14로 검사만 고친 후 통과했다. 이 자체를 생산 버그로 세지 않았다.

이번 개선 대조와 oracle은 clean seed0 개발 실험이다. 후보 score는 reference를 사용하지 않지만 이미 결과를 본 데이터에서의 개발이므로 독립 평가로 부르지 않는다. noisy 원 실패는 checkpoint 재평가했고 개선안 noisy/추가 seed는 실행하지 않았다. PINN 전체 학습은 baseline 재현·짧은 대조 3개·선별 1개 전체 확인·oracle 1개까지이며 무제한 튜닝을 하지 않았다.

TURN-062 종료까지 snapshot의 기존 코드·계획·데이터·결과 80개가 모두 불변이었다. TURN-063에서는 계획만 갱신하고 나머지 기존 79개 보존을 다시 검사한다. 과거 LOG-001–056 본문은 보존한다. 기록·계획 파일은 .gitignore 대상이므로 HEAD가 이 문서 자체의 보존을 뜻하지 않는다.
