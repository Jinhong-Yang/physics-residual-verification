# IEEE Access 초기 투고용 프로젝트

Overleaf에서 New Project → Upload Project를 선택해 `IEEE_Access_Initial_Submission_Overleaf.zip`을 업로드합니다. Main document는 `main.tex`, Compiler는 **pdfLaTeX**로 설정합니다.

- `main.tex`: 논문 본문.
- `supplementary.tex`: 구현·통계·데이터 선정·계산 재현 보충자료.
- `cover_letter.tex`: 초기 투고용 커버레터.
- `references.bib`: 참고문헌.
- `figures/`: 본문 그림과 입력 예시.
- `authors/`: 제공된 세 저자의 원본 사진. 논문 말미의 약력은 `author_metadata.txt`에 제공된 학위·경력·연구 관심사를 반영했습니다.
- `evidence/`: 선택한 실험 기록·체크포인트·코드 발췌·출처 점검 기록.
- `tools/`: 저장 결과 재계산 및 특징 캐시 기반 추가 실험 도구. NumPy가 필요하며 추가 실험에는 SciPy도 사용합니다.

저자 3명, 소속, 교신 이메일, 연구비 번호, 제공된 ORCID 2개를 반영했습니다. AI 사용 고지는 Acknowledgment에 있습니다. 저널이 배정하는 권·연도는 템플릿 자리표시자이며, 접수일·게재일·DOI를 임의로 작성하지 않았습니다.

컴파일할 문서를 바꾸려면 Main document에서 해당 `.tex`를 선택합니다. 계산 재현은 Overleaf 외부의 Python 환경에서 실행합니다.

```text
python tools/replay_tables.py
python tools/effect_sensitivity.py
python tools/run_support_study.py
python tools/score_access_video_control.py
```

두 번째 도구는 저장된 평가 단위를 하나씩 제외하는 사후 민감도 분석입니다. 새 모델 학습이나 독립 표본의 검증이 아닙니다. 결과는 `replayed/`에 생성됩니다. 원본 RGB부터 전체 예측을 재생성하는 환경은 이 패키지에 포함되지 않습니다.

세 번째 도구는 **NumPy와 SciPy**를 사용해 저장 특징에서 7개 잔차 모델을 실제로 재적합하고, 공분산 계수 4종 및 초기 Jacobian을 분석합니다. GPU 없이 실행됩니다. 132개 개발 family의 기존 6-fold를 재현하며, 이미 평가한 72개 family의 추가 결과는 사후 분석으로만 다룹니다. 기존 최종 모델과 주요 성능값은 바꾸지 않습니다. 결과·입력 캐시·수치 계산 코드를 함께 포함했습니다.

실제 제출 여부는 과학적 근거와 저자 확인을 포함해 별도로 판단해야 합니다. 저자 검토용 평가서는 업로드 ZIP 밖에 제공됩니다.

네 번째 도구는 고정 V-JEPA 2 특징에서 각 학습 fold 안에서만 PCA와 잔차 회귀를 적합합니다. 입력 특징과 결과·설정은 evidence/video_control 및 replayed/video_control에 있습니다. evidence/video_control 안의 원본 특징 추출·RGB 재현 스크립트는 원래 연구 작업공간용 소스 기록입니다. 이 소스 ZIP만으로 원본 RGB와 기반 모델이 모두 제공되는 것은 아니며, 캐시 기반 네 번째 도구와 구분합니다.
