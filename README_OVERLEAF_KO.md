# IEEE Access 초기 투고 패키지 v0.3.0

첨부 의견을 참조해 초기 투고용 본문·보충자료·커버레터를 업데이트했습니다.
외부 심사에 대한 답변서나 수정투고 문서는 포함하지 않습니다.

- 제목: Protocol-Constrained Residual Adaptation for Rigid-Body Inference: A Bounded Evaluation of Frozen Visual Features
- 수렴 Huber 대비 24.11% 개선을 기존 고정 가중치 Huber 대비 29.95%와 구분했습니다.
- 교차적합 시각 증분은 1.17%이며, 양측 구간은 여전히 0을 포함합니다.
- 채널별 개발 페널티 탐색, 직접 회귀 기준선, 프로토콜·좌표 오차, 축소 민감도를 추가했습니다.
- PyBullet 구체 보조 평가의 28.19% 개선은 원래 형상 자산의 동역학 검증과 구분했습니다.
- 물체·바닥 마찰 분리 4.38% 결과와 [-6.044, -0.116] mm 구간은 보존했습니다.
- 저자·소속·사진·ORCID·연구비·AI 고지 문장은 보존하고, 도구 인용만 Acknowledgment URL로 옮겼습니다.

`IEEE_Access_Initial_Submission_Files.zip`에 PDF 3개와 Overleaf/RGB ZIP이 들어갑니다.
Overleaf는 `main.tex`를 메인 문서로, pdfLaTeX를 컴파일러로 선택합니다.
보충자료와 커버레터는 각각 해당 tex를 메인 문서로 선택합니다.
공개 위치: https://github.com/Jinhong-Yang/physics-residual-verification/releases/tag/v0.3.0

추가 분석은 모두 사후 기술 분석이며 원래 모델을 바꾸지 않았습니다.
실행시간은 모델 적재 후 구간별 측정 합계이고, 실영상 실패의 원인은 인과적으로 확정되지 않았습니다.
수렴 MAP 두 사례의 종료 한계, 감독을 맞춘 인코더 비교 미실시, 독립 보정·신규 자산 전이 미검증도 명시했습니다.
실제 저널 제출과 게재 승인은 이 작업에 포함되지 않습니다.

추가 재현: `python tools/run_extended_study.py`. NumPy, SciPy, PyTorch가 필요합니다. 자세한 내용은 `tools/README_extended.md`에 있습니다.
