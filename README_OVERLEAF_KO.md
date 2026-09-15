# IEEE Access 초안 투고 패키지 v0.4.0

가장 강한 profiled MAP 대비 20.30% 개선을 핵심 결과로 정렬했습니다. 87차원 numeric-only, 동일 차원 Lucas-Kanade, VLM-free MAP, 실영상 단위 정정 및 신규 40개 계열 보정을 추가했습니다. 시각 우월성은 입증되지 않았으며 신규 보정 포함률 81.94%는 목표 미달입니다. 이 결과를 숨기거나 평가셋에 맞춰 재조정하지 않았습니다.

초안 투고용 본문·보충자료·커버레터이며 심사 답변서는 포함하지 않습니다. 제공된 저자 정보·사진·연구비·AI 고지는 보존했습니다. Table 10은 본문 10쪽, Data and Code Availability는 Discussion/Conclusion 뒤에 배치했습니다.

Overleaf에서 main.tex와 pdfLaTeX를 선택합니다. supplementary.tex와 cover_letter.tex는 각각 별도로 컴파일합니다. 제출 묶음에는 PDF 3개와 Overleaf 및 RGB 압축파일 3개가 들어갑니다. 신규 재현 명령은 `python tools/run_followup_study.py`입니다. 상세 결과와 재현 범위는 README.md 및 tools/README_followup.md를 참조하십시오.

공개 릴리스: https://github.com/Jinhong-Yang/physics-residual-verification/releases/tag/v0.4.0

신규 비교는 사후 분석입니다. 감독을 맞춘 인코더 비교, 자동 식별가능성 마스크, 외부 방법 이식, 물체·물리·센서 3축 검증은 완료되지 않았으며, 저널 심사 쟁점이 모두 해결되었다고 주장하지 않습니다.
