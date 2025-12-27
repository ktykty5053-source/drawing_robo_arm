# drawing_robo_arm

간단한 Pencil Recorder 웹 클라이언트와 로봇 암 펜슬 레코더 행동 모방(behavior cloning) 학습 코드를 함께 관리합니다.

## 폴더 구조
- `web/`: GitHub Pages로 배포 가능한 정적 웹 클라이언트(현재 placeholder).
- `ml/pencil_recorder_bc/`: BC 학습/추론 코드와 설정.
- `docs/`: 추가 문서(예: `CONTEXT.md`).

## Web: GitHub Pages로 실행
1. GitHub Repository Settings → Pages에서 Source를 `main` 브랜치, 폴더를 `/web`으로 선택합니다.
2. 저장하면 `https://<계정명>.github.io/<레포명>/`에서 `web/index.html`이 서빙됩니다.
3. 로컬 미리보기: 레포 루트에서 `cd web && python -m http.server 8000` 후 브라우저로 `http://localhost:8000` 접속.

## ML: 학습 실행
1. Python 환경 준비 후 필요한 패키지 설치 예시:
   ```bash
   python -m pip install torch tqdm pyyaml
   ```
2. 학습 실행(레포 루트 기준):
   ```bash
   python ml/train_bc.py --data_dir data/raw --run_dir runs/exp1 --config ml/config.yaml
   ```
   - `data/`에 원본 세션 JSON을 넣고, 출력은 `runs/` 아래에 저장됩니다.
   - 추가 옵션은 `--help`로 확인 가능합니다.

## 참고
- 대형 데이터/체크포인트(`data/`, `runs/`, `*.pt`)는 `.gitignore`로 제외되어 있습니다.
- 프로젝트 맥락 및 폴더 설명은 `docs/CONTEXT.md`를 참고하세요.
