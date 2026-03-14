# 📄 test4.py 통합 애플리케이션 분석 보고서

## 1. 개요 (Overview)
[test4.py](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py)는 사용자의 발표나 스피치 연습을 돕기 위해 기획된 **AI 기반 자세 및 스피치 교정 통합 애플리케이션(Speech Karaoke App)**입니다. 앞서 개발된 여러 모듈(`Test2_MP`, `test3`, `pose_comparator`, `gaze_anxiety_detector`, `key_pose_extractor`, `audio_analyzer` 등)을 하나로 통합하여, 실시간으로 사용자의 자세, 시선, 음성, 그리고 제스처를 종합적으로 분석하고 피드백을 제공하는 메인 엔트리 역할을 수행합니다.

## 2. 주요 기능 및 모듈 구성 (Key Features & Modules)

본 스크립트는 여러 외부 모듈을 import하여 각자의 고유 기능들을 융합합니다.
- **자세 분석 (`Test2_MP`)**: `PoseAnalyzer`, `AlertChecker`, `OverlayAlertPresenter`를 통해 어깨 비대칭, 불량한 자세, 손의 위치 등을 실시간으로 감지하고 화면에 경고 알림을 표시합니다.
- **스피치 가라오케 (`test3`)**: 자막 동기화(`_load_subtitles`, `_draw_subtitle_karaoke`) 기능을 통해 레퍼런스 영상(예: 오바마 연설)과 자막을 함께 띄워주며 사용자가 시각적으로 따라할 수 있게 돕습니다.
- **모션 유사도 비교 (`pose_comparator`)**: 슬라이딩 윈도우(Sliding Window)와 코사인 유사도/DTW 방식을 이용하여 레퍼런스 영상과 사용자의 실시간 제스처 유사도(`PoseComparator`)를 측정합니다.
- **시선 불안정 감지 (`gaze_anxiety_detector`)**: `GazeAnxietyDetector`를 통해 사용자의 시선이 불안정하게 흔들리거나(Shaking) 회피(Avoiding)하는지를 분석경고 메시지(AlertMessage)로 변환합니다.
- **핵심 자세 추출 (`key_pose_extractor`)**: `KeyPoseExtractor`를 적용하여 특정 발화 순간의 핵심 제스처(손목, 어깨 좌표)를 레퍼런스와 비교하고 구체적인 피드백 로그를 생성합니다.
- **음성 분석 (`audio_analyzer`)**: 발화 속도(WPM), 성량(Energy/Volume), 억양/스트레스(Pitch)를 분석하여 시각적인 스탯 바(Bar) 형태로 실시간 피드백을 화면에 그려줍니다.

## 3. 애플리케이션 구동 상태 (App Modes)

앱의 진행 상태는 [AppMode](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#27-33) Enum 클래스로 철저하게 분리 관리되며, 다음과 같이 5가지 상태로 나뉩니다.

1. **`DEFAULT` (기본 모션/자세 피드백 모드)**
   - 앱 실행 시 최초 나타나는 기본 화면입니다. (웹캠 단일 화면)
   - 기본적인 자세 불량 경고 및 **시선 불안정 평가** 결과가 실시간으로 화면에 오버레이됩니다.
   - 우측 상단의 `[Practice Obama]`, `[Test Obama]` 버튼을 클릭해 다른 모드로 진입할 수 있습니다.

2. **`COUNTDOWN` (테스트 준비 모드)**
   - [Test](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#45-581) 모드로 진입하기 직전 실행되며, 화면이 어두워지고 "3, 2, 1, START!" 카운트다운 텍스트를 크게 표시하여 사용자가 연설을 준비할 수 있도록 딜레이를 제공합니다.

3. **`KARAOKE_PRACTICE` (가라오케 연습 모드)**
   - 좌측에 레퍼런스 영상, 우측에 웹캠 영상이 나란히 배치되는 이중 화면 (Side-by-Side) 구도를 전환합니다.
   - 하단에 음성과 일치하는 자막이 표시되며, 실시간으로 자세 유사도(Pose Similarity) 바와 음성 분석 지표(Vol, Stress, Speed) 바가 출력되어 자가 진단을 돕습니다.
   - 키보드 숫자 키(1~5)를 통해 영상과 음성의 **재생 속도 (0.5x ~ 2.0x)**를 실시간 변경할 수 있습니다. 이 때, 오디오의 피치 변형 최소화를 위해 `pydub` 라이브러리로 배속 오디오를 미리 생성해 두고 사용합니다.

4. **`KARAOKE_TEST` (가라오케 실전 테스트 모드)**
   - 연습 모드와 레이아웃 구조는 동일하지만, 화면에 표출되던 실시간 피드백 지표나 오디오 속도 조절 힌트 등을 전부 숨기고 오직 "TEST IN PROGRESS..." 문구를 띄워 실전과 같은 부담감 있는 훈련 환경을 유도합니다.
   - 테스트 동안 사용자의 매 프레임 자세 유사도, 핵심 제스처 분석 로그결과, 그리고 내부 음성 데이터가 백그라운드에 버퍼링(배열에 append)됩니다.
   - 지정된 영상 재생이 끝나면 자동으로 점수 산출을 위해 `TEST_RESULTS` 모드로 넘어갑니다.

5. **`TEST_RESULTS` (결과 종합 화면)**
   - 테스트가 끝나면 측정된 각종 수치와 로그 데이터를 취합하여 최종 결과를 보여줍니다.
   - (자세 정확도 50% + 스피치 음성 정확도 50% 비중)으로 된 100점 만점 평가 점수가 크게 나옵니다.
   - 세부 음성 평가 항목(Accuracy, Fluency, Pronunciation)과, 실전 도중 `KeyPoseExtractor`가 잡아낸 제스처 타이밍/정확도에 대한 구체적인 피드백 텍스트 로그가 리스트로 나타납니다.

## 4. 핵심 동작 흐름 구조 (Execution Flow)

1. **초기화 ([__init__](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#46-102))**: 화면 크기를 세팅하고, `PoseAnalyzer`, `KaraokeTrainer` 등의 백엔드 모듈 연동 객체를 생성합니다. 그리고 UI 및 오디오 믹서(Pygame)를 초기화합니다.
2. **동기화 및 캐싱 ([load_karaoke_video](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#132-208))**: 레퍼런스 대상 비디오, 랜드마크 분석 정보 JSON, 자막 파일을 메모리에 적재합니다. `pydub`로 배속별 오디오 변형 파일을 백그라운드에서 임시 생성(Caching)해놓아, 끊김 없이 속도를 바꿀 수 있도록 대비합니다. 원본 자세(Raw Pose Coordinates)역시 프레임 단위로 읽어 `.npy`로 추출하여 연산에 씁니다.
3. **메인 루프 ([run](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#260-306))**: `cv2` 윈도우 창이 띄워지고 루프가 돕니다. 웹캠 이미지를 캡처하여 좌우 반전 후 `MediaPipe Holistic` 추론을 거쳐 랜드마크를 추출합니다. 이후 [AppMode](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#27-33)에 따라 각기 다른 UI 렌더링 함수([_render_default_mode](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#323-363), [_render_karaoke_mode](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#390-513) 등)를 호출하여 이미지를 합성하고 사용자 화면에 보여줍니다.
4. **제스처 매칭 (SLIDING WINDOW 기법)**: `user_pose_buffer`에 사용자 포즈 30 프레임을 누적 기록해 두고, 레퍼런스의 원본 랜드마크 프레임 배열 중 현재 재생되고 있는 동시간 대의 프레임과 매치시켜 코사인 실시간 유사도를 연산해 화면 게이지에 적용합니다.

## 5. 분석 요약 및 향후 개선점

- **종합(Pros)**: 여태까지 고도화시켜온 개별적인 AI 분석 모듈들(포스트 추적, 스피치 오디오 분석, 시선 처리 평가 도구 등)을 UI상에 효과적으로 통합시킨 메인 프레임워크입니다. 쓰레드(Thread)를 활용한 지연 평가와 Pygame을 통한 오디오 믹싱 구조 등 완성도가 높습니다.
- **향후 과제(Cons / To-Do)**: 
  1. [load_karaoke_video](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#132-208) 메서드 등지에서 대상이 되는 파일 에셋 이름들(`Obama's 2004 DNC...`)이 직접 하드코딩 되어 있습니다. 추후 File Picker 대화상자 도입이나 Playlist UI를 통해 외부 파일 확장성을 높여야 합니다.
  2. 현재 [_render_test_results](file:///Users/parksungha/Desktop/DSL/CV_Modeling/test4.py#514-581) 내부에 스피치 오디오 최종 평가(`AudioEvaluator.evaluate`) 로직이 UI 멈춤 현상(Freezing)이나 연산 지연의 우려 때문인지 더미 데이터(Dummy)로 우회되어 있습니다. 이를 확실히 백그라운드 비동기 큐 처리 방식으로 고쳐 완전한 평가가 이루어지도록 연결 지어야 합니다.
