"""
Presentation Coach — Real-time Detection Pilot
================================================
VS Code 터미널에서 바로 실행:  python realtime_coach.py
종료: q키 또는 ESC
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ═══════════════════════════════════════════════
# MediaPipe 초기화
# ═══════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ═══════════════════════════════════════════════
# 랜드마크 인덱스
# ═══════════════════════════════════════════════
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
UPPER_LIP, LOWER_LIP = 13, 14
LIP_LEFT, LIP_RIGHT  = 61, 291

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109,10]

# ═══════════════════════════════════════════════
# 색상
# ═══════════════════════════════════════════════
GREEN  = (72, 222, 128)
BLUE   = (250, 165, 96)
PURPLE = (250, 139, 167)
RED    = (113, 135, 248)
ORANGE = (60, 146, 251)
WHITE  = (230, 230, 230)
DARK   = (20, 20, 28)
GRAY   = (100, 100, 120)


# ═══════════════════════════════════════════════
# 메트릭 계산 함수
# ═══════════════════════════════════════════════
def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def calc_ear(landmarks, eye_idx):
    """Eye Aspect Ratio"""
    p = [landmarks[i] for i in eye_idx]
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h  = dist(p[0], p[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0

def calc_mar(landmarks):
    """Mouth Aspect Ratio"""
    top    = landmarks[UPPER_LIP]
    bottom = landmarks[LOWER_LIP]
    left   = landmarks[LIP_LEFT]
    right  = landmarks[LIP_RIGHT]
    v = dist(top, bottom)
    h = dist(left, right)
    return v / h if h > 0 else 0

def calc_head_pose(landmarks):
    """간이 head pose (pitch, yaw)"""
    nose       = landmarks[1]
    chin       = landmarks[152]
    forehead   = landmarks[10]
    left_cheek = landmarks[234]
    right_cheek= landmarks[454]

    cheek_mid_x = (left_cheek.x + right_cheek.x) / 2
    yaw = (nose.x - cheek_mid_x) * 200

    face_mid_y = (forehead.y + chin.y) / 2
    pitch = (nose.y - face_mid_y) * 200

    return pitch, yaw

def calc_eye_contact(landmarks, pitch, yaw):
    """시선 점수 (0~100)"""
    score = 100
    score -= min(abs(yaw) * 3, 60)
    score -= min(abs(pitch) * 2, 40)

    # iris 기반 보정 (478pt 모델)
    if len(landmarks) > 472:
        iris = landmarks[468]
        eye_l = landmarks[362]
        eye_r = landmarks[263]
        ew = dist(eye_l, eye_r)
        if ew > 0:
            ratio = (iris.x - eye_l.x) / ew
            score -= abs(ratio - 0.5) * 100

    return max(0, min(100, score))

def calc_expression(mar):
    """표정 점수"""
    if mar < 0.02: return 55    # 무표정
    if mar < 0.08: return 85    # 미소
    if mar < 0.15: return 70    # 약간 벌림
    return 40                    # 과도

def calc_posture(pose_landmarks):
    """자세 점수 + 어깨 기울기"""
    if pose_landmarks is None:
        return 75, 0

    ls = pose_landmarks.landmark[11]
    rs = pose_landmarks.landmark[12]

    if ls.visibility < 0.5 or rs.visibility < 0.5:
        return 75, 0

    dy = rs.y - ls.y
    dx = rs.x - ls.x
    tilt = abs(np.degrees(np.arctan2(dy, dx)))
    score = max(0, min(100, 100 - tilt * 6))
    return score, tilt


# ═══════════════════════════════════════════════
# 버퍼 & 스무딩
# ═══════════════════════════════════════════════
class SmoothBuffer:
    def __init__(self, size=15):
        self.buf = deque(maxlen=size)
    def push(self, val):
        self.buf.append(val)
    def avg(self):
        return sum(self.buf) / len(self.buf) if self.buf else 0
    def std(self):
        if len(self.buf) < 2: return 0
        m = self.avg()
        return np.sqrt(sum((v - m)**2 for v in self.buf) / len(self.buf))

eye_buf    = SmoothBuffer()
expr_buf   = SmoothBuffer()
post_buf   = SmoothBuffer()
head_buf   = SmoothBuffer()
yaw_hist   = SmoothBuffer(30)
pitch_hist = SmoothBuffer(30)


# ═══════════════════════════════════════════════
# 그리기 함수
# ═══════════════════════════════════════════════
def draw_bar(frame, x, y, w, h, score, label, color):
    """점수 바 그리기"""
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)

    # Fill
    fill_w = int(w * score / 100)
    bar_color = GREEN if score >= 75 else ORANGE if score >= 50 else RED
    cv2.rectangle(frame, (x, y + 2), (x + fill_w, y + h - 2), bar_color, -1)

    # Label
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # Value
    val_text = f"{int(score)}%"
    cv2.putText(frame, val_text, (x + w + 8, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1, cv2.LINE_AA)

def draw_detail(frame, x, y, label, value):
    """상세 수치 표시"""
    cv2.putText(frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 75, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

def draw_face_mesh_custom(frame, landmarks, w, h):
    """얼굴 윤곽 + 눈 + 홍채 그리기"""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(len(landmarks))]

    # 얼굴 윤곽
    for i in range(len(FACE_OVAL) - 1):
        cv2.line(frame, pts[FACE_OVAL[i]], pts[FACE_OVAL[i+1]],
                 (GREEN[0]//3, GREEN[1]//3, GREEN[2]//3), 1, cv2.LINE_AA)

    # 눈
    for eye in [LEFT_EYE, RIGHT_EYE]:
        for i in range(len(eye)):
            cv2.line(frame, pts[eye[i]], pts[eye[(i+1) % len(eye)]],
                     BLUE, 1, cv2.LINE_AA)

    # 홍채
    if len(landmarks) > 472:
        for idx in [468, 473]:
            cv2.circle(frame, pts[idx], 3, BLUE, -1, cv2.LINE_AA)

    # 코
    cv2.circle(frame, pts[1], 3, GREEN, -1, cv2.LINE_AA)

def draw_pose_custom(frame, pose_landmarks, w, h):
    """상체 포즈 그리기"""
    if pose_landmarks is None:
        return

    lm = pose_landmarks.landmark
    connections = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24)]

    for a, b in connections:
        if lm[a].visibility < 0.5 or lm[b].visibility < 0.5:
            continue
        p1 = (int(lm[a].x * w), int(lm[a].y * h))
        p2 = (int(lm[b].x * w), int(lm[b].y * h))
        cv2.line(frame, p1, p2, (PURPLE[0]//2, PURPLE[1]//2, PURPLE[2]//2), 2, cv2.LINE_AA)

    for i in range(11, 17):
        if lm[i].visibility < 0.5:
            continue
        pt = (int(lm[i].x * w), int(lm[i].y * h))
        cv2.circle(frame, pt, 4, PURPLE, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════
# Nudge 시스템
# ═══════════════════════════════════════════════
class NudgeSystem:
    def __init__(self, cooldown=8.0, display_time=4.0):
        self.cooldown = cooldown
        self.display_time = display_time
        self.last_time = 0
        self.current_msg = ""
        self.current_level = ""
        self.show_until = 0

    def trigger(self, msg, level="warn"):
        now = time.time()
        if now - self.last_time < self.cooldown:
            return
        self.last_time = now
        self.current_msg = msg
        self.current_level = level
        self.show_until = now + self.display_time

    def draw(self, frame, w):
        if time.time() > self.show_until:
            return

        color = GREEN if self.current_level == "good" else \
                ORANGE if self.current_level == "warn" else RED

        text_size = cv2.getTextSize(self.current_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = 40

        # Background pill
        pad = 14
        cv2.rectangle(frame, (tx - pad, ty - text_size[1] - pad),
                       (tx + text_size[0] + pad, ty + pad),
                       DARK, -1)
        cv2.rectangle(frame, (tx - pad, ty - text_size[1] - pad),
                       (tx + text_size[0] + pad, ty + pad),
                       color, 1)
        cv2.putText(frame, self.current_msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

nudge = NudgeSystem()


# ═══════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,    # iris 포함 478pt
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_model = mp_pose.Pose(
        model_complexity=0,        # lite
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = time.time()
    frame_idx = 0
    pose_landmarks = None

    print("=" * 50)
    print("  Presentation Coach — Pilot Test")
    print("  종료: q 또는 ESC")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # Face Mesh
        face_results = face_mesh.process(rgb)

        # Pose (매 2프레임마다)
        if frame_idx % 2 == 0:
            pose_results = pose_model.process(rgb)
            pose_landmarks = pose_results.pose_landmarks

        rgb.flags.writeable = True
        frame_idx += 1

        # ── 메트릭 계산 ──
        if face_results.multi_face_landmarks:
            fl = face_results.multi_face_landmarks[0].landmark

            ear_l = calc_ear(fl, LEFT_EYE)
            ear_r = calc_ear(fl, RIGHT_EYE)
            mar   = calc_mar(fl)
            pitch, yaw = calc_head_pose(fl)
            eye_score  = calc_eye_contact(fl, pitch, yaw)
            expr_score = calc_expression(mar)
            post_score, shoulder_tilt = calc_posture(pose_landmarks)

            # 버퍼 스무딩
            eye_buf.push(eye_score)
            expr_buf.push(expr_score)
            post_buf.push(post_score)
            yaw_hist.push(yaw)
            pitch_hist.push(pitch)

            head_stability = max(0, min(100, 100 - (yaw_hist.std() + pitch_hist.std()) * 3))
            head_buf.push(head_stability)

            s_eye  = eye_buf.avg()
            s_expr = expr_buf.avg()
            s_post = post_buf.avg()
            s_head = head_buf.avg()

            # ── 그리기: 랜드마크 ──
            draw_face_mesh_custom(frame, fl, w, h)
            draw_pose_custom(frame, pose_landmarks, w, h)

            # ── 그리기: 사이드 패널 ──
            panel_x = w - 260
            panel_y = 20
            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x - 15, panel_y - 15),
                          (w - 10, panel_y + 250), DARK, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            bar_w = 160
            draw_bar(frame, panel_x, panel_y + 20,  bar_w, 16, s_eye,  "Eye Contact", GREEN)
            draw_bar(frame, panel_x, panel_y + 65,  bar_w, 16, s_expr, "Expression",  GREEN)
            draw_bar(frame, panel_x, panel_y + 110, bar_w, 16, s_post, "Posture",     GREEN)
            draw_bar(frame, panel_x, panel_y + 155, bar_w, 16, s_head, "Head Stable", GREEN)

            # 상세 수치
            dy = panel_y + 200
            draw_detail(frame, panel_x, dy,      "Pitch",   f"{pitch:.1f}")
            draw_detail(frame, panel_x + 120, dy, "Yaw",    f"{yaw:.1f}")
            draw_detail(frame, panel_x, dy + 20,  "EAR L",  f"{ear_l:.3f}")
            draw_detail(frame, panel_x + 120, dy + 20, "EAR R", f"{ear_r:.3f}")

            # ── Nudge ──
            if s_eye < 40:
                nudge.trigger("Camera! Look here", "bad")
            elif s_post < 40:
                nudge.trigger("Straighten shoulders", "warn")
            elif s_head < 35:
                nudge.trigger("Keep head steady", "warn")
            elif s_eye > 80 and s_expr > 70 and s_post > 70:
                nudge.trigger("Great job! Keep it up!", "good")

        else:
            # 얼굴 미감지
            cv2.putText(frame, "No face detected", (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
            nudge.trigger("Face not detected!", "bad")

        # ── FPS ──
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        cv2.putText(frame, f"{fps:.0f} fps", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)

        # ── Nudge 표시 ──
        nudge.draw(frame, w)

        # ── 화면 출력 ──
        cv2.imshow("Presentation Coach - Pilot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    face_mesh.close()
    pose_model.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 종료되었습니다.")


if __name__ == "__main__":
    main()