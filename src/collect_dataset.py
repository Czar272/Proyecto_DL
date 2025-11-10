import argparse, os, time, json, cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--crop_padding", type=int, default=30)
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    label_idx = 0
    active_label = labels[label_idx]
    recording = False

    for l in labels:
        ensure_dir(os.path.join(args.out, l))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        print(
            "Press SPACE to toggle recording, number keys (1-9) to change label, Q to quit."
        )
        last_save = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w, _ = frame.shape
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x1 = max(min(xs) - args.crop_padding, 0)
                y1 = max(min(ys) - args.crop_padding, 0)
                x2 = min(max(xs) + args.crop_padding, w)
                y2 = min(max(ys) + args.crop_padding, h)
                crop = frame[y1:y2, x1:x2].copy()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                if (
                    recording and (time.time() - last_save > 0.07) and crop.size > 0
                ):  # ~14 fps
                    ts = int(time.time() * 1000)
                    fn = os.path.join(args.out, active_label, f"{ts}.jpg")
                    cv2.imwrite(fn, crop)
                    last_save = time.time()

            status = (
                f"Label: {active_label} | Recording: {'ON' if recording else 'OFF'}"
            )
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:  # space
                recording = not recording
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(labels):
                    label_idx = idx
                    active_label = labels[label_idx]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
