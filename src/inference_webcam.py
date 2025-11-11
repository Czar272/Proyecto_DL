import argparse, json, cv2, numpy as np, time, os
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CONF_THRESH = 0.70
SMOOTH_WINDOW = 6
PADDING = 60
MIN_SIDE = 96

probs_buffer = deque(maxlen=SMOOTH_WINDOW)


def preprocess(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    labels_used_path = "results/labels_used.json"
    if os.path.exists(labels_used_path):
        labels = json.load(open(labels_used_path, "r", encoding="utf-8"))
    else:
        labels = json.load(open(args.labels, "r", encoding="utf-8"))

    try:
        true_div = tf.__operators__.truediv
    except AttributeError:
        true_div = tf.math.divide
    customs = {"TrueDivide": true_div, "Divide": true_div}

    ext = os.path.splitext(args.model)[1].lower()
    if ext == ".h5":
        model = load_model(
            args.model, compile=False, custom_objects=customs, safe_mode=False
        )
    else:
        model = load_model(args.model, compile=False)

    if model.output_shape[-1] != len(labels):
        print(
            f"[WARN] Clases del modelo ({model.output_shape[-1]}) != len(labels) ({len(labels)}). "
            f"Revisa labels_used.json y el modelo."
        )
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        prev = time.time()
        fps = 0.0
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w, _ = frame.shape
            display_text = "No hand"
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x1, y1 = max(min(xs) - PADDING, 0), max(min(ys) - PADDING, 0)
                x2, y2 = min(max(xs) + PADDING, w), min(max(ys) + PADDING, h)

                if (x2 - x1) >= MIN_SIDE and (y2 - y1) >= MIN_SIDE:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        inp = preprocess(crop, args.img_size)
                        preds = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
                        probs_buffer.append(preds)
                        avg_probs = (
                            np.mean(probs_buffer, axis=0)
                            if len(probs_buffer)
                            else preds
                        )
                        i = int(np.argmax(avg_probs))
                        conf = float(avg_probs[i])

                        if conf < CONF_THRESH:
                            display_text = f"unknown ({conf*100:.1f}%)"
                        else:
                            display_text = f"{labels[i]} ({conf*100:.1f}%)"
                else:
                    display_text = "hand too small"

                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if now > prev else fps
            prev = now

            cv2.putText(
                frame,
                display_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Gesture Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
