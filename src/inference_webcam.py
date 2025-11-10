import argparse, json, cv2, numpy as np, time
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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

    labels = json.load(open(args.labels))
    model = load_model(args.model)

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
        fps = 0
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
                x1, y1 = max(min(xs) - 30, 0), max(min(ys) - 30, 0)
                x2, y2 = min(max(xs) + 30, w), min(max(ys) + 30, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    inp = preprocess(crop, args.img_size)
                    preds = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
                    i = int(np.argmax(preds))
                    conf = float(preds[i])
                    display_text = f"{labels[i]} ({conf*100:.1f}%)"

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
