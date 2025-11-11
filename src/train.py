import argparse, os, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools


def build_model(num_classes, img_size=224, base_trainable=False):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = base_trainable
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def plot_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument(
        "--fine_tune", action="store_true", help="Unfreeze top layers for fine-tuning"
    )
    ap.add_argument("--save_path", default="models/mobilenetv2.keras")
    args = ap.parse_args()

    class_names = json.load(open(args.labels, "r", encoding="utf-8"))
    num_classes = len(class_names)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )

    model, base = build_model(num_classes, args.img_size, base_trainable=False)

    inputs = keras.Input(shape=(args.img_size, args.img_size, 3))
    x = data_augmentation(inputs)
    outputs = model(x)
    model_aug = keras.Model(inputs, outputs)
    model_aug.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if not args.save_path.endswith(".keras"):
        args.save_path = os.path.splitext(args.save_path)[0] + ".keras"

    ckpt = keras.callbacks.ModelCheckpoint(
        args.save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    lr = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1)

    model_aug.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, lr, es]
    )

    if args.fine_tune:
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        model_aug.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model_aug.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(3, args.epochs // 3),
            callbacks=[ckpt, lr, es],
        )

    y_true, y_pred = [], []
    for imgs, labels_batch in val_ds:
        preds = model_aug.predict(imgs, verbose=0)
        y_true.extend(labels_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "results/confusion_matrix.png")
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)

    with open("results/labels_used.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model_aug.save(args.save_path)
    print("Saved model to", args.save_path)
    print(rep)


if __name__ == "__main__":
    main()
