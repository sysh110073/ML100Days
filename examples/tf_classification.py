"""TensorFlow/Keras：離線資料 → tf.data → fit → 驗證 → .keras 最佳模型。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf

from paired_data import make_binary_data


def main() -> None:
    tf.keras.utils.set_random_seed(42)
    x_train, y_train, x_valid, y_valid = make_binary_data()
    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
             .shuffle(len(x_train), seed=42).batch(64).prefetch(tf.data.AUTOTUNE))
    valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(64)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2),  # 原始 logits，不先 softmax
    ])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "best.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        ]
        model.fit(train, validation_data=valid, epochs=30, callbacks=callbacks, shuffle=False, verbose=0)
        best = tf.keras.models.load_model(checkpoint)
        loss, accuracy = best.evaluate(valid, verbose=0)
        print(f"TensorFlow best val_loss={loss:.4f} val_accuracy={accuracy:.3f}")
        assert accuracy > 0.9


if __name__ == "__main__":
    main()
