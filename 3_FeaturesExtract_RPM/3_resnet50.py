import numpy as np
import pandas as pd
from pathlib import Path
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model
from keras.layers import GlobalAveragePooling2D
from keras.utils import load_img, img_to_array

SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_DIR   = SCRIPT_DIR / "RPM"
OUTPUT_DIR  = SCRIPT_DIR / "Result" / "ResNet50"
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
OUT_C = 2048   # GlobalAveragePooling 后的特征维数


def build_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    out = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=out)


def load_image(path: Path) -> np.ndarray:
    img = load_img(path, target_size=IMG_SIZE)
    x = img_to_array(img)
    return preprocess_input(x)


def extract_features(model, img_paths: list) -> np.ndarray:
    n = len(img_paths)
    features = np.zeros((n, OUT_C), dtype=np.float32)
    for start in range(0, n, BATCH_SIZE):
        batch_paths = img_paths[start:start + BATCH_SIZE]
        batch = np.stack([load_image(p) for p in batch_paths])
        feat = model.predict(batch, verbose=0)
        features[start:start + len(batch_paths)] = feat
        print(f"  进度: {min(start + BATCH_SIZE, n)}/{n}")
    return features


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(INPUT_DIR.rglob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"在 {INPUT_DIR} 中未找到 PNG 文件")
    print(f"共找到 {len(img_paths)} 张图片，使用 ResNet50 提取特征...")

    model = build_model()
    features = extract_features(model, img_paths)

    filenames = [p.stem for p in img_paths]
    df = pd.DataFrame(features)
    df.insert(0, 'filename', filenames)
    out_path = OUTPUT_DIR / "resnet50_features.xlsx"
    df.to_excel(out_path, index=False)
    print(f"特征已保存: {out_path}")
    print(f"特征矩阵大小: {features.shape}")
