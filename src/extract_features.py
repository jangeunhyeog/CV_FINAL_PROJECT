"""M0/M1: 사전학습 VPR 모델 로딩 + global descriptor 추출 → .npy 캐시.

사용 예:
    python src/extract_features.py --images data/sample --out cache/sample.npy --model eigenplaces
    python src/extract_features.py --images data/nordland/database --out cache/nordland_db.npy

모델은 모두 frozen(추론만). descriptor는 L2 정규화되어 저장된다(cosine = 내적).
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_model(name: str, device: str):
    """frozen VPR 모델 로딩. README의 최신 torch.hub API 기준(2025)."""
    # EigenPlaces가 내부에서 gmberton/cosplace를 중첩 torch.hub 호출 →
    # 비대화형 환경에서 trust 프롬프트로 멈춤. 명시적으로 신뢰 처리.
    torch.hub._check_repo_is_trusted = lambda *a, **k: None
    name = name.lower()
    if name == "eigenplaces":
        # ICCV 2023, ResNet50, 2048-dim. torch.hub 한 줄 로딩.
        model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model",
            backbone="ResNet50", fc_output_dim=2048, trust_repo=True,
        )
    elif name == "salad":
        # CVPR 2024, DINOv2 기반. 로딩 API가 바뀌면 여기만 수정.
        model = torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)
    else:
        raise ValueError(f"unknown model: {name}")
    return model.eval().to(device)


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def list_images(path: str):
    if os.path.isdir(path):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True)
        return sorted(files)
    # 단일 파일 또는 glob 패턴
    return sorted(glob.glob(path))


@torch.no_grad()
def extract(model, files, tf, device, batch_size: int):
    descs = []
    batch, valid = [], []
    for f in tqdm(files, desc="extract"):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        batch.append(tf(img))
        valid.append(f)
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            d = model(x)
            descs.append(torch.nn.functional.normalize(d, dim=1).cpu())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        d = model(x)
        descs.append(torch.nn.functional.normalize(d, dim=1).cpu())
    return torch.cat(descs).numpy().astype(np.float32), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="이미지 폴더 / glob 패턴")
    ap.add_argument("--out", required=True, help="출력 .npy 경로")
    ap.add_argument("--model", default="eigenplaces")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = list_images(args.images)
    if not files:
        raise SystemExit(f"이미지를 찾지 못함: {args.images}")
    print(f"{len(files)} images | model={args.model} | device={args.device}")

    model = get_model(args.model, args.device)
    tf = build_transform(args.img_size)
    descs, valid = extract(model, files, tf, args.device, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, descs)
    # 파일명 순서도 함께 저장(평가 시 정답 매칭에 필요)
    with open(args.out + ".paths.txt", "w") as fp:
        fp.write("\n".join(valid))
    print(f"saved {descs.shape} → {args.out}")


if __name__ == "__main__":
    main()
