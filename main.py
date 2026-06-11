import torch
from torchvision import datasets, transforms

print("🚀 PyTorch起動！AIの準備を開始します...")

# 1. 画像をAIが理解できる「数字の塊（テンソル）」に変換するルールを設定
transform = transforms.ToTensor()

# 2. 世界中のAI研究者が使う「MNIST（手書きの数字画像）」データをダウンロード
print("📦 学習用の画像データを集めています（初回は少し時間がかかります）...")
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. 集めたデータの中身を確認する
print("\n--- 📊 準備完了！データセット情報 ---")
print(f"✅ 集まった画像の枚数: {len(train_data)}枚")

# 4. 試しに最初の1枚を取り出してみる
image, label = train_data[0]
print(f"✅ 1枚目の画像の正解データ（書かれている数字）: 【 {label} 】")
print(f"✅ 画像データのサイズ: {image.shape} （1色、縦28×横28ピクセル）")

print("\n🎉 Day 4クリア！AIが画像を読み込む準備が完全に整いました！")