import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("🧠 AIの脳みそ（ニューラルネットワーク）を構築中...")

# 1. データの準備（昨日の続き）
transform = transforms.ToTensor()
# 今回はすでにダウンロード済みなので download=False でOK
train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
# 6万枚の画像を「64枚ずつの束（バッチ）」に小分けする
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. AIの脳みその設計図（モデル）
class AIBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # 入力: 縦28×横28=784個のピクセルデータ
        # 隠れ層: 128個の仮想神経細胞（ニューロン）で考える
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU() # 脳を柔軟にする魔法
        # 出力層: 最終的に0〜9の「10個」の確率を出す
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 画像をAIが食べやすい「1列の細長いデータ」に引き伸ばす
        x = x.view(-1, 28 * 28) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 脳みそを実体化
model = AIBrain()

# 3. 採点係とコーチの配置
criterion = nn.CrossEntropyLoss() # 予想と正解の「ズレ（誤差）」を計算する係
optimizer = optim.Adam(model.parameters(), lr=0.001) # 誤差をもとに脳をアップデートするコーチ

print("🔥 猛特訓（学習）スタート！(PCの性能によりますが、1〜2分かかります)")

# 4. 学習ループ（今回は6万枚のデータを1周だけさせます）
epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    # 64枚の画像束を1つずつ取り出して特訓
    for i, (images, labels) in enumerate(train_loader):
        # ① AIの思考（前回の記憶）をリセット
        optimizer.zero_grad()
        
        # ② AIに画像を見せて、何の数字か予想させる（順伝播）
        outputs = model(images)
        
        # ③ 予想と本当の答えを比べて、どれくらい間違えたか計算
        loss = criterion(outputs, labels)
        
        # ④ 間違いの原因を分析して、脳内ネットワークを修正（逆伝播：ここがAIの学習の核！）
        loss.backward()
        optimizer.step() # 脳をアップデート
        
        running_loss += loss.item()
        
        # 300束（19,200枚）学習するごとに進捗を報告
        if i % 300 == 299:
            print(f"  [{epoch + 1}周目, {i + 1}バッチ] 誤差(Loss): {running_loss / 300:.3f}")
            running_loss = 0.0

print("\n🎉 特訓完了！AIが手書き数字を判別できる脳みそを手に入れました！")