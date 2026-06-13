import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 1. 脳みその設計図（器）を用意する
# ※記憶（セーブデータ）を流し込むための「空っぽの脳みそ」が先に必要です
class AIBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 空っぽの脳みそを実体化
model = AIBrain()

# 2. 魔法のコード：セーブデータ（記憶）をロードする！
model.load_state_dict(torch.load("my_model.pth"))
model.eval() # 本番モードに切り替え
print("✨ セーブデータの読み込み完了！（一瞬です）")

# 3. テスト用の画像を準備
transform = transforms.ToTensor()
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# 今回は試しに「7番目」の画像を取り出してみます（違う数字を試したければ数字を変えてOK）
test_image, real_answer = test_data[7]

# 4. 特訓なしでいきなり推論！
with torch.no_grad():
    prediction = model(test_image)
    predicted_number = prediction.argmax().item()

print("\n--- 📊 爆速・推論結果 ---")
print(f"✅ AIの予想: 【 {predicted_number} 】")
print(f"🎯 本当の正解: 【 {real_answer} 】")