import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 1. 脳みその設計図
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

# 脳みそをロード
@st.cache_resource # 画面を動かすたびに再ロードしないためのStreamlitの魔法
def load_my_model():
    model = AIBrain()
    model.load_state_dict(torch.load("my_model.pth"))
    model.eval()
    return model

model = load_my_model()

# 2. Web画面のデザイン（HTMLやCSSを書かなくてもPythonだけで作れます！）
st.title("🧠 自作AIの画像認識Webアプリ")
st.write("あなたがDay 5で猛特訓させたAIの脳みそが、Web上で働いています。")

# テストデータを読み込んで、スライダーで画像を選べるようにする
transform = transforms.ToTensor()
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# 画面にスライダー（数字を選ぶバー）を設置
image_index = st.slider("確認したいテスト画像の番号を選んでください（0〜9999）", 0, 9999, 0)

# 選ばれた画像を取り出す
test_image, real_answer = test_data[image_index]

# 画面に画像を表示する（ちょっと拡大して見やすく）
# Streamlitは画像データを簡単に画面に表示できます
st.image(test_image.numpy()[0], caption=f"テスト画像 No.{image_index}", width=150)

# 3. ボタンを押したらAIが推論する
if st.button("AIに予想させる"):
    with torch.no_grad():
        prediction = model(test_image)
        predicted_number = prediction.argmax().item()
    
    # 結果を画面にオシャレに表示
    st.subheader(f"🤖 AIの予想: 【 {predicted_number} 】")
    st.write(f"🎯 本当の正解: 【 {real_answer} 】")
    
    if predicted_number == real_answer:
        st.success("🎉 見事、大正解です！")
    else:
        st.error("💦 ハズレ！修行が足りんようです。")