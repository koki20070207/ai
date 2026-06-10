import pandas as pd
import random

print("🚀 100人分の架空データを生成中...")

# 1. 100人分のランダムなデータ（名前、年齢、言語、点数）を自動で作る
names = [f"候補生_{i}号" for i in range(1, 101)]
ages = [random.randint(18, 35) for _ in range(100)]
languages = [random.choice(['Python', 'JavaScript', 'Go', 'Ruby', 'Java']) for _ in range(100)]
scores = [random.randint(30, 100) for _ in range(100)]

data = {
    '名前': names,
    '年齢': ages,
    'プログラミング歴': languages,
    'AIテストの点数': scores
}

# データをPandasの表（データフレーム）に変換
df = pd.DataFrame(data)

# 2. データをCSVファイルとしてPCに保存する（文字化け防止のおまじない付き）
csv_filename = "ai_students_100.csv"
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"✅ 【成功】 {csv_filename} をフォルダ内に作成しました！\n")

# 3. 保存したCSVをPythonに読み込む
loaded_df = pd.read_csv(csv_filename)

# 4. 100人の点数データを一瞬で要約する（神コマンド）
print("--- 📊 100人のAIテスト点数まとめ ---")
print(loaded_df['AIテストの点数'].describe())
print("\n")

# 5. 応用：Pythonが書けて、かつ80点以上の「即戦力エリート」だけを炙り出す
elite_df = loaded_df[(loaded_df['プログラミング歴'] == 'Python') & (loaded_df['AIテストの点数'] >= 80)]

print(f"--- 🌟 Pythonが書ける80点以上の即戦力エリート（{len(elite_df)}人） ---")
print(elite_df)