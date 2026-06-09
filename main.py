import pandas as pd

# 1. 擬似的なデータ（名簿）を作成
data = {
    '名前': ['佐藤', '鈴木', '高橋', '田中', '伊藤'],
    '年齢': [21, 20, 22, 19, 23],
    'プログラミング歴': ['Python', 'JavaScript', 'Python', 'Go', 'JavaScript'],
    'AIテストの点数': [85, 40, 92, 75, 60]
}

# 2. データをPandasの「データフレーム（表）」に変換
df = pd.DataFrame(data)

print("--- すべてのデータ ---")
print(df)
print("\n")

# 3. 【データ抽出】AIテストの点数が70点以上、かつプログラミング歴が Python の人だけを絞り込む
filtered_df = df[(df['AIテストの点数'] >= 70) & (df['プログラミング歴'] == 'Python')]

print("--- 条件に合う優秀なAI候補生 ---")
print(filtered_df)