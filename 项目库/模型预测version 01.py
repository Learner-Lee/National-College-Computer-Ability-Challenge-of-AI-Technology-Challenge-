# encoding = utf-8
# 开发者：Alen
# 开发时间： 18:02 
# "Stay hungry，stay foolish."

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 读取数据
df = pd.read_csv('train.csv')

# 合并相关列形成新的特征
df['text'] = df['Ofiicial Account Name'] + ' ' + df['Title'] + ' ' + df['Report Content']

# 检查并处理缺失值，将 NaN 填充为空字符串
df['text'] = df['text'].fillna('')

# 分割数据
X = df['text']
y = df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 文本向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 训练分类器
model = LogisticRegression()
model.fit(X_train_vec, y_train)

import joblib

# 保存模型
joblib.dump(model, './model.pkl')

# 保存向量化器
joblib.dump(vectorizer, './vectorizer.pkl')

# 预测与评估
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))