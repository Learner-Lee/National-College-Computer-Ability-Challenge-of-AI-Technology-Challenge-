# encoding = utf-8
# 开发者：Alen
# 开发时间： 18:02 
# "Stay hungry，stay foolish."

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本数据转换为数值型特征向量
from sklearn.linear_model import LogisticRegression  # LogisticRegression类，作为分类模型
from sklearn.metrics import classification_report  # classification_report函数，用于输出模型预测结果的性能报告。
from scipy.sparse import hstack  # 水平拼接稀疏矩阵
import joblib  # 保存训练好的模型和向量化器

# 读取数据
df = pd.read_csv('train.csv')

df['Title'] = df['Title'].fillna('')
df['Report Content'] = df['Report Content'].fillna('')
df['Ofiicial Account Name'] = df['Ofiicial Account Name'].fillna('')

# 分别对每一列进行向量化
vectorizer_title = TfidfVectorizer(max_features=2500)
vectorizer_report = TfidfVectorizer(max_features=2500)
vectorizer_Ofiicial = TfidfVectorizer(max_features=2500)

X_title_vec = vectorizer_title.fit_transform(df['Title'])
X_report_vec = vectorizer_report.fit_transform(df['Report Content'])
X_Ofiicial_vec = vectorizer_Ofiicial.fit_transform(df['Ofiicial Account Name'])

# 将两个向量化的结果水平拼接起来
X = hstack([X_title_vec, X_report_vec, X_Ofiicial_vec])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# 训练分类器
model = LogisticRegression()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, './model.pkl')

# 保存向量化器
joblib.dump(vectorizer_title, './vectorizer_title.pkl')
joblib.dump(vectorizer_report, './vectorizer_report.pkl')
joblib.dump(vectorizer_Ofiicial, './vectorizer_Ofiicial.pkl')

# 预测与评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))