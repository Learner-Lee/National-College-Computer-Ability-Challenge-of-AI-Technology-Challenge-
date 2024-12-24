# encoding = utf-8
# 开发者：Alen
# 开发时间： 18:02 
# "Stay hungry，stay foolish."

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import joblib
from scipy.sparse import hstack
from torchvision import models, transforms
import torch
from PIL import Image


# 图像特征提取类
class ImageFeatureExtractor:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.eval()

    def extract_features(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():
            features = self.model(img_tensor).numpy()
        return features


# 读取数据
df = pd.read_csv('train.csv')

df['Title'] = df['Title'].fillna('')
df['Report Content'] = df['Report Content'].fillna('')
df['Ofiicial Account Name'] = df['Ofiicial Account Name'].fillna('')

# 定义读取 HTML 文件的函数
def read_html_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ''  # 如果文件不存在，返回空字符串

# 将 HTML 内容读取到 DataFrame 中
df['HTML Content'] = df['id'].apply(lambda x: read_html_content(f'train_x/html/{x}.html'))


# 分别对每一列进行向量化
vectorizer_title = TfidfVectorizer(max_features=2500)
vectorizer_report = TfidfVectorizer(max_features=2500)
vectorizer_Ofiicial = TfidfVectorizer(max_features=2500)
vectorizer_html = TfidfVectorizer(max_features=2500)  # 新增用于 HTML 内容的向量化器

X_title_vec = vectorizer_title.fit_transform(df['Title'])
X_report_vec = vectorizer_report.fit_transform(df['Report Content'])
X_Ofiicial_vec = vectorizer_Ofiicial.fit_transform(df['Ofiicial Account Name'])
X_html_vec = vectorizer_html.fit_transform(df['HTML Content'])  # 向量化 HTML 内容


# 初始化图像特征提取器
image_feature_extractor = ImageFeatureExtractor()

# 提取图像特征
image_features = np.array([image_feature_extractor.extract_features(f'train_x/image/{id}.png') for id in df['id']]).reshape(len(df), -1)

# 将两个向量化的结果水平拼接起来
X_text = hstack([X_title_vec, X_report_vec, X_Ofiicial_vec, X_html_vec])
X = np.hstack([X_text.toarray(), image_features])  # 合并稀疏文本特征和图像特征

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# 训练分类器
model = LogisticRegression(max_iter=3000, solver='saga', C=0.1)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, './model.pkl')

# 保存向量化器
joblib.dump(vectorizer_title, './vectorizer_title.pkl')
joblib.dump(vectorizer_report, './vectorizer_report.pkl')
joblib.dump(vectorizer_Ofiicial, './vectorizer_Ofiicial.pkl')

# HTML
joblib.dump(vectorizer_html, './vectorizer_html.pkl')

# 保存图像特征提取器
joblib.dump(image_feature_extractor, './image_feature_extractor.pkl')

# 预测与评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
