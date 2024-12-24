import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import torch
from torchvision import models, transforms
from PIL import Image

# 读取数据
df = pd.read_csv('train.csv')

# 合并相关列形成新的文本特征
df['text'] = df['Ofiicial Account Name'] + ' ' + df['Title'] + ' ' + df['Report Content']
df['text'] = df['text'].fillna('')

# 分割数据
X = df['text']
y = df['label']

# 划分训练集和测试集
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 文本向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train_text_vec = vectorizer.fit_transform(X_train_text)
X_test_text_vec = vectorizer.transform(X_test_text)

from torchvision.models import ResNet50_Weights

# 图片特征提取：使用预训练的ResNet50模型（PyTorch）
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.eval()  # 设为评估模式，避免训练中的 Dropout 等

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 定义函数提取图片特征
def extract_image_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        features = resnet_model(img)
    return features.flatten().numpy()


# 提取训练和测试集图片特征
def get_image_features(df, img_folder):
    image_features = []
    for idx, row in df.iterrows():
        img_path = os.path.join(img_folder, f"{row['id']}.png")
        if os.path.exists(img_path):
            image_features.append(extract_image_features(img_path))
        else:
            image_features.append(np.zeros((1000,)))  # 用零向量填充
    return np.array(image_features)


# 读取训练和测试集图片特征
train_image_folder = './train_x/image'
X_train_image_vec = get_image_features(df[df['id'].isin(X_train_text.index)], train_image_folder)
X_test_image_vec = get_image_features(df[df['id'].isin(X_test_text.index)], train_image_folder)

# 合并文本和图片特征
X_train_combined = np.hstack([X_train_text_vec.toarray(), X_train_image_vec])
X_test_combined = np.hstack([X_test_text_vec.toarray(), X_test_image_vec])

from sklearn.preprocessing import StandardScaler

# 标准化数据
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_test_combined = scaler.transform(X_test_combined)

# 使用 LogisticRegression，设置 max_iter 和 solver
model = LogisticRegression(max_iter=3000, solver='saga', C=0.1)
model.fit(X_train_combined, y_train)

# 保存模型和向量器
joblib.dump(model, './model.pkl')
joblib.dump(vectorizer, './vectorizer.pkl')

# 预测与评估
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))
