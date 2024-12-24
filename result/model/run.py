import os
import sys
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import torch
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor

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
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0)
            with torch.no_grad():
                features = self.model(img_tensor).numpy()
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros((1, 1000))

# 并行提取图像特征
def extract_image_features(image_paths, image_feature_extractor, batch_size=32):
    features = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = list(executor.map(image_feature_extractor.extract_features, batch_paths))
            features.extend(batch_features)
    return np.vstack(features)

# 固定区域，不能修改
def main(to_pred_dir, result_save_path):
    # 获取当前脚本所在目录
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    # 获取待预测数据文件和文件夹路径
    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")
    testa_image_dir = os.path.join(to_pred_dir, "testa_x", "image")

    # 加载模型和向量化器
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    vectorizer_title = joblib.load(os.path.join(model_dir, 'vectorizer_title.pkl'))
    vectorizer_report = joblib.load(os.path.join(model_dir, 'vectorizer_report.pkl'))
    vectorizer_ofiicial = joblib.load(os.path.join(model_dir, 'vectorizer_Ofiicial.pkl'))
    vectorizer_html = joblib.load(os.path.join(model_dir, 'vectorizer_html.pkl'))
    image_feature_extractor = joblib.load(os.path.join(model_dir, 'image_feature_extractor.pkl'))

    # 读取待预测的CSV文件
    testa = pd.read_csv(testa_csv_path)
    testa['Title'] = testa['Title'].fillna('')
    testa['Report Content'] = testa['Report Content'].fillna('')
    testa['Ofiicial Account Name'] = testa['Ofiicial Account Name'].fillna('')

    # 提取 HTML 内容
    testa['HTML Content'] = testa['id'].apply(lambda x: open(os.path.join(testa_html_dir, f"{x}.html"), "r", encoding='utf-8').read().strip() if os.path.exists(os.path.join(testa_html_dir, f"{x}.html")) else '')

    # 向量化文本数据
    X_title_vec = vectorizer_title.transform(testa['Title'])
    X_report_vec = vectorizer_report.transform(testa['Report Content'])
    X_ofiicial_vec = vectorizer_ofiicial.transform(testa['Ofiicial Account Name'])
    X_html_vec = vectorizer_html.transform(testa['HTML Content'])

    # 提取图像特征
    image_paths = [os.path.join(testa_image_dir, f"{id}.png") for id in testa['id']]
    image_features = extract_image_features(image_paths, image_feature_extractor)

    # 合并文本和图像特征
    X_text = hstack([X_title_vec, X_report_vec, X_ofiicial_vec, X_html_vec])
    X_combined = np.hstack([X_text.toarray(), image_features])

    # 使用模型进行预测
    predictions = model.predict(X_combined)

    # 将预测结果保存到 CSV 文件
    testa['label'] = predictions
    result = testa[['id', 'label']]
    result.to_csv(result_save_path, index=None, encoding='utf-8')

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]
    result_save_path = sys.argv[2]
    main(to_pred_dir, result_save_path)