import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_imdb_data(data_dir):
    """
    加载IMDB数据集
    :param data_dir: 数据集路径（包含pos/neg子目录）
    :return: 评论文本列表和对应标签列表
    """
    texts = []
    labels = []
    # 遍历正面和负面评论目录
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)  # 正面评论标记为1，负面为0
    return texts, labels

# 数据集路径配置
base_dir = r"D:\rengongzhinengdaolun\Hello World\IMDB"
train_dir = os.path.join(base_dir, 'train')  # 训练集路径
test_dir = os.path.join(base_dir, 'test')    # 测试集路径

# 加载训练集和测试集
print("Loading training data...")
X_train, y_train = load_imdb_data(train_dir)
print("Loading test data...")
X_test, y_test = load_imdb_data(test_dir)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# 文本特征提取（TF-IDF）
print("Vectorizing text data...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000,   # 保留最高频的5000个特征
                                   stop_words='english', # 移除英文停用词
                                   ngram_range=(1, 2))  # 使用1元和2元语法模型

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 训练逻辑回归模型
print("Training logistic regression model...")
logreg = LogisticRegression(penalty='l2',       # L2正则化
                            C=1.0,              # 正则化强度
                            solver='liblinear', # 适用于小数据集的求解器
                            max_iter=1000)      # 最大迭代次数

logreg.fit(X_train_tfidf, y_train)

# 预测和评估
print("Evaluating model...")
y_pred = logreg.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# 输出示例预测结果
print("\nSample Predictions:")
sample_texts = [X_test[0], X_test[1]]  # 取前两个测试样本
sample_pred = logreg.predict(tfidf_vectorizer.transform(sample_texts))
for text, pred in zip(sample_texts[:2], sample_pred[:2]):
    print(f"\nText snippet: {text[:100]}...")
    print(f"Predicted sentiment: {'Positive' if pred == 1 else 'Negative'}")