import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib
import time

# 高级文本预处理
class AdvancedPreprocessor:
    def __init__(self):
        self.emotion_words = {
            'great', 'excellent', 'wonderful', 'awful', 'terrible', 'horrible'
        }
        self.negation_map = {
            "n't": " not", "'s": " is", "'re": " are", "'ve": " have",
            "'ll": " will", "'d": " would"
        }
    
    def preprocess(self, text):
        # 处理否定和缩写
        text = text.lower()
        for k, v in self.negation_map.items():
            text = text.replace(k, v)
        
        # 保留情感词和标点
        cleaned = []
        for word in text.split():
            # 处理重复字符（如：cooooool → cool）
            if len(word) > 2:
                word = ''.join([char for i, char in enumerate(word) 
                               if i == 0 or char != word[i-1]])
            
            # 保留情感词和重要标点
            if word in self.emotion_words or any(c in word for c in {'!', '?'}):
                cleaned.append(word)
            elif word.isalnum():
                cleaned.append(word)
        
        return ' '.join(cleaned)

# 混合特征工程
def create_hybrid_features(texts):
    # 词汇级特征
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=15000,
        min_df=3,
        max_df=0.7,
        sublinear_tf=True,
        stop_words='english'
    )
    word_features = word_vectorizer.fit_transform(texts)
    
    # 字符级特征
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=5000
    )
    char_features = char_vectorizer.fit_transform(texts)
    
    # 情感符号特征
    exclamation_counts = np.array([text.count('!') for text in texts]).reshape(-1, 1)
    question_counts = np.array([text.count('?') for text in texts]).reshape(-1, 1)
    
    return hstack([word_features, char_features, exclamation_counts, question_counts])

# 数据加载
def load_data(data_dir):
    preprocessor = AdvancedPreprocessor()
    texts, labels = [], []
    for label in ['pos', 'neg']:
        path = os.path.join(data_dir, label)
        for file in os.listdir(path):
            if file.endswith('.txt'):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    text = f.read().replace('<br />', ' ')
                    texts.append(preprocessor.preprocess(text))
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

# 配置路径
base_dir = r"D:\rengongzhinengdaolun\Hello World\IMDB"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 加载数据
print("加载预处理数据...")
start = time.time()
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)
print(f"数据加载耗时: {time.time()-start:.1f}s")

# 创建混合特征
print("\n生成混合特征...")
start = time.time()
X_train_feats = create_hybrid_features(X_train)
X_test_feats = create_hybrid_features(X_test)
print(f"特征生成耗时: {time.time()-start:.1f}s")

# 构建优化模型
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    class_weight='balanced',
    max_iter=1000,
    warm_start=True,
    n_jobs=-1
)

# 超参数搜索空间
param_dist = {
    'C': np.logspace(-2, 1, 20),
    'l1_ratio': np.linspace(0, 1, 11),
    'fit_intercept': [True, False]
}

# 随机搜索
search = RandomizedSearchCV(
    model,
    param_dist,
    n_iter=50,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\n开始超参数优化...")
start = time.time()
search.fit(X_train_feats, y_train)
print(f"参数搜索耗时: {time.time()-start:.1f}s")

# 最佳模型评估
best_model = search.best_estimator_
y_pred = best_model.predict(X_test_feats)

print("\n最佳参数:", search.best_params_)
print("最终评估结果:")
print(classification_report(y_test, y_pred))
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

# 保存模型
joblib.dump(best_model, 'imdb_lr_model.pkl')