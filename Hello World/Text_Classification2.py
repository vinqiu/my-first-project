import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# 自定义停用词列表（包含常见英文停用词）
CUSTOM_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 
    'from', 'by', 'with', 'in', 'of', 'that', 'this', 'is', 'are', 'was', 
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'can', 'could', 'about', 'above', 'below', 'into',
    'through', 'during', 'before', 'after', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'don', 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

class SimplePreprocessor:
    def __init__(self):
        self.trans_table = str.maketrans('', '', "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    
    def preprocess(self, text):
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号
        text = text.translate(self.trans_table)
        
        # 简单分词并过滤停用词
        words = [word for word in text.split() if word not in CUSTOM_STOP_WORDS]
        
        return ' '.join(words)

def load_imdb_data(data_dir):
    """加载并预处理IMDB数据"""
    preprocessor = SimplePreprocessor()
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    text = f.read().replace('<br />', ' ')  # 处理换行标签
                    texts.append(preprocessor.preprocess(text))
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels

# 数据路径配置
base_dir = r"D:\rengongzhinengdaolun\Hello World\IMDB"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 加载数据
print("Loading and preprocessing data...")
X_train, y_train = load_imdb_data(train_dir)
X_test, y_test = load_imdb_data(test_dir)

# 构建处理管道
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 参数网格优化
params = {
    'tfidf__max_features': [8000, 12000],
    'tfidf__ngram_range': [(1,2), (1,3)],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['saga']
}

# 网格搜索
grid_search = GridSearchCV(pipeline, 
                          params,
                          cv=3,
                          n_jobs=-1,
                          verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# 评估最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nOptimized Evaluation:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 特征重要性分析
feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
coefs = best_model.named_steps['clf'].coef_.ravel()
top_positive = feature_names[coefs.argsort()[-10:][::-1]]
top_negative = feature_names[coefs.argsort()[:10]]

print("\nTop Positive Features:")
print(top_positive)

print("\nTop Negative Features:")
print(top_negative)