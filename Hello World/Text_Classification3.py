import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import os

# 配置参数
DATA_PATH = r"D:\rengongzhinengdaolun\Hello World\IMDB"
MAX_FEATURES = 10000  # 增加特征维度提升信息量
N_JOBS = 7           # 利用8核CPU的87.5%资源
C_VALUE = 3.5        # 精细调整的正则化参数
NGRAM_RANGE = (1, 3) # 扩展至三元语法

def load_imdb_data(path):
    """加载完整5万条数据集"""
    reviews = []
    labels = []
    
    # 加载负面评价
    neg_path = os.path.join(path, "neg")
    for file in os.listdir(neg_path)[:25000]:  # 加载全部2.5万样本
        with open(os.path.join(neg_path, file), 'r', errors='ignore') as f:
            reviews.append(f.read())
            labels.append(0)
    
    # 加载正面评价
    pos_path = os.path.join(path, "pos")
    for file in os.listdir(pos_path)[:25000]:
        with open(os.path.join(pos_path, file), 'r', errors='ignore') as f:
            reviews.append(f.read())
            labels.append(1)
    
    return reviews, labels

# 数据加载
print("▌ 正在加载完整数据集...")
texts, y = load_imdb_data(DATA_PATH)
print("✓ 已加载50,000条影评数据")

# 创建优化处理管道
pipeline = make_pipeline(
    TfidfVectorizer(max_features=MAX_FEATURES,
                   ngram_range=NGRAM_RANGE,
                   stop_words='english',
                   sublinear_tf=True,
                   min_df=3,         # 过滤低频词
                   max_df=0.85),     # 过滤高频常见词
    LogisticRegression(C=C_VALUE,
                      solver='saga',
                      max_iter=300,   # 增加迭代次数
                      tol=1e-4,       # 更严格的收敛阈值
                      n_jobs=N_JOBS,
                      random_state=42,
                      class_weight='balanced')  # 处理类别不平衡
)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, 
    test_size=0.2, 
    stratify=y,      # 保持类别分布
    random_state=42
)

# 模型训练
print("▌ 开始模型训练（预计1-2分钟）...")
pipeline.fit(X_train, y_train)
print("✓ 训练完成")

# 模型评估
print("▌ 开始模型评估")
y_pred = pipeline.predict(X_test)
print(f"测试集准确率：{accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=4))

# 特征分析示例
feature_names = pipeline.named_steps['tfidfvectorizer'].get_feature_names_out()
coefs = pipeline.named_steps['logisticregression'].coef_[0]
print("\n最具判别力的10个特征：")
print("正向词:", [feature_names[i] for i in np.argsort(coefs)[-10:]])
print("负向词:", [feature_names[i] for i in np.argsort(coefs)[:10]])