import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def load_imdb_data(dataset_dir):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for label in ['pos', 'neg']:
        for filename in os.listdir(os.path.join(dataset_dir, 'train', label)):
            with open(os.path.join(dataset_dir, 'train', label, filename), 'r', encoding='utf-8') as f:
                train_data.append(f.read())
                train_labels.append(1 if label == 'pos' else 0)
    for label in ['pos', 'neg']:
        for filename in os.listdir(os.path.join(dataset_dir, 'test', label)):
            with open(os.path.join(dataset_dir, 'test', label, filename), 'r', encoding='utf-8') as f:
                test_data.append(f.read())
                test_labels.append(1 if label == 'pos' else 0)
    return train_data, train_labels, test_data, test_labels
dataset_dir = './IMDB'
print("开始载入数据，请耐心等待。")
train_data, train_labels, test_data, test_labels = load_imdb_data(dataset_dir)
print("开始文本向量化，请耐心等待。")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
model = LogisticRegression()
print("开始训练，请耐心等待。")
model.fit(X_train, train_labels)
print("开始预测，请耐心等待。")
y_pred = model.predict(X_test)
accuracy = accuracy_score(test_labels, y_pred)
print(f"模型准确率：{accuracy:.2f}")
new_data = ["I loved this movie!","This movie was a waste of time.","Bad Movies!"]
new_X = vectorizer.transform(new_data)
new_pred = model.predict(new_X)
print("预测结果:")
for doc,pred in zip(new_data,new_pred):
    print(f"文档:{doc},预测类别: {'正向' if pred == 1 else '负向'}")