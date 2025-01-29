import jieba
import jieba.analyse
from collections import Counter
with open('兰亭序.txt','r',encoding='utf-8') as file:
    text = file.read()
    text = ''.join(filter(lambda x:'\u4e00' <= x <= '\u9fff', text))
    char_freq = Counter(text)
    most_common_char = char_freq.most_common(1)[0]
print(f"出现频率最高的字是:{most_common_char[0]}，频率为:{most_common_char[1]}")
words =jieba.cut(text)
filtered_words =[word for word in words if len(word)>1]
filtered_word_freq =Counter(filtered_words)
most_common_word = filtered_word_freq.most_common(1)[0]
print(f"出现频率最高的词是:{most_common_word[0]}，频率为:{most_common_word[1]}")
keywords = jieba.analyse.extract_tags(text,topK=10, withWeight=False)
print("提取的关键词为:", keywords)
with open('result.txt','w', encoding='utf-8') as file:
    file.write(f"出现频率最高的字是: {most_common_char[0]}，频率为:{most_common_char[1]}\n")
    file.write(f"出现频率最高的词是: {most_common_word[0]}，频率为:{most_common_word[1]}\n")
    file.write("提取的关键词为:"+ str(keywords) + "\n")