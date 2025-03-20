from collections import Counter
import re


def count_word_frequency(file_path):
    # 读取文件并统一为小写
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()

    # 使用正则表达式提取单词（过滤标点）
    words = re.findall(r'\b\w+\b', text)

    # 统计词频
    word_count = Counter(words)

    # 输出前10高频词
    print("Top 10高频词：")
    for word, count in word_count.most_common(10):
        print(f"{word}: {count}")


# 测试
count_word_frequency(r"C:\Users\29955\dian\example.txt")