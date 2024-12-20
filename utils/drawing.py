import matplotlib.pyplot as plt
import numpy as np

def generate_triples():
    # 生成一个较大的测试数据集
    # 假设有10000个不同的triples
    num_triples = 10000
    
    # 生成随机的coexist_paper_num值，假设范围在1到1000之间
    random_values = np.random.randint(1, 1001, num_triples)
    
    # 为了构建一个字典，我们需要生成10000个唯一的triple
    # 这里为了简化，我们使用简单的字符串拼接来生成triple
    triples = [("s" + str(i), "r" + str(i), "o" + str(i)) for i in range(num_triples)]
    
    # 构建字典
    triples2coexist_num = dict(zip(triples, random_values))
    
    # 输出一部分数据以供检查
    print(list(triples2coexist_num.items())[:10])
    
    return triples2coexist_num

def convert_triple2coexistPMIDs_to_triple2coexist_num(triple2coexistPMIDs: dict):
    triple2coexist_num = {}
    for triple, PMIDs in triple2coexistPMIDs.items():
        triple2coexist_num[triple] = len(PMIDs)
    return triple2coexist_num


def draw_bar(triples2coexist_num: dict):
    # 提取所有的coexist_num
    coexist_nums = list(triples2coexist_num.values())
    
    # 使用numpy的histogram函数确定分区间
    # 这里的bins参数可以根据需要调整
    # bins = np.histogram(coexist_nums, bins='auto')[1]  # 取 bins的边界值
    bins = np.histogram(coexist_nums, bins=20)[1]  # 使用固定数量的区间
    
    plt.figure(figsize=(12, 6))  # 可以调整尺寸以适应你的需要
    
    # 统计每个区间的triple数量
    hist, _ = np.histogram(coexist_nums, bins=bins)
    
    # 绘制柱状图
    plt.bar(range(len(hist)), hist, tick_label=[f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)], color='blue')
    plt.xticks(rotation=45)
    plt.bar(range(len(hist)), hist)
    plt.xlabel('Coexist Paper Num')
    
    
    plt.ylabel('Number of Triples')
    plt.title('Distribution of Triples over Coexist Paper Num')
    
    plt.show()
    
if __name__ == "__main__":
    triples2coexist_num = generate_triples()
    draw_bar(triples2coexist_num=triples2coexist_num)
    