# 摘要
>

---
# what
![过拟合](md_pic/480px-Overfitting.svg.png)
>一个假设在训练数据上能够获得比其他假设更好的拟合，但是在训练数据外的数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。
>训练集的准确率很高，但是在测试集准确率却不高，无法泛化到实际应用中。

# why
## 样本问题
- 选取的样本数据不足以代表预定的分类规则
>样本数量太少
>选样方法错误
>样本标签错误

- 样本数据存在分类决策面不唯一，随着学习的进行，BP算法使权值可能收敛过于复杂的决策面
>一个样本实际可以分成多类，但在样本标注时只标注了一个分类

- 样本噪声干扰过大，使得机器将部分噪音认为是特征从而扰乱了预设的分类规则
## 模型问题
- 假设的模型无法合理存在，或者说是假设成立的条件实际并不成立；
- 模型参数太多，模型复杂度过高
>类比高次曲线拟合
- 权值学习迭代次数太多，拟合了训练数据中的噪声和训练样例中没有代表性的特征
## 实例
[a neural network playground](http://playground.tensorflow.org/#activation=relu&regularization=L1&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2,2&seed=0.58835&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false)

# how
## 使用权值衰减的方法，即每次迭代过程中以某个小因子降低每个权值
## 选取合适的停止训练标准，使对机器的训练在合适的程度
## 保留验证数据集，对训练成果进行验证
## 获取额外数据进行交叉验证
## 增加数据
## 正则化
>在进行目标函数或代价函数优化时，在目标函数或代价函数后面加上一个正则项，一般有L1正则与L2正则等
### 损失函数正则化
#### L1
#### L2
### 网络结构正则化
#### dropout
# 实例

# 结论


---
参考资料
- [百度百科](https://baike.baidu.com/item/%E8%BF%87%E6%8B%9F%E5%90%88/3359778?fr=aladdin)
- [维基百科Overfitting](https://en.wikipedia.org/wiki/Overfitting)
- [Regularization（正则化）与Dropout](http://blog.csdn.net/u014696921/article/details/54410166)
- [5 Ways How to Reduce Overfitting](https://www.linkedin.com/pulse/5-ways-how-reduce-overfitting-tomas-nesnidal)