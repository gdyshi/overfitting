# 摘要
> 监督机器学习问题无非就是“minimizeyour error while regularizing your parameters”，也就是在规则化参数的同时最小化误差。最小化误差是为了让我们的模型拟合我们的训练数据，而规则化参数是防止我们的模型过分拟合我们的训练数据。

---
# 什么是过拟合
![过拟合](https://github.com/gdyshi/overfitting/blob/master/md_pic/480px-Overfitting.svg.png)
>一个假设在训练数据上能够获得比其他假设更好的拟合，但是在训练数据外的数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。
>模型把数据学习的太彻底，以至于把噪声数据的特征也学习到了。训练集的准确率很高，但是在测试集准确率却不高，无法泛化到实际应用中。

# 为什么会出现过拟合
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
[a neural network playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=20&networkShape=8,5,4,2&seed=0.55923&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

# 如何防止过拟合
## 重新清洗数据
## 采用更小的特征集
> 一般，减小特征集的方法有：特征选择和特征抽取。
> 特征选择是指在原有的特征中选择一部分特征来用，抛弃不重要的特征，新的特征集是原特征集的子集
> 特征抽取是指通过原有的高维特征构建新的特征，新的特征维度远远低于原有特征的维度，新的每一维特征都是原有所有特征的加权组合。最常见的特征抽取方法有主成分分析（PCA）和因子分析。

## 使用权值衰减的方法，即每次迭代过程中以某个小因子降低每个权值
## 选取合适的停止训练标准，使对机器的训练在合适的程度
## 保留验证数据集，对训练成果进行验证
## 获取额外数据进行交叉验证
## 增加训练用数据量
## 降低模型复杂度
## 增强[正则化](http://blog.csdn.net/zouxy09/article/details/24971995)
>在进行目标函数或代价函数优化时，在目标函数或代价函数后面加上一个正则项，一般有L1正则与L2正则等
假设模型参数服从先验概率，即为模型参数添加先验，只是不同的正则化方式的先验分布是不一样的。这样就规定了参数的分布，使得模型的复杂度降低
### 损失函数正则化
#### L0
> L0范数是指向量中非0的元素的个数。让参数W是稀疏的,很难优化求解

#### L1
> L1范数是指向量中各个元素绝对值之和，也叫“稀疏规则算子”。L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解

#### L2
> L2范数是指向量各元素的平方和然后求平方根。使得参数w变小加剧的效果。更小的参数值w意味着模型的复杂度更低，对训练数据的拟合刚刚好（奥卡姆剃刀），不会过分拟合训练数据，从而使得不会过拟合，以提高模型的泛化能力。
> 使矩阵可逆（存在唯一解）

### 网络结构正则化
#### dropout
> 过拟合，可以通过阻止某些特征的协同作用来缓解。在训练时候以一定的概率p来跳过一定的神经元

# 实例
[](http://playground.tensorflow.org/#activation=tanh&regularization=L2&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.01&regularizationRate=0.01&noise=20&networkShape=8,5,4,2&seed=0.55923&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
# 结论
> 从数据、模型、训练方法三个方向入手

---
参考资料
- [百度百科](https://baike.baidu.com/item/%E8%BF%87%E6%8B%9F%E5%90%88/3359778?fr=aladdin)
- [维基百科Overfitting](https://en.wikipedia.org/wiki/Overfitting)
- [Regularization（正则化）与Dropout](http://blog.csdn.net/u014696921/article/details/54410166)
- [5 Ways How to Reduce Overfitting](https://www.linkedin.com/pulse/5-ways-how-reduce-overfitting-tomas-nesnidal)
- [特征选择常用算法综述](http://www.cnblogs.com/heaad/archive/2011/01/02/1924088.html)
- [机器学习中的范数规则化之（一）L0、L1与L2范数](http://blog.csdn.net/zouxy09/article/details/24971995)
- [利用TensorFlow训练简单的二分类神经网络模型](http://blog.csdn.net/Peakulorain/article/details/76944598)