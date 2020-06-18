# 决策树在sklearn中的实现

[TOC]

## sklearn 入门

scikit-learn，又写作 sklearn，是一个开源的基于 python 语言的机器学习工具包。它通过 NumPy, SciPy 和 Matplotlib 等 python 数值计算的库实现高效的算法应用，并且涵盖了几乎所有主流机器学习算法。

http://scikit-learn.org/stable/index.html

在工程应用中，用 python 手写代码来从头实现一个算法的可能性非常低，这样不仅耗时耗力，还不一定能够写出 构架清晰，稳定性强的模型。更多情况下，是分析采集到的数据，根据数据特征选择适合的算法，在工具包中调用 算法，调整算法的参数，获取需要的信息，从而实现算法效率和效果之间的平衡。而 sklearn，正是这样一个可以 帮助我们高效实现算法应用的工具包。

sklearn 有一个完整而丰富的官网，里面讲解了基于 sklearn 对所有算法的实现和简单应用。然而，这个官网是全英 文的，并且现在没有特别理想的中文接口，市面上也没有针对 sklearn 非常好的书。因此，这门课的目的就是由简 向繁地向大家解析 sklearn 的全面应用，帮助大家了解不同的机器学习算法有哪些可调参数，有哪些可用接口，这 些接口和参数对算法来说有什么含义，又会对算法的性能及准确性有什么影响。我们会讲解 sklearn 中对算法的说 明，调参，属性，接口，以及实例应用。注意，本门课程的讲解不会涉及详细的算法原理，只会专注于算法在 sklearn 中的实现，如果希望详细了解算法的原理，建议阅读下面这本两本书：

![ScreenClip](https://raw.githubusercontent.com/protea-ban/images/master/20200505130923.png)

![ScreenClip [1]](https://gitee.com/proteaban/blogimages/raw/master/img/20200505131024.png)

# 决策树

## 1 概述

### 1.1 决策树是如何工作的

决策树(Decision Tree)是一种非参数的有监督学习方法，它能够从一系列有特征和标签的数据中总结出决策规 则，并用树状图的结构来呈现这些规则，以解决分类和回归问题。决策树算法容易理解，适用各种数据，在解决各 种问题时都有良好表现，尤其是以树模型为核心的各种集成算法，在各个行业和领域都有广泛的应用。

**关键概念：节点**

> 根节点：没有进边，有出边。包含最初的，针对特征的提问。
>
> 中间节点：既有进边也有出边，进边只有一条，出边可以有很多条。都是针对特征的提问。
>
> 叶子节点：有进边，没有出边，每个叶子节点都是一个类别标签。
>
> 子节点和父节点：在两个相连的节点中，更接近根节点的是父节点，另一个是子节点。

决策树算法的核心是要解决两个问题：

1） 如何从数据表中找出最佳节点和最佳分枝？

2） 如何让决策树停止生长，防止过拟合？

几乎所有决策树有关的模型调整方法，都围绕这两个问题展开。这两个问题背后的原理十分复杂，我们会在讲解模 型参数和属性的时候为大家简单解释涉及到的部分。在这门课中，我会尽量避免让大家太过深入到决策树复杂的原 理和数学公式中（尽管决策树的原理相比其他高级的算法来说是非常简单了），这门课会专注于实践和应用。如果 大家希望理解更深入的细节，建议大家在听这门课之前还是先去阅读和学习一下决策树的原理。

### 1.2 sklearn中的决策树

* 模块 sklearn.tree

sklearn 中决策树的类都在"tree"这个模块之下。这个模块总共包含五个类:

| 模块名                      | 类别                                  |
| --------------------------- | ------------------------------------- |
| tree.DecisionTreeClassifier | 分类树                                |
| tree.DecisionTreeRegressor  | 回归树                                |
| tree.export_graphviz        | 将生成的决策树导出为 DOT 格式，画图专用 |
| tree.ExtraTreeClassifier    | 高随机版本的分类树                    |
| tree.ExtraTreeRegressor     | 高随机版本的回归树                    |

* sklearn 的基本建模流程

在那之前，我们先来了解一下 sklearn 建模的基本流程。

![ScreenClip](https://gitee.com/proteaban/blogimages/raw/master/img/20200505170746.png)

在这个流程下，分类树对应的代码是:

```python
from sklearn import tree	#导入需要的模块
clf = tree.DecisionTreeclassifier() #实例化
clf = clf.fit (X_train, y_train) #用训练集数据训练模型
result = clf.score(X_test,y_test)	#导入测试集，从接口中调用需要的信息

```

## 2 DecisionTreeClassifier 与红酒数据集

*class* sklearn.tree.DecisionTreeclassifier *(criterion=,gini,l splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random _state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)*

### 2.1 重要参数

#### 2.1.1 criterion

为了要将表格转化为一棵树，决策树需要找出最佳节点和最佳的分枝方法，对分类树来说，衡量这个"最佳"的指标 叫做"不纯度"。通常来说，不纯度越低，决策树对训练集的拟合越好。现在使用的决策树算法在分枝方法上的核心 大多是围绕在对某个不纯度相关指标的最优化上。

不纯度基于节点来计算，树中的每个节点都会有一个不纯度，并且子节点的不纯度一定是低于父节点的，也就是 说，在同一棵决策树上，叶子节点的不纯度一定是最低的。

Criterion 这个参数正是用来决定不纯度的计算方法的。sklearn 提供了两种选择：

1)  输入”entropy"，使用信息熵(Entropy)

2)   输入"gini"，使用基尼系数(Gini Impurity)
$$
Entropy(t)=-\sum_{i=0}^{c-1}p(i|t)log_{2}p(i|t)
$$

$$
Gini(t)=1-\sum_{i=0}^{c-1}p(i|t)^{2}
$$

其中 t 代表给定的节点，i 代表标签的任意分类，$p(i|t)$ 代表标签分类 i 在节点 t 上所占的比例。注意，当使用信息熵 时，sklearn 实际计算的是基于信息熵的信息增益(Information Gain)，即父节点的信息熵和子节点的信息熵之差。

比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。但是在实际使用中，信息熵和基尼系数的效果基 本相同。信息熵的计算比基尼系数缓慢一些，因为基尼系数的计算不涉及对数。另外，因为信息熵对不纯度更加敏 感，所以信息熵作为指标时，决策树的生长会更加"精细"，因此对于高维数据或者噪音很多的数据，信息熵很容易 过拟合，基尼系数在这种情况下效果往往比较好。当模型拟合程度不足的时候，即当模型在训练集和测试集上都表 现不太好的时候，使用信息熵。当然，这些不是绝对的。

|        参数         |                          criterion                           |
| :-----------------: | :----------------------------------------------------------: |
|   如何影响模型？    | 确定不纯度的计算方法，帮忙找出最佳节点和最佳分枝，不纯度越低，决策树对训练集  的拟合越好 |
| 可能的输入有哪 些？ | 不填默认基尼系数，填写 gini 使用基尼系数，填写 entropy 使用信息增益 |
|   怎样选取参数？    | 通常就使用基尼系数  数据维度很大，噪音很大时使用基尼系数  维度低，数据比较清晰的时候，信息熵和基尼系数没区别  当决策树的拟合程度不够的时候，使用信息熵  两个都试试，不好就换另外一个 |

到这里，决策树的基本流程其实可以简单概括如下:

1. 计算全部特征的不纯度指标
2. 选取不纯度指标最优的特征来分枝
3. 在第一个特征的分枝下，计算全部特征的不纯度指标
4. 选取不纯度指标最优的特征继续分枝，直到没有更多的特征可用，或整体的不纯度指标已经最优，决策树就会停止生长。

* 建立一棵树

  1. 导入需要的算法库和模块

     ```python
     from sklearn import tree
     from sklearn.datasets import load_wine
     from sklearn.model_selection import train_test_split
     ```

     

  2. 探索数据

     ```python
     wine=load_wine()   # 加载数据
     wine.data.shape    # 数据维度
     wine.target        # 数据集的标签
     wine.feature_names # 特征名
     wine.target_names  # 标签名
     ```

     ```python
     # 将数据以表的形式呈现
     # 用到pandas
     import pandas as pd
     pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
     ```

  3. 分训练集和测试集

     ```python
     Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
     ```

     

  4. 建立模型

     ```python
     clf = tree.DecisionTreeClassifier(criterion="entropy")
     clf = clf.fit(Xtrain, Ytrain)
     score = clf.score(Xtest, Ytest)
     
     score
     ```

     

  5. 画出一棵树吧

     ```python
     feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
     import graphviz
     dot_data = tree.export_graphviz(clf
                                     ,out_file = None
                                     ,feature_names= feature_name
                                     ,class_names=["琴酒","雪莉","贝尔摩德"]
                                     ,filled=True #填充颜色
                                     ,rounded=True
                                     )
     graph = graphviz.Source(dot_data)
     graph
     ```

     

  1. 探索决策树

     看看每个特征在模型当中的重要性，即权重。

     ```python
     clf.feature_importances_
     [*zip(feature_name, clf.feature_importances_)]
     ```

我们已经在只了解一个参数的情况下，建立了一棵完整的决策树。但是回到步骤 4 建立模型，score 会在某个值附近 波动，引起步骤 5 中画出来的每一棵树都不一样。它为什么会不稳定呢？如果使用其他数据集，它还会不稳定吗？

我们之前提到过，无论决策树模型如何进化，在分枝上的本质都还是追求某个不纯度相关的指标的优化，而正如我 们提到的，不纯度是基于节点来计算的，也就是说，决策树在建树时，是靠优化节点来追求一棵优化的树，但最优 的节点能够保证最优的树吗？集成算法被用来解决这个问题：sklearn 表示，既然一棵树不能保证最优，那就建更 多的不同的树，然后从中取最好的。怎样从一组数据集中建不同的树？在每次分枝时，不从使用全部特征，而是随 机选取一部分特征，从中选取不纯度相关指标最优的作为分枝用的节点。这样，每次生成的树也就不同了。

### 2.1.2 random_state & splitter

random_state 用来设置分枝中的随机模式的参数，默认 None，在高维度时随机性会表现更明显，低维度的数据 （比如鸢尾花数据集），随机性几乎不会显现。输入任意整数，会一直长出同一棵树，让模型稳定下来。

splitter 也是用来控制决策树中的随机选项的，有两种输入值，输入"best"，决策树在分枝时虽然随机，但是还是会 优先选择更重要的特征进行分枝（重要性可以通过属性 feature_importances_查看），输入"random"，决策树在 分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这 也是防止过拟合的一种方式。当你预测到你的模型会过拟合，用这两个参数来帮助你降低树建成之后过拟合的可能 性。当然，树一旦建成，我们依然是使用剪枝参数来防止过拟合。

```python
clf = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state=30
                                  ,splitter="random"
                                 )
```

### 2.1.3 剪枝参数  

在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止。这样的决策树 往往会过拟合，这就是说，它会在训练集上表现很好，在测试集上却表现糟糕。我们收集的样本数据不可能和整体 的状况完全一致，因此当一棵决策树对训练数据有了过于优秀的解释性，它找出的规则必然包含了训练样本中的噪 声，并使它对未知数据的拟合程度不足。

```python
# 我们的树对训练集的拟合程度如何？
# 注意，此时用到的是训练集数据
score_train = clf.score(Xtrain, Ytrain)
score_train
```

为了让决策树有更好的泛化性，我们要对决策树进行剪枝。剪枝策略对决策树的影响巨大，正确的剪枝策略是优化 决策树算法的核心。sklearn 为我们提供了不同的剪枝策略：

* max_depth

  限制树的最大深度，超过设定深度的树枝全部剪掉

  这是用得最广泛的剪枝参数，在高维度低样本量时非常有效。决策树多生长一层，对样本量的需求会增加一倍，所 以限制树深度能够有效地限制过拟合。在集成算法中也非常实用。实际使用时，建议从=3 开始尝试，看看拟合的效 果再决定是否增加设定深度。

* min_samples_leaf & min_samples_split

  min_samples_leaf 限定，一个节点在分枝后的每个子节点都必须包含至少 min_samples_leaf 个^练样本，否则分 枝就不会发生，或者，分枝会朝着满足每个子节点都包含 min_samples_leaf 个样本的方向去发生

  一般搭配 max_depth 使用，在回归树中有神奇的效果，可以让模型变得更加平滑。这个参数的数量设置得太小会引 起过拟合，设置得太大就会阻止模型学习数据。一般来说，建议从=5 开始使用。如果叶节点中含有的样本量变化很 大，建议输入浮点数作为样本量的百分比来使用。同时，这个参数可以保证每个叶子的最小尺寸，可以在回归问题 中避免低方差，过拟合的叶子节点出现。对于类别不多的分类问题，=1 通常就是最佳选择。

  min_samples_split 限定，一个节点必须要包含至少 min_samples_split 个训练样本，这个节点才允许被分枝，否则 分枝就不会发生。

  ```python
  clf = tree.DecisionTreeClassifier(criterion="entropy"
                                    ,random_state=30
                                    ,max_depth=3
                                    ,splitter="random"
                                    ,min_samples_leaf=10
                                    ,min_samples_split=10
                                   )
  clf = clf.fit(Xtrain,Ytrain)
  ```

* max_features & min_impurity_decrease

  一般 max_depth 使用，用作树的”精修"

  max_features 限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。和 max_depth 异曲同工， max_features 是用来限制高维度数据的过拟合的剪枝参数，但其方法比较暴力，是直接限制可以使用的特征数量 而强行使决策树停下的参数，在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型 学习不足。如果希望通过降维的方式防止过拟合，建议使用 PCA，ICA 或者特征选择模块中的降维算法。

  min_impurity_decrease 限制信息增益的大小，信息增益小于设定数值的分枝不会发生。这是在 0.19 版本中更新的 功能，在 0.19 版本之前时使用 min_impurity_split。

* 确认最优的剪枝参数

  那具体怎么来确定每个参数填写什么值呢？这时候，我们就要使用确定超参数的曲线来进行判断了，继续使用我们 已经训练好的决策树模型 clf。超参数的学习曲线，是一条以超参数的取值为横坐标，模型的度量 J 指标为纵坐标的曲 线，它是用来衡量不同超参数取值下模型的表现的线。在我们建好的决策树里，我们的模型度量指标就是 score。

  ```python
  import matplotlib.pyplot as plt
  
  test = []
  for i in range(10):
      clf = tree.DecisionTreeClassifier(criterion="entropy"
                                        ,random_state=30
                                        ,splitter="random"
                                        ,max_depth = i + 1
                                       )
      clf = clf.fit(Xtrain,Ytrain)
      score = clf.score(Xtest,Ytest)
      test.append(score)
  
  plt.plot(range(1,11),test,color="red",label="max_depth")
  plt.legend()
  plt.show()
  ```

思考：

1. 剪枝参数一定能够提升模型在测试集上的表现吗？
   * 调参没有绝对的答案，一切都是看数据本身。
2. 这么多参数，一个个画学习曲线？
   * 在泰坦尼克号的案例中，我们会解答这个问题。

无论如何，剪枝参数的默认值会让树无尽地生长，这些树在某些数据集上可能非常巨大，对内存的消耗也非常巨 大。所以如果你手中的数据集非常巨大，你已经预测到无论如何你都是要剪枝的，那提前设定这些参数来控制树的 复杂性和大小会比较好。

### 2.1.4 目标权重参数  

* class_weight & min_weight_fraction_leaf

  完成样本标签平衡的参数。样本不平衡是指在一组数据集中，标签的一类天生占有很大的比例。比如说，在银行要 判断'一个办了信用卡的人是否会违约”，就是是 vs 否(1%： 99%)的比例。这种分类状况下，即便模型什么也不 做，全把结果预测成"否”，正确率也能有 99%。因此我们要使用 class_weight 参数对样本标签进行一定的均衡，给 少量的标签更多的权重，让模型更偏向少数类，向捕获少数类的方向建模。该参数默认 None，此模式表示自动给 与数据集中的所有标签相同的权重。

  有了权重之后，样本量就不再是单纯地记录数目，而是受输入的权重影响了，因此这时候剪枝，就需要搭配 min_ weight_fraction_leaf这个基于权重的剪枝参数来使用。另请注意，基于权重的剪枝参数(例如min_weight_ fraction_leaf)将比不知道样本权重的标准(比如 min_samples_leaf)更少偏向主导类。如果样本是加权的，则使 用基于权重的预修剪标准来更容易优化树结构，这确保叶节点至少包含样本权重的总和的一小部分。

## 2.2 重要属性和接口

属性是在模型训练之后，能够调用查看的模型的各种性质。对决策树来说，最重要的是 feature_importances_，能 够查看各个特征对模型的重要性。

sklearn 中许多算法的接口都是相似的，比如说我们之前已经用到的 fit 和 score，几乎对每个算法都可以使用。除了 这两个接口之外，决策树最常用的接口还有 apply 和 predict。apply 中输入测试集返回每个测试样本所在的叶子节 点的索引，predict 输入测试集返回每个测试样本的标签。返回的内容一目了然并且非常容易，大家感兴趣可以自己 下去试试看。

在这里不得不提的是，所有接口中要求输**A**X_train 和乂二。$七的部分，输入的特征矩阵必须至少是一个二维矩阵。

sklearn 不接受任何一维矩阵作为特征矩阵被输入。如果你的数据的确只有一个特征，那必须用 reshape(-1,1)来给 矩阵增维；如果你的数据只有一个特征和一个样本，使用 resh ape(1,-1)来给你的数据增维。

```python
# apply返回每个测试样本所在的叶子节点的索引
clf.apply(Xtest)
# predict返回每个测试样本的分类/回归结果
clf.predict(Xtest)
```

至此，我们已经学完了分类树 DecisionTreeClassifier 和用决策树绘图(export_graphviz)的所有基础。我们讲解 了决策树的基本流程，分类树的八个参数，一个属性，四个接口，以及绘图所用的代码。

八个参数：Criterion，两个随机性相关的参数(random_state，splitter)，五个剪枝参数(max_depth, min_samples_split，min_samples_leaf，max_feature，min_impurity_decrease)

一个属性：feature_importances_

四个接口： fit，score，apply，predict

有了这些知识，基本上分类树的使用大家都能够掌握了，接下来再到实例中去磨练就好。

# 3 DecisionTreeRegressor

```python
class sklearn.tree.DecisionTreeRegressor (criterion='mse‘, splitter='best‘, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0,0l min_impurity_split=None, presort=False)
```

几乎所有参数，属性及接口都和分类树一模一样。需要注意的是，在回归树种，没有标签分布是否均衡的问题，因 此没有 class_weight 这样的参数。

## 3.1 重要参数，属性及接口

#### criterion

回归树衡量分枝质量的指标，支持的标准有三种：

1)   输入"mse”使用均方误差 mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为 特征选择的标准，这种方法通过使用叶子节点的均值来最小化 L2 损失

2)   输入"friedman_mse"使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差

3)   输入”mae”使用绝对平均误差 MAE(mean absolute error)，这种指标使用叶节点的中值来最小化 L1 损失

属性中最重要的依然是 feature_importances_，接口依然是 apply, fit, predict, score 最核心。
$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(f_i-y_i)^{2}
$$
其中 N 是样本数量，i 是每一个数据样本，fi 是模型回归出的数值，yi 是样本点 i 实际的数值标签。所以 MSE 的本质， 其实是样本真实数据与回归结果的差异。**在回归树中，MSE 不只是我们的分枝质量衡量指标，也是我们最常用的衡 量回归树回归质量的指标**，当我们在使用交叉验证，或者其他方式获取回归树的结果时，我们往往选择均方误差作 为我们的评估(在分类树中这个指标是 score 代表的预测准确率)。在回归中，我们追求的是，MSE 越小越好。

然而，回归树的接口 score 返回的是 R 平方，并不是 MSE。R 平方被定义如下：
$$
R^2=1-\frac{u}{v}
$$

$$
u=\sum_{i=1}^{N}(f_i-y_i)^2 ,v=\sum_{i=1}^{N}(f_i-\widehat{y_i})^2
$$

其中$u$是残差平方和(MSE * N)，$v$是总平方和，N 是样本数量，i 是每一个数据样本，$f_i$是模型回归出的数值，$y_i$ 是样本点实际的数值标签。$\widehat{y_i}$是真实数值标签的平均数。R 平方可以为正为负(如果模型的残差平方和远远大于 模型的总平方和，模型非常糟糕，R 平方就会为负)，而均方误差永远为正。

值得一提的是，**虽然均方误差永远为正，但是 sklearn 当中使用均方误差作为评判标准时，却是计算"负均方误差“(neg_mean_squared_error)。**这是因为 sklearn 在计算模型评估指标的时候，会考虑指标本身的性质，均 方误差本身是一种误差，所以被 sklearn 划分为模型的一种损失(loss)，因此在 sklearn 当中，都以负数表示。真正的 均方误差 MSE 的数值，其实就是 neg_mean_squared_error 去掉负号的数字。

#### 简单看看回归树是怎样工作的

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state = 0)
# 交叉验证
cross_val_score(regressor
                ,boston.data
                ,boston.target
                ,cv=10
                ,scoring="neg_mean_squared_error"
               )
```

交叉验证是用来观察模型的稳定性的一种方法，我们将数据划分为 n 份，依次使用其中一份作为测试集，其他 n-1 份 作为训练集，多次计算模型的精确性来评估模型的平均准确程度。训练集和测试集的划分会干扰模型的结果，因此 用交叉验证 n 次的结果求出的平均值，是对模型效果的一个更好的度量。

![cross_val_score](https://gitee.com/proteaban/blogimages/raw/master/img/20200506113143.jpg)

## 3.2 实例：一维回归的图像绘制

接下来我们到二维平面上来观察决策树是怎样拟合一条曲线的。我们用回归树来拟合正弦曲线，并添加一些噪声来 观察回归树的表现。

1. 导入需要的库

   ```python
   import numpy as np
   from sklearn.tree import DecisionTreeRegressor
   import matplotlib.pyplot as plt
   ```

   

2. 创建一条含有噪声的正弦曲线

   在这一步，我们的基本思路是，先创建一组随机的，分布在 0~5 上的横坐标轴的取值(x),然后将这一组值放到 sin 函 数中去生成纵坐标的值(y),接着再到 y 上去添加噪声。全程我们会使用 numpy 库来为我们生成这个正弦曲线。

   决策树用到的数据必须是二维以上的。

   ```python
   rng = np.random.RandomState(1) # 随机数生成器
   X = np.sort(5 * rng.rand(80,1), axis=0)
   y = np.sin(X).ravel() # ravel降维函数
   y[::5] += 3 * (0.5 - rng.rand(16))
   ```

3. 实例化&训练模型

   ```python
   regr_1 = DecisionTreeRegressor(max_depth=2)
   regr_2 = DecisionTreeRegressor(max_depth=5)
   regr_1.fit(X, y)
   regr_2.fit(X, y)
   ```

4. 测试集导入模型，预测结果

   ```python
   X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] # 升维切片np.newaxis
   y_1 = regr_1.predict(X_test)
   y_2 = regr_2.predict(X_test)
   ```

5. 绘制图像

   ```python
   plt.figure()
   plt.scatter(X, y, s=20, edgecolors="black", c="darkorange", label="data")
   plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
   plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
   plt.xlabel("data")
   plt.ylabel("target")
   plt.title("Decision Tree Regression")
   plt.legend()
   plt.show()
   ```

![Decision Tree Regression](https://gitee.com/proteaban/blogimages/raw/master/img/20200506152801.png)

可见，回归树学习了近似正弦曲线的局部线性回归。我们可以看到，如果树的最大深度(由 max_depth 参数控制) 设置得太高，则决策树学习得太精细，它从训练数据中学了很多细节，包括噪声得呈现，从而使模型偏离真实的正 弦曲线，形成过拟合。

# 4 实例：泰坦尼克号幸存者的预测

泰坦尼克号的沉没是世界上最严重的海难事故之一，今天我们通过分类树模型来预测一下哪些人可能成为幸存者。 数据集来自[https://www.kaggle.eom/c/titanic](https://www.kaggle.com/c/titanic)。数据集包含两个 csv 格式文件，data 为我们接下来要使用的数据，test 为 kaggle 提供的测试集。

1. 导入所需要的库

   ```python
   import pandas as pd
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.model_selection import GridSearchCV
   from sklearn.model_selection import cross_val_score
   import matplotlib.pyplot as plt
   ```

2. 导入数据集，探索数据

   pd.read_csv 函数读取文件时有两种填写路径的方法：
   1. 在文件路径前加上 r ，此时是右斜杠：pd.read_csv(r"D:\data\data.csv")
   2. 无需 r 将右斜杠改成左斜杆：pd.read_csv("D:/data/data.csv")

   ```python
   data = pd.read_csv("taitanic_data.csv")
   data.head()
   data.info()
   ```

3. 对数据集进行预处理

   决策树的结点必须为数字，对象特征要转成数字。

   ```python
   # 删除缺失值过多的列，和观察判断来说和预测的y没有关系的列
   data.drop(["Cabin","Name","Ticket"], inplace=True,axis=1)
   
   # 处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法
   data["Age"] = data["Age"].fillna(data["Age"].mean())
   data = data.dropna()
   
   # 将二分类变量转换为数值型变量
   # astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这个方式可以很便捷地将二分类特征转换为0~1
   data["Sex"] = (data["Sex"] == "male").astype("int")
   
   # 将三分类变量转换为数值型变量
   labels = data["Embarked"].unique().tolist()
   data["Embarked"] = data["Embarked"].apply(lambda x:labels.index(x))
   ```

4. 提取标签和特征矩阵，分测试集和训练集

   ```python
   X = data.iloc[:,data.columns != "Survived"]
   y = data.iloc[:,data.columns == "Survived"]
   
   Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
   ```

   只要索引变乱，而且不是有意保持乱序索引的，就要将索引重新排序（从 0 开始）：
   Xtrain.index = range(Xtrain.shape[0])
   X.reset_index(drop=True,inplace=True)

   ```python
   # 修正测试集和训练集的索引
   for i in [Xtrain, Xtest, Ytrain, Ytest]:
       i.index = range(i.shape[0])
   ```

5. 导入模型，粗略跑一下查看结果

   ```python
   clf = DecisionTreeClassifier(random_state=25)
   clf = clf.fit(Xtrain, Ytrain)
   score_ = clf.score(Xtest, Ytest)
   score_
   
   #交叉验证
   score = cross_val_score(clf,X,y,cv=10).mean()
   score
   ```

6. 在不同 max_depth 下观察模型的拟合状况

   画出训练集和测试集的评分结果，可以有效查看模型是否过拟合。

   ```python
   tr = []
   te = []
   for i in range(10):
       clf = DecisionTreeClassifier(random_state=25
                                    ,max_depth=i+1
                                    ,criterion="entropy"
                                   )
       clf = clf.fit(Xtrain,Ytrain)
       score_tr = clf.score(Xtrain, Ytrain)
       score_te = cross_val_score(clf,X,y,cv=10).mean()
       tr.append(score_tr)
       te.append(score_te)
   print(max(te))
   plt.plot(range(1,11),tr,color="red",label="train")
   plt.plot(range(1,11),te,color="blue",label="test")
   plt.xticks(range(1,11))
   plt.legend()
   plt.show()
   ```

   注意：这里为什么^"entropy”？因为我们注意到，在最大深度=3 的时候，模型拟合不足，在训练集和测试集上的表现接 近，但却都不是非常理想，只能够达到 83%左右，所以我们要使用 entropy。

   ![train_test](https://gitee.com/proteaban/blogimages/raw/master/img/20200506161720.png)

7. 用网格搜索调整参数

   ```python
   import numpy as np
   gini_thresholds = np.linspace(0,0.5,20)
   
   parameters = {'splitter':('best','random')
                   ,'criterion':("gini","entropy")
                   ,"max_depth":[*range(1,10)]
                   ,'min_samples_leaf':[*range(1,50,5)]
                   ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
                }
   clf = DecisionTreeClassifier(random_state=25)
   GS = GridSearchCV(clf,parameters,cv=10)
   GS.fit(Xtrain,Ytrain)
   
   GS.best_params_
   GS.best_score_
   ```

   网格搜索就是在给定参数中完成所有可能的组合，通过`best_parms_` 和 `best_score_` 可查看最好模型对应的参数和结果。

   网格搜索的最好结果不一定会比之前自己选择时强，因为它只能在给定的参数组合中选择，这是它的缺点。

# 5 决策树的优缺点

**决策树优点**：

1. 易于理解和解释，因为树木可以画出来被看见

2. 需要很少的数据准备。其他很多算法通常都需要数据规范化，需要创建虚拟变量并删除空值等。但请注意， sklearn 中的决策树模块不支持对缺失值的处理。

3. 使用树的成本（比如说，在预测数据的时候）是用于训练树的数据点的数量的对数，相比于其他算法，这是 一个很低的成本。

4. 能够同时处理数字和分类数据，既可以做回归又可以做分类。其他技术通常专门用于分析仅具有一种变量类 型的数据集。

5. 能够处理多输出问题，即含有多个标签的问题，注意与一个标签中含有多种标签分类的问题区别开

6. 是一个白盒模型，结果很容易能够被解释。如果在模型中可以观察到给定的情况，则可以通过布尔逻辑轻松 解释条件。相反，在黑盒模型中（例如，在人工神经网络中），结果可能更难以解释。

7. 可以使用统计测试验证模型，这让我们可以考虑模型的可靠性。

8. 即使其假设在某种程度上违反了生成数据的真实模型，也能够表现良好。

**决策树的缺点：**

1. 决策树学习者可能创建过于复杂的树，这些树不能很好地推广数据。这称为过度拟合。修剪，设置叶节点所 需的最小样本数或设置树的最大深度等机制是避免此问题所必需的，而这些参数的整合和调整对初学者来说 会比较晦涩

2. 决策树可能不稳定，数据中微小的变化可能导致生成完全不同的树，这个问题需要通过集成算法来解决。

3. 决策树的学习是基于贪婪算法，它靠优化局部最优（每个节点的最优）来试图达到整体的最优，但这种做法 不能保证返回全局最优决策树。这个问题也可以由集成算法来解决，在随机森林中，特征和样本会在分枝过 程中被随机采样。

4. 有些概念很难学习，因为决策树不容易表达它们，例如 XOR，奇偶校验或多路复用器问题。

5. 如果标签中的某些类占主导地位，决策树学习者会创建偏向主导类的树。因此，建议在拟合决策树之前平衡 数据集。

# 6 附录

## 6.1 分类树参数列表

| 参数                         | 意义                                                         |
| ---------------------------- | :----------------------------------------------------------- |
| **criterion**                | 字符型，可不填，默认基尼系数 Cgini')  <br />用来衡量分枝质量的指标，即衡量不纯度的指标输入"gini"使用基尼系数，或输入"entropy* 使用信息增益(Information Gain) |
| **splitter**                 | 字符型，可不填，默认最佳分枝('best')  <br />确定每个节点的分枝策略  <br />输入“best”使用最佳分枝，或输入"random”使用最佳随机分枝 |
| **max_depth**                | 整数或 None,可不填，默认 None  <br />树的最大深度。如果是 None,树会持续生长直到所有叶子节点的不纯度为 0,或者直到每个 叶子节点所含的样本量都小于参数 min_samples_split 中输入的数字 |
| **min_samples_ split**       | 整数或浮点数，可不填，默认=2  <br />一个中间节点要分枝所需要的最小样本量。如果—节点包含的样本量小于 min_samples_split 中填写的数字，这个节点的分枝就不会发生，也就是说，这个节点一^会  成为 f 叶子节点  <br />1)  如果输入整数，则认为输入的数字是分枝所需的最 1 辟本量  <br />2) 如果输入浮点数，则认为输入的浮点数是比例，输入的浮点数输入模型的数据集的样  本量(n samples)是分枝所需的最小样本量  <br />浮点功能是 0.18 版本以上的 sklearn 才可以使用 |
| **min_sample leaf**          | 整数浮点数，可不填，默认=1  <br />一个叶节点要存在所需要的最®本量• f 节点在分枝后的每个子节点中，必须要包含至 少 min_sample_leaf 个训练样本，否则分枝就不会发生.这个参数可能会有着使模型更平滑 的效果，尤其是在回归中  <br />1)  如果输入整数，则认为输入的数字是叶节点存在所需的最小样本量 <br />2) 如果输入浮点数，则认为输入的浮点数是比例，输入模型的数据集的样本量(n samples)是叶节点存在所需的最小样本量 |
| **min_weight fraction_leaf** | 浮点数，可不填，默认=0. <br />一个叶节点要存在所需要的权重占输入模型的数据集的总权重的比例.  <br />总权重由 fit 接口中的 sample_weight 参数确定，当 sample_weight 是 None 时，默认所有样 本的权重相同 |
| **max_features**             | 整数，浮点数，字符型或 None,可不填，默认 None  <br />在做最佳分枝的时候，考虑的特征个数  <br />1)  输入整数，则每一次分枝都考虑 max_features个特征  <br />2) 输入浮点数，则认为输入的浮点数是比例，每次分枝考虑的特征数目是max_features 输入模型的数据集的特征个数(n_features)  <br />3)  输入"auto",采用n_features的平方根作为分枝时考虑毓征数目  <br />4)  输入"sqrt",采用n_features的平方根作为分枝时考虑的特征数目  <br />5)  输入"log2",采用log2(n_/reatures)作为分枝时考虑的特征数目  <br />6)  输入"None” , n_features 就是分枝时考虑的特征数目  <br />注意：如果在限制的 max_features 中，决策树无法找到节点样本上至少 f 敬曲分枝，那 对分枝的搜索不会停止，决策树搭会检直比限制的 max_features 数目更多的特征 |
| **random_state**             | 整数，sklearn 中设定好的 RandomState 实例，或 None,可不填，默认 None  <br />1)输入整数，random_state 是由随机数生成器生成的随机数种子  <br />2)输入 Randomstate 实例，则 random_state 是一个随机数生成器<br />3)输入 None,随机数生成器会是 np.random 模块中的一个 RandomState 实例 |
| **max_leaf_nodes**           | 整数或 None,可不填，默认 None  <br />最大叶节点数量.在最佳枝方式下，以 max_leaf_nodes 为限制来生长树.如果是 None, 则没有叶节点数量的限制. |
| **min_impurity_ decrease**   | 浮点数，可以不填，默认=0.  <br />当一个节点的分枝后引起的不纯度的降低大于或等于 min_impurity_decrease中输入的数 值，则这个分枝则会被保留，不会被剪枝。  <br />带权重的不纯度下降可以表示为：  <br />$\frac{N_t}{N}$:不纯度<br />$\frac{N_{t_R}}{N}$：右侧树枝的不纯度<br />$\frac{N_{t_L}}{N}$：左侧树枝的不纯度<br />其中N是样本总量，$N_t$是节点t中的样本量，$N_{t_L}$是左侧子节点的样本量，$N_{t_R}$是右侧子节 点的样本量  <br />注意：如果sample_weight在fit接口中有值，则N, $N_t$, $N_{t_R}$, $N_{t_L}$都是指样本量的权重，而非单纯的样本数量  <br />仅在 0.19 以本中提供此功能 |
| **class_weight**             | 字典，字典的列表，"balanced”或者“None”，默认 None  <br />与标签相关联的权重，表现方式是{标签的值：权重}.如果为 None,则默认所有的标签持有 相同的权重。对于多输出问题，字典中权重的顺序需要与各个 y 在标签数据集中睇冽顺序相同  <br /><br />注意，对于多输出问题（包括多标签问题），定义的权重必须具体到每个标签下的每个类， 其中类是字典键值对中的键，权重是键值对中的值.比如说，对于有四个标签，且每个标签  是二分类（0 和 1）的分类问题而言，权重应该被表示为：<br />  [{0:1,1:1）, （0:1,1:5）, {0:1,1:1）, （0:1,1:1）]  <br />而不是：  <br />[{1:1}, {2:5}, （3:1}, （4:1}]  <br />如果使用"balanced"模式，将会使用 y 的值自动调整与输入数据中的类频率成反比的权 重，比如<br />$\frac{N_{samples}}{n_{classes}*np.bincount(y)}$<br />对于多输出问题，每一列 y 的权重将被相乘  <br />注意：如果指定了 sample_weight,这些权重将通过 fit 接口与 sample_weight 相乘 |
| **min_impurity_ split**      | 浮点数 <br />防止树生长的阈值之一.如果 f 节点的不纯度高于 min_impurity_split,这个节点就会被分 枝，否则的话这个节点就只能是叶子节点.  <br /><br />在 0.19 以上版本中,这个参数的功能由被 min_impurity_decrease 取代,在 0.21 版本中这 个参会被删除,请使用 min_impurity_decrease |
| **presort**                  | 布尔值，可不填，默认 False  <br />是否预先分配数据以加快枇合中最隹分枝的发现.在大型数据集上使用默认设置决策树时.  将这个参数设置为 true 可能会延长训隧过程，降低训练速度.当使用较小的数据集或限制书 的深度时，设置这个参数为 true 可能会加快训练速度. |

## 6.2 分类树属性列表

| 参数                     | 意义                                                         |
| ------------------------ | ------------------------------------------------------------ |
| **classes_**             | 输出组(array)或者 f 数组的列表(list),结构为标签的数目(n_classes) 输出所有标签 |
| **feature_importances_** | 输出组，结构为特征的数目(n_features)  返回每个特征的重要性，一般是这个特征在多次分枝中产生的信息增益的综合，也被称为  "基尼重要性"(Gini Importance) |
| **max_features_**        | 输出整数  <br />参数 m a xfeatu res 的推断值                   |
| **n_classes_**           | 输出整数或列表 <br />标签类别的数据                          |
| **n_features_**          | 在训练模型(fit)时使用翊匏馒                                  |
| **n_outputs_**           | 在训练模型(fit)时输出的结果的个数                            |
| **tree_**                | 输出 f  可以导出建好的树结构的端口，通过这个端口，可以访问树的结构种球属性，包括但不仅限于查看：  <br />1)   二叉树的结构  <br />2)   每个节点的深度以及它是否是叶子  <br />3)   使用 decision_path 方法的示例到达的节点  <br />4)   用 apply 这个接口取样出的叶子  <br />5)   用于预测样本的规则  <br />6)  一组样本共享的决策路径 |

tree_的更多内容可以参考：

https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py

## 6.3 分类树接口列表

| 接口                                            | 意义                                                         |
| ----------------------------------------------- | :----------------------------------------------------------- |
| **apply  (X[, check input])**                   | 输入测试集或样本点，返回每个样本被分到的叶节点的索引  <br />check_input 是接口 apply 的参数，输入布尔值，默认 True,通常不使用 |
| **decision_path(X[, check_ input])**            | 输入测试集或样本点，返回树中的决策树结构  <br />Check_input 同样是参数 |
| **fit(X,y[, sample weight, check  input,...])** | 训练模型的接口，其中 X 代表训练样本的特征，y 代表目标数据，即标签，X 和 y 都必须是类数组结构，一般我们都使用 ndarray 来导入  <br />sample_weight 是 fit 的参数，用来为样本标签设置权重，输入的格式是一个和测试集样 本量一致长度的数字数组，数组中所带有的数字表示每个样本量所占的权重，数组中数 字的综合代表整个测试集权重总数  <br />返回训练完毕的模型 |
| **get _params([deep])**                         | 布尔值，获取这个模型评估对象的参数.接口本身的参数 deep,默认为 True,表示返 回此估计器的参数并包含作为饰器的子对象.  <br />返回模型评估对象在实例化时的参数设置 |
| **predict(X[,  check input])**                  | 预测所提供的测试集 X 中样本点的标签,这里的测试集 X 必须和 fit 中提供的训练集结构  一致  <br />返回模型预测的浜假样本的标签或回归值 |
| **predict_log_proba(X)**                        | 预测所提供的测试集 X 中样本点归属于各个标签的对数概率          |
| **predict_proba(XL check_ input])**             | 预测所提供的测试集 X 中样本点归属于各个标签的概率  <br />返回测试集中每个样本点对应的每个标签的概率，各个标签按词典顺序扣 E 序.预测的类 概率是叶中相同类的样本的分数. |
| **score(X, y[, sample weig ht])**               | 用给定测试数据和标签的平均准确度作为模型的评分标准，分数越高模型越好.其中 X 是测试集，y 是测试集的真实标签.sample_weight 是 score 的参数，用法与 fit 的参数一致  <br />返回给定策树数据和标签的平均准确度，在多标签分类中，这个指标是子集精度. |
| **set_params(\**params)**                       | 可以为已经建立的评估器重设参数  <br />返回重新设置的评估器本身 |

