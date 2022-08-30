### 8-16 高老师讨论记录
前一个月工作
1. 每周腾讯会议讨论机制
2. github项目，代码、文档管理
3. overleaf 写论文
4. 腾讯文档difference letter 模板  （马悦有latex版本 ）
5. 文献整理表格，增加40篇，有一篇强相关（会议版没有引用，现在找不出在这篇基础上明显的提升点）



主要问题:
1. 找不出在强相关文献基础上明显的提升点
2. 神经网络的代码工作进展很慢 （有代码没调通）

可能还可以做的
1. 审稿意见整理
2. 组合优化方面GBS和BS还是有一点区别的，没有写的太明白
3. 神经网络结构方面：有一些花头，但是不会分析

Difference
- 毕设工作中发现 加东西没有提升，如何做期刊版本增补：    咨询周浩麟、刘一鸣， 做了繁复操作后效果提升不明显，应当如何写论文
- 框架图重画
- 实验部分加了数据集
- 技术上改动： 高宇岑 将直接拼接改成gating，会好一些
- 加toy example
- 加discussion: 
- 李赵睿 learning tour 找高远宁要一下原稿

- 仔细分析DTRP是怎么转成期刊版， 分析出其difference letter，然后做出我们的

statistical visualization: 高宇岑做 spherical clustering 聚类TSP文章，在Intro里面写了做聚类， 数据集有均匀分布有非均匀分布。 与旅游推荐类似，其中有历史规律性，这种规律让深度学习模型合理。 由此加toy example来说明 

- 建模上：表示学习部分可以创新。 马鸣谦已经在related work部分写，移到正文中的建模部分。最后数值实验，得出某一种最好
    - 编码器LSTM改成GNN，弄出网络
    - 搬AAAI2020的模型 geo-dialated LSTM

- 会议总结：
    - 1. 大框架SEQ2SEQ不变，LSTM不变， GBS不变
    - 2. 能变的只有ENCODER部分的2个表示方法， 以及合并方法
        - User encoder: 林世源做了工作，没有用。验证了这部分表示是合理的
        - Geo encoder: 可以改很多  
            - Temporal因素， 隐含在LSTM里面（改动下面的encoder的网络结构）
    - 最后就改地理位置部分的表示，把代码弄通。 加入AAAI2020或者IJCAI2018的设计。分工
        - 张嘉乐：从底层构建代码，跑通，加内容
        - 马鸣谦：整理相关的表示，时间、空间、轨迹是三种不同的表达形式，分别可以做encoding。 跑毕设代码。

如何调程序：（目前的瓶颈就在这里，一定要跑通）
- 先跑白板
- 一点点加回来
- Bi-LSTM可以问周昕逸、陈家栋
- 山大同学的代码尝试一下
找一些TKDE方向的论文，要增加实验工程量（咨询周浩麟、刘一鸣）， 建模设计分析不是太多。
统一用Torch框架下的



### 8-9 交流记录     
1. 建立difference letter的模板， 在原来板子上改内容， 填写进度能够表征转刊工作的进度
2. 参考强相关文献DTRP，分析我们在此基础上的进展（不同点）是什么？
3. DTRP代码找不到（github等网上找不到），邮件发了没回。 建议看一下作者情况，向会议一作、期刊二作要一下代码
4. 传统算法对于历史数据没有分析，  另外是只能推荐整个序列，无法单独往后加： 先做进表里面，后面再加解说文字
5. 如何确定我们模型的增补内容： DTRP强相关文献分析；
    - 我们的文章在编码结构LSTM上面做了新的设计（双向）。 参考神经网络设计的文章来写如何分析新结构的作用。
    - 针对数据的某个特点，设计了某种结构： 双向LSTM可以去掉时间方向
6. 具体下一步工作： 拆解mattrip与DTRP在神经网络结构设计上的区别， 找到对应的机器学习领域当时做出这些设计的论文，分析其作用
7. 代码工作：Baselines正在配环境，跑代码（马鸣谦）； 原文代码继续拆解，跑通各个模块（张嘉乐）
8. 组合优化的baseline没找到特别合适的， 这个放一下，需要继续加深对文章技术的理解。
模型中LSTM- 双向LSTM - Attention (Document网站上)
9. 序列生成模型，联系 陈家栋 （张嘉乐）。 
小结：钻研原理方面内容，加深对文章的神经网络模型理解： 对比DTRP，回顾DCE资料网站上相关（广告组的）报告，找机器学习领域相关论文


8-2讨论
1. 文献整理表，各列内容讨论。 接下来可以正式动手写related work部分内容
2. baselines： 
 - LSTPM 代码有实现， 但是数据集的预处理有点复杂。 最好要向原作者要一下数据集。
 - 基础LSTM，一般论文通用baseline
 - DTRP这篇强相关
 - 组合类型，待定
3. 林世源毕设讨论： 相关工作部分介绍很好，但是对于前沿涉猎不够； 实验部分只能说明编码没用，不能直接借鉴。
 - 如何增补期刊内容： 加社交网络； 加时间段方面的考虑，拆POI的考虑，参考LSTPM的文章（序号10，写的也比较清楚）

接下来工作： 写文字（精读论文找思路）&调程序

## Week 3 内容总结

## 马鸣谦

### 文献整理表

链接：【腾讯文档】MatTrip相关文献整理表 https://docs.qq.com/sheet/DTElDSkROQmpTbVhS

交我算文档：见repo根目录

## Week 2 内容总结

## 马鸣谦

### 扫文

- 补充了九篇文章，内容包括

  - 其他functional考虑（天气、人群、交通等）作为related works 相关semantic部分

  - Optimization类型的POI recommendation 主要是一些启发式算法

- 做baseline实现，选择的是扫文里第十篇LSTPM，源代码找到fork在GitHub里了，能跑通源代码，在理解和实现

论文具体内容：
问题：已知当前user过往路径集合S，和现在路径里l1到l(n-1)，预测下一个POI的前k个可能结果


## Week 1 内容总结

### 扫文

#### 文章算法、内容

- 极强相关 -- 2 篇

  为WISE 2017及其转期刊版本

  2017的内容和现在的非常相似 算法是LSTM+beam search

  2020的期刊版增加了heterogeneous network考虑relationship的方法；用了更多dataset

- 强相关 -- 7篇

  内容多为各类深度学习的方法做next/next k poi recommendation

- 中等相关 -- 11篇

  区别多在问题设定、考虑的因素有较大差别（NLP的text识别推荐类型，QA，其他类型的learning等），可以做相关领域在POI recommendation的方法

- 低相关/不相关 

  差别较为明显，不太能用

#### Dataset

常用：Foursquare, Gowalla, Yelp, Weeplaces, Brightkite, Flikr

数据集链接：

Foursquare: https://archive.org/details/201309_foursquare_dataset_umn

Gowalla: https://www.yongliu.org/datasets/

Yelp: https://www.yelp.com/dataset

Brightkite: http://snap.stanford.edu/data/loc-Brightkite.html.

Flikr: https://sites.google.com/site/limkwanhui/datacode#ijcai15.

Weeplaces: https://www.yongliu.org/datasets/

#### 想法

- 感觉现有内容可能可以从模型上进行一些改进，用更新的network或者增加一些考虑因素，区别于2017那篇
- 增加实验数据集规模（或者sigir审稿意见里提到的预测规模？）
- 增加评价指标 （MRR，Edit Distance，Sensitivity）

