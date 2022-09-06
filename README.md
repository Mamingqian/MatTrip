9-6推进
1. 昨天和高老师交流了一下， 继续设计实验，同时开始写作
2. github使用问题：先在网页上修改
3. 测试代码evaluation.py跑通 (目前像原程序一样输出如下即可)。 路径输出的部分在generator2.py中的BeamSearch()函数，可以参考
City : Osaka
Method : NaMNTR
F1 :  0.7439216223978723
Recall :  0.8001080358914575
Precision :  0.7505864764684729
Time :  6.169641971588135  ms
MaxTime :  23.999691009521484  ms
MinTime :  2.998828887939453  ms
4. 讨论具体的实验设计， 和哪些模型比？ （这个之前讨论过）， 怎么比？（上次说有个问题，这次看看能不能解决）
    - 参考DTRP期刊版， nextPOI, general Planning做两方面实验，分别比较。  nextPOI需要改模型
        - ①nextPOI: 有必要比较，体现encoder部分的功能。  另外体现dynamic revision。 
        - 路径推荐方面
            - ② F1 score: 原文的标准
            - ③ mandatory POI:  100次路径生成中包含强制点的次数 44 Travel Itinerary Recommendations with Must-see Points-of-Interest 
5. 八个数据集已做预处理： POI在ficklr原来里面，fake_user, fake_path均已出来出来 （user preference 建模的计算式修改）    
6. baseline已经实现了
    - LSTPM(AAAI2020), Next POI
	- PLSPL(TKDE2022), Next POI
	- ST-RNN(AAAI2016), Next POI
7. encoder的改动？继承与8-23的讨论
    - 可以用的目前只有 AAAI2020 ： geodialated ，可以使用，但是时间影响还需要进一步建模）
    - 另外还可以看的一篇 IJCAI 2018： Exploiting POI-Specific Geographical Influence for Point-of-Interest Recommendation   



小结：接下来工作 —— 实验设计、encoder改动
1.  ①nextPOI: 有必要比较，体现encoder部分的功能。  另外体现dynamic revision。 数据集都一样，和LSTPM, PLSPL, ST-RNN, 会议版mattrip, 期刊版mattrip。  需要改decodere输出代码的。 用本文的8个数据集，标准acc-x，本文(张嘉乐) , baseline（马鸣谦）  
2.  实验设计，需要进一步看文章考虑的， ②路径推荐F1 score: 本文会议版的标准。 baseline还需要设计    马鸣谦
3.  实验设计，需要进一步看文章考虑的，③ mandatory POI:  100次路径生成中包含强制点的次数 44 Travel Itinerary Recommendations with Must-see Points-of-Interest   张嘉乐
4. encoder的改动。 AAAI2020 ：geodialated ，可以使用，但是时间影响还需要进一步建模 (生成的路径中缺少时间标签，需要估计。nextPOI是有标签的)  张嘉乐
5. encoder的改动。 高宇岑的两篇论文阅读 
    - An Attention-Based Bi-GRU for Route Planning and Order Dispatch of Bus-Booking Platform(DASFAA)
    - DQN selector      
    
    
### 8-30推进
1. 代码运行问题： 回退torch版本
2. User interest抽取部分，得到的fake_user不是按照论文写的方法算的： 
- 解决方法2.1 使用毕设/DTRP里面的写法，用每个类别访问次数除以总访问次数。这个容易实现。√ 然后要讨论合理性，
    - 2.1.1 是否有文献支持？
    - 2.2.2 会不会引起其他问题：网络表达能力退化？ （无法区分不同用户）
- 解决方法2.2: 逆向考虑原来的fake_user是怎么凑出来的  ×
3. 序列生成和点生成没法比： baseline模型只能生成下一个点，无法生成完整序列。 如果按照baseline的标准的话我们的模型比较吃亏
解决方法：
    - 3.1 增加实验类型，参考DTRP，按照已有baseline能够接受的比较方式来设计实验。 
        - 序列生成和序列生成的比。
    - 3.2. MATTRIP能否每次预测下一个点，如果预测错的话使用标签信息进行纠正然后预测。 张嘉乐考虑一下）
    - 3.3 如果不行的话： 实验标准是，我们生成一个序列，比如正确序列是ABCZZZZDE， 我们生成了AFGXXXXDE. 这个标准有点赖

小结：
1. 跑通原来的模型生成代码 张嘉乐
2. 数据集处理问题：使用毕设/DTRP里面的写法，对所有数据集统一处理出来。 （很快）
3. 实验比较问题： 增加实验类型，序列生成vs序列生成, 序列生成vs点生成用两种实验标准来做
4. 实验比较问题： MATTRIP能否每次预测下一个点，如果预测错的话使用标签信息进行纠正然后预测，使得可以序列生成vs点生成。 张嘉乐考虑一下mattrip要怎么改

1. 实验部分增补

2. 模型部分模型部分的改动


### 8-23 推进
1. 跑通程序
2. DTRP文章的转刊思路仔细分析
    - Model部分加了表示学习
    - 实验部分加了数据集    
3. 讨论： encoder部分可以加些什么，怎么加
    - 加表示学习部分： 文献整理
        - AAAI2020 ： 第十篇LSTPM，   可以加入time-weighted operation （这部分不行，9-6） 
        - AAAI2020 ： geodialated  （可以使用，但是时间影响还需要进一步建模）
        可以用，原本是给出next poi的。 我们在用的时候应该讨论用于我们的问题会造成什么影响。 最后p= spftmax ()这一步要和grid beam search 结合起来
            - 原本程序train大约是半小时
        - IJCAI 2018： Exploiting POI-Specific Geographical Influence for Point-of-Interest Recommendation   看一下
        - learning tour可以直接用的
        - 高宇岑gating研究可以交流一下
    - 代码参考： pytorch文档、encoder-decoder的开源代码
4. 已经跑的baseline和数据集
    - 另一篇TKDE论文的baseline在Foursquare数据集上跑了
    - LSTPM在YFCC和gowalla数据集上跑了


接下来的工作
1. 设计一下实验增补方案， 写一下文章experiment部分第一节
    - 整理一下可用数据集 马鸣谦
    - 整理一下baseline,写进论文: LSTPM(AAAI2020), PLSPL(TKDE2022), ST-RNN（AAAI2016）,原会议版 张嘉乐
    - 评价指标设计 （F1-score，能加的可能有MRR，Edit Distance）  张嘉乐
2. 本文模型encoder部分的改动
    - 文献阅读 IJCAI 2018 (不用了)
    - 改动实现 geodialated 马鸣谦
    - 跑序列生成部分GBS代码 张嘉乐

    


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

