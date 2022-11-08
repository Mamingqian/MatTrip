### 11-8讨论
1. 修改框架确定，6点，一人改三点
- 张嘉乐
    - 问题定义可以改， 文章写作上修改比较大           
    - GBS升维留给新的文章，参考TC2016，《算法导论》， 再加上GNN，pointer network技术（转刊版本二维GBS加点讨论、证明，也参考TC2016，《算法导论》，《人工智能：一种现代方法》中 ）  
    - 文献调研加ML4CO。（问 高宇岑 要一下文献调研成果， 有KDD2022实验代码graph2route）。写文章，重画图，做表
- 马鸣谦
   - 加FM做预处理部分（ 转刊difference,核心技术变化)       
   - 实验部分可以， 根据实际情况加（有就用，没有就去掉）       
   - 文献调研改表，写文章，重画图，做表   
2. 调研表修改： 左侧加分类树状结构，把机器学习右侧比较稀疏的合并。 下次在处理打勾稀疏问题。
3. FM问题，写到文章
    - 31. 问题定义： 矩阵缺失值填充问题
    - 32. 关于用户的定义， 单个序列看做用户不合理，修改。
    - 33. 关于为何选择Funk-SVD的讨论
    - 34. 关于填充结果，和类别标签的对比实验， 可以调参（对于多次访问算几次的处理）
4. 实验设计    
    - 数据集增加：找带category的 可以自己处理路径
    - Route Recommendation：f-1 score, 增加edit distance 和文章里新提到的算法做对比
    - **Dynamic Revison的实验修改**： 不加GBS维度，当用户修改单个点时直接修改RNN输出（即下一时刻输出）状态为one-hot，并且清空GBS此前的内容，从这一点继续往后用GBS。
    如何评价dynamic revision的表现： case study, 用一些例子，在第一个出现不符合历史数据的点上dynamic revision, 观察后续点是否会吻合的更好/更差，在地图上可视化。
    - Mandatory, Unavailable: 作为预实验 随机指定若干unavailable/mandatory 看100次能满足几次。 注意随机生成时需要和测试集相符，即强制点在测试集的Ground Truth内，unavailable的不在。 
    unavailable的时间窗口是否考虑：可以加，随机生成点对（POI,不能出现在序列的第几个）。 随机指定若干unavailable/mandatory， 看100次中其他baseline能满足几次。 
    - Next POI（前面基本够了，可能不做了）：做acc1/5/10，ndcg1/5/10 比较和现有next poi sota算法差距

- 另外聊天：
    - JOJ团队，有个法国老师带的，一直有新人。  https://www.ji.sjtu.edu.cn/cn/about/faculty-staff/faculty-directory/faculty-detail/76/
    - 数模组队建议，实验室； 准备建议，可视化和各种工具




### 11-1讨论
一. 内容讨论： 细化深化对文章内容技术的理解程度， 为通过分析评价来增补内容做准备
论文第三部分， Section V,VI
- Section V presents the trace datasets and experiment results. 
- Section VI concludes the paper andshows our future directions.

二. Difference大纲梳理
    - a. 重新定义问题，   
        增加了更多限制（从2维->3维）：       lunch break, Compactniss
    - b. 扩展GBS算法， 引入动态规划思想， 进一步实现了对2个旅游路线功能的支持
    - c. 增加讨论：对使用编码-解码结构，LSTM, GBS来做旅游路径推荐问题这一技术路线的合理性进行讨论？？ （这部分目前未落实）
    - d. 增加实验
    - e. 增加文献
    - f. 语言文字修改
    （如有其它顺手实现了的修改点，务必记录下来，之后讨论整合到哪一点或者新开）

为每一处改动提供动机、 资料
考虑能否落实
  - a. 重新定义问题，   增加了更多限制（从2维->3维）： 
    - lunch break,     ×
        - 可行性。dp状态设计，可以，dp[15， 30， 45, 60]表示在这个休息时长下的一套路线（路线数量为beam宽度）。  dp转移方程  
        - 应用性  不佳
    - Compactniss，总路线物理长度（曼哈顿距离），表达要换：  ×
        - 可行性。 dp[1,2,3,4,...300]。 这一状态维度的数量会爆炸。
        - 应用性  不错
    - weather: incorperated into Unavailable。  
        - 奖励条件怎么设？
        - 具体增益如何量化？
        （查文献，短期没法落地）
    - Dynamic Revison:  
        - 可行性：增加维度 dp[路线点数，20][强制点数，3] [最后的点,30]， 存的是一个beam的路线及其概率（5条）
        - 应用性：有用
            - 值得写：为什么要这么复杂来做dynamic revision.  原文的方式会损失信息。 
        - 技术性：算法设计不优雅

    
  - b. 扩展GBS算法， 引入动态规划思想， 进一步实现了对2个旅游路线功能的支持
  - c. 增加讨论：对使用编码-解码结构，LSTM, GBS来做旅游路径推荐问题这一技术路线的合理性进行讨论？？ （这部分目前未落实）
  - d. 增加实验
  - e. 增加文献
- f. 语言文字修改
    （如有其它顺手实现了的修改点，务必记录下来，之后讨论整合到哪一点或者新开）


三、具体的一些技术细节讨论， 实际能做的一些事情
1. GBS过程中，RNN网络输出P(y_t|0：y_{t-1})是怎么实现的？
设5个点， 第一个选1号， 第二个是[0.1, 0.2, 0.3, 0.4,0.5]
如果第一个选2号，第二个是？
out  ○  ○   ○   ○(概率向量)
hid  ○  ○   ○   ○
inp  ○  ○   ○   ○
理解了，可以实现Dynamic Revision  (张嘉乐)
2. 数据预处理部分， 
    - 矩阵分解，预测每个人对每个POI的偏好。   随便用一个推荐系统处理数据稀疏的矩阵分解方法。
    - 奇异值分解SVD， 目前有技术实现过程(可以尝试， 马鸣谦)
    - 稍微整理一下矩阵分解，甚至是推荐系统处理数据稀疏的矩阵分解方法。
3. 问题建模 （张嘉乐，见上一次）
4. 实验部分   （马鸣谦）
    - 可以加数据集
    - Dynamic Revison的实验设计 (现在的情况比会议版的改进)
        - 思路是加了三维动态规划之后保留semantic，预测出错时修正单点， 能够比二维不保留semantic的情况下，在之后的预测上有更好的表现
    - Next POI实验：体现单点预测能力,加了Dynamic Revison后效果会好一些。 top-k频率。 需要继续完善
    - mandatory, unavailable的实验设计： 其他模型，生成100次，能选出几次符合条件。  可以细化设计一下。可以作为预实验，放在第三章，配合建模的。

### 10-25讨论
- 已做修改：Related Work
    - 1. A.传统优化、推荐方法。 B.语义推荐模型及其局限性。 C. 到最相似的工作DTRP
    - 2. 表格参考之前论文的源码来填内容


1. 内容讨论： 细化深化对文章内容技术的理解程度， 为通过分析评价来增补内容做准备
论文第二部分， Section III,IV
- Section III formulates the problem precisely.
- Section IV presents the proposed MatTrip model, with detailed framework descriptions. 
(仔细的，带着评论的心态去读论文)

III.
A. 问题初步概念，2个
- POI ： 点，包含3个属性
- Visit: 二元关系
*缺少了游客u的定义，可能可以加，构成一个异构图（知识图谱）的数据结构

B. 问题定义
- 目标定义： similar不够科学，参考推荐系统的论文的问题Formulation方式
- 约束条件1,2,3：  起终点、路线长度、必经点集合、（不可经过点）。 
- *约束条件4)写的有问题，意思
    - 可能是每个点都在开放时间访问   
    - 可能是不可经过点。 符号重新整。 
- 时间估计： 需要当地旅游景点游玩时间经验
    - 假设知道t_s,t_d，是可以得到每个景点时间。 （模型理论上可以讨论， 数据集还不支持）


IV.主模型4阶段结构—— 数据预处理建模、 2个编码器、 1个注意力解码器、 GBS输出
- 1.1. 数据预处理 （由于数据来源问题没解决，模型最初步站不住，暂不考虑修改）想办法从数据集里搞到t_d (搞到有t_d的数据集)
Q1： Bridge The Gap Between Dataset and Model: How to get $t_s$, $t_d$ form Geo-tagged photo?
    - Very Important Part. Used in Fig.1 and Experiment Settings
    
- 1.2. 编码器
    - 为何编码器是双向的，解码器却要用有方向的LSTM？
- 1.3. 注意力解码器
    - 对数损失函数是不是写的有问题？ 可能交叉熵？
Q3: 还有没有看到类似论文中多写的内容本文没有的？ （如实验部分与另一篇TKDE2022论文对比发现基本一致）
- 1.4. GBS
    - RNN的输出优化问题，复杂度，束搜索（BS）优化
    - Functionality问题：其中强制点用BS不太好实现， 不可经过点是可以的，长度可能可以？
    - Naive解决方法及其局限性
    - 网格束搜索GBS的方法描述、复杂度上的优化
        - 优势：用dp的思路把长度、强制点建模成了2个维度，来保持语义的同时进行优化选择。  是否可以继续加维度，增加functionality
        - DTRP的期刊版用BS是直接把强制点人为加入路径的（这样做会破坏神经网络建模的semantic），否定的话会导致Dynamic Revision站不住
    - Dynamic Revision:强制修改一个输出点，是否导致GBS崩溃。 GBS加一个维度： 关系户数量  [0,1,2,3]表示被人工硬加的点数


2. 修改想法
    - 引用文献，讨论分析模型设计合理性，如“Reported by [25] that splitting “statistically independent" features in a multi-modal model can achieve better result” (这个例子还可以更多引用25内容进行展开)
    - IV.E部分揉了问题背景、定义、解决方法，需要把定义提到前面的III部分 （存在什么困难？ 
        - 1. 章节III的组织，
        - 2. 章节IV.E的重新组织）
        - 3. 需要重新审视是否有其他可以解决类似问题的baseline

3. 工作思路整理：再看Difference Letter,对比一下我们目前已有的修改想法，看看
    - 1. 建模——数据集不匹配问题：解决数据稀疏问题。（重要的Difference点） 本希望找到有t_d数据集（暂时找不到），只能暂时用现在的数据集，也不是完全不行，但是有访问时间的数据非常少。 理想情况是每个(u,p)对，得到一个平均访问时间。 Baseline:  每个POI可以算平均访问时间、 User一部分可以算。 
        - 进阶：矩阵分解 （可以稍微了解一下，用小矩阵的乘积来表示低秩大矩阵， 难度高） 
    - 2. 时间估计在数据集上的实现也是一个问题（ 目前暂时用在路径里面的 位次/总路线长度来计算）
    - 3. 把GBS优势进一步扩大：用dp的思路把长度、强制点建模成了2个维度，来保持语义的同时进行优化选择。  是否可以继续加维度，增加functionality?  （难度中等，主要困难是要选什么functionality）：动态规划处理的问题特点——局部能够算，整体要求状态存在或者发生几次
        - 比如路线的休闲程度——总路线距离：Compactniss 。 搞成一个新的dp维度。
        - lunch break: 有/无，时间有多长（15， 30， 45, 60） [0~5]   
        - weather: incorperated into Unavailable.  天气增加有一些景点的吸引力。在BS选点的时候判断是否符合奖励条件，符合就给softmax之前增益一点。
            - 此处奖励条件可以搞更多标准。
        - Grouping: 要社交关系，不行。×
    - 4. 继续拆解第三部分，与相关论文的实验部分作对比

和TMC的difference比对一下，已经完全足够了，把“工作思路”里面的3做出，1,2根据实际情况做一部分，就可以满足转刊的技术创新需求，然后完成页数要求。




### 10-18 讨论记录
1. 论文内容梳理（分3部分，背景准备部分、 模型算法部分、 实验总结部分）
    - Intro
        - 1. 旅游推荐是什么。 现在有大量的数据集能支持这件事。应该是比较符合TKDE的风格的。
        - 2. 现有模型数学优化为何不行，举例说明
        - 3. 进一步解释问题的复杂性，路径中“语义”的定义
        - 4. 现有模型神经网络的缺陷。 说明Request, functionality是什么。 神经网络模型通常难以考虑这个
        - 5. 模型概况：名称、解决的2大关键问题、主要涉及到的深度学习技术
        - 6. （有点问题，需要再提炼）创新点小结
    - Related work：reviews related studies in tour recommendation, especially recommendation with functionalities and machine learning based methods
    - Section III formulates the problem precisely.
    - Section IV presents the proposed MatTrip model, with detailedframework descriptions. 
    - Section V presents the trace datasets and experiment results. 
    - Section VI concludes the paper andshows our future directions.

2. 发现的可修改点、审稿意见 再读
    
    - * 加建模： 解释建模动机（虽然其实顺序不是这样）。 有一个重构的想法，把几个functionality直接提到问题建模里面去，要求设计模型能满足这些个条件
        - weather dependency
        - POI opening hours
        - restricted sequence length
        - mandatory POIs
        - dynamic route revision
    逻辑上： 首先论证，旅游推荐的路线推荐，为什么需要这些功能（查文献，游客通常需要些什么。也可结合自己的旅行经验来写）
    - * 补一个相关工作实现functionality的列表 （很多文章都是空的，除了DTRP），
    - * related work 部分增补关于现有 数学优化模型 方向工作的缺陷分析： 计算复杂，人工特征不能覆盖所有信息... 写作时注意引经据典。
    - * related work II.B 第二段增补近年工作用到了他人信息的，但是也存在冷启动、数据稀疏等问题。

3. 工作思路整理
    - 先着手从小处改起来，写文章“登门槛”
    - 两个关键点 implicit information，张嘉乐去探究，继续看问题建模和深度网络部分； trip planning functionalities 马鸣谦去拆，按2中修改点中related work部分增补



### 10-11交流记录
- 1. 整理审稿意见,见共享腾讯文档。 仔细梳理文章内容，通过解答问题把文章技术内容理清楚。
- 2. 写作， 
    - 数据集处理的细节没有写清楚。 可以归纳之前做的数据工作。马鸣谦
    - Semantic到底什么意思，和NLP不一样。 说清楚
    - User interest不应该用RNN，是不是CNN更好？ 甚至图神经网络。 原因。 Preference Vector
    - 实验： 显著性检验（同类）
- 4. 增加论证：比如RNN抽取特征，找一找这样做的文献。(RNN路径推荐， RNN时序预测）  RNN建模能力较强，查一下Bi-LSTM的好处。  

进一步工作： 各种技术整理一下。
    - 问题，baseline。（推荐系统相关的参考文献不足） 马鸣谦
    - 技术（Decoder和 Grid Beam Search 连接的细节，技术到底怎么回事。Dynamic Revision很有兴趣， Online Learning。如果使用GBS，能否实现改正错误预测点的情况下，继续预测后面的路径（即允许用户修正）  张嘉乐

### 9-27 具体的文章写作点
增补写作的方法
1. 重述文章，类似于重新做会议报告。 找到原文中缺少的部分   张嘉乐
2. 发掘参考文献和文章内容的关联点。（再读扫文表中标出的重点文章，找到与本文的关联点，引上） 马鸣谦 张嘉乐
3. 阅读审稿意见，找一些细节修改点   
    - 1. Symbol Table整一下
    - 2. 实验结果的解读阐释
4. **实验部分还不能写**：首先是整体故事逻辑不明（张嘉乐再作重述，提炼整体逻辑）。 另外想要证明的事情太多，有些没想好怎么证明，有些是证明出来并不那么好。还有些是baseline不能用来比（问题定义不太一样）
    - ① 原来的整条路径与历史的匹配度 （baseline没找齐，最好能有更合适的。 DTRP还没有要到代码，可能需要自己写。）  
    - ② 能够实现functionality  （有效果，不知道如何量化）
    - ③ next POI （已经证明不如专业的next POI模型）
对策：充分整理已有文献的价值（甚至可以在细节处直接复制），写作时量米下锅。 更合适的baseline随缘遇，遇不到就算了。

9-20 
1. 关于实验   马鸣谦
    - next POI： 因此要用正确标签修正预测的，在generation.py的118行处改下个输入
    - 标准： 分预测的点数是路线的第几个（第三，第五，第十这样，因为点数不同影响剩余点数，对准确率影响较大）， 与nextPOI的其他baseline比较
2. 模型修改    张嘉乐
    - 好实现 √, 效果：应该一般
    - 论证：（转刊版在文章中的technical difference）   需要三部分。  1 motivation: 阐述要用的原因（现实情况需要增加非相邻点之间影响） 
       2 使用的依据（参考文献）   3 实验效果的分析，结合实际说明原因  （在LSTPM文章都有一些，应用的RNN2017需要看一下，扩充，写进文章里面）  
3. 序列预测
    - baseline还没有：暂定2个，  有一个LP（44号文献，不一定能跑起来），+ LSTM (已有，代码中的learning tour)
    - 评价指标：原来的F1 score,  edit distance  √
    - Functionality 实现评价： 预测100次看must visit被成功访问的百分比这个实验是不是要做？（暂时不做，等文章逻辑出来之后，可以锦上添花）

增加页数
- 实验部分: 作图、加表、 解释、 增加评价指标(不少文字)
- 模型变化（加公式、图片、伪代码）
- related work逻辑卡住了，还要看主逻辑


9-13 工作小结
1. 改刊的技术差别：  geodialated ☆  （重点改这个）     张嘉乐  看文章解释性+改代码实现
        - 优点： 有时候影响是由跳步的点决定的。  LSTPM第4页， 还要参看以下Dilated RNN文章
        - 工作
            - 训练时，序列采样部分增加跳点采样的样本。 生成是不变
            （找一下代码里面采样的部分在哪里，改对应部分）
2. 改代码，将原模型改成支持NextPOI的方法：  将训练好的encoder, decoder拿出来，不用GBS直接输出  马鸣谦 写代码
（先做这两个，周中有进展了随时交流）
3. 继续在这次整理的数据集上，跑baseline的实验测试。 关于标准， nextPOI(基线齐全)实验，跑acc-x 和 MAP/NDCG； 序列生成实验（基线很烂还要更新），跑F1-score   马鸣谦
4. 在这次整理的数据集上，跑本文改过的模型的测试，标准同3.    张嘉乐


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

