背景说明
- 旅游推荐，见论文，略


文件说明
版本 torch.__version__    '1.3.1+cpu'
（高版本可能导致测试时报错）


- 1.   train.py:  定义了训练的函数，并在主函数中使用。 对其他文件的调用关系为
	- data.py : 读取数据输入并且统一格式的函数
	- model.py ： 定义网络结构的函数 


- 2. plot.py: 画出在地图上生成的路径
	- generator.py
	- generator2.py
	- generation.py ： 更新时间最晚（20年8月）


- 3. evaluation.py : 测试的函数，更新时间更晚一些，应该是这个   
- 4. __evaluation.py
