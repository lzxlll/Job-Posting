# 招聘启事
原始职位发布数据的观察总数为 140,107,287。

该项目旨在：

1. 识别并删除重复/缺失值的职位空缺发布 (job_group_test.py)。
2. 提取1000个最频繁的职位发布标题，然后手动映射到标准的中文职业代码（job_group_test.py）。
3. 将发布职位名称映射到中文职业分类和代码（GB/T 6565-2015）（连续词袋模型（CBOW））。
4. 通过比较手动映射的 1000 个最常见职位发布与算法映射来评估映射的质量。
5. 从职位描述中提取信息。 

## ***匹配过程***

** 1. 识别并删除重复/缺失值的职位空缺发布**
我们首先删除缺少职位信息（obs = ?）的观测值，这占总观测值的 ?%。 我们有以下信息：
| Variable | Format |
| --- | --- |
|招聘主键ID  | bigint(20)|
|公司ID  | bigint(20)|
|公司名称 | varchar(255)|
|城市名称 | varchar(255)|
|公司所在区域 | varchar(255)|
|工作薪酬 | varchar(255)|
|教育要求 | varchar(255)|
|工作经历 | varchar(255)|
|工作描述 | varchar(255)|
|职位名称 | varchar(255)|
|工作名称 | varchar(255)|
|招聘数量 | varchar(255)|
|发布日期  | datetime|
|行业名称 | varchar(255)|
|来源 | varchar(255)|