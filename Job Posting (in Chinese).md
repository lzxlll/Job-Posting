# 招聘数据
原始职位发布数据的观察总数为 140,107,287。

该项目旨在：

1. 将庞大的原始文本数据文件拆分为 142 个文件，每个文件包含 1,000,000 个在线职位发布 (job_posting_subset.ipynb)。 识别并删除重复/缺失值的职位空缺发布 (job_group_test.py)。
2. 提取1000个最频繁的职位发布标题，然后手动映射到标准的中文职业代码（job_group_test.py）。
3. 将发布职位名称映射到中文职业分类和代码（GB/T 6565-2015）（连续词袋模型（CBOW））。
4. 通过比较手动映射的 1000 个最常见职位发布与算法映射来评估映射的质量。
5. 从职位描述中提取信息。 

## 匹配过程

### 1. 识别并删除重复/缺失值的职位空缺发布
我们首先删除缺少职位信息（obs = ?）的观测值，这占总观测值的 ?%。 我们有以下信息：

| Variable | Format | Variable | Format | Variable | Format |
| --- | --- |  --- | --- |   --- | --- |
|招聘主键ID  | bigint(20)| 工作薪酬 | varchar(255)| 工作名称 | varchar(255)|
|公司ID  | bigint(20)| 教育要求 | varchar(255)| 招聘数量 | varchar(255)|
|公司名称 | varchar(255)| 工作经历 | varchar(255)|发布日期  | datetime|
|城市名称 | varchar(255)| 工作描述 | varchar(255)|行业名称 | varchar(255)|
|公司所在区域 | varchar(255)| 职位名称 | varchar(255)|来源 | varchar(255)|

其次，我们按照一下标准删除重复的职位发布：我们按日期升序对具有相同公司 ID、公司位置和职位名称的观察进行排序。重复的职位发布是指具有相同公司ID和公司位置并在同一月份发布的职位信息。 这个过程删除了 ？个观测值这相当于总观察值的 ?%。

:tada: **统计数据**
- [ ] 由于公司 ID、公司位置、发布日期、职位和职位描述中的 N/A 而减少的观测值？
- [ ] 由于职位发布重复而减少的观测值？
- [ ] 发布日期的频率分布是什么，按年-日？
- [ ] 职位发布的频率分布按照来源分类是怎样的？ 



### 2. 提取 1000 个最常见的职位发布标题
为了确定 1000 个最常见的职位发布标题，我们按职位对这些职位发布进行分组。 然后计算每个组内的频率。 有 28,949,287 个观察共享 1000 个最常见的职位发布标题，占总观察的 ?%。



### 3. 将网络标题映射到中文职业分类和代码（GB/T 6565-2015）
中国职业分类与代码（GB/T 6565-2015）包含4个级别的职业分类：一般（8组）、中等（66组）、细部（413组）和1838个职业。

对于每个职业，都附有工作描述和定义。 对于我们网络发布数据中的每个职位 t，我们计算了 t 与中国职业分类和代码（GB/T 6565-2015）中出现的所有职位 τ 之间的相似度。对于每个标准化职位 τ，我们观察一个 GB/T 职业代码。 对于职位 t，我们将标准化职位 τ 的 GB/T 职业代码赋给 t。 在我们的网络职位发布数据中至少出现两次的任何职位我们通过算法进行匹配。


### 4. 算法匹配有效性检查
我们依靠比较手动映射和算法映射的结果来验证有效性。在`提取 1000 个最常见的职位发布标题`步骤中，我们按频率对网络标题进行排名，并选择前 1000 个作为手动映射的候选对象。我们使用人类知识，仅依靠“职位”信息将前 1000 名网络职位映射到中国职业分类和代码（GB/T 6565-2015）[our_chinese_mapping.xlsx](https://github.com/lzxlll/Job-Posting/files/7537941/our_chinese_mapping.xlsx)。人工手动匹配的频率最高的前1000个职位的匹配结果：[top1000_manual_mapping.xlsx](https://github.com/lzxlll/Job-Posting/files/7783017/top1000_manual_mapping.xlsx)。
 
这种有效性检查依赖于两个假设：（i）基于人类知识的手动匹配是最精确的。 (ii) 网络职位发布的标题和描述应匹配。也就是说，职位名称“计算机工程师”应该在职位描述中包含“计算机工程师”相关信息，而不是其他任意描述。

:tada: 统计数据

我们将这个手动匹配结果与《将网络帖子标题映射到中国职业分类和代码（GB/T 6565-2015）》中基于算法的结果进行如下比较：

- 对于每个排名前 1000 的网络职位，我们检查基于算法的映射中出现频率最高的匹配与手动匹配具有相同匹配结果的百分比。更具体地说，我们在上面提到过`28,949,287 个观察共享 1000 个最常见的职位发布标题`。

  假设有 1,000 个职位的网络职位名称为“销售代理”，我们在 GB/T 6565-2015 中手动将其匹配为“销售员”。另一方面，通过算法，30% 的“销售代理”可以匹配到“老师”，20% 可以匹配到“厨师”，50% 可以匹配到“推销员”。那么，在这种情况下，“销售员”是出现频率最高的匹配，同时它与手动映射结果匹配。我们计算所有排名前 1000 的网络职位的手动与算法相同的百分比。

- 我们比较了基于手动映射和基于算法映射的 28,949,287 个观测值的频率分布。这就是，我们比较了基于匹配和基于算法算法匹配的结果的 GB/T 6565-2015 标题频率分布。理想情况下，两个分布应该共享相似的模式。


<hr />

## 提取信息过程

我们通过搜索关键词来提取网络职位描述中的信息。我们的关键词文件[our_chinese_mapping.xlsx](https://github.com/lzxlll/Job-Posting/files/7537944/our_chinese_mapping.xlsx)
包括三个电子表格（技术、字符、O*NET），三个电子表格的格式相同，包括：`工作特征`，以及`我们的判断`。`工作特征`说明关键词的类型，`我们的判断`为关键词列表。 对于每个`工作特征`，因为性格、财务能力、问题管理能力以及ONET的原始语言为英语，我们首先使用Google translator将每一个特征翻译为中文。然而这将导致产生多个同义词，因而大大缩减特征词的数量。针对这一问题，我们使用“万磁王”（https://wantwords.thunlp.org/home/） 扩充相似词汇（https://github.com/thunlp/WantWords.git）。

`工作特征`包括以下三类：

- 使用不同的电脑技术（例如，Microsoft Word、Python、Matlab）；
- 性格、财务能力、问题管理能力等；
- O*NET 工作风格、技能、知识要求和活动。


我们的目的是统计关键字在描述中的出现次数。通过这样做，我们将文本描述分解为一组虚拟变量。例如，在我们编译的关键字文件中，有 67 个与技术相关的工作特征。然后，我们创建了 67 列，每列对应一个技术关键字。我们计算每个关键字的外观并为每个观察值（职位描述）填写单元格。



