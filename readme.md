# Few-shot Learning Algorithms and Practice with Python

Digital edition of Learning Algorithms and Practice with Python

<!-- PROJECT SHIELDS -->

[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->

<br />

<p align="center">
  <a href="https://github.com/dingjicai/Book_fewshot_learning">
    <img src="cover.jpg" alt="Logo" width="100" height="120">
  </a>

<h3 align="center">小样本机器学习Python算法与实践</h3>
  <p align="center">
    Few-shot Learning Algorithms and Practice with Python
    <br />

</p>

本书纸质版由石油工业出版社出版

## 目录

- [Few-shot Learning Algorithms and Practice with Python](#few-shot-learning-algorithms-and-practice-with-python)
  - [目录](#目录)
    - [文件目录说明](#文件目录说明)
    - [环境配置建议](#环境配置建议)
    - [作者的话](#作者的话)


### 文件目录说明

本书共分七章，分别是小样本问题与机器学习综述、数据预处理和分析、数据增强、 传统机器学习算法、不完全监督学习、迁移学习和元学习。其中第一章主要涉及小样本问题总体解决思路和机器学习框架，第二章针对机器学习数据预处理和分析，第三章主要包括传统数据增强和基于深度学习的数据增强，第四章集中介绍传统机器学习领域解决小样本问题的常用算法，第五章至第七章集中介绍深度学习领域小样本算法。

```
filetree 
├── LICENSE
├── README.md
└── /code
  ├── /code/Chapter_01 - 小样本学习和机器学习综述
  ├── /code/Chapter_02 - 数据预处理和分析
  ├── /code/Chapter_03 - 数据增强
  ├── /code/Chapter_04 - 传统机器学习
  ├── /code/Chapter_05 - 不完全监督学习
  ├── /code/Chapter_06 - 迁移学习
  └── /code/Chapter_07 - 元学习
```

### 环境配置建议

1. tensorflow   2.5.0
2. pytorch      1.11.0
3. Paddlepaddle 2.3.2
4. Scikit-learn 1.0.2


### 作者的话

第一次接触机器学习是1997年（华东）石油大学做本科毕业设计，题目是利用BP网络自动拾取地震波初至（地震波到达时），耗时几个月用Fortran语言（4000多行代码）实现了简单的BP网络，最终效果是在人工合成数据上获得95%以上的精度。在当时这是一个比较令人振奋的结果了。对比20多年后的今天，完成上述本科毕业设计可能只需要几天时间，少于100行的python代码，而在实际资料上获得很好的结果。机器学习尤其深度学习发展速度令人瞠目结舌。

第一次在企业或高校做人工智能相关培训是2019年左右。目标是培训深度学习入门。但培训效果并不理想，主要原因是没有一本合适的教材。当下关于机器学习的书籍很多，以翻译作品为主，翻译质量也参差不齐。编制一本学员手册迫在眉睫，手册编制不局限于某一种机器学习平台，不严格区分传统机器学习和深度学习，并且所有资源是开放共享的，适合包括高校和科研单位的工作者使用。本书正是在培训手册基础上不断丰富形成的。

笔者近些年一直在能源领域辛勤耕耘，有别于“学术派”人工智能，一直追求以深度学习落地为奋斗目标。作者深深的感触到，在标注数据足够多的前提下，深度学习的确展示出了无穷魅力，但是更多的残酷事实是能源领域并没有充足的标注数据“喂饱”深度学习算法。反观互联网领域，如图像、声音和语言等方向，人工智能如日中天，的确有一种“热闹是他们的，我什么也没有”（朱自清的散文《荷塘月色》）的凄凉。数据、场景、算力、算法“四位一体”是人工智能成熟的必备因素。实际上，不仅仅是能源行业，很多其它领域开展人工智能研究同样面临着标注数据不足的问题。小样本问题首先是数据问题，其次是算法问题。针对小样本数据进行分析和运用是本书的主要目标。

基于以上等诸多因素，作者立志写一本目前行业内还未见到相关专题的书籍，以期许能对目前奋斗在人工智能领域或者即将进入人工智能领域的科研工作者有所帮助。

感谢中海油的同事。

在写书过程中拜读了许多国内外人工智能工作者文章和书籍，受益匪浅，对人工智能科研工作者同行深表感谢。

在写书过程中，参阅了很多人工智能大神们在github、博客、知乎、百度文库、简书等媒介上的文章和代码，深受启发。

在机器学习研究过程中，还从很多优秀的网络课程上获得给养，如吴恩达老师、台大教授李宏毅老师、菜菜TsaiTsai 老师等机器学习课程，课程精彩绝伦，对本书的很多内容起到了不可替代的作用。

天下人工智能工作者是一家，作为这个大家庭中的普通一员，出版本书仅仅是想为人工智能的发展贡献一份微薄之力，如果有侵犯版权等问题出现，请第一时间联系作者，在此深表歉意，并在后续出版过程中进行修正。
<br />

<p align="center">
  <a href="https://github.com/dingjicai/Book_fewshot_learning">
    <img src="findme.jpg" alt="Logo" width="100" height="100">
  </a>
<h4 align="center">更多内容请关注公众号：AI行在路上</h4>
</p>