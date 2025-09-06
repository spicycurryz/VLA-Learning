# Robocasa

## Introduction
这篇文章总结的三个主要贡献：
1. 提出了一个simulation framework Robocasa,用于生成多样真实的厨房场景，用AI工具生成纹理和3D物品
2. 设计了100个任务，前25个是原子任务，后75个是在LLM帮助下生成的复合长流程任务
3. 提供了大规模多任务数据集，包含了大量合成的轨迹

## Related Work

1. **Simulation Frameworks for Robotics**
   - 别的现有的一些框架存在一些缺点：仅支持桌面操作/场景规模太小/缺乏真实物理模拟。Robaocasa在这些方面都有优势，同时还拥有照片级渲染，支持多种不同类型的机器人（通用性更强）
   - 现有框架要么规模大但无生成式AI，要么有生成式AI但规模小,Robocasa是only one两者兼顾的框架，理论上可以实现“无限多样性”
   - 现有一些框架需要用户自行收集数据，Robocasa直接提供大规模演示数据集
2. **Datasets and Benchmarks for Robotics**
   1. **Datasets**:
      1. 自监督学习：反复试错，耗时长，数据质量受到“试错策略”的影响
      2. 人工遥操作：数据质量高，但成本高，复现难
      3. 算法生成轨迹：预编程生成数据“依赖特权信息”和“手工式启发”（也就是迁移性比较差），利用LLM也需设计流程，人力成本比较高
      4. Robocasa的方案：人工遥感收集少量高质量演示数据，再用MimicGen将这些演示适配到新场景
   2. **Learning from Large Offline Datasets**
      1. Behavior Clonning(BC)
      2. Offline RL:可从低质量数据集中学习，但奖励函数难以设计
      3. Robocasa具体学习方案：Transformer框架，offline datasets

## Robocasa Simulation

1. Core Simulation Platform:Robosuite
   1. Robosuite三大优势
      1. 物理真实度高
      2. 仿真速度快
      3. 模块化设计
   2. 继承了Robosuite的核心组件，新增了对移动机器人的支持
   3. 使用了高质量的渲染方案，解决传统仿真中“视觉失真”的问题
2. Kitchen Scenes
   1. 这一场景任务复杂交互多
   2. 设计的核心准则之一是全交互(fully interactive)
   3. kitchen布局设计参考了家居杂志，避免偏离失真，同时保证多样化避免过拟合
   4. 考虑多种风格和户型，给予不同的纹理，颜色等等，共计120个kitchen场景。除此以外，还利用AI工具生成了多种纹理，以随机选择纹理组合的形式保证数据集的visual diversity;户型方面也涵盖了从基础到高端的各类，确保机器人适应不同空间约束下的操作
3. Assets
   1. Interactable Furniture and Appliances
      1. 从各处收集一些资产基础模型，并转换成MuJoCo支持的MJCF格式
      2. 对资产进行后处理的核心要素是把整体设备拆分成带关节的实体
      3. 在此基础上还有对电器功能逻辑上的模拟
   2. Kitchen Objects
      1. 物体库来自开源数据集和Luma.ai文本to3D的生成，涵盖厨房里会出现的多种类型
      2. 对objects进行了筛选，最终得到2509 个高质量物体，覆盖 153 个类别，其中1592 个来自 Luma.ai

## Robocasa Activity Dataset

1. **Atomic Tasks:Building Blocks of Behavior**
   - 8大核心“感知-运动功能”
   - Atomic Tasks设计的逻辑：每个任务聚焦一种核心技能，且包含“不同场景变体”；展望：暂未包含“柔性物体操作”
2. **Creating Composite Tasks with Large Language Models**
   1. 复合任务本质上是把多个原子任务串联起来，完成有语义的完整活动
   2. 尤其需要注意的是“ecological statistics”,也就是考虑任务分布与真实人类活动一致(Eg:在厨房里洗碗的频率显然会显著高于烘焙蛋糕)，同时要避免设计严重脱离现实情境的任务(Eg:将杯子从橱柜里反复拿出和放入)
   3. 使用LLM指导任务生成的原因：LLM基于海量人类知识可以理解人类活动逻辑，耗时短，节省人工
   4. 具体步骤：
      1. step1:prompt GPT-4生成并筛选出20个厨房中的高频、可操作活动
      2. step2:prompt GPT-4和Gemini 1.5为每个活动提出具有代表性的任务，对LLM偶尔的逻辑漏洞进行筛选修正
      3. LLM输出任务蓝图，再代码实现
3. **RoboCasa Datasets**
   1. **Collecting a base set of demonstrations through human teleoperation**:
      人工遥控操作，采集了1250条数据，但发现仍然无法覆盖各项任务和场景，于是决定引入data generation tools
   2. **Leveraging automated trajectory generation methods to synthesize demonstrations**
      使用MimicGen工具自动生成轨迹，核心算法如下:
      - 分解此前的human demonstration为 **"object-centric"** 的一系列操作
         > 这里可以举例理解一下：譬如说把“拾取杯子放入橱柜”分解成：
         > 1. segment1:拾取杯子：机械臂移动到杯子位置→闭合抓手→提起杯子
         > 2. segment2:放入橱柜：机械臂移动到橱柜位置→张开抓手→放下杯子
      - 在新场景中变换上述片段：
         > 同样来举例说明一下：
         > 假设原本的demonstration中杯子在台面右侧，且橱柜为单开门；novel scene中杯子在台面左侧，且橱柜为双开门。那么MimicGen会修改segment1的机械臂路径，修改segment2的打开橱柜的方式
      - 拼接片段生成新的demonstration:按照原先的逻辑顺序拼接

      MimicGen的使用是有assumption的，Robocasa恰好满足适配：
      1. MimicGen要求任务可以被分解成已知的"object-centric" subtasks序列，前面提到，Robocasa设计了8个核心技能，而atom task正是以此为基础建立的，同样的core skill对应的atom task拆解出来的子任务序列其实是一样的，只有reference object有所不同，同样举个简单的例子予以说明：
         > 针对pick and place这个core skill,假设atom_task1是“拾取杯子到橱柜”，atom_task2是“拾取苹果到水槽”，那么这两者的子任务序列均为“拾取subtask1+放置subtask2”,只有reference object(杯子，橱柜，苹果，水池)不一样
         这样显然可以减少人力消耗。

      2. 不仅如此，MimicGen要求提供的human demonstration是有子任务边界标注的（比如说什么时候算抓取结束，什么时候算开始放置等），Robocasa可以通过自动检测的方式完成标注，且只要对8个core skill各标注一次即可
   
      对MimicGen生成的轨迹还需要进行成功条件判断筛选，同时考虑效率采用并行方式生成数据

## Experiments

首先搞清楚本文实验聚焦解决的三个问题内容,这非常重要：

1. MimicGen生成的轨迹对learning multi-task policies效果如何？与人类演示相比呢？
2. 随着训练数据集的扩增，imitation learning policy的泛化性能如何？
3. 大规模仿真数据集能否促进downstream tasks的知识迁移以及真实世界任务的政策学习？

1. **Imitation learning for Atom Tasks**
   1. 实验设计
      1. 这个实验主要聚焦于研究上述第一个问题，实验在四个数据集上展开：Human-50(只有1250条human data),Generated-100,Generated-300,  Generated-3000
      2. 实验training data采用AI生成纹理的图像，evaluation则使用人类精选的纹理。每组实验采用同样的机器人和结构
      3. 政策模型方面采用了RoboMimic中的BC-Transformer
   2. 评估方法
      1. 每个任务在 5 个固定测试场景（不同户型 + 风格）中各运行 50 次，测试用的物体是 “训练中未见过的实例”，且个测试场景中有 2 个是 “训练中未见过的风格”，充分验证了泛化性
   3. 实验结果：生成数据集更优，展现出“规模效应”，技能学习难度存在显著差异（高多样性，高灵活度需求的更难）
2. **Imitation Learning for composite tasks**
   本实验针对复合任务展开，针对每个任务单独设计策略
   1. 实验比较了两种训练设置：
      1. Scratch(从零训练)
      2. Fine-tuning(微调)：在atom task上预训练的模型再在50条human demonstration上微调
   2. 选择了5个复合任务
   3. 实验结果表明，Scratch在成功率方面基本挂零，Fine-tuning则实现了零突破，但性能仍有很大瓶颈。后续有望在policy architecture, learning algorithm, and fine-tuning strategy等方面做出优化

3. **Transfer to Real World Environments**
   1. 首先明确为什么要设计这个实验，是为了证明仿真数据的价值。因为我们最终的目的是要落地现实世界，本实验正是证明了即使在控制器、光照、相机等方面存在差异，仿真数据仍然能提升real world机器人的性能
   2. 基于三个“拾取放置”类任务，同时考虑见过和没见过的物体，发现加入仿真数据相对而言显著提升了任务成功率

## Conclusion

1. 总结了一下RoboCasa这个仿真框架的功能、核心技术与优势
2. 探讨未来的改进方向：
   1. 当前架构在复合任务上效果比较糟糕，未来会研究更强大的策略架构和学习算法，并进一步改进数据集的质量
   2. 对LLM的应用仍然需要一定量的人工参与，考虑能否利用 LLMs 提出数千个新场景与新任务，并在极少人工指导的情况下自动编写代码，实现这些场景与任务的部署
   3. 考虑更广阔的任务场景
   4. 多源数据融合