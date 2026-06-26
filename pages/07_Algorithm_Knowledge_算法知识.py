"""
算法知识学习模块
提供机器学习算法的理论知识讲解
包含算法原理、应用场景、优缺点分析等内容
"""

import streamlit as st

# 设置页面配置
st.set_page_config(page_title="算法知识学习", page_icon="📚", layout="wide")

# 页面标题和介绍
st.title("📚 机器学习算法知识库")
st.markdown("> 深入理解机器学习算法的原理、应用和特点")

# 侧边栏导航
st.sidebar.title("📖 算法导航")
algorithm_choice = st.sidebar.selectbox(
    "选择要学习的算法",
    ["线性回归", "KNN分类", "K-Means聚类", "神经网络", "机器学习基础"]
)

# 线性回归知识模块
if algorithm_choice == "线性回归":
    st.header("📈 线性回归算法")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 算法原理")
        st.markdown("""
        **线性回归**是一种用于预测连续值的监督学习算法。
        
        ### 核心思想
        - 寻找特征与目标变量之间的线性关系
        - 通过最小化预测值与真实值的差距来优化模型
        - 使用梯度下降等优化算法找到最佳参数
        
        ### 数学模型
        $$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$
        
        - $y$: 预测的目标变量
        - $x_i$: 输入特征
        - $w_i$: 权重参数
        - $b$: 偏置项
        """)
        
        st.subheader("🎯 应用场景")
        st.markdown("""
        - **房价预测**: 根据房屋面积、位置等特征预测价格
        - **销售预测**: 基于历史数据预测未来销售额
        - **趋势分析**: 分析变量间的线性关系
        """)
    
    with col2:
        st.subheader("⚖️ 优缺点分析")
        
        st.info("**优点：**")
        st.markdown("""
        - 简单易懂，计算效率高
        - 可解释性强
        - 理论基础扎实
        """)
        
        st.warning("**缺点：**")
        st.markdown("""
        - 假设特征与目标是线性关系
        - 对异常值敏感
        - 无法处理复杂非线性关系
        """)
        
        st.subheader("📊 关键参数")
        st.markdown("""
        - **学习率**: 控制参数更新步长
        - **迭代次数**: 训练轮数
        - **正则化**: 防止过拟合
        """)

# KNN分类知识模块
elif algorithm_choice == "KNN分类":
    st.header("🌸 K-最近邻分类算法")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 算法原理")
        st.markdown("""
        **KNN**是一种基于实例的学习算法，通过多数投票原则进行分类。
        
        ### 核心思想
        - "物以类聚"：相似的数据点应该属于同一类别
        - 根据待分类样本的K个最近邻的类别进行投票
        - 距离度量决定"相似性"的标准
        
        ### 工作流程
        1. 计算待分类样本与所有训练样本的距离
        2. 选择距离最近的K个样本
        3. 统计K个样本中各类别的数量
        4. 将待分类样本归为数量最多的类别
        """)
        
        st.subheader("🎯 应用场景")
        st.markdown("""
        - **图像识别**: 手写数字、人脸识别
        - **推荐系统**: 基于用户相似性的推荐
        - **医疗诊断**: 疾病分类和预测
        """)
    
    with col2:
        st.subheader("⚖️ 优缺点分析")
        
        st.info("**优点：**")
        st.markdown("""
        - 简单直观，易于实现
        - 无需训练过程（惰性学习）
        - 对数据分布没有假设
        """)
        
        st.warning("**缺点：**")
        st.markdown("""
        - 计算复杂度高（需要存储所有数据）
        - 对K值选择敏感
        - 对不平衡数据敏感
        """)
        
        st.subheader("📊 关键参数")
        st.markdown("""
        - **K值**: 最近邻的数量
        - **距离度量**: 欧氏距离、曼哈顿距离等
        - **权重**: 统一权重或距离权重
        """)

# K-Means聚类知识模块
elif algorithm_choice == "K-Means聚类":
    st.header("🍰 K-Means聚类算法")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 算法原理")
        st.markdown("""
        **K-Means**是一种无监督学习算法，将数据划分为K个簇。
        
        ### 核心思想
        - 将相似的数据点聚集到同一簇中
        - 不同簇的数据点尽可能不同
        - 通过迭代优化簇中心和簇分配
        
        ### 工作流程
        1. 随机选择K个初始簇中心
        2. 将每个数据点分配到最近的簇中心
        3. 重新计算每个簇的中心点
        4. 重复步骤2-3直到收敛
        """)
        
        st.subheader("🎯 应用场景")
        st.markdown("""
        - **客户细分**: 根据消费行为对客户分组
        - **图像分割**: 将图像像素分成不同区域
        - **异常检测**: 识别与其他数据不同的点
        """)
    
    with col2:
        st.subheader("⚖️ 优缺点分析")
        
        st.info("**优点：**")
        st.markdown("""
        - 简单高效，易于实现
        - 适用于大规模数据集
        - 结果可解释性强
        """)
        
        st.warning("**缺点：**")
        st.markdown("""
        - 需要预先指定K值
        - 对初始中心点敏感
        - 只能发现球状簇
        """)
        
        st.subheader("📊 关键参数")
        st.markdown("""
        - **K值**: 簇的数量
        - **初始化方法**: K-Means++或随机初始化
        - **最大迭代次数**: 防止无限循环
        """)

# 神经网络知识模块
elif algorithm_choice == "神经网络":
    st.header("🧠 神经网络算法")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 算法原理")
        st.markdown("""
        **神经网络**是一种模仿人脑神经元连接方式的深度学习算法。
        
        ### 核心思想
        - 通过多层非线性变换学习复杂模式
        - 前向传播计算预测值
        - 反向传播调整参数
        - 梯度下降优化损失函数
        
        ### 网络结构
        - **输入层**: 接收原始数据
        - **隐藏层**: 进行特征提取和变换
        - **输出层**: 产生最终预测结果
        """)
        
        st.subheader("🎯 应用场景")
        st.markdown("""
        - **图像识别**: 人脸识别、物体检测
        - **自然语言处理**: 机器翻译、情感分析
        - **语音识别**: 语音转文字、语音合成
        - **推荐系统**: 个性化内容推荐
        """)
    
    with col2:
        st.subheader("⚖️ 优缺点分析")
        
        st.info("**优点：**")
        st.markdown("""
        - 强大的表达能力
        - 自动特征学习
        - 适用于复杂任务
        - 端到端学习
        """)
        
        st.warning("**缺点：**")
        st.markdown("""
        - 需要大量数据
        - 计算资源要求高
        - 可解释性差
        - 容易过拟合
        """)
        
        st.subheader("📊 关键参数")
        st.markdown("""
        - **网络层数**: 深度影响模型能力
        - **神经元数量**: 宽度影响模型容量
        - **学习率**: 控制参数更新步长
        - **激活函数**: 引入非线性变换
        """)

# 机器学习基础知识模块
else:
    st.header("🤖 机器学习基础概念")
    
    tab1, tab2, tab3 = st.tabs(["基本概念", "学习类型", "评估指标"])
    
    with tab1:
        st.subheader("🔤 基本术语")
        
        st.markdown("""
        ### 特征 (Feature)
        描述数据的属性或变量，如房屋的面积、位置等。
        
        ### 标签 (Label)
        要预测的目标变量，如房屋的价格、疾病的类型等。
        
        ### 训练集 (Training Set)
        用于训练模型的数据集，包含特征和对应的标签。
        
        ### 测试集 (Test Set)
        用于评估模型性能的独立数据集。
        
        ### 过拟合 (Overfitting)
        模型在训练集上表现很好，但在新数据上表现差。
        
        ### 欠拟合 (Underfitting)
        模型在训练集和新数据上都表现不佳。
        """)
    
    with tab2:
        st.subheader("📚 学习类型")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 监督学习
            **特点**: 有标签数据
            **任务**: 分类、回归
            **示例**: 
            - 垃圾邮件分类
            - 房价预测
            """)
        
        with col2:
            st.markdown("""
            ### 无监督学习
            **特点**: 无标签数据
            **任务**: 聚类、降维
            **示例**: 
            - 客户细分
            - 异常检测
            """)
        
        with col3:
            st.markdown("""
            ### 强化学习
            **特点**: 智能体与环境交互
            **任务**: 决策优化
            **示例**: 
            - 游戏AI
            - 机器人控制
            """)
    
    with tab3:
        st.subheader("📊 评估指标")
        
        st.markdown("""
        ### 分类任务指标
        - **准确率**: 正确分类的样本比例
        - **精确率**: 预测为正例中真正正例的比例
        - **召回率**: 真正正例中被正确预测的比例
        - **F1分数**: 精确率和召回率的调和平均
        
        ### 回归任务指标
        - **均方误差**: 预测值与真实值差的平方的平均
        - **R²分数**: 模型解释的方差比例
        
        ### 聚类任务指标
        - **轮廓系数**: 衡量聚类效果的指标
        - **惯性**: 样本到簇中心的距离平方和
        """)

# 学习建议和练习
st.markdown("---")
st.header("💡 学习建议")

col_advice1, col_advice2 = st.columns(2)

with col_advice1:
    st.markdown("""
    ### 📝 理论学习
    1. **理解原理**: 先理解算法的数学原理
    2. **掌握概念**: 熟悉关键术语和概念
    3. **对比分析**: 比较不同算法的优缺点
    4. **应用场景**: 了解算法的适用场景
    """)

with col_advice2:
    st.markdown("""
    ### 🔧 实践练习
    1. **参数调节**: 尝试不同参数组合
    2. **可视化观察**: 通过图表理解算法行为
    3. **故障诊断**: 故意设置问题并解决
    4. **项目应用**: 将算法应用到实际问题
    """)

# 常见问题解答
st.markdown("---")
st.header("❓ 常见问题解答")

with st.expander("如何选择适合的机器学习算法？"):
    st.markdown("""
    选择算法时考虑以下因素：
    - **问题类型**: 分类、回归、聚类还是其他
    - **数据规模**: 数据量大小和特征数量
    - **数据质量**: 是否有缺失值、异常值
    - **可解释性**: 是否需要理解模型决策
    - **计算资源**: 可用的计算能力和时间
    """)

with st.expander("什么是过拟合？如何避免？"):
    st.markdown("""
    **过拟合**是指模型在训练数据上表现很好，但在新数据上表现差。
    
    **避免方法**:
    - 增加训练数据量
    - 使用正则化技术
    - 简化模型复杂度
    - 使用交叉验证
    - 早停法（Early Stopping）
    """)

with st.expander("梯度下降为什么重要？"):
    st.markdown("""
    **梯度下降**是优化机器学习模型的核心算法：
    - **作用**: 通过迭代找到损失函数的最小值
    - **原理**: 沿着梯度反方向更新参数
    - **变体**: 批量梯度下降、随机梯度下降、小批量梯度下降
    - **关键**: 学习率的选择影响收敛速度和稳定性
    """)

# 学习资源推荐
st.markdown("---")
st.header("📚 推荐学习资源")

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.markdown("""
    ### 📖 书籍推荐
    - **《机器学习》** - 周志华
    - **《统计学习方法》** - 李航
    - **《Python机器学习》** - Sebastian Raschka
    - **《深度学习》** - Ian Goodfellow
    """)

with col_res2:
    st.markdown("""
    ### 🌐 在线资源
    - **Coursera**: 吴恩达机器学习课程
    - **Kaggle**: 实践项目和竞赛
    - **GitHub**: 开源代码和项目
    - **官方文档**: scikit-learn、PyTorch文档
    """)

# 核心代码示例
st.markdown("---")
st.header("💻 核心代码示例")

code_tab1, code_tab2, code_tab3, code_tab4 = st.tabs(["线性回归", "KNN分类", "K-Means聚类", "神经网络"])

with code_tab1:
    st.markdown("### 📈 线性回归核心代码")
    st.code("""# 梯度下降实现
def gradient_descent(X, y, lr=0.01, epochs=100):
    w, b = 0.0, 0.0  # 初始化参数
    n = len(X)
    
    for epoch in range(epochs):
        # 前向传播
        y_pred = w * X + b
        
        # 计算损失
        loss = np.mean((y_pred - y)**2)
        
        # 计算梯度
        dw = (2/n) * np.dot(X, (y_pred - y))
        db = (2/n) * np.sum(y_pred - y)
        
        # 参数更新
        w = w - lr * dw
        b = b - lr * db
        
    return w, b, loss
""", language="python")
    
    st.markdown("**代码讲解：**")
    st.markdown("""
    - **参数初始化**: 权重w和偏置b初始化为0
    - **前向传播**: 计算预测值 y_pred = w*X + b
    - **损失计算**: 使用均方误差衡量预测准确性
    - **梯度计算**: 对w和b分别求偏导数
    - **参数更新**: 沿着梯度反方向更新参数
    """)

with code_tab2:
    st.markdown("### 🌸 KNN分类核心代码")
    st.code("""# KNN分类实现
def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    
    for test_point in X_test:
        # 计算距离
        distances = []
        for i, train_point in enumerate(X_train):
            dist = np.linalg.norm(test_point - train_point)  # 欧氏距离
            distances.append((dist, y_train[i]))
        
        # 选择最近的k个邻居
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # 多数投票
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        # 预测结果
        prediction = max(votes.items(), key=lambda x: x[1])[0]
        predictions.append(prediction)
    
    return predictions
""", language="python")
    
    st.markdown("**代码讲解：**")
    st.markdown("""
    - **距离计算**: 使用欧氏距离衡量样本相似性
    - **邻居选择**: 选择距离最近的k个训练样本
    - **多数投票**: 根据k个邻居的标签进行投票
    - **预测结果**: 选择票数最多的类别作为预测
    """)

with code_tab3:
    st.markdown("### 🍰 K-Means聚类核心代码")
    st.code("""# K-Means聚类实现
def kmeans_clustering(X, k=3, max_iters=100):
    # 随机初始化簇中心
    centers = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        # 分配样本到最近的簇
        labels = []
        for point in X:
            distances = [np.linalg.norm(point - center) for center in centers]
            labels.append(np.argmin(distances))
        
        # 更新簇中心
        new_centers = []
        for i in range(k):
            cluster_points = X[np.array(labels) == i]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(centers[i])
        
        # 检查收敛
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return labels, centers
""", language="python")
    
    st.markdown("**代码讲解：**")
    st.markdown("""
    - **初始化**: 随机选择k个样本作为初始簇中心
    - **分配步骤**: 将每个样本分配到最近的簇中心
    - **更新步骤**: 重新计算每个簇的中心点
    - **收敛判断**: 当簇中心不再变化时停止迭代
    """)

with code_tab4:
    st.markdown("### 🧠 神经网络核心代码")
    st.code("""# 简单神经网络实现
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        # 前向传播
        self.z1 = X @ self.w1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU激活
        self.z2 = self.a1 @ self.w2 + self.b2
        return self.z2
    
    def backward(self, X, y, lr=0.01):
        # 反向传播
        m = len(X)
        dz2 = (self.z2 - y) / m
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU导数
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # 参数更新
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
""", language="python")
    
    st.markdown("**代码讲解：**")
    st.markdown("""
    - **网络结构**: 输入层 → 隐藏层 → 输出层
    - **前向传播**: 线性变换 + 激活函数
    - **反向传播**: 计算梯度并更新参数
    - **激活函数**: ReLU引入非线性
    """)

# 学习建议
st.markdown("---")
st.success("💡 **学习建议**: 理论与实践相结合，通过可视化工具加深理解，多动手实践才能真正掌握机器学习算法。")

st.caption("🎓 本知识库旨在帮助高职学生系统学习机器学习算法，建议按照顺序学习并完成相应的实践练习。")