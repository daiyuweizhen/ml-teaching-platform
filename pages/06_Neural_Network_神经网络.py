"""
神经网络可视化教学工具
专门针对高职学生理解神经网络核心概念
包含前向传播、反向传播、梯度下降、过拟合等可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 延迟导入，提高页面加载速度
@st.cache_resource
def import_pytorch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return torch, nn, optim

@st.cache_resource
def import_sklearn():
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    return make_classification, make_regression, train_test_split, StandardScaler

# 导入必要的库
# 设置页面配置
st.set_page_config(page_title="神经网络可视化", page_icon="🧠", layout="wide")

torch, nn, optim = import_pytorch()
make_classification, make_regression, train_test_split, StandardScaler = import_sklearn()


# 页面标题和介绍
st.title("🧠 神经网络可视化：深度学习核心概念")
st.markdown("> 通过交互式可视化理解前向传播、反向传播、梯度下降等抽象概念")

# 主界面布局
col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("⚙️ 网络配置")
    
    # 网络结构配置
    st.markdown("**📐 网络结构**")
    layer_choice = st.selectbox(
        "网络层数", 
        ["单层感知机", "双层神经网络", "三层神经网络", "自定义层数"],
        help="选择网络深度，观察深度对模型能力的影响",
        key="layer_choice"
    )

    if layer_choice == "自定义层数":
        num_layers = st.slider("隐藏层数量", 1, 5, 2, key="num_layers")
    else:
        num_layers = {"单层感知机": 0, "双层神经网络": 1, "三层神经网络": 2}[layer_choice]

    hidden_units = st.slider("每层神经元数量", 2, 32, 8,
                            help="神经元数量影响模型容量和拟合能力",
                            key="hidden_units")

    # 训练参数配置
    st.markdown("**🎯 训练参数**")
    learning_rate = st.slider("学习率", 0.001, 1.0, 0.1, 0.001,
                             help="控制参数更新步长，过大导致震荡，过小收敛慢",
                             key="learning_rate")
    batch_size = st.slider("批次大小", 8, 128, 32, 8,
                          help="小批次训练更稳定，大批次训练更快",
                          key="batch_size")
    epochs = st.slider("训练轮数", 10, 500, 100, 10,
                      help="训练迭代次数，影响模型收敛程度",
                      key="epochs")

    # 激活函数选择
    activation_func = st.selectbox(
        "激活函数",
        ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU"],
        help="不同激活函数影响梯度流动和模型表达能力",
        key="activation_func"
    )

    # 故障场景模拟
    st.markdown("**🔧 故障场景**")
    fault_scenario = st.selectbox(
        "故障模拟",
        ["正常训练", "学习率过大", "梯度消失", "过拟合", "欠拟合"],
        help="故意设置故障场景，让学生通过调整参数修复模型",
        key="fault_scenario"
    )

    # 应用故障场景
    if fault_scenario == "学习率过大":
        learning_rate = 1.0
    elif fault_scenario == "梯度消失":
        activation_func = "Sigmoid"
        learning_rate = 0.001
    elif fault_scenario == "过拟合":
        epochs = 500
        hidden_units = 32
    elif fault_scenario == "欠拟合":
        epochs = 10
        hidden_units = 2

with col_left:
    st.subheader("📊 训练过程可视化")
    
    # 生成模拟数据
    np.random.seed(42)
    if st.checkbox("使用分类任务"):
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                  n_informative=2, n_clusters_per_class=1, 
                                  random_state=42)
        task_type = "classification"
    else:
        X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
        y = y.reshape(-1, 1)
        task_type = "regression"
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # 转换为PyTorch张量，确保维度匹配
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)  # 确保是二维张量
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)    # 确保是二维张量
    
    # 定义神经网络
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, activation):
            super(SimpleNN, self).__init__()
            
            self.layers = nn.ModuleList()
            
            # 输入层到第一个隐藏层
            self.layers.append(nn.Linear(input_size, hidden_size))
            
            # 隐藏层
            for _ in range(num_layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            
            # 输出层
            self.layers.append(nn.Linear(hidden_size, output_size))
            
            # 激活函数
            if activation == "ReLU":
                self.activation = nn.ReLU()
            elif activation == "Sigmoid":
                self.activation = nn.Sigmoid()
            elif activation == "Tanh":
                self.activation = nn.Tanh()
            elif activation == "Leaky ReLU":
                self.activation = nn.LeakyReLU(0.1)
        
        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = self.activation(layer(x))
            x = self.layers[-1](x)
            return x
    
    # 初始化模型
    input_size = X_train.shape[1]
    output_size = 1
    model = SimpleNN(input_size, hidden_units, output_size, num_layers, activation_func)
    
    # 定义损失函数和优化器
    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练按钮
    train_btn = st.button("🚀 开始训练", type="primary", use_container_width=True)
    
    if train_btn:
        # 存储训练历史
        train_losses = []
        test_losses = []
        gradients = []
        
        # 训练过程
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 添加梯度裁剪和数值稳定性检查
        max_grad_norm = 1.0
        
        for epoch in range(epochs):
            # 前向传播
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                st.warning("训练损失为NaN，请减小学习率或调整网络参数")
                break
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 记录梯度
            total_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad += torch.norm(param.grad).item()
            gradients.append(total_grad)
            
            # 参数更新，添加学习率衰减
            current_lr = learning_rate * (1 - epoch / epochs)  # 线性衰减
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.step()
            
            # 计算测试集损失
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                
                # 检查测试损失是否为NaN
                if torch.isnan(test_loss):
                    st.warning("测试损失为NaN，模型可能发散")
                    break
            
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            
            # 更新进度
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"训练进度: {epoch+1}/{epochs}轮, 训练损失: {loss.item():.4f}")
        
        # 创建可视化图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('损失曲线变化', '梯度流动情况', '训练/测试损失对比', '决策边界可视化'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(x=list(range(len(train_losses))), y=train_losses, 
                      mode='lines', name='训练损失', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 梯度流动
        fig.add_trace(
            go.Scatter(x=list(range(len(gradients))), y=gradients,
                      mode='lines', name='梯度范数', line=dict(color='red')),
            row=1, col=2
        )
        
        # 训练/测试损失对比
        fig.add_trace(
            go.Scatter(x=list(range(len(train_losses))), y=train_losses,
                      mode='lines', name='训练损失', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(test_losses))), y=test_losses,
                      mode='lines', name='测试损失', line=dict(color='orange')),
            row=2, col=1
        )
        
        # 决策边界可视化（仅适用于2D分类）
        if task_type == "classification" and X.shape[1] == 2:
            # 创建网格点
            h = 0.02
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # 预测网格点
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_tensor = torch.FloatTensor(grid_points)
            with torch.no_grad():
                Z = torch.sigmoid(model(grid_tensor)).numpy()
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            fig.add_trace(
                go.Contour(x=xx[0, :], y=yy[:, 0], z=Z, 
                          colorscale='Blues', showscale=False),
                row=2, col=2
            )
            
            # 绘制数据点
            fig.add_trace(
                go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', 
                          name='类别0', marker=dict(color='red')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', 
                          name='类别1', marker=dict(color='green')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示训练结果
        st.subheader("📈 训练结果分析")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("最终训练损失", f"{train_losses[-1]:.4f}")
        
        with col_result2:
            st.metric("最终测试损失", f"{test_losses[-1]:.4f}")
        
        with col_result3:
            overfit_ratio = test_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else 1.0
            st.metric("过拟合程度", f"{overfit_ratio:.2f}", 
                     delta="过拟合" if overfit_ratio > 1.5 else "正常")
        
        # ====== 差异化功能：逐层激活值可视化 ======
        st.markdown("---")
        st.subheader("🔬 逐层激活值分析")
        st.caption("观察信息在神经网络各层之间的流动（取前10个样本）")
        
        with torch.no_grad():
            sample_input = X_train_tensor[:min(10, X_train_tensor.shape[0])]
            activations = []
            x_sample = sample_input
            for layer in model.layers[:-1]:
                x_sample = model.activation(layer(x_sample))
                activations.append(x_sample.numpy())
        
        n_layers = len(activations)
        if n_layers > 0:
            fig_act = make_subplots(rows=1, cols=n_layers,
                                    subplot_titles=[f'第{i+1}层' for i in range(n_layers)])
            for i, act in enumerate(activations):
                act_flat = act.flatten()
                fig_act.add_trace(
                    go.Histogram(x=act_flat, nbinsx=20, marker_color='#1565C0', opacity=0.7),
                    row=1, col=i+1
                )
            fig_act.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_act, use_container_width=True)
            st.caption("💡 激活值分布反映每层神经元的活跃程度。Sigmoid容易出现大量饱和区（接近0或1），ReLU则保持更健康的分布。")
        
        # ====== 差异化功能：梯度消失/爆炸对比 ======
        st.markdown("---")
        st.subheader("⚡ 梯度消失 vs 梯度爆炸 对比")
        st.caption("对比当前激活函数与 Sigmoid 在深层网络中的梯度表现")
        
        if activation_func != "Sigmoid":
            try:
                model_sigmoid = SimpleNN(input_size, hidden_units, output_size, num_layers, "Sigmoid")
                optimizer_sig = optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
                grad_cur = []
                grad_sig = []
                
                for epoch in range(min(50, epochs)):
                    outputs = model(X_train_tensor)
                    loss_cur = criterion(outputs, y_train_tensor)
                    optimizer.zero_grad()
                    loss_cur.backward()
                    total_g = sum(torch.norm(p.grad).item() for p in model.parameters() if p.grad is not None)
                    grad_cur.append(total_g)
                    optimizer.step()
                
                for epoch in range(min(50, epochs)):
                    outputs_s = model_sigmoid(X_train_tensor)
                    loss_s = criterion(outputs_s, y_train_tensor)
                    optimizer_sig.zero_grad()
                    loss_s.backward()
                    total_gs = sum(torch.norm(p.grad).item() for p in model_sigmoid.parameters() if p.grad is not None)
                    grad_sig.append(total_gs)
                    optimizer_sig.step()
                
                fig_grad = go.Figure()
                fig_grad.add_trace(go.Scatter(
                    x=list(range(len(grad_cur))), y=grad_cur, mode='lines',
                    name=f'{activation_func}(当前)', line=dict(color='#1565C0', width=2)
                ))
                fig_grad.add_trace(go.Scatter(
                    x=list(range(len(grad_sig))), y=grad_sig, mode='lines',
                    name='Sigmoid(对比)', line=dict(color='#EF5350', width=2)
                ))
                fig_grad.update_layout(
                    height=300,
                    xaxis_title="训练步数",
                    yaxis_title="梯度范数",
                    legend=dict(x=0.6, y=0.95)
                )
                st.plotly_chart(fig_grad, use_container_width=True)
                st.caption(f"💡 {activation_func}的梯度范数明显大于Sigmoid，说明梯度流动更通畅。Sigmoid在深层网络中容易导致梯度消失。")
            except Exception as e:
                st.info(f"💡 梯度对比需要重新实例化模型，当前环境受限（{e}）。可切换到ReLU观察梯度流动的改善效果。")
        else:
            st.info("💡 当前已选择Sigmoid激活函数。可切换到ReLU观察梯度流动的改善效果。")

# 教学指导（在右侧列中显示）
with col_right:
    st.subheader("🎓 教学指导")
    
    # 当前场景分析
    st.info(f"**当前场景**: {fault_scenario}")
    
    if fault_scenario == "学习率过大":
        st.warning("""
        **问题**: 学习率过大导致参数更新步长过大
        **现象**: 损失函数震荡或发散
        **解决方法**: 降低学习率到0.01-0.1之间
        """)
    
    elif fault_scenario == "梯度消失":
        st.warning("""
        **问题**: Sigmoid激活函数导致梯度消失
        **现象**: 梯度值接近0，模型无法学习
        **解决方法**: 使用ReLU激活函数，增大学习率
        """)
    
    elif fault_scenario == "过拟合":
        st.warning("""
        **问题**: 模型过于复杂，过度拟合训练数据
        **现象**: 训练损失低，测试损失高
        **解决方法**: 减少网络层数或神经元数量
        """)
    
    elif fault_scenario == "欠拟合":
        st.warning("""
        **问题**: 模型过于简单，无法捕捉数据规律
        **现象**: 训练和测试损失都较高
        **解决方法**: 增加网络层数或神经元数量
        """)
    
    else:
        st.success("""
        **正常训练场景**
        - 观察损失曲线平稳下降
        - 梯度值保持合理范围
        - 训练/测试损失差距适中
        """)
    
    st.markdown("---")
    st.subheader("📚 核心概念")
    
    tab1, tab2, tab3 = st.tabs(["前向传播", "反向传播", "梯度下降"])
    
    with tab1:
        st.markdown("""
        ### 前向传播 (Forward Propagation)
        - **定义**: 输入数据通过网络层层传递
        - **过程**: 输入 → 隐藏层 → 输出层
        - **作用**: 计算预测值和损失函数
        - **可视化**: 观察网络输出如何变化
        """)
    
    with tab2:
        st.markdown("""
        ### 反向传播 (Backward Propagation)  
        - **定义**: 从输出层反向计算梯度
        - **过程**: 损失函数 → 输出层 → 隐藏层
        - **作用**: 计算每个参数的梯度
        - **可视化**: 观察梯度流动情况
        """)
    
    with tab3:
        st.markdown("""
        ### 梯度下降 (Gradient Descent)
        - **定义**: 沿着梯度反方向更新参数
        - **公式**: θ = θ - η·∇J(θ)
        - **作用**: 最小化损失函数
        - **可视化**: 观察参数如何收敛
        """)

# 实验场景设计
st.markdown("---")
st.subheader("🔬 实验场景设计")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    st.markdown("""
    ### 场景1: 学习率调节
    **目标**: 理解学习率对训练的影响
    **步骤**:
    1. 设置学习率为0.001（过小）
    2. 设置学习率为1.0（过大）
    3. 找到最佳学习率(0.01-0.1)
    **观察**: 损失曲线变化
    """)

with col_exp2:
    st.markdown("""
    ### 场景2: 网络深度实验
    **目标**: 理解网络深度的影响
    **步骤**:
    1. 使用单层感知机
    2. 使用三层神经网络
    3. 比较拟合能力
    **观察**: 决策边界复杂度
    """)

with col_exp3:
    st.markdown("""
    ### 场景3: 过拟合诊断
    **目标**: 识别和解决过拟合
    **步骤**:
    1. 设置复杂网络(32神经元)
    2. 训练500轮
    3. 观察训练/测试差距
    **观察**: 过拟合程度指标
    """)

st.markdown("---")
st.caption("💡 教学提示：通过调节参数和观察可视化效果，学生可以直观理解神经网络的工作原理和参数影响。")