"""
线性回归可视化页面
演示梯度下降过程和线性回归算法
包含学习率调节、迭代次数控制等功能
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置页面配置
st.set_page_config(page_title="线性回归可视化", page_icon="📈", layout="wide")

# 页面标题和介绍
st.title("📈 线性回归可视化：梯度下降过程")
st.markdown("> 调整学习率和迭代次数，观察模型是如何一步步拟合数据的")

# 生成模拟数据（房价预测场景）
np.random.seed(42)
n_samples = 120
X = np.random.rand(n_samples, 1) * 10
true_slope = 2.5
true_intercept = 1.2
y = true_slope * X.squeeze() + true_intercept + np.random.randn(n_samples) * 1.5

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ 参数调节")
    
    # 控件区域
    lr = st.slider("📌 学习率 (Learning Rate)", 0.001, 0.3, 0.05, step=0.001,
                   help="控制每次参数更新的步长；过大导致震荡，过小收敛缓慢")
    epochs = st.slider("🔄 迭代次数 (Epochs)", 10, 500, 100, step=10,
                       help="参数更新的轮数；足够多才能收敛")
    
    # 初始化方式选择
    init_method = st.radio("初始化方式", ["零初始化", "随机初始化"], index=0)
    
    st.markdown("---")
    train_btn = st.button("🚀 开始训练", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**📖 教学提示**")
    st.info("""
    - **学习率过小（<0.01）** ：损失下降缓慢，需要更多迭代
    - **学习率过大（>0.1）** ：损失可能震荡或发散
    - **迭代次数不够**：模型未收敛，拟合效果差
    """)

with col_right:
    # 初始化session_state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'training_params' not in st.session_state:
        st.session_state.training_params = {}
    
    # 检查参数是否变化
    current_params = (lr, epochs, init_method)
    params_changed = st.session_state.training_params != current_params
    
    # 当点击按钮或参数变化时重新训练
    if train_btn or not st.session_state.trained or params_changed:
        st.session_state.trained = True
        st.session_state.training_params = current_params
        
        # 梯度下降实现
        if init_method == "零初始化":
            w, b = 0.0, 0.0
        else:
            w, b = np.random.randn() * 0.1, np.random.randn() * 0.1  # 缩小初始化范围
        
        losses = []
        weights = []
        biases = []
        
        # 添加梯度裁剪和数值稳定性检查
        max_grad_norm = 10.0  # 梯度裁剪阈值
        
        for i in range(epochs):
            y_pred = w * X.squeeze() + b
            loss = np.mean((y_pred - y) ** 2)
            
            # 计算梯度
            dw = (2/len(X)) * np.dot(X.squeeze(), (y_pred - y))
            db = (2/len(X)) * np.sum(y_pred - y)
            
            # 梯度裁剪，防止梯度爆炸
            grad_norm = np.sqrt(dw**2 + db**2)
            if grad_norm > max_grad_norm:
                dw = dw * max_grad_norm / grad_norm
                db = db * max_grad_norm / grad_norm
            
            # 参数更新，添加学习率自适应
            adaptive_lr = lr / (1 + 0.01 * i)  # 随着迭代逐渐减小学习率
            w = w - adaptive_lr * dw
            b = b - adaptive_lr * db
            
            # 检查数值稳定性
            if np.isnan(w) or np.isnan(b) or np.abs(w) > 1e10 or np.abs(b) > 1e10:
                st.warning("参数发散，请减小学习率或调整初始化方式")
                break
            
            losses.append(loss)
            weights.append(w)
            biases.append(b)
        
        # 使用sklearn的线性回归作为基准
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        sklearn_w = lr_model.coef_[0]
        sklearn_b = lr_model.intercept_
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('数据拟合效果', '损失函数变化'),
            column_widths=[0.6, 0.4]
        )
        
        # 数据拟合图
        fig.add_trace(
            go.Scatter(x=X.squeeze(), y=y, mode='markers', name='原始数据',
                      marker=dict(size=8, opacity=0.7)),
            row=1, col=1
        )
        
        # 绘制最终拟合线
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred_final = w * x_range + b
        fig.add_trace(
            go.Scatter(x=x_range, y=y_pred_final, mode='lines', 
                      name=f'梯度下降拟合 (w={w:.3f}, b={b:.3f})',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # 绘制sklearn基准线
        y_sklearn = sklearn_w * x_range + sklearn_b
        fig.add_trace(
            go.Scatter(x=x_range, y=y_sklearn, mode='lines',
                      name=f'Sklearn基准 (w={sklearn_w:.3f}, b={sklearn_b:.3f})',
                      line=dict(color='green', dash='dash', width=2)),
            row=1, col=1
        )
        
        # 损失函数图
        fig.add_trace(
            go.Scatter(x=list(range(len(losses))), y=losses, mode='lines',
                      name='损失值', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True, title_text="线性回归训练过程")
        fig.update_xaxes(title_text="特征X", row=1, col=1)
        fig.update_yaxes(title_text="目标y", row=1, col=1)
        fig.update_xaxes(title_text="迭代次数", row=1, col=2)
        fig.update_yaxes(title_text="损失值", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            # 防止数值溢出，限制显示范围
            w_display = w if abs(w) < 1e10 else np.sign(w) * 1e10
            st.metric("最终权重 (w)", f"{w_display:.4f}", 
                     delta=f"{(w - true_slope):.4f}" if abs(w) < 1e10 else "数值溢出")
        
        with col_metrics2:
            # 防止数值溢出，限制显示范围
            b_display = b if abs(b) < 1e10 else np.sign(b) * 1e10
            st.metric("最终偏置 (b)", f"{b_display:.4f}",
                     delta=f"{(b - true_intercept):.4f}" if abs(b) < 1e10 else "数值溢出")
        
        with col_metrics3:
            # 防止数值溢出，限制显示范围
            loss_display = losses[-1] if losses[-1] < 1e10 else 1e10
            st.metric("最终损失", f"{loss_display:.4f}")
        
        # ====== 差异化功能1：残差诊断图 ======
        st.markdown("---")
        st.subheader("🔍 残差诊断图（Residual Diagnostic）")
        y_pred_all = w * X.squeeze() + b
        residuals = y - y_pred_all
        
        fig_res = make_subplots(rows=1, cols=2, subplot_titles=('残差 vs 预测值', '残差分布'))
        fig_res.add_trace(
            go.Scatter(x=y_pred_all, y=residuals, mode='markers',
                      marker=dict(size=8, opacity=0.6, color='#1565C0'),
                      name='残差'),
            row=1, col=1
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        fig_res.add_trace(
            go.Histogram(x=residuals, nbinsx=20, marker_color='#42A5F5',
                        name='残差分布'),
            row=1, col=2
        )
        fig_res.update_layout(height=350, showlegend=False)
        fig_res.update_xaxes(title_text="预测值", row=1, col=1)
        fig_res.update_yaxes(title_text="残差", row=1, col=1)
        fig_res.update_xaxes(title_text="残差值", row=1, col=2)
        fig_res.update_yaxes(title_text="频数", row=1, col=2)
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("💡 理想情况下残差应随机分布在0轴两侧，无明显模式。若出现漏斗形或曲线形，说明模型假设不满足。")
        
        # ====== 差异化功能2：学习率动态对比 ======
        st.markdown("---")
        st.subheader("📊 学习率对比实验")
        st.caption("同时对比小(0.01)、中(当前)、大(0.2)三种学习率的训练轨迹")
        
        lr_list = [0.01, lr, 0.2]
        lr_labels = [f"小学习率(0.01)", f"当前学习率({lr})", f"大学习率(0.2)"]
        colors = ['#42A5F5', '#1565C0', '#EF5350']
        
        fig_lr = go.Figure()
        for lr_val, lbl, clr in zip(lr_list, lr_labels, colors):
            w_t, b_t = 0.0, 0.0
            losses_lr = []
            for i in range(epochs):
                y_p = w_t * X.squeeze() + b_t
                loss = np.mean((y_p - y) ** 2)
                dw = (2/len(X)) * np.dot(X.squeeze(), (y_p - y))
                db = (2/len(X)) * np.sum(y_p - y)
                w_t -= lr_val * dw
                b_t -= lr_val * db
                losses_lr.append(loss)
            fig_lr.add_trace(go.Scatter(
                x=list(range(len(losses_lr))), y=losses_lr,
                mode='lines', name=lbl, line=dict(color=clr, width=2)
            ))
        
        fig_lr.update_layout(
            height=350,
            xaxis_title="迭代次数",
            yaxis_title="损失值",
            legend=dict(x=0.6, y=0.95)
        )
        st.plotly_chart(fig_lr, use_container_width=True)
        st.caption("💡 小学习率收敛慢但稳定，大学习率收敛快但可能震荡。选择合适的学习率是训练的关键。")


st.markdown("---")
st.markdown("""
### 🎓 教学要点
1. **梯度下降原理**：通过不断调整参数使损失函数最小化
2. **学习率影响**：控制参数更新的步长，影响收敛速度和稳定性
3. **初始化策略**：不同的初始化方式会影响收敛路径
4. **收敛判断**：损失函数不再明显下降时模型收敛
""")