"""
KNN分类器可视化页面
演示K-最近邻算法的分类过程和决策边界
包含K值调节、距离度量选择等功能
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 设置页面配置
st.set_page_config(page_title="KNN分类可视化", page_icon="🌸", layout="wide")

# 页面标题和介绍
st.title("🌸 KNN分类可视化：决策边界探索")
st.markdown("> 调节K值和距离度量，观察KNN算法如何根据最近邻进行分类")

# 生成模拟数据
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          random_state=42)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ 参数调节")
    
    # 控件区域
    k_value = st.slider("📌 K值 (Number of Neighbors)", 1, 15, 5,
                       help="选择最近邻的数量；过小容易过拟合，过大容易欠拟合")
    
    distance_metric = st.selectbox("📏 距离度量", 
                                  ["欧氏距离", "曼哈顿距离", "切比雪夫距离"],
                                  help="计算样本间距离的方法")
    
    weights_method = st.selectbox("⚖️ 权重方法", 
                                 ["统一权重", "距离权重"],
                                 help="近邻的投票权重分配方式")
    
    # 映射选择到实际参数
    metric_map = {"欧氏距离": "euclidean", "曼哈顿距离": "manhattan", "切比雪夫距离": "chebyshev"}
    weights_map = {"统一权重": "uniform", "距离权重": "distance"}
    
    st.markdown("---")
    train_btn = st.button("🚀 开始训练", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**📖 教学提示**")
    st.info("""
    - **K值过小（<3）**：决策边界复杂，容易过拟合
    - **K值过大（>10）**：决策边界平滑，容易欠拟合
    - **距离权重**：近邻的投票权重与距离成反比
    """)

with col_right:
    # 初始化session_state
    if 'knn_trained' not in st.session_state:
        st.session_state.knn_trained = False
    if 'knn_params' not in st.session_state:
        st.session_state.knn_params = {}
    
    # 检查参数是否变化
    current_params = (k_value, distance_metric, weights_method)
    params_changed = st.session_state.knn_params != current_params
    
    # 当点击按钮或参数变化时重新训练
    if train_btn or not st.session_state.knn_trained or params_changed:
        st.session_state.knn_trained = True
        st.session_state.knn_params = current_params
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # 训练KNN模型
        knn = KNeighborsClassifier(
            n_neighbors=k_value,
            metric=metric_map[distance_metric],
            weights=weights_map[weights_method]
        )
        knn.fit(X_train, y_train)
        
        # 预测
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('决策边界可视化', '不同K值准确率对比'),
            column_widths=[0.6, 0.4]
        )
        
        # 创建网格点用于决策边界
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 预测网格点
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 决策边界图
        fig.add_trace(
            go.Contour(x=xx[0, :], y=yy[:, 0], z=Z, 
                      colorscale='Viridis', showscale=False),
            row=1, col=1
        )
        
        # 绘制训练数据点
        colors = ['red', 'blue']
        for i, color in enumerate(colors):
            mask = y_train == i
            fig.add_trace(
                go.Scatter(x=X_train[mask, 0], y=X_train[mask, 1],
                          mode='markers', marker=dict(size=8, color=color),
                          name=f'训练集-类别{i}', showlegend=True),
                row=1, col=1
            )
        
        # 不同K值的准确率对比
        k_range = range(1, 16)
        accuracies = []
        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            knn_temp.fit(X_train, y_train)
            y_pred_temp = knn_temp.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred_temp))
        
        fig.add_trace(
            go.Scatter(x=list(k_range), y=accuracies, mode='lines+markers',
                      name='准确率', line=dict(color='orange', width=3)),
            row=1, col=2
        )
        
        # 标记当前K值
        fig.add_trace(
            go.Scatter(x=[k_value], y=[accuracy], mode='markers',
                      marker=dict(size=12, color='red'),
                      name=f'当前K值 ({k_value})'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="特征1", row=1, col=1)
        fig.update_yaxes(title_text="特征2", row=1, col=1)
        fig.update_xaxes(title_text="K值", row=1, col=2)
        fig.update_yaxes(title_text="准确率", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric("测试准确率", f"{accuracy:.3f}")
        
        with col_metrics2:
            st.metric("K值", f"{k_value}")
        
        with col_metrics3:
            best_k = k_range[np.argmax(accuracies)]
            st.metric("最佳K值", f"{best_k}", 
                     delta=f"{best_k - k_value}" if best_k != k_value else "0")

st.markdown("---")
st.markdown("""
### 🎓 教学要点
1. **KNN原理**：根据K个最近邻的多数投票进行分类
2. **K值选择**：平衡模型的偏差和方差
3. **距离度量**：影响"相似性"的计算方式
4. **决策边界**：K值越小边界越复杂，越大边界越平滑
""")