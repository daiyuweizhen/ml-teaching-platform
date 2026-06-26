"""
K-Means聚类可视化页面
演示K-Means算法的聚类过程和肘部法则
包含簇数K调节、初始化方法选择等功能
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 设置页面配置
st.set_page_config(page_title="K-Means聚类可视化", page_icon="🍰", layout="wide")

# 页面标题和介绍
st.title("🍰 K-Means聚类可视化：无监督学习探索")
st.markdown("> 调节簇数K和初始化方法，观察K-Means如何将数据划分为不同簇")

# 生成模拟数据
np.random.seed(42)
X, y_true = make_blobs(n_samples=200, centers=3, n_features=2, 
                      random_state=42, cluster_std=0.8)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ 参数调节")
    
    # 控件区域
    k_value = st.slider("📌 簇数K (Number of Clusters)", 2, 10, 3,
                       help="指定要划分的簇数量；使用肘部法则确定最佳K值")
    
    init_method = st.selectbox("🎯 初始化方法", 
                              ["K-Means++", "随机初始化"],
                              help="初始簇中心的选择方式")
    
    max_iter = st.slider("🔄 最大迭代次数", 100, 500, 300, step=50,
                        help="算法最大迭代轮数")
    
    random_state = st.slider("🎲 随机种子", 0, 100, 42,
                            help="控制随机性，确保结果可复现")
    
    # 映射选择到实际参数
    init_map = {"K-Means++": "k-means++", "随机初始化": "random"}
    
    st.markdown("---")
    cluster_btn = st.button("🚀 开始聚类", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**📖 教学提示**")
    st.info("""
    - **肘部法则**：寻找惯性下降的"肘点"确定最佳K值
    - **K-Means++**：智能初始化，减少对随机性的依赖
    - **轮廓系数**：衡量聚类效果的指标，值越大越好
    """)

with col_right:
    # 初始化session_state
    if 'kmeans_trained' not in st.session_state:
        st.session_state.kmeans_trained = False
    
    # 当点击按钮或参数变化时重新训练
    if cluster_btn or not st.session_state.kmeans_trained:
        st.session_state.kmeans_trained = True
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练K-Means模型
        kmeans = KMeans(
            n_clusters=k_value,
            init=init_map[init_method],
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )
        y_pred = kmeans.fit_predict(X_scaled)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X_scaled, y_pred)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('聚类结果可视化', '肘部法则分析'),
            column_widths=[0.6, 0.4]
        )
        
        # 聚类结果图
        colors = px.colors.qualitative.Set3
        for i in range(k_value):
            mask = y_pred == i
            fig.add_trace(
                go.Scatter(x=X_scaled[mask, 0], y=X_scaled[mask, 1],
                          mode='markers', marker=dict(size=8, color=colors[i]),
                          name=f'簇 {i+1}', showlegend=True),
                row=1, col=1
            )
        
        # 绘制簇中心
        fig.add_trace(
            go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                      mode='markers', marker=dict(size=15, color='black', symbol='x'),
                      name='簇中心', showlegend=True),
            row=1, col=1
        )
        
        # 肘部法则分析
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
            
            if k > 1:  # 轮廓系数需要至少2个簇
                y_pred_temp = kmeans_temp.predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, y_pred_temp))
        
        # 惯性图（肘部法则）
        fig.add_trace(
            go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                      name='惯性', line=dict(color='blue', width=3)),
            row=1, col=2
        )
        
        # 标记当前K值
        fig.add_trace(
            go.Scatter(x=[k_value], y=[kmeans.inertia_], mode='markers',
                      marker=dict(size=12, color='red'),
                      name=f'当前K值 ({k_value})'),
            row=1, col=2
        )
        
        # 轮廓系数图（第二个y轴）
        fig.add_trace(
            go.Scatter(x=list(range(2, 11)), y=silhouette_scores, 
                      mode='lines+markers', name='轮廓系数',
                      line=dict(color='green', width=2), yaxis="y2"),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500, 
            showlegend=True,
            yaxis2=dict(
                title="轮廓系数",
                overlaying="y",
                side="right"
            )
        )
        fig.update_xaxes(title_text="特征1", row=1, col=1)
        fig.update_yaxes(title_text="特征2", row=1, col=1)
        fig.update_xaxes(title_text="簇数K", row=1, col=2)
        fig.update_yaxes(title_text="惯性", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 性能指标
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric("轮廓系数", f"{silhouette_avg:.3f}",
                     delta="好" if silhouette_avg > 0.5 else "一般")
        
        with col_metrics2:
            st.metric("簇内惯性", f"{kmeans.inertia_:.1f}")
        
        with col_metrics3:
            best_k_silhouette = k_range[np.argmax(silhouette_scores)]
            st.metric("最佳K值", f"{best_k_silhouette}",
                     delta=f"{best_k_silhouette - k_value}" if best_k_silhouette != k_value else "0")

st.markdown("---")
st.markdown("""
### 🎓 教学要点
1. **K-Means原理**：通过迭代优化簇中心和簇分配
2. **肘部法则**：寻找惯性下降的转折点确定最佳K值
3. **轮廓系数**：衡量聚类质量，值越接近1越好
4. **初始化影响**：好的初始化能加速收敛并改善结果
""")