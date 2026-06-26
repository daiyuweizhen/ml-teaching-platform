"""
模型对比分析页面
在同一数据集上对比不同机器学习算法的性能
支持分类、回归、聚类任务的算法对比
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)

# 设置页面配置
st.set_page_config(page_title="模型对比分析", page_icon="📊", layout="wide")

# 页面标题和介绍
st.title("📊 模型对比分析：多算法性能评估")
st.markdown("> 在同一数据集上对比不同机器学习算法的性能表现")

col_config, col_main = st.columns([1, 2])

with col_config:
    st.subheader("⚙️ 对比配置")
    
    # 任务类型选择
    task_type = st.selectbox("🎯 任务类型", 
                            ["分类", "回归", "聚类"],
                            help="选择要对比的机器学习任务类型")
    
    # 数据集选择
    if task_type == "分类":
        dataset_options = {"鸢尾花数据集": load_iris, "模拟分类数据": "synthetic"}
    elif task_type == "回归":
        dataset_options = {"模拟回归数据": "synthetic"}
    else:
        dataset_options = {"模拟聚类数据": "synthetic"}
    
    dataset_choice = st.selectbox("📁 数据集", list(dataset_options.keys()))
    
    # 算法选择
    st.subheader("🤖 对比算法")
    
    if task_type == "分类":
        algorithms = {
            "逻辑回归": "logistic",
            "KNN分类": "knn_class",
            "决策树": "decision_tree",
            "支持向量机": "svm"
        }
    elif task_type == "回归":
        algorithms = {
            "线性回归": "linear",
            "KNN回归": "knn_reg",
            "决策树回归": "tree_reg"
        }
    else:
        algorithms = {
            "K-Means": "kmeans",
            "DBSCAN": "dbscan"
        }
    
    selected_algorithms = []
    for algo_name, algo_key in algorithms.items():
        if st.checkbox(algo_name, value=True):
            selected_algorithms.append((algo_name, algo_key))
    
    st.markdown("---")
    compare_btn = st.button("🚀 开始对比", type="primary", use_container_width=True)

with col_main:
    # 初始化session_state
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
    
    # 当点击按钮或参数变化时重新对比
    if compare_btn or not st.session_state.comparison_done:
        st.session_state.comparison_done = True
        
        # 加载数据
        if dataset_choice == "模拟分类数据":
            X, y = make_classification(n_samples=200, n_features=4, 
                                      n_redundant=0, n_informative=4,
                                      random_state=42)
        elif dataset_choice == "模拟回归数据":
            X, y = make_regression(n_samples=200, n_features=3, 
                                  noise=10, random_state=42)
        elif dataset_choice == "模拟聚类数据":
            X, y = make_blobs(n_samples=200, centers=3, n_features=2,
                             random_state=42)
        else:
            data = dataset_options[dataset_choice]()
            X, y = data.data, data.target
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if task_type != "聚类":
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
        
        # 存储结果
        results = []
        
        # 训练和评估每个算法
        for algo_name, algo_key in selected_algorithms:
            try:
                if task_type == "分类":
                    if algo_key == "logistic":
                        model = LogisticRegression(random_state=42)
                    elif algo_key == "knn_class":
                        model = KNeighborsClassifier(n_neighbors=5)
                    elif algo_key == "decision_tree":
                        model = DecisionTreeClassifier(random_state=42)
                    elif algo_key == "svm":
                        model = SVC(random_state=42)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    results.append({
                        'algorithm': algo_name,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                
                elif task_type == "回归":
                    if algo_key == "linear":
                        model = LinearRegression()
                    elif algo_key == "knn_reg":
                        model = KNeighborsRegressor(n_neighbors=5)
                    elif algo_key == "tree_reg":
                        model = DecisionTreeRegressor(random_state=42)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results.append({
                        'algorithm': algo_name,
                        'mse': mse,
                        'r2': r2
                    })
                
                else:  # 聚类
                    if algo_key == "kmeans":
                        model = KMeans(n_clusters=3, random_state=42)
                        y_pred = model.fit_predict(X_scaled)
                        
                        silhouette = silhouette_score(X_scaled, y_pred)
                        inertia = model.inertia_
                        
                        results.append({
                            'algorithm': algo_name,
                            'silhouette': silhouette,
                            'inertia': inertia
                        })
            
            except Exception as e:
                st.warning(f"算法 {algo_name} 训练失败: {str(e)}")
        
        # 创建可视化图表
        if results:
            if task_type == "分类":
                # 分类任务对比图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('准确率对比', '精确率对比', '召回率对比', 'F1分数对比')
                )
                
                algorithms = [r['algorithm'] for r in results]
                accuracies = [r['accuracy'] for r in results]
                precisions = [r['precision'] for r in results]
                recalls = [r['recall'] for r in results]
                f1_scores = [r['f1'] for r in results]
                
                # 准确率
                fig.add_trace(
                    go.Bar(x=algorithms, y=accuracies, name='准确率',
                          marker_color=px.colors.qualitative.Set3),
                    row=1, col=1
                )
                
                # 精确率
                fig.add_trace(
                    go.Bar(x=algorithms, y=precisions, name='精确率',
                          marker_color=px.colors.qualitative.Set1),
                    row=1, col=2
                )
                
                # 召回率
                fig.add_trace(
                    go.Bar(x=algorithms, y=recalls, name='召回率',
                          marker_color=px.colors.qualitative.Set2),
                    row=2, col=1
                )
                
                # F1分数
                fig.add_trace(
                    go.Bar(x=algorithms, y=f1_scores, name='F1分数',
                          marker_color=px.colors.qualitative.Pastel),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                
            elif task_type == "回归":
                # 回归任务对比图
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('均方误差对比', 'R²分数对比')
                )
                
                algorithms = [r['algorithm'] for r in results]
                mse_scores = [r['mse'] for r in results]
                r2_scores = [r['r2'] for r in results]
                
                # 均方误差（越低越好）
                fig.add_trace(
                    go.Bar(x=algorithms, y=mse_scores, name='MSE',
                          marker_color='red'),
                    row=1, col=1
                )
                
                # R²分数（越高越好）
                fig.add_trace(
                    go.Bar(x=algorithms, y=r2_scores, name='R²',
                          marker_color='green'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400)
                
            else:
                # 聚类任务对比图
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('轮廓系数对比', '簇内惯性对比')
                )
                
                algorithms = [r['algorithm'] for r in results]
                silhouettes = [r['silhouette'] for r in results]
                inertias = [r['inertia'] for r in results]
                
                # 轮廓系数（越高越好）
                fig.add_trace(
                    go.Bar(x=algorithms, y=silhouettes, name='轮廓系数',
                          marker_color='blue'),
                    row=1, col=1
                )
                
                # 簇内惯性（越低越好）
                fig.add_trace(
                    go.Bar(x=algorithms, y=inertias, name='惯性',
                          marker_color='orange'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示详细结果
            st.subheader("📈 详细性能指标")
            
            if task_type == "分类":
                df_results = pd.DataFrame(results)
                st.dataframe(df_results.style.format({
                    'accuracy': '{:.3f}', 'precision': '{:.3f}', 
                    'recall': '{:.3f}', 'f1': '{:.3f}'
                }))
            
            elif task_type == "回归":
                df_results = pd.DataFrame(results)
                st.dataframe(df_results.style.format({
                    'mse': '{:.3f}', 'r2': '{:.3f}'
                }))
            
            else:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results.style.format({
                    'silhouette': '{:.3f}', 'inertia': '{:.1f}'
                }))

st.markdown("---")
st.markdown("""
### 🎓 教学要点
1. **算法选择**：不同任务需要不同的算法
2. **性能指标**：准确理解各指标的含义和适用场景
3. **对比分析**：通过可视化直观比较算法优劣
4. **实际应用**：根据具体问题选择合适的算法
""")