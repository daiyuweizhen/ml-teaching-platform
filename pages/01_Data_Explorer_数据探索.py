"""
数据集探索分析页面
探索多种机器学习数据集的特性、分布和相关性
包含数据概览、特征分析、相关性热力图、PCA降维等功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer,
    fetch_california_housing, make_classification
)
from sklearn.preprocessing import StandardScaler

# 设置页面配置
st.set_page_config(page_title="数据集探索分析", page_icon="🔍", layout="wide")

# 页面标题和介绍
st.title("🔍 数据集探索分析：数据理解与可视化")
st.markdown("> 探索不同机器学习数据集的特性、分布和相关性")

# 数据集选项
dataset_options = {
    "鸢尾花 (Iris)": {
        "loader": load_iris,
        "type": "分类",
        "description": "经典的分类数据集，包含3种鸢尾花的4个特征"
    },
    "手写数字 (Digits)": {
        "loader": load_digits,
        "type": "分类", 
        "description": "手写数字识别数据集，包含0-9的数字图像"
    },
    "葡萄酒 (Wine)": {
        "loader": load_wine,
        "type": "分类",
        "description": "葡萄酒化学成分数据集，包含3种葡萄酒类型"
    },
    "乳腺癌 (Breast Cancer)": {
        "loader": load_breast_cancer,
        "type": "分类",
        "description": "乳腺癌诊断数据集，良性/恶性分类"
    },
    "加州房价 (California Housing)": {
        "loader": fetch_california_housing,
        "type": "回归",
        "description": "加州地区房价预测数据集"
    }
}

col_config, col_main = st.columns([1, 2])

with col_config:
    st.subheader("⚙️ 数据集选择")
    
    # 数据集选择
    dataset_choice = st.selectbox("📁 选择数据集", list(dataset_options.keys()))
    
    dataset_info = dataset_options[dataset_choice]
    
    st.info(f"**类型**: {dataset_info['type']}\n**描述**: {dataset_info['description']}")
    
    st.markdown("---")
    
    # 分析选项
    st.subheader("📊 分析选项")
    
    show_stats = st.checkbox("显示数据统计信息", value=True)
    show_distribution = st.checkbox("显示特征分布", value=True)
    show_correlation = st.checkbox("显示相关性热力图", value=True)
    
    st.markdown("---")
    explore_btn = st.button("🚀 开始探索", type="primary", use_container_width=True)

with col_main:
    # 初始化session_state
    if 'exploration_done' not in st.session_state:
        st.session_state.exploration_done = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    # 当点击按钮或数据集变化时重新探索
    dataset_changed = st.session_state.current_dataset != dataset_choice
    if explore_btn or not st.session_state.exploration_done or dataset_changed:
        st.session_state.exploration_done = True
        st.session_state.current_dataset = dataset_choice
        
        # 加载数据集
        data_loader = dataset_info["loader"]
        data = data_loader()
        
        if hasattr(data, 'data') and hasattr(data, 'target'):
            X = data.data
            y = data.target
            feature_names = data.feature_names if hasattr(data, 'feature_names') else [f"特征{i+1}" for i in range(X.shape[1])]
            target_names = data.target_names if hasattr(data, 'target_names') else [f"类别{i}" for i in np.unique(y)]
        else:
            # 对于fetch_california_housing等数据集
            X = data.data
            y = data.target
            feature_names = data.feature_names if hasattr(data, 'feature_names') else [f"特征{i+1}" for i in range(X.shape[1])]
            target_names = ["目标变量"]
        
        # 创建DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # 数据概览
        st.subheader("📋 数据概览")
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.metric("样本数量", len(df))
        
        with col_info2:
            st.metric("特征数量", len(feature_names))
        
        with col_info3:
            st.metric("类别数量", len(np.unique(y)))
        
        with col_info4:
            st.metric("缺失值", df.isnull().sum().sum())
        
        # 显示数据统计信息
        if show_stats:
            st.subheader("📊 统计信息")
            
            # 数值特征统计
            numeric_stats = df[feature_names].describe()
            st.dataframe(numeric_stats.style.format("{:.2f}"))
        
        # 特征分布可视化
        if show_distribution and len(feature_names) > 0:
            st.subheader("📈 特征分布")
            
            # 选择要可视化的特征
            selected_features = st.multiselect(
                "选择要可视化的特征",
                feature_names,
                default=feature_names[:min(4, len(feature_names))]
            )
            
            if selected_features:
                # 创建分布图
                fig_dist = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_features[:4]
                )
                
                for i, feature in enumerate(selected_features[:4]):
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    
                    fig_dist.add_trace(
                        go.Histogram(x=df[feature], name=feature, nbinsx=20),
                        row=row, col=col
                    )
                
                fig_dist.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # 相关性热力图
        if show_correlation and len(feature_names) > 1:
            st.subheader("🔥 相关性热力图")
            
            # 计算相关性矩阵
            corr_matrix = df[feature_names].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                hoverongaps=False,
                colorbar=dict(title="相关系数")
            ))
            
            fig_corr.update_layout(
                title="特征相关性热力图",
                xaxis_title="特征",
                yaxis_title="特征",
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # 目标变量分析
        st.subheader("🎯 目标变量分析")
        
        if dataset_info["type"] == "分类":
            # 分类任务：类别分布
            class_counts = pd.Series(y).value_counts()
            
            fig_class = px.pie(
                values=class_counts.values,
                names=[target_names[i] for i in class_counts.index],
                title="类别分布"
            )
            st.plotly_chart(fig_class, use_container_width=True)
        
        else:
            # 回归任务：目标变量分布
            fig_target = px.histogram(
                df, x='target', 
                title="目标变量分布",
                nbins=30
            )
            st.plotly_chart(fig_target, use_container_width=True)
        
        # 特征与目标关系
        if len(feature_names) >= 2:
            st.subheader("🔗 特征与目标关系")
            
            # 使用session_state来跟踪特征选择，避免不必要的重新渲染
            if 'selected_x_feature' not in st.session_state:
                st.session_state.selected_x_feature = feature_names[0]
            if 'selected_y_feature' not in st.session_state:
                st.session_state.selected_y_feature = feature_names[1]
            
            # 只有当数据集变化时才重置特征选择
            if st.session_state.current_dataset != dataset_choice:
                st.session_state.selected_x_feature = feature_names[0]
                st.session_state.selected_y_feature = feature_names[1]
            
            # 安全检查：确保session_state中的特征在当前数据集中存在
            if st.session_state.selected_x_feature not in feature_names:
                st.session_state.selected_x_feature = feature_names[0]
            if st.session_state.selected_y_feature not in feature_names:
                st.session_state.selected_y_feature = feature_names[1]
            
            # 获取特征索引
            x_index = feature_names.index(st.session_state.selected_x_feature)
            y_index = feature_names.index(st.session_state.selected_y_feature)
            
            x_feature = st.selectbox("X轴特征", feature_names, index=x_index)
            y_feature = st.selectbox("Y轴特征", feature_names, index=y_index)
            
            # 更新session_state
            st.session_state.selected_x_feature = x_feature
            st.session_state.selected_y_feature = y_feature
            
            if dataset_info["type"] == "分类":
                fig_scatter = px.scatter(
                    df, x=x_feature, y=y_feature, color='target',
                    title=f"{x_feature} vs {y_feature} (按类别着色)",
                    color_continuous_scale='viridis'
                )
            else:
                fig_scatter = px.scatter(
                    df, x=x_feature, y=y_feature, color='target',
                    title=f"{x_feature} vs {y_feature}",
                    color_continuous_scale='viridis'
                )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.markdown("""
### 🎓 教学要点
1. **数据理解**：探索数据集的基本特性和分布
2. **特征分析**：了解每个特征的统计属性和分布
3. **相关性分析**：发现特征间的相互关系
4. **数据质量**：检查缺失值和异常值
""")