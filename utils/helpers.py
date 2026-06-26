"""
公共工具函数库
包含机器学习可视化平台中常用的辅助函数
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler


def generate_sample_data(n_samples=100, data_type="regression", random_state=42):
    """
    生成模拟数据用于演示
    
    Parameters:
    -----------
    n_samples : int
        样本数量
    data_type : str
        数据类型："regression", "classification", "clustering"
    random_state : int
        随机种子
        
    Returns:
    --------
    X : numpy array
        特征数据
    y : numpy array
        目标变量
    """
    np.random.seed(random_state)
    
    if data_type == "regression":
        # 生成线性回归数据
        X = np.random.rand(n_samples, 1) * 10
        true_slope = 2.5
        true_intercept = 1.2
        y = true_slope * X.squeeze() + true_intercept + np.random.randn(n_samples) * 1.5
        
    elif data_type == "classification":
        # 生成二分类数据
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
    elif data_type == "clustering":
        # 生成聚类数据
        centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        X = []
        y = []
        
        for i, center in enumerate(centers):
            cluster_points = center + 0.5 * np.random.randn(n_samples // 4, 2)
            X.extend(cluster_points)
            y.extend([i] * (n_samples // 4))
        
        X = np.array(X)
        y = np.array(y)
        
    else:
        raise ValueError("不支持的data_type，请选择 'regression', 'classification', 或 'clustering'")
    
    return X, y


def create_gradient_descent_animation(X, y, learning_rate=0.05, epochs=100, init_method="zero"):
    """
    创建梯度下降过程的动画数据
    
    Parameters:
    -----------
    X : numpy array
        特征数据
    y : numpy array
        目标变量
    learning_rate : float
        学习率
    epochs : int
        迭代次数
    init_method : str
        初始化方法："zero" 或 "random"
        
    Returns:
    --------
    weights : list
        权重变化历史
    biases : list
        偏置变化历史
    losses : list
        损失值变化历史
    """
    if init_method == "zero":
        w, b = 0.0, 0.0
    else:
        w, b = np.random.randn(), np.random.randn()
    
    weights = []
    biases = []
    losses = []
    
    for i in range(epochs):
        y_pred = w * X.squeeze() + b
        loss = np.mean((y_pred - y) ** 2)
        
        # 计算梯度
        dw = (2/len(X)) * np.dot(X.squeeze(), (y_pred - y))
        db = (2/len(X)) * np.sum(y_pred - y)
        
        # 参数更新
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        weights.append(w)
        biases.append(b)
        losses.append(loss)
    
    return weights, biases, losses


def plot_decision_boundary(model, X, y, h=0.02):
    """
    绘制分类器的决策边界
    
    Parameters:
    -----------
    model : sklearn classifier
        训练好的分类器
    X : numpy array
        特征数据（2D）
    y : numpy array
        目标变量
    h : float
        网格步长
        
    Returns:
    --------
    fig : plotly figure
        决策边界图
    """
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加决策区域
    fig.add_trace(go.Contour(
        x=xx[0, :], y=yy[:, 0], z=Z,
        colorscale='Viridis',
        showscale=False,
        opacity=0.3
    ))
    
    # 添加数据点
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            marker=dict(size=8, color=colors[i % len(colors)]),
            name=f'类别 {label}'
        ))
    
    fig.update_layout(
        title="决策边界可视化",
        xaxis_title="特征1",
        yaxis_title="特征2",
        showlegend=True
    )
    
    return fig


def calculate_model_metrics(y_true, y_pred, problem_type="classification"):
    """
    计算模型评估指标
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    problem_type : str
        问题类型："classification" 或 "regression"
        
    Returns:
    --------
    metrics : dict
        评估指标字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, r2_score
    )
    
    if problem_type == "classification":
        metrics = {
            "准确率": accuracy_score(y_true, y_pred),
            "精确率": precision_score(y_true, y_pred, average='weighted'),
            "召回率": recall_score(y_true, y_pred, average='weighted'),
            "F1分数": f1_score(y_true, y_pred, average='weighted')
        }
    elif problem_type == "regression":
        metrics = {
            "均方误差": mean_squared_error(y_true, y_pred),
            "R2分数": r2_score(y_true, y_pred)
        }
    else:
        raise ValueError("不支持的problem_type，请选择 'classification' 或 'regression'")
    
    return metrics


def create_comparison_radar_chart(metrics_dict, algorithms):
    """
    创建算法对比雷达图
    
    Parameters:
    -----------
    metrics_dict : dict
        各算法的指标字典
    algorithms : list
        算法名称列表
        
    Returns:
    --------
    fig : plotly figure
        雷达图
    """
    metrics = list(metrics_dict[algorithms[0]].keys())
    
    fig = go.Figure()
    
    for algo in algorithms:
        values = [metrics_dict[algo][metric] for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 闭合雷达图
            theta=metrics + [metrics[0]],
            fill='toself',
            name=algo
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="算法性能雷达图"
    )
    
    return fig


def analyze_feature_correlation(df, target_col=None):
    """
    分析特征相关性
    
    Parameters:
    -----------
    df : pandas DataFrame
        数据框
    target_col : str, optional
        目标列名
        
    Returns:
    --------
    corr_matrix : pandas DataFrame
        相关性矩阵
    high_corr_pairs : list
        高相关性特征对
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if target_col and target_col in numeric_cols:
        # 计算与目标变量的相关性
        target_corr = df[numeric_cols].corr()[target_col].drop(target_col)
        
        # 计算特征间相关性
        feature_corr = df[numeric_cols].corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_val = abs(feature_corr.iloc[i, j])
                if corr_val > 0.8 and feature_corr.columns[i] != target_col and feature_corr.columns[j] != target_col:
                    high_corr_pairs.append({
                        '特征1': feature_corr.columns[i],
                        '特征2': feature_corr.columns[j],
                        '相关性': feature_corr.iloc[i, j]
                    })
        
        return feature_corr, high_corr_pairs, target_corr
    
    else:
        # 只计算特征间相关性
        corr_matrix = df[numeric_cols].corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append({
                        '特征1': corr_matrix.columns[i],
                        '特征2': corr_matrix.columns[j],
                        '相关性': corr_matrix.iloc[i, j]
                    })
        
        return corr_matrix, high_corr_pairs, None


def create_interactive_plot(x_data, y_data, title="", x_label="", y_label="", plot_type="scatter"):
    """
    创建交互式图表
    
    Parameters:
    -----------
    x_data : array-like
        x轴数据
    y_data : array-like
        y轴数据
    title : str
        图表标题
    x_label : str
        x轴标签
    y_label : str
        y轴标签
    plot_type : str
        图表类型："scatter", "line", "bar"
        
    Returns:
    --------
    fig : plotly figure
        交互式图表
    """
    if plot_type == "scatter":
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='markers'))
    elif plot_type == "line":
        fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines'))
    elif plot_type == "bar":
        fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
    else:
        raise ValueError("不支持的plot_type，请选择 'scatter', 'line', 或 'bar'")
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    return fig