"""
机器学习可视化增强工具库
提供更丰富的可视化功能和错误处理
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)


class MLVisualizer:
    """机器学习可视化器类"""
    
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    def create_gradient_descent_visualization(self, X, y, learning_rate=0.05, epochs=100, 
                                             init_method="zero", random_state=42):
        """
        创建梯度下降过程的可视化
        
        Args:
            X: 特征数据
            y: 目标变量
            learning_rate: 学习率
            epochs: 迭代次数
            init_method: 初始化方法
            random_state: 随机种子
            
        Returns:
            dict: 包含可视化数据和结果的字典
        """
        np.random.seed(random_state)
        
        try:
            # 参数初始化
            if init_method == "zero":
                w, b = 0.0, 0.0
            else:
                w, b = np.random.randn(), np.random.randn()
            
            # 存储训练过程
            weights = []
            biases = []
            losses = []
            gradients_w = []
            gradients_b = []
            
            # 梯度下降过程
            for i in range(epochs):
                y_pred = w * X.squeeze() + b
                loss = np.mean((y_pred - y) ** 2)
                
                # 计算梯度
                dw = (2/len(X)) * np.dot(X.squeeze(), (y_pred - y))
                db = (2/len(X)) * np.sum(y_pred - y)
                
                # 参数更新
                w = w - learning_rate * dw
                b = b - learning_rate * db
                
                # 记录过程
                weights.append(w)
                biases.append(b)
                losses.append(loss)
                gradients_w.append(dw)
                gradients_b.append(db)
            
            # 创建可视化图表
            fig = self._create_gradient_descent_figure(X, y, weights, biases, losses, epochs)
            
            return {
                'success': True,
                'final_weights': w,
                'final_biases': b,
                'final_loss': losses[-1],
                'figure': fig,
                'training_history': {
                    'weights': weights,
                    'biases': biases,
                    'losses': losses,
                    'gradients_w': gradients_w,
                    'gradients_b': gradients_b
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"梯度下降可视化失败: {str(e)}"
            }
    
    def _create_gradient_descent_figure(self, X, y, weights, biases, losses, epochs):
        """创建梯度下降过程图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('数据拟合效果', '损失函数变化', '权重变化', '偏置变化'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # 数据拟合图
        fig.add_trace(
            go.Scatter(x=X.squeeze(), y=y, mode='markers', name='原始数据',
                      marker=dict(size=6, opacity=0.7)),
            row=1, col=1
        )
        
        # 绘制最终拟合线
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred_final = weights[-1] * x_range + biases[-1]
        fig.add_trace(
            go.Scatter(x=x_range, y=y_pred_final, mode='lines', 
                      name=f'最终拟合', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # 损失函数图
        fig.add_trace(
            go.Scatter(x=list(range(len(losses))), y=losses, mode='lines',
                      name='损失值', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # 权重变化图
        fig.add_trace(
            go.Scatter(x=list(range(len(weights))), y=weights, mode='lines',
                      name='权重', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # 偏置变化图
        fig.add_trace(
            go.Scatter(x=list(range(len(biases))), y=biases, mode='lines',
                      name='偏置', line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="梯度下降训练过程")
        fig.update_xaxes(title_text="特征X", row=1, col=1)
        fig.update_yaxes(title_text="目标y", row=1, col=1)
        fig.update_xaxes(title_text="迭代次数", row=1, col=2)
        fig.update_yaxes(title_text="损失值", row=1, col=2)
        fig.update_xaxes(title_text="迭代次数", row=2, col=1)
        fig.update_yaxes(title_text="权重值", row=2, col=1)
        fig.update_xaxes(title_text="迭代次数", row=2, col=2)
        fig.update_yaxes(title_text="偏置值", row=2, col=2)
        
        return fig
    
    def create_knn_decision_boundary(self, X, y, k_value=5, metric="euclidean", 
                                    weights="uniform", test_size=0.3, random_state=42):
        """
        创建KNN决策边界可视化
        
        Args:
            X: 特征数据
            y: 目标变量
            k_value: K值
            metric: 距离度量
            weights: 权重方法
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            dict: 包含可视化数据和结果的字典
        """
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # 数据预处理
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 分割数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # 训练KNN模型
            knn = KNeighborsClassifier(
                n_neighbors=k_value,
                metric=metric,
                weights=weights
            )
            knn.fit(X_train, y_train)
            
            # 预测
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 创建决策边界
            fig = self._create_decision_boundary_figure(knn, X_scaled, y, X_train, y_train, 
                                                       X_test, y_test, y_pred)
            
            # 计算不同K值的准确率
            k_range = range(1, 16)
            accuracies = []
            for k in k_range:
                knn_temp = KNeighborsClassifier(n_neighbors=k)
                knn_temp.fit(X_train, y_train)
                y_pred_temp = knn_temp.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred_temp))
            
            return {
                'success': True,
                'accuracy': accuracy,
                'figure': fig,
                'k_accuracy_data': {
                    'k_values': list(k_range),
                    'accuracies': accuracies
                },
                'model_info': {
                    'k_value': k_value,
                    'metric': metric,
                    'weights': weights
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"KNN决策边界可视化失败: {str(e)}"
            }
    
    def _create_decision_boundary_figure(self, model, X, y, X_train, y_train, X_test, y_test, y_pred):
        """创建决策边界图表"""
        # 创建网格点
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 预测网格点
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('决策边界可视化', '测试集分类结果'),
            specs=[[{}, {}]]
        )
        
        # 决策边界图
        unique_labels = np.unique(y)
        colors = self.colors[:len(unique_labels)]
        
        # 绘制决策区域
        for i, label in enumerate(unique_labels):
            mask = Z == label
            if np.any(mask):
                fig.add_trace(
                    go.Contour(
                        x=xx[0, :], y=yy[:, 0], z=Z,
                        colorscale=[[0, colors[0]], [0.5, colors[1]], [1, colors[2]]],
                        showscale=False,
                        opacity=0.3
                    ),
                    row=1, col=1
                )
        
        # 绘制训练数据点
        for i, color in enumerate(colors):
            mask = y_train == i
            fig.add_trace(
                go.Scatter(
                    x=X_train[mask, 0], y=X_train[mask, 1],
                    mode='markers',
                    marker=dict(size=8, color=color, line=dict(width=1, color='black')),
                    name=f'训练集-类别{i}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 测试集分类结果图
        correct_mask = y_pred == y_test
        
        # 正确分类的点
        for i, color in enumerate(colors):
            mask = (y_test == i) & correct_mask
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=X_test[mask, 0], y=X_test[mask, 1],
                        mode='markers',
                        marker=dict(size=10, color=color, symbol='circle'),
                        name=f'正确-类别{i}',
                        showlegend=True
                    ),
                    row=1, col=2
                )
        
        # 错误分类的点
        for i, color in enumerate(colors):
            mask = (y_test == i) & ~correct_mask
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=X_test[mask, 0], y=X_test[mask, 1],
                        mode='markers',
                        marker=dict(size=12, color=color, symbol='x', line=dict(width=2)),
                        name=f'错误-类别{i}',
                        showlegend=True
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="特征1", row=1, col=1)
        fig.update_yaxes(title_text="特征2", row=1, col=1)
        fig.update_xaxes(title_text="特征1", row=1, col=2)
        fig.update_yaxes(title_text="特征2", row=1, col=2)
        
        return fig


def create_performance_comparison_chart(algorithm_results):
    """
    创建算法性能对比图表
    
    Args:
        algorithm_results: 算法结果列表
        
    Returns:
        plotly.figure: 性能对比图表
    """
    algorithms = []
    accuracies = []
    
    for result in algorithm_results:
        if result['success']:
            algorithms.append(result['algorithm_name'])
            accuracies.append(result['accuracy'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=accuracies,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
        text=[f'{acc:.3f}' for acc in accuracies],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="算法性能对比",
        xaxis_title="算法",
        yaxis_title="准确率",
        yaxis_range=[0, 1]
    )
    
    return fig


def validate_input_data(X, y, algorithm_type):
    """
    验证输入数据的有效性
    
    Args:
        X: 特征数据
        y: 目标变量
        algorithm_type: 算法类型
        
    Returns:
        tuple: (是否有效, 错误信息)
    """
    if X is None or y is None:
        return False, "输入数据不能为空"
    
    if len(X) != len(y):
        return False, "特征数据和目标变量长度不一致"
    
    if algorithm_type == "classification" and len(np.unique(y)) < 2:
        return False, "分类任务需要至少2个类别"
    
    return True, "数据验证通过"


def generate_sample_dataset(dataset_type, n_samples=100, random_state=42):
    """
    生成示例数据集
    
    Args:
        dataset_type: 数据集类型
        n_samples: 样本数量
        random_state: 随机种子
        
    Returns:
        tuple: (X, y)
    """
    np.random.seed(random_state)
    
    if dataset_type == "regression":
        X = np.random.rand(n_samples, 1) * 10
        true_slope = 2.5
        true_intercept = 1.2
        y = true_slope * X.squeeze() + true_intercept + np.random.randn(n_samples) * 1.5
        
    elif dataset_type == "classification":
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
    elif dataset_type == "clustering":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, 
                         random_state=random_state, cluster_std=0.8)
    
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return X, y