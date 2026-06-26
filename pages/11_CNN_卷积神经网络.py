"""
CNN卷积神经网络可视化页面
演示卷积核滑动、特征图提取、池化操作、全连接分类
专为高职学生理解CNN核心机制设计
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# st.set_page_config already in Home.py, removed to avoid conflict

st.title("🔲 CNN卷积神经网络可视化")
st.markdown("> 通过手写数字识别案例，直观理解卷积核如何提取特征")

# ========== 生成示例数据 ==========
# 模拟一个手写数字"7"的图像 (8x8)
np.random.seed(42)
digit_image = np.array([
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0]
], dtype=float)

# 添加少量噪声
digit_image += np.random.randn(8, 8) * 0.1
digit_image = np.clip(digit_image, 0, 1)

# ========== 侧边栏 ==========
with st.sidebar:
    st.subheader("⚙️ 卷积核选择")
    kernel_type = st.selectbox(
        "卷积核类型",
        ["边缘检测(水平)", "边缘检测(垂直)", "模糊(平均)", "锐化", "自定义"],
        help="不同卷积核提取不同类型的特征"
    )
    
    if kernel_type == "边缘检测(水平)":
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernel_desc = "检测水平边缘——上方暗、下方亮的变化"
    elif kernel_type == "边缘检测(垂直)":
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_desc = "检测垂直边缘——左侧暗、右侧亮的变化"
    elif kernel_type == "模糊(平均)":
        kernel = np.ones((3, 3)) / 9
        kernel_desc = "平滑图像，降低噪声"
    elif kernel_type == "锐化":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        kernel_desc = "增强边缘和细节"
    else:
        kernel = np.array([[-1, -1, 1], [0, 2, 0], [1, 1, -1]])
        kernel_desc = "自定义混合卷积核"
    
    st.caption(f"💡 {kernel_desc}")
    
    padding = st.selectbox("填充方式", ["valid (无填充)", "same (零填充)"])
    stride = st.slider("步长", 1, 3, 1)
    
    st.markdown("---")
    st.subheader("📊 池化方式")
    pool_type = st.selectbox("池化类型", ["最大池化", "平均池化"])
    pool_size = st.slider("池化窗口大小", 2, 4, 2)

# ========== 主内容区 ==========

# ---- 步骤1: 原始图像 ----
st.subheader("📷 步骤1: 原始输入图像 (8×8)")
st.caption("模拟手写数字'7'的灰度图像，值越大越亮")

fig1 = go.Figure(data=go.Heatmap(
    z=digit_image,
    colorscale='Greys',
    showscale=True,
    text=np.round(digit_image, 2),
    texttemplate="%{text}",
    textfont={"size": 10}
))
fig1.update_layout(height=350, width=400)
st.plotly_chart(fig1, use_container_width=False)

# ---- 步骤2: 卷积操作 ----
st.markdown("---")
st.subheader("🔍 步骤2: 卷积核滑动")

# 卷积实现
def convolve2d(image, kernel, padding='valid', stride=1):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    if padding == 'same':
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        padded = image
    
    out_h = (padded.shape[0] - k_h) // stride + 1
    out_w = (padded.shape[1] - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            patch = padded[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = np.sum(patch * kernel)
    
    return output, padded

feature_map, padded_img = convolve2d(digit_image, kernel, padding.split()[0], stride)

# 展示卷积核
col_k, col_fm = st.columns([1, 2])
with col_k:
    st.markdown("**卷积核 (3×3):**")
    fig_k = go.Figure(data=go.Heatmap(
        z=kernel,
        colorscale='RdBu',
        showscale=True,
        text=kernel,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig_k.update_layout(height=250, width=280)
    st.plotly_chart(fig_k, use_container_width=False)

with col_fm:
    st.markdown(f"**特征图 ({feature_map.shape[0]}×{feature_map.shape[1]}):**")
    st.caption(f"padding={padding.split()[0]}, stride={stride}")
    fig_fm = go.Figure(data=go.Heatmap(
        z=feature_map,
        colorscale='Viridis',
        showscale=True,
        text=np.round(feature_map, 2),
        texttemplate="%{text}",
        textfont={"size": 11}
    ))
    fig_fm.update_layout(height=350, width=450)
    st.plotly_chart(fig_fm, use_container_width=False)

# 卷积过程动画式展示——展示第一个窗口
st.markdown("**🔎 卷积过程示意（第1个窗口位置）：**")
col_patch, col_result = st.columns([1, 1])
with col_patch:
    if padding.split()[0] == 'same':
        first_patch = padded_img[1:4, 1:4]
    else:
        first_patch = padded_img[0:3, 0:3]
    fig_patch = go.Figure(data=go.Heatmap(
        z=first_patch,
        colorscale='Greys',
        text=np.round(first_patch, 2),
        texttemplate="%{text}"
    ))
    fig_patch.update_layout(height=250, title="输入窗口 (3×3)")
    st.plotly_chart(fig_patch, use_container_width=False)

with col_result:
    product = first_patch * kernel
    result_val = np.sum(product)
    fig_prod = go.Figure(data=go.Heatmap(
        z=product,
        colorscale='RdBu',
        text=np.round(product, 2),
        texttemplate="%{text}"
    ))
    fig_prod.update_layout(height=250, title=f"点乘结果 = {result_val:.3f}")
    st.plotly_chart(fig_prod, use_container_width=False)

# ---- 步骤3: 池化 ----
st.markdown("---")
st.subheader("📐 步骤3: 池化降维")

def pool2d(feature_map, pool_size, pool_type='max'):
    h, w = feature_map.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            patch = feature_map[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            if pool_type == '最大池化':
                output[i, j] = np.max(patch)
            else:
                output[i, j] = np.mean(patch)
    return output

pooled = pool2d(feature_map, pool_size, 'max' if '最大' in pool_type else 'mean')

col_before, col_arrow, col_after = st.columns([2, 1, 2])
with col_before:
    st.markdown(f"**池化前：{feature_map.shape[0]}×{feature_map.shape[1]}**")
    fig_before = go.Figure(data=go.Heatmap(
        z=feature_map, colorscale='Viridis',
        text=np.round(feature_map, 2), texttemplate="%{text}"
    ))
    fig_before.update_layout(height=300)
    st.plotly_chart(fig_before, use_container_width=False)

with col_arrow:
    st.markdown(f"<br><br><br><br><h1 style='text-align:center'>{pool_type}<br>↓<br>{pool_size}×{pool_size}</h1>", unsafe_allow_html=True)

with col_after:
    st.markdown(f"**池化后：{pooled.shape[0]}×{pooled.shape[1]}**")
    fig_after = go.Figure(data=go.Heatmap(
        z=pooled, colorscale='Viridis',
        text=np.round(pooled, 2), texttemplate="%{text}"
    ))
    fig_after.update_layout(height=300)
    st.plotly_chart(fig_after, use_container_width=False)

st.caption(f"💡 池化将特征图从 {feature_map.shape[0]}×{feature_map.shape[1]} 压缩到 {pooled.shape[0]}×{pooled.shape[1]}，减少了 {int((1 - pooled.size/feature_map.size)*100)}% 的计算量，同时保留了关键特征。")

# ---- 步骤4: 全连接分类 ----
st.markdown("---")
st.subheader("🔗 步骤4: 全连接层分类")

flattened = pooled.flatten()
st.markdown(f"**展平：** 将 {pooled.shape[0]}×{pooled.shape[1]} 的特征图展开为 {len(flattened)} 维向量")
st.code(f"向量 = {np.round(flattened, 3).tolist()}")

# 模拟全连接层分类
st.markdown("**模拟全连接层 → 输出分类概率：**")
fc_weights = np.random.randn(10, len(flattened)) * 0.5
fc_bias = np.random.randn(10) * 0.1
logits = fc_weights @ flattened + fc_bias
probs = np.exp(logits) / np.sum(np.exp(logits))

digits = list(range(10))
fig_fc = go.Figure(data=go.Bar(
    x=digits, y=probs,
    marker_color=['#1565C0' if i == 7 else '#BBDEFB' for i in digits],
    text=[f'{p:.1%}' for p in probs],
    textposition='outside'
))
fig_fc.update_layout(
    height=300,
    xaxis=dict(title="数字类别", tickmode='array', tickvals=digits),
    yaxis=dict(title="概率", range=[0, max(probs)*1.3])
)
st.plotly_chart(fig_fc, use_container_width=True)
st.caption("💡 示例中数字'7'的概率最高，模型正确识别。实际CNN通过反向传播学习卷积核和全连接层的权重。")

# ========== 教学要点 ==========
st.markdown("---")
st.markdown("""
### 🎓 CNN核心概念总结

| 组件 | 作用 | 本页演示 |
|------|------|---------|
| **卷积层** | 用卷积核扫描图像，提取局部特征（边缘、纹理等） | 3×3卷积核在8×8图像上滑动，生成特征图 |
| **激活函数** | 引入非线性，让网络能学习复杂模式 | ReLU将负值置0，保留正值特征 |
| **池化层** | 降维、减少计算量、增强平移不变性 | 最大池化/平均池化压缩特征图尺寸 |
| **全连接层** | 将提取的特征综合，输出分类结果 | 展开→加权求和→Softmax概率输出 |

### 🔬 建议实验
1. 切换不同卷积核，观察特征图的变化——边缘检测核产生明显的边缘响应
2. 对比 valid 和 same 填充方式，理解padding如何保持输出尺寸
3. 调整步长，观察输出特征图尺寸的变化（步长越大，输出越小）
4. 对比最大池化和平均池化的效果——最大池化保留最显著特征，平均池化保留整体趋势
""")
