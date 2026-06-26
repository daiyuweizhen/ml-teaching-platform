"""
AI驱动的机器学习教学闭环平台 - 首页
基于Streamlit的交互式教学平台
专为高职《人工智能导论》课程设计
"""

import streamlit as st
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="AI驱动的机器学习教学闭环平台",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题和介绍
st.title("🔄 AI驱动的机器学习教学闭环平台")
st.markdown("> **AI贯穿教学全流程**：AI辅助实验 → AI智能判题 → AI教学诊断 → 精准教学决策")
st.markdown("---")

# 四步教学闭环流程图
st.markdown("""
<style>
.ai-step-row {
    display: flex;
    justify-content: space-between;
    margin: 24px 0;
    gap: 12px;
}
.ai-step-card {
    flex: 1;
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    border-left: 4px solid #1565C0;
    border-radius: 8px;
    padding: 18px 14px;
    text-align: center;
}
.ai-step-card .step-num {
    font-size: 28px;
    font-weight: 700;
    color: #1565C0;
    margin-bottom: 6px;
}
.ai-step-card .step-title {
    font-size: 15px;
    font-weight: 600;
    color: #1A202C;
    margin-bottom: 4px;
}
.ai-step-card .step-desc {
    font-size: 12px;
    color: #546E7A;
    line-height: 1.5;
}
.ai-arrow {
    display: flex;
    align-items: center;
    font-size: 24px;
    color: #1565C0;
    font-weight: 700;
}
</style>
<div class="ai-step-row">
  <div class="ai-step-card">
    <div class="step-num">①</div>
    <div class="step-title">AI辅助实验</div>
    <div class="step-desc">8个交互式可视化实验<br>参数实时调节、现象即时观察</div>
  </div>
  <div class="ai-arrow">→</div>
  <div class="ai-step-card">
    <div class="step-num">②</div>
    <div class="step-title">AI智能判题</div>
    <div class="step-desc">DeepSeek语义判题<br>AI生成个性化学习建议</div>
  </div>
  <div class="ai-arrow">→</div>
  <div class="ai-step-card">
    <div class="step-num">③</div>
    <div class="step-title">AI教学诊断</div>
    <div class="step-desc">全班数据智能分析<br>自动识别薄弱知识点</div>
  </div>
  <div class="ai-arrow">→</div>
  <div class="ai-step-card">
    <div class="step-num">④</div>
    <div class="step-title">精准教学决策</div>
    <div class="step-desc">教师看板实时数据<br>学生成绩一键导出</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 功能模块介绍
st.subheader("🔧 功能模块")

# 第一行功能模块
col_mod1, col_mod2, col_mod3 = st.columns(3)

with col_mod1:
    st.markdown("""
    ### 📈 线性回归可视化
    - 梯度下降过程演示
    - 学习率调节效果
    - 参数收敛可视化
    """)

with col_mod2:
    st.markdown("""
    ### 🌸 KNN分类可视化
    - 决策边界探索
    - K值影响分析
    - 距离度量比较
    """)

with col_mod3:
    st.markdown("""
    ### 🍰 K-Means聚类
    - 肘部法则分析
    - 簇数K选择
    - 轮廓系数评估
    """)

# 第二行功能模块
col_mod4, col_mod5, col_mod6 = st.columns(3)

with col_mod4:
    st.markdown("""
    ### 📊 模型对比分析
    - 多算法性能对比
    - 分类/回归/聚类任务
    - 可视化性能评估
    """)

with col_mod5:
    st.markdown("""
    ### 🔍 数据探索分析
    - 多数据集特性分析
    - 相关性热力图
    - 特征分布可视化
    """)

with col_mod6:
    st.markdown("""
    ### 🧠 神经网络教学
    - 前向/反向传播可视化
    - 逐层激活值分析
    - 梯度消失/爆炸演示
    - 过拟合自动诊断
    """)

# 第三行功能模块
col_mod7, col_mod8, col_mod9 = st.columns(3)

with col_mod7:
    st.markdown("""
    ### 📚 算法知识库
    - 算法原理详细讲解
    - 应用场景分析
    - 学习资源推荐
    """)

with col_mod8:
    st.markdown("""
    ### 🎯 AI理解力检验
    - 5道核心理解题
    - DeepSeek语义判题
    - AI动态变体出题
    - 即时反馈与指导
    - 实验→理解闭环
    """)

with col_mod9:
    st.markdown("""
    ### 📊 教师数据看板
    - 全班理解率统计
    - 薄弱知识点定位
    - 学生答题明细
    - 教学决策支持
    """)

# 第四行功能模块
col_mod10, col_mod11, col_mod12 = st.columns(3)

with col_mod10:
    st.markdown("""
    ### 🔲 CNN卷积神经网络
    - 卷积核滑动可视化
    - 特征图逐层展示
    - 池化操作演示
    - 全连接层分类
    """)

with col_mod11:
    st.markdown("""&nbsp;""")

with col_mod12:
    st.markdown("""&nbsp;""")

st.markdown("---")

# 技术架构展示
with st.expander("🔧 技术架构说明"):
    st.markdown("""
    - **前端/后端**：Streamlit（自动生成Web界面）
    - **机器学习**：scikit-learn（经典ML算法）+ PyTorch（神经网络）
    - **可视化**：Plotly（动态交互式图表）+ Matplotlib
    - **数据处理**：Pandas、NumPy
    - **AI教学引擎**：DeepSeek API（智能判题 · 变体出题 · 教学诊断 · 个性化建议）
    - **数据持久化**：SQLite（学生答题记录与教学数据，零依赖）
    - **教学闭环**：实验→判题→诊断→决策，数据驱动教学改进
    """)

# 学习路径说明
with st.expander("🧭 推荐学习路径"):
    st.markdown("""
    ### 📚 从实践到理论的学习顺序
    
    1. **数据探索** (数据探索分析) - 理解数据集特性
    2. **线性回归** - 掌握监督学习基础
    3. **KNN分类** - 学习基于实例的分类
    4. **K-Means聚类** - 理解无监督学习
    5. **模型对比** - 综合比较算法性能
    6. **神经网络** - 探索深度学习原理
    7. **算法知识** - 理论知识总结深化
    
    ### 🎯 教学理念
    - **实践导向**：先动手体验，后理论总结
    - **循序渐进**：从简单到复杂的学习路径
    - **直观理解**：通过可视化降低学习门槛
    - **问题驱动**：通过故障场景培养调试能力
    """)

st.markdown("---")
st.caption("© 2026 AI驱动的机器学习教学闭环平台 | DeepSeek深度赋能的交互式教学系统 | 数据驱动 · 智能诊断 · 精准教学")
