"""
机器学习算法可视化工坊 - 首页
AI驱动的交互式教学与智能诊断平台
专为高职《人工智能导论》课堂限时教学场景设计
"""

import streamlit as st

st.set_page_config(
    page_title="AI教学诊断平台 - 首页",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 自定义CSS ----------
st.markdown("""
<style>
.banner {
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.banner h1 {
    color: white !important;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.banner p {
    color: #BBDEFB !important;
    font-size: 1.05rem;
}
.step-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #1565C0;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%;
}
.step-card h3 {
    color: #1565C0;
    margin-bottom: 0.5rem;
}
.step-card .step-num {
    display: inline-block;
    background: #1565C0;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-weight: bold;
    margin-right: 0.5rem;
    font-size: 0.9rem;
}
.highlight-box {
    background: #E3F2FD;
    border: 1px solid #90CAF9;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.feature-item {
    background: white;
    border: 1px solid #E8ECF1;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- 顶部横幅 ----------
st.markdown("""
<div class="banner">
    <h1>🎯 AI驱动的交互式教学与智能诊断平台</h1>
    <p>动手实验 → 理解检验 → DeepSeek智能判题 → 教师数据诊断 · 四步闭环 · 专为高职课堂限时教学场景设计</p>
</div>
""", unsafe_allow_html=True)

# ---------- 问题背景 ----------
col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("📌 课堂痛点")
    st.markdown("""
    <div style="line-height:2.0; margin-top:0.5rem;">
    <p>❌ <b>动手了但没理解</b>——学生调完参数说不清"为什么"</p>
    <p>❌ <b>教师靠感觉判断</b>——55人的课堂，谁真懂谁没懂全靠猜</p>
    <p>❌ <b>反馈延迟</b>——作业收上来下周才批完，学生早已忘了实验过程</p>
    <p>❌ <b>没有数据支撑</b>——教务要求的过程性评价材料无从下手</p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("💡 本平台方案")
    st.markdown("""
    <div style="line-height:2.0; margin-top:0.5rem;">
    <p>✅ <b>实验后立即检验</b>——每完成一个可视化实验，马上做理解题</p>
    <p>✅ <b>AI自动判题</b>——DeepSeek语义评判，秒级反馈，精准到概念点</p>
    <p>✅ <b>教师可视化看板</b>——全班通过率、薄弱知识点一目了然</p>
    <p>✅ <b>教学数据自动积累</b>——每节课自动生成过程性评价材料</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- 四步教学流程 ----------
st.subheader("🔄 课堂四步教学闭环")

step_cols = st.columns(4)
steps = [
    ("1", "动手实验", "操作8个交互式可视化页面，调节参数观察算法行为变化"),
    ("2", "理解检验", "选择知识点回答理解题，用自己的话解释'为什么'"),
    ("3", "AI判题", "DeepSeek语义评判，给出分数和概念级改进建议"),
    ("4", "数据诊断", "教师看板展示全班通过率分布，定位教学薄弱点")
]

for i, (num, title, desc) in enumerate(steps):
    with step_cols[i]:
        st.markdown(f"""
        <div class="step-card">
            <h3><span class="step-num">{num}</span>{title}</h3>
            <p style="color:#555; font-size:0.92rem; margin-top:0.3rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------- 核心功能 ----------
st.subheader("🔧 核心功能")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="highlight-box">
        <h4>📈 可视化实验（8个模块）</h4>
        <p style="font-size:0.9rem; color:#555;">
        线性回归 · KNN分类 · K-Means聚类<br>
        神经网络 · 模型对比 · 数据探索<br>
        算法知识库 · 参数故障模拟<br>
        <b>55人课堂验证，85%好评</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="highlight-box">
        <h4>🎯 AI理解力检验</h4>
        <p style="font-size:0.9rem; color:#555;">
        5道核心理解题覆盖全部可视化模块<br>
        DeepSeek语义判题 · 秒级反馈<br>
        支持课堂限时考试模式<br>
        错题自动追踪 · 个性化推荐
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="highlight-box">
        <h4>📊 教师数据看板</h4>
        <p style="font-size:0.9rem; color:#555;">
        全班通过率柱状图 · 薄弱知识点定位<br>
        学生答题明细表 · 个人统计<br>
        成绩单一键导出Excel<br>
        过程性评价材料自动积累
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- 技术架构 + 学习路径 ----------
col_tech, col_path = st.columns(2)

with col_tech:
    with st.expander("🔧 技术架构"):
        st.markdown("""
        | 层次 | 技术 | 用途 |
        |------|------|------|
        | 界面层 | Streamlit | Web交互界面 |
        | AI引擎 | DeepSeek API | 语义判题与智能反馈 |
        | 算法层 | scikit-learn | 机器学习模型 |
        | 可视化 | Plotly | 动态交互图表 |
        | 存储层 | SQLite | 学生答题数据持久化 |
        | 部署 | Streamlit Cloud | 一键部署上课 |
        """)

with col_path:
    with st.expander("🧭 课堂使用流程"):
        st.markdown("""
        **教师课前：** 配置DeepSeek API Key，开启考试模式
        
        **课堂中（20分钟）：**
        1. 学生操作可视化页面实验（5分钟）
        2. 切换到AI理解力检验页面答题（10分钟）
        3. AI实时判题反馈（即时）
        4. 教师查看看板，定位薄弱点（5分钟）
        
        **课后：** 导出全班成绩Excel，自动积累过程性评价材料
        """)

st.markdown("---")
st.caption("© 2026 AI驱动交互式教学与智能诊断平台 | DeepSeek · Streamlit · SQLite | 开源可复现")
