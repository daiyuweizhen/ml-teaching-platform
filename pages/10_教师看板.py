"""
教师看板 - 教学数据汇总与学情分析
从 SQLite 读取所有学生答题记录，生成可视化统计
"""

import streamlit as st
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.db import (
    init_db, get_topic_stats, get_all_submissions_detail,
    get_all_topics, get_exercises_by_topic, get_improvement_stats
)
from utils.deepseek import generate_teaching_diagnosis

st.set_page_config(page_title="教师看板", page_icon="📊", layout="wide")

init_db()

st.title("📊 教学数据看板")
st.markdown("> 实时掌握全班学生的算法理解情况，精准定位教学薄弱点 —— AI驱动教学决策")

# ========== 数据加载 ==========
topic_stats = get_topic_stats()
all_detail = get_all_submissions_detail()

# 总览指标
total_exercises = len(get_exercises_by_topic())
total_submissions = len(all_detail)
if all_detail:
    passed_count = sum(1 for d in all_detail if d["passed"])
    total_score = sum(d["score"] for d in all_detail)
    avg_score = round(total_score / len(all_detail), 1)
    pass_rate = round(passed_count / len(all_detail) * 100, 1) if all_detail else 0
else:
    passed_count = 0
    avg_score = 0
    pass_rate = 0

# 参与学生数
student_set = set(d["name"] for d in all_detail) if all_detail else set()
student_count = len(student_set)

# ========== 顶部总览卡片 ==========
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📚 题库数量", f"{total_exercises} 题")
with col2:
    st.metric("📝 答题次数", f"{total_submissions} 次")
with col3:
    st.metric("👥 参与学生", f"{student_count} 人")
with col4:
    st.metric("✅ 通过率", f"{pass_rate}%")
with col5:
    st.metric("📈 平均分", f"{avg_score}")

# ========== 知识点通过率柱状图 ==========
st.markdown("---")
st.subheader("📋 各知识点理解率")

if topic_stats:
    df_topic = pd.DataFrame(topic_stats)
    df_topic["通过率"] = df_topic.apply(
        lambda r: round(r["passed_count"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
        axis=1
    )
    df_topic = df_topic.sort_values("通过率")

    st.bar_chart(
        df_topic.set_index("topic")["通过率"],
        use_container_width=True,
        horizontal=True
    )

    # 教学效果证据
    st.markdown("---")
    st.subheader("📈 教学效果证据")
    imp = get_improvement_stats()
    if imp["count"] > 0:
        col_ev1, col_ev2, col_ev3 = st.columns(3)
        with col_ev1:
            st.metric("持续学习学生数", f"{imp['count']}人", help="有2次及以上答题记录的学生")
        with col_ev2:
            st.metric("首次答题平均分", f"{imp['avg_first_score']}分")
        with col_ev3:
            delta_str = f"+{imp['improvement']}" if imp['improvement'] >= 0 else str(imp['improvement'])
            st.metric("AI反馈后平均分", f"{imp['avg_latest_score']}分", delta=delta_str)
        if imp["improvement"] > 0:
            st.success(f"✅ 经过AI判题反馈后重新学习，{imp['count']}名持续学习的学生平均分从 **{imp['avg_first_score']}** 提升至 **{imp['avg_latest_score']}**，进步 **{imp['improvement']}分**")
    else:
        st.info("📭 暂无重复答题数据（需有学生在AI反馈后再次答题才能生成教学效果证据）")

    # 知识点评分表格
    st.markdown("### 知识点详情")
    col_show = df_topic[["topic", "total", "passed_count", "通过率", "avg_score"]].copy()
    col_show.columns = ["知识点", "答题次数", "通过次数", "通过率(%)", "平均分"]
    st.dataframe(col_show, use_container_width=True, hide_index=True)

    # 薄弱知识点高亮
    weak_topics = df_topic[df_topic["通过率"] < 70]
    if not weak_topics.empty:
        st.warning(f"⚠️ 以下知识点通过率低于70%，建议重点讲解：")
        for _, row in weak_topics.iterrows():
            st.markdown(f"- **{row['topic']}**：通过率 {row['通过率']}%，平均分 {row['avg_score']}")
else:
    st.info("📭 暂无学生答题数据。请先让学生在「AI理解力检验」页面完成答题。")

# ========== 学生答题明细 ==========
st.markdown("---")
st.subheader("👤 学生答题明细")

if all_detail:
    df_detail = pd.DataFrame(all_detail)
    df_detail["状态"] = df_detail["passed"].apply(lambda x: "✅" if x else "❌")
    df_detail["提交时间"] = df_detail["submitted_at"].apply(lambda x: x[:16] if x else "")

    col_show = df_detail[["name", "topic", "状态", "score", "提交时间"]].copy()
    col_show.columns = ["姓名", "知识点", "状态", "得分", "提交时间"]
    col_show = col_show.sort_values("提交时间", ascending=False)

    st.dataframe(col_show, use_container_width=True, hide_index=True)

    # 学生个人统计
    st.markdown("### 学生个人统计")
    df_student = df_detail.groupby("name").agg(
        答题次数=("score", "count"),
        通过次数=("passed", "sum"),
        平均分=("score", "mean")
    ).round(1)
    df_student["通过率"] = (df_student["通过次数"] / df_student["答题次数"] * 100).round(1)
    df_student = df_student.sort_values("平均分", ascending=False)
    st.dataframe(df_student, use_container_width=True)
else:
    st.info("📭 暂无学生答题记录")

# ========== AI教学诊断 ==========
st.markdown("---")
st.subheader("🤖 AI教学诊断报告")

# 从侧边栏获取API Key
api_key_for_diag = st.sidebar.text_input(
    "DeepSeek API Key（用于AI诊断）",
    type="password",
    key="teacher_api_key",
    help="输入API Key后AI将自动分析全班数据并生成教学建议",
    placeholder="sk-..."
)

if api_key_for_diag:
    if st.button("🔍 生成AI教学诊断", type="primary"):
        with st.spinner("🤖 AI正在分析全班教学数据..."):
            # 构建数据文本
            df_for_diag = pd.DataFrame(topic_stats)
            diag_text = "全班AI理解力检验数据分析：\n\n"
            for _, row in df_for_diag.iterrows():
                rate = round(row['passed_count']/row['total']*100, 1) if row['total'] > 0 else 0
                diag_text += f"- {row['topic']}：答题{row['total']}次，通过{row['passed_count']}次（通过率{rate}%），平均分{row['avg_score']}\n"
            diag_text += f"\n全班{student_count}名学生，总答题{total_submissions}次，整体通过率{pass_rate}%，平均分{avg_score}。"
            
            diagnosis = generate_teaching_diagnosis(diag_text, api_key_for_diag)
            st.markdown(diagnosis)
else:
    st.info("💡 在侧边栏输入 DeepSeek API Key 后，点击按钮即可生成 AI 教学诊断报告")

# ========== 成绩单导出 ==========
st.markdown("---")
st.subheader("📥 成绩单导出")
if st.button("📊 导出成绩单Excel（含答题明细/学生汇总/知识点统计）", type="primary", use_container_width=True):
    from io import BytesIO
    import pandas as pd
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet1: 答题明细
        df_detail = pd.DataFrame(all_detail)
        df_detail["状态"] = df_detail["passed"].apply(lambda x: "通过" if x else "未通过")
        df_detail["提交时间"] = df_detail["submitted_at"].apply(lambda x: x[:16] if x else "")
        df_detail[["name", "topic", "状态", "score", "提交时间"]].to_excel(
            writer, sheet_name="答题明细", index=False
        )
        
        # Sheet2: 学生汇总
        df_student = pd.DataFrame(all_detail)
        df_summary = df_student.groupby("name").agg(
            答题次数=("score", "count"),
            通过次数=("passed", "sum"),
            平均分=("score", "mean")
        ).round(1)
        df_summary["通过率"] = (df_summary["通过次数"] / df_summary["答题次数"] * 100).round(1)
        df_summary.sort_values("平均分", ascending=False).to_excel(writer, sheet_name="学生汇总")
        
        # Sheet3: 知识点统计
        df_topic_export = pd.DataFrame(topic_stats)
        df_topic_export["通过率"] = df_topic_export.apply(
            lambda r: round(r["passed_count"]/r["total"]*100, 1) if r["total"]>0 else 0, axis=1
        )
        df_topic_export[["topic", "total", "passed_count", "通过率", "avg_score"]].to_excel(
            writer, sheet_name="知识点统计", index=False
        )
    
    st.download_button(
        label="⬇️ 下载成绩单",
        data=output.getvalue(),
        file_name="AI教学诊断_成绩单.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.caption("💡 所有数据存储在本地 `teaching_data.db` 文件中，可随时备份或导出。AI教学诊断由DeepSeek驱动。")
