"""
AI编程实训 - 理解力检验页面
学生：选择知识点 → 回答问题 → AI判题 → 即时反馈
教师：配置API Key，查看教学效果
"""

import streamlit as st
import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.db import (
    init_db, get_or_create_student, get_exercises_by_topic,
    get_all_topics, save_submission, get_student_submissions
)
from utils.deepseek import judge_answer, generate_personalized_guidance, generate_variant_question

st.set_page_config(page_title="AI理解力检验", page_icon="🎯", layout="wide")

# 初始化数据库
init_db()

st.title("🎯 AI 理解力检验")
st.markdown("> 完成可视化实验后，来检验你是否真正理解了算法原理")

# ========== 侧边栏：模式选择与API Key ==========
with st.sidebar:
    st.header("⚙️ 设置")
    mode = st.radio("使用模式", ["🎓 学生练习", "👨‍🏫 教师管理"], key="exercise_mode")

    # 考试模式
    exam_mode = st.checkbox("🏫 开启考试模式", help="启用倒计时，模拟课堂限时考试场景")
    if exam_mode:
        exam_time = st.slider("考试限时（分钟）", 5, 60, 15, 5)
        if "exam_time_remaining" not in st.session_state:
            st.session_state["exam_time_remaining"] = exam_time * 60
        if "exam_started" not in st.session_state:
            st.session_state["exam_started"] = False
        if "exam_finished" not in st.session_state:
            st.session_state["exam_finished"] = False

    st.markdown("---")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=st.session_state.get("deepseek_api_key", ""),
        help="请输入 DeepSeek API Key，用于AI判题",
        placeholder="sk-..."
    )
    if api_key:
        st.session_state["deepseek_api_key"] = api_key

# ============================================================
# 教师管理模式
# ============================================================
if mode == "👨‍🏫 教师管理":
    st.subheader("📋 题库管理")

    # 筛选
    topics = ["全部"] + get_all_topics()
    selected_topic = st.selectbox("按知识点筛选", topics)

    exercises = get_exercises_by_topic(selected_topic if selected_topic != "全部" else None)

    if not exercises:
        st.info("暂无题目，请先初始化数据库")
    else:
        for i, ex in enumerate(exercises):
            with st.expander(f"题{i+1} | {ex['topic']} | {ex['related_page']}"):
                st.markdown(f"**题目：** {ex['question']}")
                st.markdown(f"**标准关键词：** `{ex['keywords']}`")
                st.markdown(f"**关联可视化页面：** `{ex['related_page']}`")

    st.markdown("---")
    st.info("💡 题库为预设的5道理解题，覆盖全部可视化模块。教师无需手动出题。")

# ============================================================
# 学生练习模式
# ============================================================
else:
    # 身份确认
    if "student_name" not in st.session_state:
        st.session_state["student_name"] = ""
        st.session_state["student_id"] = None

    col_name, col_confirm = st.columns([3, 1])
    with col_name:
        student_name = st.text_input("请输入你的姓名", value=st.session_state["student_name"])
    with col_confirm:
        if st.button("确认身份", type="primary", use_container_width=True) and student_name.strip():
            st.session_state["student_name"] = student_name.strip()
            st.session_state["student_id"] = get_or_create_student(student_name.strip())
            st.rerun()

    if not st.session_state["student_id"]:
        st.info("👆 请先输入姓名确认身份")
        st.stop()

    # 已确认身份
    st.success(f"当前学生：**{st.session_state['student_name']}**")

    # 错题追踪：获取未通过的知识点，优先推荐
    from utils.db import get_unpassed_topics
    track = get_unpassed_topics(st.session_state["student_id"])
    if track["failed"]:
        failed_tags = "、".join(track["failed"])
        st.warning(f"📌 你还有未通过的知识点：**{failed_tags}**，建议优先复习")
        default_topic = track["failed"][0]
    elif track["pending"]:
        pending_tags = "、".join(track["pending"])
        st.info(f"📋 你还有未答的知识点：**{pending_tags}**")
        default_topic = track["pending"][0]
    else:
        default_topic = None

    # 选择知识点
    topics = get_all_topics()
    if default_topic and default_topic in topics:
        default_idx = topics.index(default_topic)
    else:
        default_idx = 0
    selected_topic = st.selectbox("选择要检验的知识点", topics, index=default_idx, key="student_topic")

    # 获取该知识点题目
    exercises = get_exercises_by_topic(selected_topic)

    if not exercises:
        st.warning("该知识点暂无题目")
        st.stop()

    # 取该知识点的第一道题（预设每个知识点一道题）
    current_ex = exercises[0]

    st.markdown("---")
    st.subheader(f"📝 题目：{current_ex['topic']}")

    # 提示先去操作可视化页面
    st.info(f"💡 **提示**：回答前建议先去 **{current_ex['related_page']}** 页面实际操作可视化实验，再回来回答问题。")

    # 变体出题
    col_question, col_variant = st.columns([4, 1])
    with col_question:
        st.markdown(f"**{current_ex['question']}**")
    with col_variant:
        if st.button("🔄 AI换题", help="AI生成同知识点不同问法的变体题目", use_container_width=True):
            variant = generate_variant_question(
                topic=current_ex["topic"],
                base_question=current_ex["question"],
                keywords=current_ex["keywords"],
                api_key=api_key
            )
            if variant and variant != current_ex["question"]:
                st.session_state["variant_question"] = variant
                st.rerun()
    
    # 如果有变体题目，使用变体题目
    question_text = st.session_state.get("variant_question", current_ex["question"])
    if "variant_question" in st.session_state:
        st.info(f"🔄 AI已生成变体题目，核心概念不变")

    # 答题区
    answer = st.text_area(
        "你的回答",
        placeholder="请详细阐述你的理解...（至少20个字）",
        height=150,
        key=f"answer_{current_ex['id']}"
    )

    # 考试模式：倒计时显示
    js_timer = ""
    if exam_mode and st.session_state.get("exam_started") and not st.session_state.get("exam_finished"):
        js_timer = """
        <div style="background:#E3F2FD; padding:12px; border-radius:8px; text-align:center; margin:10px 0;">
            <span style="font-size:18px; font-weight:700; color:#1565C0;">⏱ 考试剩余时间</span>
            <span id="exam-timer" style="font-size:24px; font-weight:700; color:#D32F2F; margin-left:8px;"></span>
        </div>
        <script>
        var remaining = """ + str(st.session_state.get("exam_time_remaining", 900)) + """;
        function updateTimer() {
            var m = Math.floor(remaining / 60);
            var s = remaining % 60;
            document.getElementById('exam-timer').textContent = m + '分' + (s<10?'0':'') + s + '秒';
            if (remaining <= 0) {
                document.getElementById('exam-timer').textContent = '时间到！';
                document.getElementById('exam-timer').style.color = '#D32F2F';
                return;
            }
            remaining--;
            setTimeout(updateTimer, 1000);
        }
        updateTimer();
        </script>
        """
        st.markdown(js_timer, unsafe_allow_html=True)
    
    # 考试模式：开始按钮
    if exam_mode and not st.session_state.get("exam_started"):
        if st.button("🚀 开始考试", type="primary", use_container_width=True):
            st.session_state["exam_started"] = True
            st.session_state["exam_time_remaining"] = exam_time * 60
            st.session_state["exam_finished"] = False
            st.rerun()

    col_submit, col_clear = st.columns([1, 3])
    with col_submit:
        submit_clicked = st.button("📤 提交回答", type="primary", use_container_width=True)

    if submit_clicked:
        if not api_key:
            st.error("❌ 请先让教师在侧边栏配置 DeepSeek API Key")
        elif len(answer.strip()) < 10:
            st.warning("⚠️ 回答过于简短，请至少写10个字以上")
        else:
            with st.spinner("🤖 AI正在评判你的回答..."):
                result = judge_answer(
                    question=current_ex["question"],
                    keywords=current_ex["keywords"],
                    student_answer=answer,
                    api_key=api_key
                )

            # 存入数据库
            save_submission(
                student_id=st.session_state["student_id"],
                exercise_id=current_ex["id"],
                answer=answer,
                passed=result["passed"],
                score=result["score"],
                feedback=result["feedback"]
            )

            # 考试模式：记录完成
            if exam_mode:
                st.session_state["exam_finished"] = True

            # 展示结果
            st.markdown("---")
            if result["passed"]:
                st.success(f"✅ 通过！得分：{result['score']}/100")
            else:
                st.error(f"❌ 未通过。得分：{result['score']}/100")

            st.markdown(f"**AI评语：** {result['feedback']}")

            # AI个性化学习建议
            if api_key:
                guidance = generate_personalized_guidance(
                    question_topic=current_ex["topic"],
                    keywords=current_ex["keywords"],
                    related_page=current_ex["related_page"],
                    student_answer=answer,
                    passed=result["passed"],
                    score=result["score"],
                    api_key=api_key
                )
                if guidance:
                    st.markdown(f"💡 **AI学习建议：** {guidance}")

            if not result["passed"] and current_ex["related_page"]:
                st.markdown(f"📖 **建议**：回去 **{current_ex['related_page']}** 页面重新操作一遍可视化实验，再回来重新回答。")

            # 清除变体题目（如果存在）
            if "variant_question" in st.session_state:
                del st.session_state["variant_question"]

    # 历史答题记录
    st.markdown("---")
    with st.expander("📜 我的答题记录"):
        records = get_student_submissions(st.session_state["student_id"])
        if not records:
            st.info("暂无答题记录")
        else:
            for r in records:
                emoji = "✅" if r["passed"] else "❌"
                st.markdown(f"{emoji} **{r['topic']}** | 得分 {r['score']} | {r['submitted_at'][:16]}")
                st.markdown(f"> {r['feedback'][:100]}")
                st.markdown("---")
