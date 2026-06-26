"""
DeepSeek API 调用封装
用于判题——将学生回答与标准关键词比对，由AI语义判断
"""

import json
import requests


def call_deepseek(prompt, api_key, temperature=0.3, max_tokens=1024):
    """
    调用 DeepSeek API
    
    Args:
        prompt: 提示词
        api_key: API密钥
        temperature: 温度参数 (判题用低温度，保证一致性)
        max_tokens: 最大输出 token 数
    
    Returns:
        dict: {"success": bool, "content": str, "error": str}
    """
    if not api_key:
        return {"success": False, "content": "", "error": "未配置 API Key"}

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return {"success": True, "content": content, "error": ""}
        else:
            error_msg = response.json().get("error", {}).get("message", f"HTTP {response.status_code}")
            return {"success": False, "content": "", "error": error_msg}
    
    except requests.exceptions.Timeout:
        return {"success": False, "content": "", "error": "API 请求超时，请检查网络"}
    except Exception as e:
        return {"success": False, "content": "", "error": f"API 调用异常: {str(e)}"}


def judge_answer(question, keywords, student_answer, api_key):
    """
    判题：将题目、标准关键词和学生回答发给 DeepSeek 进行语义判断
    
    Args:
        question: 题目文本
        keywords: 标准答案关键词（逗号分隔）
        student_answer: 学生回答文本
        api_key: DeepSeek API Key
    
    Returns:
        dict: {"passed": bool, "score": int, "feedback": str, "raw": str}
    """
    prompt = f"""你是一位严格但耐心的AI课程助教。请根据以下标准评判学生回答。

【题目】
{question}

【标准答案关键词】
{keywords}

【学生回答】
{student_answer}

【评分规则】
- 学生回答覆盖了大部分关键词且理解正确 → passed=true, score=80-100
- 学生回答覆盖了部分关键词、理解基本正确但不够完整 → passed=true, score=60-79
- 学生回答理解有偏差、遗漏重要概念 → passed=false, score=30-59
- 学生回答完全错误或答非所问 → passed=false, score=0-29

【要求】
请严格返回以下JSON格式，不要有任何额外文字：
{{"passed": true或false, "score": 0到100的整数, "feedback": "你的评语，指出哪里对、哪里不足、建议如何改进，用中文，150字以内"}}

如果学生回答过于简短（少于10个字），直接判定为未通过，feedback中提醒"请更详细地回答"。
"""
    
    result = call_deepseek(prompt, api_key, temperature=0.1)
    
    if not result["success"]:
        return {
            "passed": False,
            "score": 0,
            "feedback": f"判题系统异常: {result['error']}",
            "raw": ""
        }
    
    raw = result["content"]
    
    # 解析 DeepSeek 返回的 JSON
    try:
        # 清洗可能的 markdown 代码块包裹
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
        
        parsed = json.loads(clean)
        return {
            "passed": bool(parsed.get("passed", False)),
            "score": int(parsed.get("score", 0)),
            "feedback": str(parsed.get("feedback", "")),
            "raw": raw
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # JSON 解析失败，尝试从文本中提取
        return {
            "passed": False,
            "score": 0,
            "feedback": f"判题结果解析异常，原始返回：{raw[:200]}",
            "raw": raw
        }


def generate_personalized_guidance(question_topic, keywords, related_page, student_answer, passed, score, api_key):
    """
    生成个性化学习建议——告诉学生应该回去做哪个实验、调节哪个参数
    """
    if not api_key:
        return ""
    
    if passed:
        prompt = f"""学生已经通过了"{question_topic}"知识点的检验（得分{score}）。请给出1-2条进阶建议，包括：建议在"{related_page}"可视化页面中做哪个更深入的实验、调节哪个参数来加深理解。用中文，80字以内。"""
    else:
        prompt = f"""学生在"{question_topic}"知识点未通过（得分{score}）。标准答案是{keywords}。请给出针对性建议：告诉他应该回到"{related_page}"可视化页面中，具体调节哪个参数、观察什么现象来理解薄弱的概念。用中文，80字以内。"""
    
    result = call_deepseek(prompt, api_key, temperature=0.3, max_tokens=256)
    return result["content"] if result["success"] else ""


def generate_variant_question(topic, base_question, keywords, api_key):
    """
    基于原题生成同知识点的变体题目，防止学生背答案
    """
    if not api_key:
        return base_question  # fallback to original
    
    prompt = f"""你是一位AI课程教师。请基于以下信息，生成一道同知识点的变体题目（换一个问法/换一个场景，但检验的是相同的核心概念）。

知识点：{topic}
原题：{base_question}
核心概念关键词：{keywords}

要求：
1. 题目从不同角度提问（例如原题问"为什么导致震荡"，变体可问"如何判断学习率是否合适"）
2. 保持相同的难度水平
3. 用中文，50-150字
4. 只返回题目文本，不要任何额外说明"""

    result = call_deepseek(prompt, api_key, temperature=0.7, max_tokens=512)
    if result["success"] and len(result["content"]) > 10:
        return result["content"].strip()
    return base_question


def generate_teaching_diagnosis(topic_stats_text, api_key):
    """
    根据全班统计数据生成AI教学诊断报告
    """
    if not api_key:
        return "（需要配置 DeepSeek API Key 才能生成 AI 教学诊断）"
    
    prompt = f"""你是一位教学数据分析专家。请根据以下全班学生AI理解力检验数据，生成一份简明的教学诊断报告。

数据：
{topic_stats_text}

请从以下角度分析（200-400字，用中文）：
1. 班级整体掌握情况概述（最弱知识点的风险等级）
2. 建议的课堂教学策略调整（重点讲解哪个知识点、建议用什么可视化辅助）
3. 对学生个别辅导的建议（分层教学建议）

格式要求：分3段，每段一个小标题。语言简洁专业，像给教师同事的建议。"""

    result = call_deepseek(prompt, api_key, temperature=0.5, max_tokens=1024)
    if result["success"]:
        return result["content"]
    return "AI诊断生成失败，请检查API Key配置。"
