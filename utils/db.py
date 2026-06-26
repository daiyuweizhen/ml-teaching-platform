"""
数据库管理模块 - SQLite 本地存储
零依赖，Python 内置 sqlite3
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "teaching_data.db")

# 预设的理解题——直接对应现有的可视化页面（共15题，覆盖线性回归/KNN/K-Means/神经网络/CNN/综合）
PRESET_EXERCISES = [
    # ==================== 线性回归（3题） ====================
    {
        "topic": "线性回归",
        "question": "在梯度下降可视化中，如果把学习率从0.05调到0.5，损失曲线出现剧烈震荡。请解释：为什么学习率过大会导致震荡？",
        "keywords": "太大,步长,越过最小值,震荡,发散",
        "related_page": "02_Linear_Regression_线性回归"
    },
    {
        "topic": "线性回归",
        "question": "在线性回归中，损失函数通常使用均方误差（MSE）而非绝对误差（MAE）。请说明：MSE相比MAE有哪些优势？为什么梯度下降配合MSE效果更好？",
        "keywords": "均方误差,MAE,可导性,梯度,异常值,MSE",
        "related_page": "02_Linear_Regression_线性回归"
    },
    {
        "topic": "线性回归",
        "question": "当特征值范围差异很大时（如年龄0-100、收入0-100000），不进行归一化会怎样影响梯度下降？请解释为什么要做特征缩放，以及常用的归一化和标准化方法（Min-Max / Z-Score）的区别。",
        "keywords": "特征缩放,归一化,标准化,梯度下降,等高线,收敛速度",
        "related_page": "02_Linear_Regression_线性回归"
    },
    # ==================== KNN分类（2题） ====================
    {
        "topic": "KNN分类",
        "question": "KNN分类中，当K=1时决策边界非常曲折、锯齿状明显；当K=15时边界变得平滑。请问：K太小和K太大分别会导致什么问题？实际应用中如何选择合适的K值？",
        "keywords": "过拟合,欠拟合,噪声敏感,过于平滑,交叉验证,奇数",
        "related_page": "03_KNN_Classifier_KNN分类"
    },
    {
        "topic": "KNN分类",
        "question": "KNN算法的核心是距离度量。欧氏距离在高维空间中会出现'维度灾难'问题——所有点之间的距离趋于相等。请解释：为什么高维数据下欧氏距离失效？曼哈顿距离和余弦相似度分别适用于什么场景？",
        "keywords": "维度灾难,欧氏距离,曼哈顿距离,余弦相似度,高维,稀疏",
        "related_page": "03_KNN_Classifier_KNN分类"
    },
    # ==================== K-Means聚类（2题） ====================
    {
        "topic": "K-Means聚类",
        "question": "K-Means聚类中，选择K=3和K=10得到的结果完全不同。请说明：K值选择对聚类结果有什么影响？如何利用肘部法则（Elbow Method）或轮廓系数确定合适的K值？",
        "keywords": "肘部法则,轮廓系数,簇内平方和,过分割,欠分割,业务含义",
        "related_page": "04_KMeans_Clustering_K均值聚类"
    },
    {
        "topic": "K-Means聚类",
        "question": "K-Means对初始质心位置非常敏感，不同的随机初始化可能导致收敛到不同的局部最优解。请解释：K-Means++初始化算法如何改善这一问题？除了K-Means++，还有哪些方法可以避免陷入局部最优？",
        "keywords": "K-Means++,初始化,局部最优,多次运行,全局最优,质心",
        "related_page": "04_KMeans_Clustering_K均值聚类"
    },
    # ==================== 神经网络（3题） ====================
    {
        "topic": "神经网络",
        "question": "在神经网络可视化中，把隐藏层从1层逐渐增加到5层时，训练准确率持续上升但测试准确率先升后降。请解释：为什么会发生这种情况？这种现象叫什么？如何解决？",
        "keywords": "过拟合,泛化能力,模型容量,正则化,Dropout,早停",
        "related_page": "06_Neural_Network_神经网络"
    },
    {
        "topic": "神经网络",
        "question": "神经网络中常用的激活函数有Sigmoid、Tanh、ReLU和Leaky ReLU。请比较这四种激活函数的优缺点，并说明为什么现代深度学习中ReLU及其变体更受欢迎？Sigmoid在什么场景下仍然有用？",
        "keywords": "激活函数,梯度消失,Sigmoid,ReLU,Leaky ReLU,输出层",
        "related_page": "06_Neural_Network_神经网络"
    },
    {
        "topic": "神经网络",
        "question": "在训练神经网络时，学习率设置为0.001收敛很慢，设为0.1则完全不收敛。请解释：学习率如何影响训练过程？自适应优化器（如Adam）相比固定学习率的SGD有什么优势？Adam为什么通常能自动调整等效学习率？",
        "keywords": "学习率,Adam,SGD,自适应,动量,收敛",
        "related_page": "06_Neural_Network_神经网络"
    },
    # ==================== CNN卷积神经网络（2题） ====================
    {
        "topic": "CNN",
        "question": "卷积神经网络（CNN）中，一个3x3的卷积核在32x32的图像上滑动，stride=1、padding=same。请问：卷积核的每一个权重参数代表什么含义？不同的卷积核（如边缘检测核、模糊核）提取的特征有什么不同？为什么CNN能用远少于全连接层的参数提取图像特征？",
        "keywords": "卷积核,参数共享,局部感受野,边缘检测,特征图,权重",
        "related_page": "06_Neural_Network_神经网络"
    },
    {
        "topic": "CNN",
        "question": "CNN中池化层（Pooling）通常跟在卷积层之后。最大池化（Max Pooling）和平均池化（Average Pooling）的作用分别是什么？池化层为什么能使模型获得平移不变性？去掉所有池化层、只用步长大于1的卷积来降维，会有什么优缺点？",
        "keywords": "池化,Max Pooling,平移不变性,降维,步长,感受野",
        "related_page": "06_Neural_Network_神经网络"
    },
    # ==================== 综合题（3题） ====================
    {
        "topic": "模型对比",
        "question": "在模型对比页面中，同一数据集上KNN的准确率达95%而线性回归只有72%。请问：为什么不同算法在同一数据集上表现差异这么大？选择算法时应该考虑哪些因素？",
        "keywords": "数据分布,线性可分,非线性,模型假设,偏差方差,适用场景",
        "related_page": "05_Model_Comparison_模型对比"
    },
    {
        "topic": "综合",
        "question": "在一个实际的机器学习项目中，你发现增加训练样本从1000条到10000条后，线性回归的准确率几乎没有提升，但KNN的准确率大幅提升。请从两种算法的模型容量和数据需求角度解释这个现象，并给出何时应该花时间收集更多数据的建议。",
        "keywords": "样本量,模型容量,数据需求,线性模型,非参数模型,数据收集",
        "related_page": "05_Model_Comparison_模型对比"
    },
    {
        "topic": "综合",
        "question": "在机器学习项目中，模型在训练集上表现完美（99%准确率）但在测试集上只有65%。请从偏差-方差权衡的角度分析：这是高偏差还是高方差问题？使用正则化、增加训练数据、减少模型复杂度三种策略分别适用于哪种情况？请举例说明。",
        "keywords": "偏差方差权衡,过拟合,欠拟合,正则化,模型复杂度,泛化",
        "related_page": "05_Model_Comparison_模型对比"
    }
]


def get_db_path():
    """获取数据库文件路径"""
    return DB_PATH


def init_db():
    """初始化数据库，创建表并插入预设题目"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 学生表
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class_name TEXT DEFAULT '25D智能技术班',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 题目表
    c.execute("""
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            question TEXT NOT NULL,
            keywords TEXT NOT NULL,
            related_page TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 答题记录表
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            exercise_id INTEGER,
            answer TEXT,
            passed INTEGER DEFAULT 0,
            score INTEGER DEFAULT 0,
            feedback TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (exercise_id) REFERENCES exercises(id)
        )
    """)

    # 插入预设题目（逐条幂等：按 question 文本判断是否已存在，支持增量新增）
    for ex in PRESET_EXERCISES:
        c.execute("SELECT COUNT(*) FROM exercises WHERE question = ?", (ex["question"],))
        if c.fetchone()[0] == 0:
            c.execute(
                "INSERT INTO exercises (topic, question, keywords, related_page) VALUES (?, ?, ?, ?)",
                (ex["topic"], ex["question"], ex["keywords"], ex["related_page"])
            )

    conn.commit()
    conn.close()


def ensure_db():
    """确保数据库已初始化，返回连接"""
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_or_create_student(name, class_name="25D智能技术班"):
    """获取或创建学生记录，返回学生ID"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("SELECT id FROM students WHERE name = ?", (name,))
    row = c.fetchone()
    if row:
        student_id = row["id"]
    else:
        c.execute("INSERT INTO students (name, class_name) VALUES (?, ?)", (name, class_name))
        conn.commit()
        student_id = c.lastrowid
    conn.close()
    return student_id


def get_exercises_by_topic(topic=None):
    """获取题目列表，可按知识点筛选"""
    conn = ensure_db()
    c = conn.cursor()
    if topic and topic != "全部":
        c.execute("SELECT * FROM exercises WHERE topic = ? ORDER BY id", (topic,))
    else:
        c.execute("SELECT * FROM exercises ORDER BY id")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_topics():
    """获取所有知识点列表"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("SELECT DISTINCT topic FROM exercises ORDER BY topic")
    rows = c.fetchall()
    conn.close()
    return [r["topic"] for r in rows]


def save_submission(student_id, exercise_id, answer, passed, score, feedback):
    """保存一次答题记录"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO submissions (student_id, exercise_id, answer, passed, score, feedback) VALUES (?, ?, ?, ?, ?, ?)",
        (student_id, exercise_id, answer, 1 if passed else 0, score, feedback)
    )
    conn.commit()
    conn.close()


def get_student_submissions(student_id):
    """获取某个学生的所有答题记录"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("""
        SELECT s.id, e.topic, e.question, s.answer, s.passed, s.score, s.feedback, s.submitted_at
        FROM submissions s
        JOIN exercises e ON s.exercise_id = e.id
        WHERE s.student_id = ?
        ORDER BY s.submitted_at DESC
    """, (student_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_topic_stats():
    """获取各知识点的答题统计"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("""
        SELECT e.topic,
               COUNT(s.id) as total,
               SUM(s.passed) as passed_count,
               ROUND(AVG(s.score), 1) as avg_score
        FROM exercises e
        LEFT JOIN submissions s ON e.id = s.exercise_id
        GROUP BY e.topic
        ORDER BY avg_score ASC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_submissions_detail():
    """获取所有学生答题明细"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("""
        SELECT st.name, e.topic, e.question, s.answer, s.passed, s.score, s.submitted_at
        FROM submissions s
        JOIN students st ON s.student_id = st.id
        JOIN exercises e ON s.exercise_id = e.id
        ORDER BY s.submitted_at DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unpassed_topics(student_id):
    """
    获取学生未通过的知识点列表（从未答过或最近一次未通过）
    用于错题追踪：学生再次进入时优先展示这些知识点
    """
    conn = ensure_db()
    c = conn.cursor()
    
    # 获取所有知识点
    all_topics = set()
    c.execute("SELECT DISTINCT topic FROM exercises")
    for row in c.fetchall():
        all_topics.add(row["topic"])
    
    # 获取该学生最近一次通过的知识点
    c.execute("""
        SELECT e.topic, s.passed, s.submitted_at
        FROM submissions s
        JOIN exercises e ON s.exercise_id = e.id
        WHERE s.student_id = ?
        ORDER BY s.submitted_at DESC
    """, (student_id,))
    
    latest_by_topic = {}
    for row in c.fetchall():
        topic = row["topic"]
        if topic not in latest_by_topic:
            latest_by_topic[topic] = row["passed"]
    
    conn.close()
    
    # 未通过的知识点 = 全部知识点 - 最近一次通过的知识点
    failed = [t for t in all_topics if latest_by_topic.get(t, 0) == 0]
    done = [t for t in all_topics if latest_by_topic.get(t, 0) == 1]
    pending = [t for t in all_topics if t not in latest_by_topic]
    
    return {
        "failed": failed,
        "done": done,
        "pending": pending
    }


def get_student_stat(student_id):
    """获取单个学生的统计"""
    conn = ensure_db()
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*) as total,
               SUM(passed) as passed,
               ROUND(AVG(score), 1) as avg_score
        FROM submissions
        WHERE student_id = ?
    """, (student_id,))
    row = c.fetchone()
    conn.close()
    return {
        "total": row["total"] or 0,
        "passed": row["passed"] or 0,
        "avg_score": row["avg_score"] or 0
    }


def get_improvement_stats():
    """
    获取学生重复答题的进步统计（用于教学效果证据）
    统计有多次提交的学生的分数变化趋势
    """
    conn = ensure_db()
    c = conn.cursor()
    
    # 找有多次提交的学生（至少2次提交）
    c.execute("""
        SELECT student_id, COUNT(*) as cnt
        FROM submissions
        GROUP BY student_id
        HAVING cnt >= 2
    """)
    multi_students = [row["student_id"] for row in c.fetchall()]
    
    if not multi_students:
        conn.close()
        return {"count": 0, "avg_first_score": 0, "avg_latest_score": 0, "improvement": 0}
    
    first_scores = []
    latest_scores = []
    
    for sid in multi_students:
        c.execute("""
            SELECT score FROM submissions
            WHERE student_id = ?
            ORDER BY submitted_at ASC
            LIMIT 1
        """, (sid,))
        first = c.fetchone()
        if first:
            first_scores.append(first["score"])
        
        c.execute("""
            SELECT score FROM submissions
            WHERE student_id = ?
            ORDER BY submitted_at DESC
            LIMIT 1
        """, (sid,))
        latest = c.fetchone()
        if latest:
            latest_scores.append(latest["score"])
    
    conn.close()
    
    if first_scores and latest_scores:
        avg_first = round(sum(first_scores) / len(first_scores), 1)
        avg_latest = round(sum(latest_scores) / len(latest_scores), 1)
        return {
            "count": len(multi_students),
            "avg_first_score": avg_first,
            "avg_latest_score": avg_latest,
            "improvement": round(avg_latest - avg_first, 1)
        }
    return {"count": 0, "avg_first_score": 0, "avg_latest_score": 0, "improvement": 0}
