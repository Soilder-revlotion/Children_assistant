"""统一数据 Schema 和工具函数"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

AGE_RANGES = ["备孕", "孕期", "0-1月", "1-6月", "6-12月", "1-3岁", "3-6岁", "6岁+", "通用"]

CATEGORIES = [
    "喂养", "睡眠", "健康", "发育", "早教", "心理",
    "安全", "疫苗", "疾病", "孕期保健", "产后护理",
    "家庭教育", "幼儿园", "行为习惯", "营养", "皮肤护理",
    "常见病", "用药", "急救", "其他"
]


@dataclass
class KnowledgeItem:
    """育儿知识统一条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""           # 来源：matinf/dxy/zhihu/babybook/xiaohongshu
    type: str = ""             # 类型：qa/article/experience/knowledge
    title: str = ""
    question: str = ""         # QA 类的问题
    content: str = ""          # 正文/回答
    age_range: str = "通用"     # 适用年龄段
    category: str = "其他"      # 育儿子分类
    quality_score: float = 0.0
    url: str = ""
    crawled_at: str = field(default_factory=lambda: datetime.now().isoformat())
    content_hash: str = ""     # 用于去重

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(
                self.content.encode("utf-8")
            ).hexdigest()


AGRE_BOOST_KEYWORDS = {
    # 年龄段关键词映射
    "备孕": ["备孕", "怀孕前", "准备怀孕", "孕前", "叶酸", "排卵"],
    "孕期": ["孕期", "怀孕", "孕妇", "产检", "胎动", "孕吐", "妊娠", "分娩", "坐月子"],
    "0-1月": ["新生儿", "满月", "月子", "初生", "刚出生"],
    "1-6月": ["婴儿", "小月龄", "母乳", "配方奶", "胀气", "肠绞痛", "夜醒"],
    "6-12月": ["辅食", "爬行", "出牙", "断奶", "学步"],
    "1-3岁": ["幼儿", "断奶", "学步", "说话", "如厕", "任性", "打人"],
    "3-6岁": ["学龄前", "幼儿园", "入园", "社交", "识字", "专注力"],
}

CATEGORY_KEYWORDS = {
    "喂养": ["喂奶", "母乳", "配方奶", "辅食", "断奶", "厌奶", "奶粉", "奶瓶", "吃饭", "挑食"],
    "睡眠": ["睡觉", "哄睡", "夜醒", "睡眠", "入睡", "夜奶", "白天觉", "作息"],
    "健康": ["体检", "身高", "体重", "发育", "生长", "头围", "视力", "听力"],
    "发育": ["翻身", "爬行", "走路", "说话", "出牙", "精细动作", "大运动", "认知"],
    "早教": ["早教", "启蒙", "绘本", "亲子阅读", "游戏", "玩具", "识字", "英语"],
    "心理": ["情绪", "安全感", "分离焦虑", "粘人", "哭闹", "胆小"],
    "安全": ["安全", "摔倒", "烫伤", "窒息", "触电", "溺水", "出行安全"],
    "疫苗": ["疫苗", "接种", "预防针", "免疫"],
    "疾病": ["发烧", "感冒", "咳嗽", "腹泻", "湿疹", "过敏", "肺炎", "黄疸"],
    "孕期保健": ["产检", "B超", "唐筛", "糖耐", "胎心", "宫缩", "见红"],
    "产后护理": ["产后", "恶露", "盆底肌", "腹直肌", "月子", "哺乳期"],
    "行为习惯": ["如厕", "刷牙", "洗手", "收纳", "礼貌", "分享"],
}

BANNED_KEYWORDS = ["广告", "招商", "加盟", "代理", "加微信", "扫码", "点击购买", "下单"]


def guess_age_range(text: str) -> str:
    """根据文本内容猜测适用年龄段"""
    scores = {}
    for age, keywords in AGRE_BOOST_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[age] = score
    if not scores:
        return "通用"
    return max(scores, key=scores.get)


def guess_category(text: str) -> str:
    """根据文本内容猜测分类"""
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[cat] = score
    if not scores:
        return "其他"
    return max(scores, key=scores.get)


def is_valid_knowledge(item: KnowledgeItem) -> bool:
    """检查条目是否有效"""
    if not item.content or len(item.content) < 20:
        return False
    text = item.title + item.question + item.content
    for kw in BANNED_KEYWORDS:
        if kw in text:
            return False
    return True


def chinese_char_ratio(text: str) -> float:
    """计算中文字符占比"""
    if not text:
        return 0.0
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    return chinese / len(text)
