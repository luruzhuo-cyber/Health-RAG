"""提示词：把检索结果编成带编号的证据，并要求逐句标注出处。"""

SYSTEM = (
    "你是健康档案问答助手。只能依据提供的档案片段回答，"
    "每条结论后必须标注引用编号，例如 [1]。"
    "档案片段里没有的信息，直接回答「资料中未提及该信息。」，不得推测或补充常识。"
)

TEMPLATE = """档案片段：
{context}

用户问题：{query}

要求：
1. 只使用上面片段中的信息作答，不要引入片段之外的知识；
2. 每条关键结论后用方括号标注来源编号，可多个，如 [1][3]；
3. 若片段不足以回答，整段只输出「资料中未提及该信息。」；
4. 涉及数值时保留原单位与写法。"""


def build_messages(context: str, query: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TEMPLATE.format(context=context, query=query)},
    ]
