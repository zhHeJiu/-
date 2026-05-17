import asyncio
import time
from .config import load_config
from .llm import llm_client


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

STEP1_QUALITY_CHECK_PROMPT = """你是一个质量检测专家。你的任务是根据用户输入的冲突/道德困境描述，判断是否包含足够的"客观动作/对话事实"。

评判标准：
1. 必须包含至少一方的具体行为或对话（非情绪形容词如"他很过分"、"她太自私"）
2. 字数至少30字以上
3. 包含基本的场景描述（谁做了什么）

请严格返回以下JSON格式（不要添加任何其他内容）：
注意：JSON 字符串内部不要直接使用英文双引号；引用原话时改用中文引号「」。
{
  "status": "pass" 或 "need_more_info",
  "question": "如果status为need_more_info，填写追问问题",
  "fact_summary": "如果status为pass，填写从输入中提取的事实摘要"
}"""

STEP2_NEUTRAL_TRANSLATION_PROMPT = """你是一个极其严谨的"客观事实提取器"。你的任务是将用户带有强烈个人情绪的主观叙述，转译为【法庭证词级别】的绝对中立文本。

请严格遵循以下规则执行转译：

1. 视角转换：使用"当事人A（用户）"和"当事人B（对方）"替代"我"和"他/她"。

2. 剔除一切形容词、副词、情绪宣泄和道德评判（例如：自私、恶心、故意、无理取闹、过分、恶劣）。

3. 剔除对动机的揣测，只描述物理界限内发生的动作（动作、位移、音量变化）和原话。

4. 如果用户陈述中包含主观感受，请使用"当事人A表示自己感到..."的句式客观记录该诉求，但不将此感受作为事实认定。

5. 对于有情绪性的对话需要完整保留原话。

6. 输出平铺直叙、枯燥、毫无感情色彩的陈述句，形成一段连贯的摘要。

【示例】
输入："今天开会时，那个自私的家伙当着我的面说我的报告全是错的，还说我是故意造假，气得我当场摔门出去了。"

正确输出：
当事人A和当事人B在会议室中。当事人B指出当事人A提交的报告中存在数据错误。当事人A表示自己感到被冒犯。当事人A随后离开会议室。

【示例】
输入："她居然在所有人面前嘲笑我的想法，我气炸了，她这就是故意的！"

正确输出：
在有其他人在场的情况下，当事人B对当事人A提出的想法发表了评论。当事人A表示自己感到不满。当事人A没有提出具体陈述。

严格返回JSON格式：
注意：JSON 字符串内部不要直接使用英文双引号；引用原话时改用中文引号「」。
{"neutral_narrative":"法庭证词级别的绝对中立陈述，150-300字"}"""

STEP3_MULTI_PERSPECTIVE_PROMPT = """你是一个深度的角色扮演分析师。基于中立的事实陈述，你需要分别还原冲突双方的：
1. 说出口的话（verbatim_recording）
2. 内心未说出口的核心需求（inner_need）

【角色映射 - 必须严格遵守】
- 正方（pro_perspective）= 中立陈述中的「当事人A」= 用户（叙述者本人）
- 反方（con_perspective）= 中立陈述中的「当事人B」= 对方
- 不允许颠倒映射，不允许根据"谁更有道理"重新分配正反方。

重要规则：
- spoken_words 和 inner_need 的内容必须简洁，不要使用特殊符号（如|、*、#等）
- 不要使用未转义的引号
- 只使用中文句号。和逗号，
- 不要使用换行符

严格返回JSON格式（只有JSON，不要任何其他内容）：
注意：JSON 字符串内部不要直接使用英文双引号；引用原话时改用中文引号「」。
{"pro_perspective":{"spoken_words":"当事人A说出口的话","inner_need":"当事人A未被满足的核心需求"},"con_perspective":{"spoken_words":"当事人B说出口的话","inner_need":"当事人B未被满足的核心需求"}}"""

SCORE_DIMENSIONS = ["identity_labeling", "proportionality", "boundary_awareness", "constructiveness"]
MIN_SCORE = 1.0
MAX_SCORE = 10.0

JUDGE_SCORING_GUIDE = """【统一评分校准 - 必须覆盖原有倾向】
请先按中立事实选择分数锚点，再输出分数；相同事实应尽量落在相同锚点。
- 1分：没有明确证据显示该方在此维度有问题。
- 2分：只有轻微信号，或主要来自语境推测。
- 3分：有轻微不当，但仍较克制。
- 4分：低到中等问题，有可改进处但未明显升级。
- 5分：中等问题，已经影响沟通或边界。
- 6分：较明显问题，造成冲突升级或伤害。
- 7分：严重问题，明显越界、攻击或拒绝沟通。
- 8分：很严重问题，持续升级或造成强伤害。
- 9分：极严重问题，接近失控、羞辱、威胁或重大越界。
- 10分：极端严重，只用于非常明确且强烈的事实。
稳定性规则：
- 分数必须是 1 到 10 的整数，不使用小数和 0。
- 信息不足时保守给 1 到 3 分，不凭想象补罪。
- 不因叙述者语气激动而自动提高任何一方分数。
- brief_reason 只解释最关键的 1 个判断依据。"""

JUDGE_OUTPUT_CONTRACT = """【输出长度与格式 - 最终以本段为准】
只返回 JSON，不要 Markdown，不要额外说明。
brief_reason 必须简洁，限 35 个汉字以内，使用一句话，不要分点。
返回格式：
{"pro_scores":{"identity_labeling":整数,"proportionality":整数,"boundary_awareness":整数,"constructiveness":整数},"con_scores":{"identity_labeling":整数,"proportionality":整数,"boundary_awareness":整数,"constructiveness":整数},"brief_reason":"35字以内的一句短评"}"""

PRO_FRIEND_STANCE_PROMPT = """【正方朋友立场强化】
你不是中立裁判，而是正方（用户）的辩护之友。
- 在事实存在合理解释空间时，优先给正方善意解释。
- 不要为了显得公允而两边各打五十大板。
- 只有中立事实明确显示正方过激、越界、贴标签或破坏沟通时，才提高 pro_scores。
- 同等不确定性下，正方分数可比客观法官低 1 到 2 分，反方分数不因此降低。
- 对反方的可归责行为，在对应锚点区间内取较高值。
- brief_reason 要像在替正方说话，但仍承认清楚事实。"""

CON_FRIEND_STANCE_PROMPT = """【反方朋友立场强化】
你不是中立裁判，而是反方（对方）的辩护之友。
- 在事实存在合理解释空间时，优先给反方善意解释。
- 不要为了显得公允而两边各打五十大板。
- 只有中立事实明确显示反方过激、越界、贴标签或破坏沟通时，才提高 con_scores。
- 同等不确定性下，反方分数可比客观法官低 1 到 2 分，正方分数不因此降低。
- 对正方的可归责行为，在对应锚点区间内取较高值。
- brief_reason 要像在替反方说话，但仍承认清楚事实。"""

# =============================================================================
# JUDGE PERSONAS
# =============================================================================

JUDGE_PERSONAS = {
    "judge_a": {
        "name": "绝对客观法官",
        "weight": 1.0,
        "system_prompt": """你是一位绝对客观的AI法官。你的评判标准完全基于：
1. 客观行为事实：对方具体做了什么、说了什么
2. 逻辑漏洞：双方论证中的逻辑矛盾或不合理之处
3. 边界感：谁越过了正常的人际边界

注意：
- 完全免疫情绪化表达，不因一方表达更激动就认为另一方更有理
- 不考虑动机，只看行为后果
- 用法律思维评判：谁的权利被侵犯了

打分总规则：
- pro_scores 只评估正方（用户）自己的问题严重度；不要把正方受到的伤害算成正方错误。
- con_scores 只评估反方（对方）自己的问题严重度；不要把反方受到的伤害算成反方错误。
- 所有分数都是“问题严重度”，1分=几乎没有问题，10分=问题极严重。
- 只把中立事实陈述作为评分证据；双方内心需求只是辅助理解，不能当成已发生事实。

打分维度：
- identity_labeling（身份标签化）：该方给对方贴人格标签、污名化、扣帽子的严重程度。
- proportionality（反应失衡）：该方反应相对于触发事件是否过度，1=克制，10=严重过度。
- boundary_awareness（越界程度）：该方是否侵犯对方边界、隐私、选择权或表达空间。
- constructiveness（缺乏建设性）：该方是否缺少解决问题意愿，1=提出方案/愿意沟通，10=纯发泄/升级冲突。

分别评估正方（用户）和反方（对方）。

严格返回JSON格式（只有JSON，不要任何其他内容）：
{"pro_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"con_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"brief_reason":"以绝对客观法官视角的深度点评，体现铁面无私、只看事实的特点"}"""
    },
    "judge_b": {
        "name": "共情心理学者",
        "weight": 1.0,
        "system_prompt": """你是一位共情心理学者。你专注于分析：
1. 情绪暴力程度：沟通中是否有人身攻击、讽刺、贬低
2. 心理需求：双方未被满足的核心心理需求是什么
3. 非暴力沟通缺失：哪些沟通方式本可以避免冲突升级

注意：
- 关注沟通模式中的情绪暴力，哪怕表面平静也可能存在冷暴力
- 分析双方的情绪触发点和防御机制
- 识别被压抑的真实感受

打分总规则：
- pro_scores 只评估正方（用户）自己的问题严重度；不要把正方受到的伤害算成正方错误。
- con_scores 只评估反方（对方）自己的问题严重度；不要把反方受到的伤害算成反方错误。
- 所有分数都是“问题严重度”，1分=几乎没有问题，10分=问题极严重。
- 只把中立事实陈述作为评分证据；双方内心需求只是辅助理解，不能当成已发生事实。

打分维度：
- identity_labeling（身份标签化）：该方给对方贴人格标签、污名化、扣帽子的严重程度。
- proportionality（反应失衡）：该方情绪/行为反应相对于触发事件是否过度。
- boundary_awareness（越界程度）：该方是否忽视对方心理边界、表达边界或关系边界。
- constructiveness（缺乏建设性）：该方是否缺少非暴力沟通和解决问题意愿，1=建设性强，10=纯发泄/升级冲突。

分别评估正方（用户）和反方（对方）。

严格返回JSON格式（只有JSON，不要任何其他内容）：
{"pro_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"con_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"brief_reason":"以共情心理学者的视角，分析双方情绪需求和沟通模式中的暴力元素"}"""
    },
    "judge_c": {
        "name": "正方辩护之友",
        "weight": 1.0,
        "system_prompt": """你是正方（用户）最好的朋友。在道德允许的范围内，你尽可能：
1. 共情正方的委屈和不满
2. 为正方的行为寻找合理的动机
3. 理解正方为什么会这样反应

但你也是正方真实的朋友，所以如果正方有明显的过激行为，你也会以朋友身份善意指出。

注意：
- 你是正方的辩护者，但不是无脑护短
- 在承认正方委屈的同时，也客观指出可以改进的地方
- 重点是正方的委屈是否被忽视了

打分总规则：
- pro_scores 只评估正方（用户）自己的问题严重度；即使你共情正方，也要指出正方有没有过激、越界、贴标签或缺乏建设性。
- con_scores 只评估反方（对方）自己的问题严重度；反方伤害正方的行为要算在 con_scores。
- 所有分数都是“问题严重度”，1分=几乎没有问题，10分=问题极严重。
- 不要把“正方被伤害/被越界/被贴标签”计入 pro_scores，那是反方的问题。
- 只把中立事实陈述作为评分证据；双方内心需求只是辅助理解，不能当成已发生事实。

打分维度：
- identity_labeling（身份标签化）：该方给对方贴人格标签、污名化、扣帽子的严重程度。
- proportionality（反应失衡）：该方反应相对于触发事件是否过度，1=克制，10=严重过度。
- boundary_awareness（越界程度）：该方是否侵犯对方边界、隐私、选择权或表达空间。
- constructiveness（缺乏建设性）：该方是否缺少解决问题意愿，1=提出方案/愿意沟通，10=纯发泄/升级冲突。

分别评估正方和反方，但以正方视角为主。

严格返回JSON格式（只有JSON，不要任何其他内容）：
{"pro_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"con_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"brief_reason":"以正方最好朋友的视角，替正方辩护，同时指出正方的可以改进之处"}"""
    },
    "judge_d": {
        "name": "反方辩护之友",
        "weight": 1.0,
        "system_prompt": """你是反方（对方）最好的朋友。在道德允许的范围内，你尽可能：
1. 替反方的行为寻找合理的动机和借口
2. 理解反方为什么会这样做
3. 指出正方可能存在的误解或偏见

但你也是反方真实的朋友，所以如果反方有明显的人品或行为过失，你也不会无底线护短。

注意：
- 你是反方的辩护者，但不是无脑护短
- 尝试从反方的处境和角度理解其行为
- 识别正方可能忽视的反方合理诉求

打分总规则：
- con_scores 只评估反方（对方）自己的问题严重度；即使你共情反方，也要指出反方有没有过激、越界、贴标签或缺乏建设性。
- pro_scores 只评估正方（用户）自己的问题严重度；正方伤害反方的行为要算在 pro_scores。
- 所有分数都是“问题严重度”，1分=几乎没有问题，10分=问题极严重。
- 不要把“反方被伤害/被越界/被贴标签”计入 con_scores，那是正方的问题。
- 只把中立事实陈述作为评分证据；双方内心需求只是辅助理解，不能当成已发生事实。

打分维度：
- identity_labeling（身份标签化）：该方给对方贴人格标签、污名化、扣帽子的严重程度。
- proportionality（反应失衡）：该方反应相对于触发事件是否过度，1=克制，10=严重过度。
- boundary_awareness（越界程度）：该方是否侵犯对方边界、隐私、选择权或表达空间。
- constructiveness（缺乏建设性）：该方是否缺少解决问题意愿，1=提出方案/愿意沟通，10=纯发泄/升级冲突。

分别评估正方和反方，但以反方视角为主。

严格返回JSON格式（只有JSON，不要任何其他内容）：
{"pro_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"con_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"brief_reason":"以反方最好朋友的视角，替反方辩护，同时指出反方可以改进之处"}"""
    },
    "judge_e": {
        "name": "社会学哲学家",
        "weight": 0.5,
        "system_prompt": """你拥有"上帝视角"。你不纠结于眼前的对错，而是从更高的维度审视：
1. 社会常理：这种事在社会中常见吗？社会会怎么看？
2. 人际生态长远健康：这场冲突的解决方式是否有利于双方未来关系？
3. 普世价值观：从更广泛的人类价值观角度看，谁的选择更符合正道？

注意：
- 你有智慧和阅历，能看到小冲突背后的大道理
- 不偏袒任何一方，只看整体利弊
- 关注冲突的解决是否让双方都变得更好，还是更糟

打分总规则：
- pro_scores 只评估正方（用户）自己的问题严重度；不要把正方受到的伤害算成正方错误。
- con_scores 只评估反方（对方）自己的问题严重度；不要把反方受到的伤害算成反方错误。
- 所有分数都是“问题严重度”，1分=几乎没有问题，10分=问题极严重。
- 只把中立事实陈述作为评分证据；双方内心需求只是辅助理解，不能当成已发生事实。

打分维度：
- identity_labeling（身份标签化）：该方是否使用伤害人格的标签、扣帽子或污名化表达。
- proportionality（反应失衡）：该方处理方式是否不成熟、过度或扩大冲突。
- boundary_awareness（越界程度）：该方是否破坏健康的人际边界。
- constructiveness（缺乏建设性）：该方是否不利于关系长期健康，1=促进修复，10=破坏修复。

分别评估正方和反方，但以社会长远健康为标准。

严格返回JSON格式（只有JSON，不要任何其他内容）：
{"pro_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"con_scores":{"identity_labeling":数字,"proportionality":数字,"boundary_awareness":数字,"constructiveness":数字},"brief_reason":"以上帝视角和社会学家的智慧，指出这场冲突对双方未来关系的影响"}"""
    }
}


# =============================================================================
# WORKFLOW PIPELINE
# =============================================================================

class MoralDilemmaWorkflow:
    """Main workflow orchestrator for moral dilemma assessment."""

    async def run(self, user_input: str) -> dict:
        """
        Execute the full pipeline.
        Returns a dict with all step results and final aggregated scores.
        """
        result = {"user_input": user_input, "steps": {}}
        total_tokens = 0

        def consume_tokens(step_result):
            nonlocal total_tokens
            if isinstance(step_result, dict):
                token_info = step_result.pop("_token_info", None)
                if token_info:
                    total_tokens += token_info.get("total_tokens", 0)

        # Step 1: Quality Check
        step1_result = await self._step1_quality_check(user_input)
        consume_tokens(step1_result)
        result["steps"]["step1_quality"] = step1_result

        if step1_result.get("status") != "pass":
            result["status"] = "need_more_info"
            result["question"] = step1_result.get("question", "请补充更多具体事实、原话和时间顺序。")
            result["total_tokens"] = total_tokens
            return result

        # Step 2: Neutral Translation
        step2_result = await self._step2_neutral_translate(user_input)
        consume_tokens(step2_result)
        result["steps"]["step2_neutral"] = step2_result

        # Step 3: Multi-Perspective
        step3_result = await self._step3_multi_perspective(step2_result["neutral_narrative"])
        consume_tokens(step3_result)
        result["steps"]["step3_perspective"] = step3_result

        # Step 4: Multi-Agent Jury (5 concurrent judges)
        step4_results = await self._step4_multi_agent_jury(
            neutral_narrative=step2_result["neutral_narrative"],
            pro_perspective=step3_result["pro_perspective"],
            con_perspective=step3_result["con_perspective"],
        )
        for judge_result in step4_results:
            consume_tokens(judge_result)
        result["steps"]["step4_judges"] = step4_results

        # Step 5: Weighted Aggregation
        step5_result = self._step5_weighted_aggregate(step4_results)
        result["steps"]["step5_aggregate"] = step5_result

        result["status"] = "complete"
        result["final_scores"] = step5_result
        result["total_tokens"] = total_tokens

        return result

    async def _step1_quality_check(self, user_input: str) -> dict:
        result = await llm_client.chat_async(
            STEP1_QUALITY_CHECK_PROMPT,
            user_input,
            temperature=self._default_temperature(),
        )
        return self._normalize_step1_result(result)

    async def _step2_neutral_translate(self, user_input: str) -> dict:
        result = await llm_client.chat_async(
            STEP2_NEUTRAL_TRANSLATION_PROMPT,
            user_input,
            temperature=self._default_temperature(),
        )
        result["neutral_narrative"] = self._require_non_empty_str(
            result, "neutral_narrative", "step2.neutral_narrative"
        )
        return result

    async def _step3_multi_perspective(self, neutral_narrative: str) -> dict:
        result = await llm_client.chat_async(
            STEP3_MULTI_PERSPECTIVE_PROMPT,
            neutral_narrative,
            temperature=self._default_temperature(),
        )
        return self._normalize_perspective_result(result)

    async def _step4_multi_agent_jury(
        self, neutral_narrative: str, pro_perspective: dict, con_perspective: dict
    ) -> list:
        """Run 5 different judge personas concurrently."""
        tasks = []

        for judge_id, persona in JUDGE_PERSONAS.items():
            user_msg = self._build_judge_user_message(
                neutral_narrative=neutral_narrative,
                pro_perspective=pro_perspective,
                con_perspective=con_perspective,
            )
            tasks.append(
                self._call_judge_safe(
                    judge_id=judge_id,
                    persona=persona,
                    user_msg=user_msg,
                )
            )

        return await asyncio.gather(*tasks)

    async def _step4_multi_agent_jury_stream(
        self, neutral_narrative: str, pro_perspective: dict, con_perspective: dict
    ):
        """Yield judge results as soon as each concurrent call finishes."""
        tasks = []

        for judge_id, persona in JUDGE_PERSONAS.items():
            user_msg = self._build_judge_user_message(
                neutral_narrative=neutral_narrative,
                pro_perspective=pro_perspective,
                con_perspective=con_perspective,
            )
            tasks.append(
                asyncio.create_task(
                    self._call_judge_safe_timed(
                        judge_id=judge_id,
                        persona=persona,
                        user_msg=user_msg,
                    )
                )
            )

        try:
            for completed in asyncio.as_completed(tasks):
                yield await completed
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    def _build_judge_user_message(
        self, neutral_narrative: str, pro_perspective: dict, con_perspective: dict
    ) -> str:
        return f"""【角色映射 - 必须严格遵守】
- 正方（pro_scores 评估对象）= 中立陈述中的「当事人A」= 用户（叙述者本人）
- 反方（con_scores 评估对象）= 中立陈述中的「当事人B」= 对方
- 不允许根据"谁更委屈/更有理"重新分配正反方。

中立事实陈述：
{neutral_narrative}

评分证据约束：
1. 只能根据"中立事实陈述"中已经发生的动作、原话、时间顺序打分。
2. 下方"视角重构"的需求是推测，用于理解语境，不能当成行为事实。
3. pro_scores 表示正方（当事人A）自己的错误严重度；con_scores 表示反方（当事人B）自己的错误严重度。
4. 不要因为一方更委屈，就把"被伤害"算成这一方的错误。

正方视角（当事人A / 用户）：
说：{pro_perspective['spoken_words']}
需求：{pro_perspective['inner_need']}

反方视角（当事人B / 对方）：
说：{con_perspective['spoken_words']}
需求：{con_perspective['inner_need']}"""

    def _build_judge_system_prompt(self, judge_id: str, persona: dict) -> str:
        parts = [persona["system_prompt"], JUDGE_SCORING_GUIDE]
        if judge_id == "judge_c":
            parts.append(PRO_FRIEND_STANCE_PROMPT)
        elif judge_id == "judge_d":
            parts.append(CON_FRIEND_STANCE_PROMPT)
        parts.append(JUDGE_OUTPUT_CONTRACT)
        return "\n\n".join(parts)

    def _sort_judge_results(self, judge_results: list) -> list:
        order = {judge_id: index for index, judge_id in enumerate(JUDGE_PERSONAS.keys())}
        return sorted(judge_results, key=lambda r: order.get(r.get("judge_id"), len(order)))

    def _judge_count(self) -> int:
        return len(JUDGE_PERSONAS)

    async def _call_judge_safe(self, judge_id: str, persona: dict, user_msg: str) -> dict:
        try:
            return await self._call_judge(judge_id=judge_id, persona=persona, user_msg=user_msg)
        except Exception as exc:
            return {
                "judge_id": judge_id,
                "judge_name": persona["name"],
                "weight": persona["weight"],
                "error": str(exc),
            }

    async def _call_judge_safe_timed(self, judge_id: str, persona: dict, user_msg: str) -> dict:
        started_at = time.perf_counter()
        result = await self._call_judge_safe(judge_id=judge_id, persona=persona, user_msg=user_msg)
        result["_duration_seconds"] = round(time.perf_counter() - started_at, 3)
        return result

    async def _call_judge(self, judge_id: str, persona: dict, user_msg: str) -> dict:
        """Call a single judge with their specific persona."""
        result = await llm_client.chat_async(
            self._build_judge_system_prompt(judge_id, persona),
            user_msg,
            temperature=self._judge_temperature(),
        )
        result = self._normalize_judge_result(result)
        result["judge_id"] = judge_id
        result["judge_name"] = persona["name"]
        result["weight"] = persona["weight"]
        return result

    def _step5_weighted_aggregate(self, judge_results: list) -> dict:
        """Aggregate scores with weights applied per judge."""
        dimensions = SCORE_DIMENSIONS
        valid_results = [
            r for r in judge_results
            if self._is_valid_judge_result(r, dimensions)
        ]
        failed_judges = [
            {
                "judge_id": r.get("judge_id"),
                "judge_name": r.get("judge_name"),
                "error": r.get("error", "invalid judge result"),
            }
            for r in judge_results
            if not self._is_valid_judge_result(r, dimensions)
        ]
        expected_judge_count = len(judge_results)

        if not valid_results:
            return {
                "error": "All judge calls failed",
                "pro_avg": {},
                "con_avg": {},
                "pro_total": 0,
                "con_total": 0,
                "judge_scores": [],
                "failed_judges": failed_judges,
                "valid_judge_count": 0,
                "expected_judge_count": expected_judge_count,
                "confidence": "low",
                "total_weight": 0,
                "verdict_summary": {
                    "title": "无法形成结论",
                    "detail": "所有陪审团调用失败，无法进行可靠聚合。",
                    "primary_issue": "证据不足",
                    "mediation_advice": "建议稍后重试，或补充更具体的事实与原话。",
                },
            }

        def normalized_score(value):
            if value is None:
                return None
            try:
                score = float(value)
            except (TypeError, ValueError):
                return None
            return min(MAX_SCORE, max(MIN_SCORE, score))

        # Store per-judge scores for display
        judge_scores = []
        for r in valid_results:
            judge_scores.append({
                "judge_id": r.get("judge_id"),
                "judge_name": r.get("judge_name"),
                "weight": r.get("weight", 1.0),
                "pro_scores": self._normalize_score_dict(r.get("pro_scores", {}), dimensions),
                "con_scores": self._normalize_score_dict(r.get("con_scores", {}), dimensions),
                "brief_reason": r.get("brief_reason", ""),
            })

        # Weighted average calculation
        total_weight = sum(r.get("weight", 1.0) for r in valid_results)

        pro_weighted_sum = {dim: 0.0 for dim in dimensions}
        con_weighted_sum = {dim: 0.0 for dim in dimensions}
        pro_dimension_weight = {dim: 0.0 for dim in dimensions}
        con_dimension_weight = {dim: 0.0 for dim in dimensions}

        for r in valid_results:
            weight = r.get("weight", 1.0)
            for dim in dimensions:
                pro_score = normalized_score(r.get("pro_scores", {}).get(dim))
                con_score = normalized_score(r.get("con_scores", {}).get(dim))
                if pro_score is not None:
                    pro_weighted_sum[dim] += pro_score * weight
                    pro_dimension_weight[dim] += weight
                if con_score is not None:
                    con_weighted_sum[dim] += con_score * weight
                    con_dimension_weight[dim] += weight

        pro_avg = {
            dim: round(pro_weighted_sum[dim] / pro_dimension_weight[dim], 2)
            if pro_dimension_weight[dim] else 0
            for dim in dimensions
        }
        con_avg = {
            dim: round(con_weighted_sum[dim] / con_dimension_weight[dim], 2)
            if con_dimension_weight[dim] else 0
            for dim in dimensions
        }

        pro_available_dims = [dim for dim in dimensions if pro_dimension_weight[dim]]
        con_available_dims = [dim for dim in dimensions if con_dimension_weight[dim]]
        pro_total = round(sum(pro_avg[dim] for dim in pro_available_dims) / len(pro_available_dims), 2) if pro_available_dims else 0
        con_total = round(sum(con_avg[dim] for dim in con_available_dims) / len(con_available_dims), 2) if con_available_dims else 0
        confidence = self._confidence_label(len(valid_results), expected_judge_count)
        verdict_summary = self._build_verdict_summary(pro_avg, con_avg, pro_total, con_total, confidence)

        return {
            "pro_avg": pro_avg,
            "con_avg": con_avg,
            "pro_total": pro_total,
            "con_total": con_total,
            "judge_scores": judge_scores,
            "failed_judges": failed_judges,
            "valid_judge_count": len(valid_results),
            "expected_judge_count": expected_judge_count,
            "confidence": confidence,
            "verdict_summary": verdict_summary,
            "total_weight": total_weight,
        }

    def _normalize_score_dict(self, scores: dict, dimensions: list[str]) -> dict:
        normalized = {}
        for dim in dimensions:
            value = scores.get(dim)
            try:
                normalized[dim] = round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 2)
            except (TypeError, ValueError):
                normalized[dim] = None
        return normalized

    def _default_temperature(self) -> float:
        return load_config()["default_temperature"]

    def _judge_temperature(self) -> float:
        return load_config()["judge_temperature"]

    def _normalize_step1_result(self, result: dict) -> dict:
        if not isinstance(result, dict):
            raise ValueError("step1 returned non-object JSON")
        if result.get("status") == "pass":
            result["question"] = result.get("question", "")
            result["fact_summary"] = result.get("fact_summary", "")
            return result
        result["status"] = "need_more_info"
        result["question"] = result.get("question") or "请补充更多具体事实、原话和时间顺序。"
        result["fact_summary"] = result.get("fact_summary", "")
        return result

    def _require_non_empty_str(self, data: dict, key: str, label: str) -> str:
        if not isinstance(data, dict):
            raise ValueError(f"{label} returned non-object JSON")
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty {label}")
        return value.strip()

    def _normalize_perspective_result(self, result: dict) -> dict:
        if not isinstance(result, dict):
            raise ValueError("step3 returned non-object JSON")
        for side in ("pro_perspective", "con_perspective"):
            perspective = result.get(side)
            if not isinstance(perspective, dict):
                raise ValueError(f"Missing {side}")
            for key in ("spoken_words", "inner_need"):
                value = perspective.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing {side}.{key}")
                perspective[key] = value.strip()
        return result

    def _normalize_judge_result(self, result: dict) -> dict:
        if not isinstance(result, dict):
            raise ValueError("judge returned non-object JSON")
        normalized = dict(result)
        normalized["pro_scores"] = self._normalize_required_score_dict(
            result.get("pro_scores"), "pro_scores"
        )
        normalized["con_scores"] = self._normalize_required_score_dict(
            result.get("con_scores"), "con_scores"
        )
        normalized["brief_reason"] = self._shorten_brief_reason(result.get("brief_reason", ""))
        return normalized

    def _normalize_required_score_dict(self, scores: dict, label: str) -> dict:
        if not isinstance(scores, dict):
            raise ValueError(f"Missing {label}")
        normalized = {}
        for dim in SCORE_DIMENSIONS:
            if dim not in scores:
                raise ValueError(f"Missing {label}.{dim}")
            try:
                value = float(scores[dim])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {label}.{dim}") from exc
            normalized[dim] = int(round(min(MAX_SCORE, max(MIN_SCORE, value))))
        return normalized

    def _is_valid_judge_result(self, result: dict, dimensions: list[str]) -> bool:
        if not isinstance(result, dict) or "error" in result:
            return False
        return all(
            isinstance(result.get(side), dict)
            and all(isinstance(result[side].get(dim), (int, float)) for dim in dimensions)
            for side in ("pro_scores", "con_scores")
        )

    def _shorten_brief_reason(self, reason: str, limit: int = 35) -> str:
        text = " ".join(str(reason or "").split())
        if not text:
            return "依据有限，按事实保守评分。"
        if len(text) <= limit:
            return text
        for separator in ("。", "；", "，", ","):
            index = text.find(separator)
            if 0 < index + 1 <= limit:
                return text[: index + 1]
        return text[:limit].rstrip("，,；;。") + "。"

    def _confidence_label(self, valid_count: int, expected_count: int) -> str:
        if expected_count <= 0:
            return "low"
        ratio = valid_count / expected_count
        if ratio >= 0.8:
            return "high"
        if ratio >= 0.6:
            return "medium"
        return "low"

    def _build_verdict_summary(
        self, pro_avg: dict, con_avg: dict, pro_total: float, con_total: float, confidence: str
    ) -> dict:
        labels = {
            "identity_labeling": "身份标签化",
            "proportionality": "反应失衡",
            "boundary_awareness": "越界程度",
            "constructiveness": "缺乏建设性",
        }
        combined = {
            dim: max(pro_avg.get(dim, 0), con_avg.get(dim, 0))
            for dim in labels
        }
        peak_score = max(combined.values()) if combined else 0
        # 当所有维度都接近最低错误度，没有任何"主要问题"可言。
        if peak_score <= 1.5:
            primary_dim = None
            primary_issue = "无明显问题"
            mediation_advice = "双方在四个维度上的错误度都很低，本次冲突可视为非典型摩擦，建议先沟通确认是否存在未捕捉到的事实。"
        else:
            primary_dim = max(combined, key=combined.get)
            primary_issue = labels[primary_dim]
            advice_by_issue = {
                "identity_labeling": "先停止人格化标签，改用可核验的动作、原话和影响来表达不满。",
                "proportionality": "先降低反应强度，把回应拆成事实确认、感受表达和具体请求。",
                "boundary_awareness": "先明确哪些行为越过了隐私、选择权或表达空间，再谈补救。",
                "constructiveness": "先提出一个可执行的下一步，例如道歉、澄清、补偿或重新约定边界。",
            }
            mediation_advice = advice_by_issue[primary_dim]

        diff = round(con_total - pro_total, 2)

        if abs(diff) < 0.75:
            title = "双方责任接近"
            detail = f"正方总错误度 {pro_total}，反方总错误度 {con_total}，差值较小。更适合先处理共同模式，而不是急于判定单方全责。"
        elif diff > 0:
            title = "反方问题更重"
            detail = f"反方总错误度 {con_total} 高于正方 {pro_total}，主要应优先修正反方行为，同时保留正方可改进处。"
        else:
            title = "正方问题更重"
            detail = f"正方总错误度 {pro_total} 高于反方 {con_total}，主要应优先修正正方行为，同时确认反方是否也有触发因素。"

        return {
            "title": title,
            "detail": detail,
            "primary_issue": primary_issue,
            "mediation_advice": mediation_advice,
            "score_gap": abs(diff),
            "confidence": confidence,
        }
