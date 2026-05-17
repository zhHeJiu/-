from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time

from .config import ConfigError, public_config, update_config
from .workflow import MoralDilemmaWorkflow

app = FastAPI(title="Moral Dilemma Assessor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow = MoralDilemmaWorkflow()


class AssessRequest(BaseModel):
    user_input: str


class ConfigUpdateRequest(BaseModel):
    minimax_api_key: str | None = None
    clear_api_key: bool = False
    minimax_base_url: str | None = None
    model: str | None = None
    default_temperature: float | None = None
    judge_temperature: float | None = None
    llm_max_retries: int | None = None


def format_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def public_error_message(exc: Exception) -> str:
    message = str(exc).splitlines()[0].strip()
    if not message:
        return "分析过程中出现未知错误"
    return message[:300]


@app.get("/")
async def root():
    return {"message": "Moral Dilemma Assessor API", "version": "1.0.0"}


@app.get("/api/config")
async def get_config():
    try:
        return public_config()
    except (ConfigError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=public_error_message(exc)) from exc


@app.post("/api/config")
async def save_config(request: ConfigUpdateRequest):
    try:
        update_config(request.model_dump(exclude_unset=True))
        return public_config()
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=public_error_message(exc)) from exc


@app.post("/api/assess/stream")
async def assess_stream(request: AssessRequest):
    """
    Streaming assessment endpoint using Server-Sent Events.
    Streams step-by-step progress with token counts.
    """
    if not request.user_input or len(request.user_input.strip()) < 10:
        raise HTTPException(status_code=400, detail="输入内容过短，至少需要10个字符")

    async def event_generator():
        total_tokens = 0
        total_ai_calls = 3 + workflow._judge_count()
        ai_calls_started = 0
        ai_calls_completed = 0
        active_ai_calls = 0

        def extract_token_info(result: dict) -> dict:
            return result.pop("_token_info", {}) if isinstance(result, dict) else {}

        def add_tokens(token_info: dict, duration_seconds: float | None = None) -> dict:
            nonlocal total_tokens
            step_tokens = token_info.get("total_tokens", 0) if token_info else 0
            total_tokens += step_tokens
            payload = {
                **token_info,
                "step_tokens": step_tokens,
                "cumulative_total_tokens": total_tokens,
            }
            if duration_seconds is not None:
                duration = max(0.001, duration_seconds)
                payload["duration_seconds"] = round(duration, 3)
                payload["tps_observed"] = round(step_tokens / duration, 2) if step_tokens else 0
            return payload

        def start_ai_calls(count: int = 1) -> None:
            nonlocal ai_calls_started, active_ai_calls
            ai_calls_started += count
            active_ai_calls += count

        def finish_ai_calls(count: int = 1) -> None:
            nonlocal ai_calls_completed, active_ai_calls
            ai_calls_completed += count
            active_ai_calls = max(0, active_ai_calls - count)

        def usage_payload(extra: dict | None = None) -> dict:
            payload = {
                "total_tokens": total_tokens,
                "ai_calls_started": ai_calls_started,
                "ai_calls_completed": ai_calls_completed,
                "active_ai_calls": active_ai_calls,
                "total_ai_calls": total_ai_calls,
            }
            if extra:
                payload.update(extra)
            return payload

        # Step 1: Quality Check
        start_ai_calls()
        yield format_sse("step", usage_payload({
            "step": 1,
            "name": "quality_check",
            "label": "质量检测",
            "status": "running",
        }))
        step_started_at = time.perf_counter()
        try:
            step1_result = await workflow._step1_quality_check(request.user_input)
        finally:
            step1_duration = time.perf_counter() - step_started_at
            finish_ai_calls()
        step1_tokens = add_tokens(extract_token_info(step1_result), step1_duration)
        yield format_sse("step", usage_payload({
            "step": 1, "name": "quality_check", "label": "质量检测",
            "status": "done" if step1_result.get("status") == "pass" else "need_info",
            "tokens": step1_tokens,
            "result": step1_result
        }))

        if step1_result.get("status") != "pass":
            yield format_sse("done", usage_payload({
                "status": "need_more_info",
                "question": step1_result.get("question", "请补充更多具体事实、原话和时间顺序。"),
            }))
            return

        # Step 2: Neutral Translation
        start_ai_calls()
        yield format_sse("step", usage_payload({
            "step": 2,
            "name": "neutral_translate",
            "label": "中立化转译",
            "status": "running",
        }))
        step_started_at = time.perf_counter()
        try:
            step2_result = await workflow._step2_neutral_translate(request.user_input)
        finally:
            step2_duration = time.perf_counter() - step_started_at
            finish_ai_calls()
        step2_tokens = add_tokens(extract_token_info(step2_result), step2_duration)
        yield format_sse("step", usage_payload({
            "step": 2, "name": "neutral_translate", "label": "中立化转译",
            "status": "done", "tokens": step2_tokens, "result": step2_result
        }))

        # Step 3: Multi-Perspective
        start_ai_calls()
        yield format_sse("step", usage_payload({
            "step": 3,
            "name": "multi_perspective",
            "label": "多视角重构",
            "status": "running",
        }))
        step_started_at = time.perf_counter()
        try:
            step3_result = await workflow._step3_multi_perspective(step2_result["neutral_narrative"])
        finally:
            step3_duration = time.perf_counter() - step_started_at
            finish_ai_calls()
        step3_tokens = add_tokens(extract_token_info(step3_result), step3_duration)
        yield format_sse("step", usage_payload({
            "step": 3, "name": "multi_perspective", "label": "多视角重构",
            "status": "done", "tokens": step3_tokens, "result": step3_result
        }))

        # Step 4: Multi-Agent Jury
        total_judges = workflow._judge_count()
        start_ai_calls(total_judges)
        yield format_sse("step", usage_payload({
            "step": 4,
            "name": "judging",
            "label": "AI 陪审团评分",
            "status": "running",
            "completed_judges": 0,
            "total_judges": total_judges,
        }))
        step4_results = []
        judge_tokens = 0

        async for judge_result in workflow._step4_multi_agent_jury_stream(
            neutral_narrative=step2_result["neutral_narrative"],
            pro_perspective=step3_result["pro_perspective"],
            con_perspective=step3_result["con_perspective"],
        ):
            finish_ai_calls()
            judge_duration = judge_result.pop("_duration_seconds", None)
            judge_token_info = extract_token_info(judge_result)
            judge_token_payload = add_tokens(judge_token_info, judge_duration)
            judge_tokens += judge_token_payload["step_tokens"]
            step4_results.append(judge_result)
            yield format_sse("usage", usage_payload({
                "source": "judging",
                "judge_id": judge_result.get("judge_id"),
                "judge_name": judge_result.get("judge_name"),
                "delta_tokens": judge_token_payload["step_tokens"],
                "duration_seconds": judge_token_payload.get("duration_seconds"),
                "tps_observed": judge_token_payload.get("tps_observed"),
                "completed_judges": len(step4_results),
                "total_judges": total_judges,
            }))

        step4_results = workflow._sort_judge_results(step4_results)
        yield format_sse("step", usage_payload({
            "step": 4, "name": "judging", "label": "AI 陪审团评分",
            "status": "done",
            "tokens": {"total_tokens": judge_tokens, "cumulative_total_tokens": total_tokens},
            "completed_judges": total_judges,
            "total_judges": total_judges,
            "result": step4_results
        }))

        # Step 5: Aggregation
        yield format_sse("step", usage_payload({
            "step": 5,
            "name": "aggregation",
            "label": "数据聚合",
            "status": "running",
        }))
        step5_result = workflow._step5_weighted_aggregate(step4_results)
        yield format_sse("step", usage_payload({
            "step": 5, "name": "aggregation", "label": "数据聚合",
            "status": "done",
            "tokens": {"total_tokens": 0, "cumulative_total_tokens": total_tokens},
            "result": step5_result
        }))

        # Final result
        final_result = {
            "user_input": request.user_input,
            "steps": {
                "step1_quality": step1_result,
                "step2_neutral": step2_result,
                "step3_perspective": step3_result,
                "step4_judges": step4_results,
                "step5_aggregate": step5_result,
            },
            "status": "complete",
            "final_scores": step5_result,
            "total_tokens": total_tokens,
        }
        yield format_sse("done", usage_payload(final_result))

    async def safe_event_generator():
        try:
            async for event in event_generator():
                yield event
        except Exception as exc:
            yield format_sse("error", {
                "status": "error",
                "message": public_error_message(exc),
            })

    return StreamingResponse(
        safe_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/assess")
async def assess_dilemma(request: AssessRequest):
    """Original non-streaming assessment endpoint."""
    if not request.user_input or len(request.user_input.strip()) < 10:
        raise HTTPException(status_code=400, detail="输入内容过短，至少需要10个字符")

    result = await workflow.run(request.user_input)
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok"}
