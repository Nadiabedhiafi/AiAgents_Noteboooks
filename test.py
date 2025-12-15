import os, re, json, base64, mimetypes
from typing import Any, Dict, List, TypedDict, Optional

import pandas as pd
import httpx
from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, END


# ----------------------------
# Your LLM client setup (as-is)
# ----------------------------
from pygenai import LangchainGenAIChat

load_dotenv()

http_client = httpx.Client(verify=False)
ahttp_client = httpx.AsyncClient(verify=False)

model_name = "mistral-medium-2505"
client = LangchainGenAIChat(
    client_id=os.getenv("AC_LLM_USER_NAME"),
    client_secret=os.getenv("AC_LLM_PWD"),
    auth_token_url=os.getenv("AC_TOKEN_URL"),
    base_url=os.getenv("AC_LLM_URL"),
    cert=(os.getenv("AC_LLM_CERT_PATH"), os.getenv("AC_LLM_KEY_PATH")),
    http_client=http_client,
    http_async_client=ahttp_client,
    model_name=model_name,
)


# ----------------------------
# Utils
# ----------------------------
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df


def ensure_execute_python_tags(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text).strip()
    if "<execute_python>" not in text:
        text = f"<execute_python>\n{text}\n</execute_python>"
    return text


def extract_code_from_tags(llm_text: str) -> str:
    m = re.search(r"<execute_python>([\s\S]*?)</execute_python>", llm_text)
    return (m.group(1).strip() if m else "").strip()


def encode_image_b64(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    media_type = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


def safe_exec_matplotlib(code: str, df: pd.DataFrame):
    """
    Executes generated code in a restricted-ish namespace.
    Still uses exec (like your notebook), but keeps globals tight.
    """
    exec_globals = {"df": df}
    exec(code, exec_globals)


# ----------------------------
# Agents (LLM calls)
# ----------------------------
def generator_agent(instruction: str, out_path: str) -> str:
    prompt = f"""
You are a data visualization expert.

Return your answer *strictly* in this format:

<execute_python>
# valid python code here
</execute_python>

Do not add explanations, only the tags and the code.

The code should create a visualization from a DataFrame 'df' with these columns:
- date (M/D/YY)
- time (HH:MM)
- cash_type (card or cash)
- card (string)
- price (number)
- coffee_name (string)
- quarter (1-4)
- month (1-12)
- year (YYYY)

User instruction: {instruction}

Requirements for the code:
1. Assume the DataFrame is already loaded as 'df'.
2. Use matplotlib for plotting.
3. Add clear title, axis labels, and legend if needed.
4. Save the figure as '{out_path}' with dpi=300.
5. Do not call plt.show().
6. Close all plots with plt.close().
7. Add all necessary import statements.
"""
    resp = client.invoke(prompt)
    # resp might be object w/ .content in your environment:
    return resp.content if hasattr(resp, "content") else str(resp)


def critic_agent_with_image(
    instruction: str,
    out_path_next: str,
    code_current: str,
    chart_path: str,
) -> str:
    media_type, b64 = encode_image_b64(chart_path)

    # IMPORTANT: your client.invoke(messages) expects the "messages" format
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Output MUST follow the requested strict format."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
You are a data visualization expert.
Critique the attached chart and the current code against the instruction, then return improved matplotlib code.

Instruction:
{instruction}

Current code (context):
{code_current}

OUTPUT FORMAT (STRICT):
1) First line: a valid JSON object with ONLY the "feedback" field.
Example: {{"feedback":"Legend unclear; Q1 filter wrong."}}

2) After a newline, output ONLY the refined Python code wrapped in:
<execute_python>
...
</execute_python>

Hard constraints:
- No markdown / no backticks / no extra prose.
- pandas + matplotlib only (no seaborn).
- Assume df already exists; do not read files.
- Save to '{out_path_next}' with dpi=300.
- Always plt.close(); no plt.show().
- Include all necessary imports.
""" },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
            ],
        },
    ]

    resp = client.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


def parse_feedback_and_code(critic_text: str) -> tuple[str, str]:
    lines = critic_text.strip().splitlines()
    json_line = lines[0].strip() if lines else "{}"

    feedback = ""
    try:
        obj = json.loads(json_line)
        feedback = str(obj.get("feedback", "")).strip()
    except Exception:
        # fallback: best-effort JSON extraction
        m = re.search(r"\{.*?\}", critic_text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                feedback = str(obj.get("feedback", "")).strip()
            except Exception:
                feedback = "Failed to parse feedback JSON."

    code = extract_code_from_tags(critic_text)
    return feedback, code


def should_stop(feedback: str) -> bool:
    """
    Simple heuristic: stop if feedback indicates it's good enough.
    Adapt this to your needs (keywords, score, etc.).
    """
    f = feedback.lower()
    good_signals = [
        "looks good", "good enough", "no issues", "meets the instruction",
        "satisfies", "correct", "clear and accurate"
    ]
    return any(s in f for s in good_signals)


# ----------------------------
# LangGraph state + nodes
# ----------------------------
class VizState(TypedDict):
    instruction: str
    csv_path: str
    out_dir: str
    max_iters: int
    iter: int

    df: Any
    code: str
    chart_path: str
    feedback: str
    done: bool


def node_load_data(state: VizState) -> VizState:
    df = load_and_prepare_data(state["csv_path"])
    state["df"] = df
    return state


def node_generate(state: VizState) -> VizState:
    out_path = os.path.join(state["out_dir"], f"chart_v{state['iter']}.png")
    llm_text = generator_agent(state["instruction"], out_path)
    state["code"] = extract_code_from_tags(llm_text) or extract_code_from_tags(ensure_execute_python_tags(llm_text))
    state["chart_path"] = out_path
    return state


def node_execute(state: VizState) -> VizState:
    safe_exec_matplotlib(state["code"], state["df"])
    return state


def node_critic(state: VizState) -> VizState:
    out_path_next = os.path.join(state["out_dir"], f"chart_v{state['iter'] + 1}.png")
    critic_text = critic_agent_with_image(
        instruction=state["instruction"],
        out_path_next=out_path_next,
        code_current=state["code"],
        chart_path=state["chart_path"],
    )
    feedback, refined_code = parse_feedback_and_code(critic_text)
    state["feedback"] = feedback
    if refined_code:
        state["code"] = refined_code
        state["chart_path"] = out_path_next
    state["done"] = should_stop(feedback)
    return state


def node_iterate(state: VizState) -> VizState:
    state["iter"] += 1
    if state["iter"] >= state["max_iters"]:
        state["done"] = True
    return state


def route_after_critic(state: VizState) -> str:
    return END if state["done"] else "execute_refined"


# Build graph
graph = StateGraph(VizState)

graph.add_node("load_data", node_load_data)
graph.add_node("generate", node_generate)
graph.add_node("execute", node_execute)
graph.add_node("critic", node_critic)
graph.add_node("execute_refined", node_execute)  # re-use executor for refined code
graph.add_node("iterate", node_iterate)

graph.set_entry_point("load_data")
graph.add_edge("load_data", "generate")
graph.add_edge("generate", "execute")
graph.add_edge("execute", "critic")
graph.add_conditional_edges("critic", route_after_critic, {END: END, "execute_refined": "execute_refined"})
graph.add_edge("execute_refined", "iterate")
graph.add_edge("iterate", "critic")  # loop back with new chart_v{iter+1}.png

app = graph.compile()


# ----------------------------
# Run
# ----------------------------
initial_state: VizState = {
    "instruction": "Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in coffee_sales.csv.",
    "csv_path": "coffee_sales.csv",
    "out_dir": ".",          # or "/data/git/aicredit-doc-extraction/notebooks"
    "max_iters": 3,
    "iter": 1,

    "df": None,
    "code": "",
    "chart_path": "",
    "feedback": "",
    "done": False,
}

final_state = app.invoke(initial_state)
print("FINAL CHART:", final_state["chart_path"])
print("FINAL FEEDBACK:", final_state["feedback"])