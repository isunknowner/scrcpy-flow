"""把 AI 规划结果落库：生成流程 / 生成区块项目。"""
from __future__ import annotations

from .db import Database, make_alias

NODE_W = 260
NODE_H = 120


def create_flow_from_plan(db: Database, plan: dict, x: float = 120, y: float = 160) -> int:
    """按 plan["steps"] 建一条流程，每步一个 AI 决策节点，首尾串成链。"""
    name = str(plan.get("name") or "AI 流程")[:40]
    description = str(plan.get("description") or "")
    flow_id = db.create_flow(db.unique_flow_name(name), description)
    steps = [step for step in (plan.get("steps") or []) if isinstance(step, dict)]
    if not steps:
        steps = [{"name": name, "goal": description or name}]
    prev = None
    for index, step in enumerate(steps):
        goal = str(step.get("goal") or step.get("name") or "").strip() or description or name
        node_id = db.add_node(
            flow_id,
            name=str(step.get("name") or f"步骤 {index + 1}")[:24],
            kind="ai",
            action_type="ai",
            action={"goal": goal},
            threshold=0.82,
            is_entry=index == 0,
            is_end=False,
            x=float(x + index * NODE_W),
            y=float(y),
        )
        if prev is not None:
            db.add_edge(flow_id, prev, node_id, weight=1.0)
        prev = node_id
    if prev is not None:
        db.update_node(prev, is_end=True)
    return flow_id


def create_project_from_plan(db: Database, plan: dict) -> int:
    """按 plan["blocks"] 生成一个区块项目；区块内流程用 AI 步骤自动创建。"""
    name = str(plan.get("name") or "AI 项目")[:40]
    description = str(plan.get("description") or "")
    project_id = db.create_project(
        db.unique_project_name(name),
        description=description,
        settings={"mode": "sequence", "stop_on_error": True},
    )
    used: set[str] = set()
    calls: list[str] = []
    blocks = [item for item in (plan.get("blocks") or []) if isinstance(item, dict)]
    if not blocks:
        blocks = [{"name": name, "flow_goal": description or name, "steps": []}]
    for block in blocks:
        block_name = str(block.get("name") or "区块")
        flow_goal = str(block.get("flow_goal") or block_name)
        flow_id = create_flow_from_plan(
            db,
            {
                "name": block_name,
                "description": flow_goal,
                "steps": block.get("steps") or [{"name": block_name, "goal": flow_goal}],
            },
        )
        alias = make_alias(block_name, used)
        used.add(alias)
        db.add_project_block(project_id, flow_id, alias=alias)
        calls.append(f"{alias}()")
    db.update_project(project_id, python_code="\n".join(calls) or f"# {name}")
    return project_id
