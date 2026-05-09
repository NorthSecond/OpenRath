import json
import os
from collections.abc import Mapping
from typing import Any

import rath.flow as flow
from rath.flow.tool import (
    FlowToolCall,
    ToolRegistration,
    flow_tool_command_run,
    register_global_tool,
)
from rath.session import Session, run_session_loop

BIGMODEL_IMAGES_URL = "https://open.bigmodel.cn/api/paas/v4/images/generations"


class ImageGenTool(ToolRegistration):
    """HTTP POST to the image API using a ``curl`` argv list."""

    def __init__(self) -> None:
        super().__init__(
            name="image_gen",
            builder=ImageGenTool._build,
            description=(
                "Generate an image with Zhipu GLM-Image (glm-image) via curl. "
                "Stdout is the API JSON (image URL under data[0].url). "
                "Needs ZHIPU_API_KEY or OPENAI_API_KEY and curl on PATH."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Text prompt (see GLM-Image limits, max ~1000 characters)."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "description": (
                            "Output size, e.g. 1280x1280, 1568x1056 (per docs)."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        )

    @staticmethod
    def _build(args: Mapping[str, Any]) -> FlowToolCall:
        """Return a command-run tool call for one GLM-Image request."""
        prompt = str(args["prompt"])
        size = str(args.get("size") or "1280x1280")
        key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set ZHIPU_API_KEY or OPENAI_API_KEY")
        body = json.dumps(
            {"model": "glm-image", "prompt": prompt, "size": size},
            ensure_ascii=False,
        )
        return flow_tool_command_run(
            cmd=[
                "curl",
                "-sS",
                "-X",
                "POST",
                BIGMODEL_IMAGES_URL,
                "-H",
                "Content-Type: application/json",
                "-H",
                f"Authorization: Bearer {key}",
                "-d",
                body,
            ],
            timeout=180.0,
        )


if __name__ == "__main__":
    register_global_tool(ImageGenTool())

    agent = flow.Agent(
        agent_session=Session.from_agent_prompt(
            "You have an image_gen tool for Zhipu GLM-Image. "
            "When the user asks for an image, call image_gen with a concise prompt "
            "and optional size (default 1280x1280). The tool returns API JSON; "
            "mention the image URL from the response."
        ),
        provider=flow.Provider(model="glm-5.1"),
    )
    user_session = Session.from_user_message(
        "Generate a simple cartoon cat on a sofa (no text in the image). "
        "Use image_gen once, then answer in one short sentence. "
        "Save in ./custom_tool_usage.png"
    ).to("local", spec="./")

    out_session = run_session_loop(
        user_session=user_session,
        agent_session=agent.agent_session,
        agent_provider=agent.provider,
    )
    last = out_session.chunk_table.rows[-1].payload
    print(last.get("content") or last)
