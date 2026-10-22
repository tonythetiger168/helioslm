"""hsh: the HeliosLM harness, after DeepSeek Harness's everything-is-a-plugin design."""
from .kernel import (Context, EventModeError, Fork, KernelError,  # noqa: F401
                     ListenerErrors, Service)
from .session import (Session, SessionError, Sessions,  # noqa: F401
                      build_request, derive_messages, fold_request_header,
                      rebuild_requests, register_event_type, request_hash,
                      usage_summary)
from .schema import SchemaError, check_schema, validate  # noqa: F401
from .tools import Execution, ToolDefinition, ToolResult, Tools, tool  # noqa: F401
from .llm import (LLM, LLMError, OpenAICompatAdapter, ScriptedAdapter,  # noqa: F401
                  TextModelAdapter, render_transcript)
from .loop import Agent, Agents  # noqa: F401
