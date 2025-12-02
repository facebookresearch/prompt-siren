# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import models, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from typing_extensions import assert_never

from ..environments.abstract import AbstractEnvironment
from ..providers import infer_model
from ..types import (
    InjectableModelRequest,
    InjectableUserContent,
    InjectableUserPromptPart,
    InjectionAttack,
    InjectionAttacksDict,
)
from .abstract import AbstractAgent
from .states import (
    EndState,
    ExecutionEndedError,
    ExecutionState,
    FinishReason,
    InjectableModelRequestState,
    ModelRequestState,
    ModelResponseState,
    NoPreviousStateError,
)
from .utils import (
    contents_contain_only_user_request_content,
    extract_tool_call_parts,
    handle_tool_calls,
    inject_injectable_model_request,
    parts_contain_only_model_request_parts,
    query_model,
    restore_state_context,
    serialize_tool_return_parts,
    ToolResultSerializationMode,
)

EnvStateT = TypeVar("EnvStateT")
RawOutputT = TypeVar("RawOutputT")
FinalOutputT = TypeVar("FinalOutputT")
InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)


class PlainAgentConfig(BaseModel):
    """Configuration for PlainAgent.

    This is a simple agent configuration that only needs the model
    and optional model settings.
    """

    model: models.Model = Field(
        description="The language model to use (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4', 'bedrock:...')"
    )
    model_settings: ModelSettings = Field(
        default_factory=ModelSettings,
        description="Optional model settings (temperature, max_tokens, etc.)",
    )
    tool_result_serialization_mode: ToolResultSerializationMode = Field(
        default="json", description="How to serialize tool outputs."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("model", mode="before")
    @classmethod
    def infer_model_from_string(cls, v: models.Model | models.KnownModelName | str) -> models.Model:
        """Infer the model from string/known model name, checking custom providers first."""
        # If it's already a Model object, return it as-is (for tests)
        if isinstance(v, models.Model):
            return v

        return infer_model(v)


@dataclass(frozen=True)
class PlainAgent(AbstractAgent):
    agent_type: ClassVar[str] = "plain"
    _config: PlainAgentConfig

    @property
    def config(self) -> PlainAgentConfig:
        return self._config

    def get_agent_name(self) -> str:
        """Get a descriptive name for this agent (used for filenames and logging)."""
        return f"plain:{self.config.model.model_name}"

    async def run(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> RunContext[EnvStateT]:
        result_state = None
        async for state in self.iter(
            environment,
            env_state,
            user_prompt,
            message_history=message_history,
            toolsets=toolsets,
            usage_limits=usage_limits,
            usage=usage,
            attacks=attacks,
            instrument=instrument,
        ):
            result_state = state

        if result_state is None:
            raise RuntimeError("No loop iteration was executed when running `agent.iter`.")

        return result_state.run_ctx

    async def resume_iter_from_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        # Restore state context to ensure correct environment state
        current_state = await restore_state_context(current_state, toolsets)

        while not isinstance(current_state, EndState):
            current_state = await self.next_state(
                current_state=current_state,
                toolsets=toolsets,
                usage_limits=usage_limits,
                attacks=attacks,
                instrument=instrument,
            )
            yield current_state

    async def prev_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        prev_state = current_state._previous_state
        if prev_state is None:
            raise NoPreviousStateError(
                "You're trying to get `prev_state` of a state which is the initial state."
            )

        # Restore the previous state's environment context
        return await restore_state_context(prev_state, toolsets)

    def create_initial_request_state(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        usage: RunUsage | None = None,
    ) -> (
        ModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
        | InjectableModelRequestState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]
    ):
        message_history = message_history or []

        usage = usage or RunUsage()
        # Model is already inferred during config validation
        run_ctx: RunContext[EnvStateT] = RunContext(
            deps=env_state,
            model=self.config.model,
            usage=usage,
            messages=list(message_history),
        )

        if contents_contain_only_user_request_content(user_prompt):
            model_request = ModelRequest(parts=[UserPromptPart(user_prompt)])
            return ModelRequestState(run_ctx, environment, model_request, _previous_state=None)

        model_request = InjectableModelRequest(parts=[InjectableUserPromptPart(user_prompt)])
        return InjectableModelRequestState(
            run_ctx,
            environment,
            [InjectableUserPromptPart(user_prompt)],
            _previous_state=None,
        )

    async def iter(
        self,
        environment: AbstractEnvironment[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        env_state: EnvStateT,
        user_prompt: str | Sequence[UserContent | InjectableUserContent],
        *,
        message_history: Sequence[ModelMessage] | None = None,
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> AsyncGenerator[ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]]:
        initial_state = self.create_initial_request_state(
            environment,
            env_state,
            user_prompt,
            message_history=message_history,
            usage=usage,
        )
        yield initial_state

        async for current_state in self.resume_iter_from_state(
            current_state=initial_state,
            toolsets=toolsets,
            usage_limits=usage_limits,
            attacks=attacks,
            instrument=instrument,
        ):
            yield current_state

    async def next_state(
        self,
        *,
        current_state: ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT],
        toolsets: Sequence[AbstractToolset[EnvStateT]],
        usage_limits: UsageLimits | None = None,
        attacks: InjectionAttacksDict[InjectionAttackT] | None = None,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> ExecutionState[EnvStateT, RawOutputT, FinalOutputT, InjectionAttackT]:
        match current_state:
            case ModelRequestState(run_ctx, environment, model_request):
                model_response, reply_ctx = await query_model(
                    model_request,
                    run_ctx,
                    usage_limits,
                    self.config.model_settings,
                    toolsets,
                    instrument,
                )
                return ModelResponseState(reply_ctx, environment, model_response, current_state)
            case InjectableModelRequestState(run_ctx, environment, injectable_model_request_parts):
                injected_model_request = await inject_injectable_model_request(
                    environment,
                    injectable_model_request_parts,
                    attacks,
                    self.config.tool_result_serialization_mode,
                )
                return ModelRequestState(
                    run_ctx, environment, injected_model_request, current_state
                )
            case ModelResponseState(run_ctx, environment, model_response):
                tool_call_parts = extract_tool_call_parts(run_ctx)
                if not tool_call_parts:
                    return EndState(
                        run_ctx,
                        environment,
                        FinishReason.AGENT_LOOP_END,
                        current_state,
                    )

                # handle_tool_calls manages environment copying and returns new run_ctx with updated env_state
                results_parts, new_run_ctx = await handle_tool_calls(
                    run_ctx, environment, tool_call_parts, toolsets
                )
                if parts_contain_only_model_request_parts(results_parts):
                    serialized_parts = serialize_tool_return_parts(
                        results_parts, self.config.tool_result_serialization_mode
                    )
                    return ModelRequestState(
                        new_run_ctx,
                        environment,
                        ModelRequest(serialized_parts),
                        current_state,
                    )
                return InjectableModelRequestState(
                    new_run_ctx, environment, results_parts, current_state
                )
            case EndState():
                raise ExecutionEndedError("The loop has already ended.")
            case _:
                assert_never(current_state)


def create_plain_agent(config: PlainAgentConfig, context: None = None) -> PlainAgent:
    """Factory function to create a PlainAgent from its configuration.

    Args:
        config: Agent configuration
        context: Optional context parameter (unused by agents, for registry compatibility)
    """
    # PlainAgent accepts model as str | KnownModelName | Model
    return PlainAgent(_config=config)
