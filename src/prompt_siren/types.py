# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeGuard, TypeVar

import pydantic
from pydantic import BaseModel
from pydantic_ai.messages import (
    AudioMediaType,
    BinaryContent,
    DocumentMediaType,
    ImageMediaType,
    ModelMessage,
    ModelRequestPart,
    UserContent,
)

InjectionVectorID = str
TaskCoupleID = str
ExecutionMode = Literal["benign", "attack"]


@dataclass(frozen=True)
class StrContentAttack:
    content: str
    kind: Literal["str"] = "str"


@dataclass(frozen=True)
class BinaryContentAttack:
    content: bytes
    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
    kind: Literal["binary"] = "binary"


InjectionAttack = Annotated[StrContentAttack | BinaryContentAttack, pydantic.Discriminator("kind")]

InjectionAttackT = TypeVar("InjectionAttackT", bound=InjectionAttack)

InjectionAttacksDict: TypeAlias = dict[InjectionVectorID, InjectionAttackT]

InjectionAttacksDictTypeAdapter = pydantic.TypeAdapter(
    InjectionAttacksDict,
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes="base64", val_json_bytes="base64"),
)


def _inject_str(
    content: str,
    default: InjectionAttacksDict[StrContentAttack],
    attacks: InjectionAttacksDict[InjectionAttack] | None,
) -> str:
    result = content
    if attacks is None:
        to_inject = default
    else:
        relevant_attacks = {k: v for k, v in attacks.items() if k in default}
        to_inject = default | relevant_attacks
    for vector_id, attack in to_inject.items():
        if not isinstance(attack, StrContentAttack):
            raise InvalidAttackError(
                f"Expected f`{StrContentAttack.__name__}`, got `{type(attack).__name__}`"
            )
        result = result.replace(vector_id, attack.content)
    return result


class InvalidAttackError(Exception): ...


@dataclass(frozen=True)
class InjectableStrContent:
    content: str
    default: InjectionAttacksDict[StrContentAttack]
    kind: Literal["str-content"] = "str-content"

    @property
    def vector_ids(self) -> list[InjectionVectorID]:
        return list(self.default.keys())

    def inject(self, attacks: InjectionAttacksDict[InjectionAttack] | None) -> str:
        return _inject_str(self.content, self.default, attacks)


@dataclass(frozen=True)
class InjectableBinaryContent:
    vector_id: InjectionVectorID
    default: BinaryContentAttack
    content: None = None
    kind: Literal["binary-content"] = "binary-content"

    @property
    def vector_ids(self) -> list[InjectionVectorID]:
        return [self.vector_id]

    def inject(self, attacks: InjectionAttacksDict[InjectionAttack] | None) -> BinaryContent:
        attack = attacks[self.vector_id] if attacks else self.default
        if not isinstance(attack, BinaryContentAttack):
            raise InvalidAttackError(
                f"Expected `{BinaryContentAttack.__name__}`, got `{type(attack).__name__}`"
            )
        if attack.media_type != self.default.media_type:
            raise RuntimeError(
                f"Attack media type mismatches with placeholder media type. Expected `{self.default.media_type}`, got `{attack.media_type}`."
            )
        return BinaryContent(data=attack.content, media_type=self.default.media_type)


InjectableUserContent = Annotated[
    InjectableStrContent | InjectableBinaryContent,
    pydantic.Discriminator("kind"),
]


@cache
def _get_injectable_user_part_vector_ids(
    part: InjectableUserPromptPart,
) -> list[InjectionVectorID]:
    """Needed because lru_cache should not be used for methods.

    https://docs.astral.sh/ruff/rules/cached-instance-method/#why-is-this-bad.
    """
    all_ids = []
    for content in part.content:
        if isinstance(content, InjectableStrContent | InjectableBinaryContent):
            all_ids.append(content.vector_ids)
            continue

    return all_ids


@dataclass(frozen=True)
class InjectableUserPromptPart:
    content: Sequence[InjectableUserContent | UserContent]
    part_kind: Literal["user-prompt"] = "user-prompt"

    @property
    def vector_ids(self) -> list[InjectionVectorID]:
        return _get_injectable_user_part_vector_ids(self)

    def inject(
        self, attacks: InjectionAttacksDict[InjectionAttack] | None
    ) -> Sequence[UserContent]:
        injected_content_parts = []
        for content_part in self.content:
            if isinstance(content_part, UserContent):
                injected_content_parts.append(content_part)
                continue
            injected_content_parts.append(content_part.inject(attacks))
        return injected_content_parts


def get_vector_ids(
    part: InjectableToolReturnPart | InjectableRetryPromptPart | InjectableStrContent,
) -> list[InjectionVectorID]:
    return list(part.default.keys())


@dataclass(frozen=True)
class InjectableToolReturnPart(Generic[InjectionAttackT]):
    tool_name: str
    content: Any
    tool_call_id: str
    default: dict[InjectionVectorID, InjectionAttackT]
    part_kind: Literal["tool-return"] = "tool-return"

    @property
    def vector_ids(self) -> list[InjectionVectorID]:
        return get_vector_ids(self)


@dataclass(frozen=True)
class InjectableRetryPromptPart:
    content: str
    tool_name: str | None
    tool_call_id: str
    default: InjectionAttacksDict[StrContentAttack]
    part_kind: Literal["retry-prompt"] = "retry-prompt"

    @property
    def vector_ids(self) -> list[InjectionVectorID]:
        return get_vector_ids(self)

    def inject(self, attacks: InjectionAttacksDict[InjectionAttack] | None) -> str:
        return _inject_str(self.content, self.default, attacks)


InjectableModelRequestPart = Annotated[
    InjectableUserPromptPart | InjectableToolReturnPart | InjectableRetryPromptPart,
    pydantic.Discriminator("part_kind"),
]


def is_injectable_model_request_part(
    part: InjectableModelRequestPart | ModelRequestPart,
) -> TypeGuard[InjectableModelRequestPart]:
    return isinstance(
        part,
        InjectableUserPromptPart | InjectableToolReturnPart | InjectableRetryPromptPart,
    )


@dataclass(frozen=True)
class InjectableModelRequest:
    parts: list[InjectableModelRequestPart | ModelRequestPart]
    kind: Literal["injectable-request"] = "injectable-request"


PIModelMessage = Annotated[ModelMessage | InjectableModelRequest, pydantic.Discriminator("kind")]


InjectableModelMessagesTypeAdapter = pydantic.TypeAdapter(
    list[PIModelMessage],
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes="base64", val_json_bytes="base64"),
)

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class AttackMetadata(BaseModel, Generic[ConfigT]):
    """Metadata for an attack file with attack configuration."""

    name: str
    config: ConfigT | None = None


class AttackFile(BaseModel, Generic[ConfigT]):
    """Schema for attack files storing injection attacks by task"""

    metadata: AttackMetadata[ConfigT]
    attacks: dict[TaskCoupleID, dict[str, dict[str, Any]]]

    @classmethod
    def from_attacks_dict(
        cls,
        attacks: dict[TaskCoupleID, InjectionAttacksDict[InjectionAttack]],
        name: str,
        config: ConfigT | None = None,
    ) -> AttackFile[ConfigT]:
        """Convert a dictionary of attacks to an AttackFile object

        Args:
            attacks: Dictionary mapping task IDs to attack dictionaries
            name: Name of the attack
            description: Optional description of the attack
            author: Optional author of the attack

        Returns:
            AttackFile object with all attacks
        """

        # Convert attacks to serializable format
        serialized_attacks = {}
        for task_id, attack_dict in attacks.items():
            serialized_attacks[task_id] = InjectionAttacksDictTypeAdapter.dump_python(attack_dict)

        return cls(
            metadata=AttackMetadata[ConfigT](name=name, config=config),
            attacks=serialized_attacks,
        )

    def to_attacks_dict(
        self,
    ) -> tuple[dict[TaskCoupleID, InjectionAttacksDict[InjectionAttack]], str]:
        """Convert an AttackFile to a dictionary of attacks

        Returns:
            Tuple of (attacks dict, attack name)
        """
        attack_data: dict[TaskCoupleID, InjectionAttacksDict[InjectionAttack]] = {}

        # Convert serialized attacks back to attack objects
        for task_id, vectors in self.attacks.items():
            attack_data[task_id] = InjectionAttacksDictTypeAdapter.validate_python(vectors)

        return attack_data, self.metadata.name
