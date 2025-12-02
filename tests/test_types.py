# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for serialization and deserialization of types in prompt_siren."""

import base64
import json
from collections.abc import Sequence

from prompt_siren.types import (
    BinaryContentAttack,
    InjectableModelMessagesTypeAdapter,
    InjectableModelRequest,
    InjectableStrContent,
    InjectableUserPromptPart,
    InjectionAttacksDict,
    InjectionAttacksDictTypeAdapter,
    StrContentAttack,
)
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart


def test_injectable_model_messages_adapter_roundtrip():
    """Test serialization and deserialization of list[PIModelMessage] using InjectableModelMessagesTypeAdapter."""
    # Create a simple list of messages: one regular, one injectable
    regular_msg = ModelRequest(
        parts=[
            UserPromptPart(content="Regular message"),
            UserPromptPart(
                content=[BinaryContent(data=b"This is a PNG image", media_type="image/png")]
            ),
        ]
    )

    injectable_content = InjectableStrContent(
        content="Injectable content", default={"vector1": StrContentAttack("a")}
    )
    injectable_part = InjectableUserPromptPart(content=[injectable_content])
    injectable_msg = InjectableModelRequest(parts=[injectable_part])

    # Create the list
    messages = [regular_msg, injectable_msg]

    # Serialize to JSON and verify binary encoding
    json_data = InjectableModelMessagesTypeAdapter.dump_json(messages)

    # Verify binary content is correctly encoded as base64 in the JSON
    parsed_json = json.loads(json_data)
    binary_part = parsed_json[0]["parts"][1]["content"][0]
    assert binary_part["kind"] == "binary"
    # Check that data is a valid base64 string
    binary_data = base64.b64decode(binary_part["data"])
    assert binary_data == b"This is a PNG image"

    # Deserialize from JSON
    deserialized = InjectableModelMessagesTypeAdapter.validate_json(json_data)

    # Verify structure is preserved
    assert len(deserialized) == 2

    # Verify first message (regular)
    assert isinstance(deserialized[0], ModelRequest)
    assert len(deserialized[0].parts) == 2
    assert isinstance(deserialized[0].parts[0], UserPromptPart)
    assert deserialized[0].parts[0].content == "Regular message"
    assert isinstance(deserialized[0].parts[1], UserPromptPart)
    assert isinstance(deserialized[0].parts[1].content, Sequence)
    assert len(deserialized[0].parts[1].content) == 1
    assert isinstance(deserialized[0].parts[1].content[0], BinaryContent)
    assert deserialized[0].parts[1].content[0].data == b"This is a PNG image"
    assert deserialized[0].parts[1].content[0].media_type == "image/png"

    # Verify second message (injectable)
    assert isinstance(deserialized[1], InjectableModelRequest)
    assert deserialized[1].kind == "injectable-request"
    assert len(deserialized[1].parts) == 1
    first_part = deserialized[1].parts[0]
    assert isinstance(first_part, InjectableUserPromptPart)
    assert first_part.part_kind == "user-prompt"
    assert len(first_part.content) == 1
    injectable_content_item = first_part.content[0]
    assert isinstance(injectable_content_item, InjectableStrContent)
    assert injectable_content_item.content == "Injectable content"
    assert list(injectable_content_item.default.keys()) == ["vector1"]


def test_injection_attacks_dict_adapter_roundtrip():
    """Test serialization and deserialization of InjectionAttacksDict."""
    # Create instances of both attack types
    str_attack1 = StrContentAttack(content="First malicious payload")
    str_attack2 = StrContentAttack(content="Second malicious payload")
    binary_attack = BinaryContentAttack(content=b"Binary malicious payload", media_type="image/png")

    # Create dictionaries for each type of attack
    str_attacks: InjectionAttacksDict[StrContentAttack] = {
        "vector1": str_attack1,
        "vector2": str_attack2,
    }

    binary_attacks: InjectionAttacksDict[BinaryContentAttack] = {
        "vector3": binary_attack,
    }

    # Test JSON serialization and deserialization using the TypeAdapter
    json_str_attacks = InjectionAttacksDictTypeAdapter.dump_json(str_attacks)
    deserialized_str_attacks = InjectionAttacksDictTypeAdapter.validate_json(json_str_attacks)

    # Verify structure is preserved
    assert len(deserialized_str_attacks) == 2
    assert isinstance(deserialized_str_attacks["vector1"], StrContentAttack)
    assert isinstance(deserialized_str_attacks["vector2"], StrContentAttack)
    assert deserialized_str_attacks["vector1"].content == "First malicious payload"
    assert deserialized_str_attacks["vector2"].content == "Second malicious payload"

    json_bin_attacks = InjectionAttacksDictTypeAdapter.dump_json(binary_attacks)

    # Verify binary content is correctly encoded as base64 in the JSON
    parsed_bin_attacks = json.loads(json_bin_attacks)
    binary_attack_json = parsed_bin_attacks["vector3"]
    assert binary_attack_json["kind"] == "binary"
    # Check that content is a valid base64 string
    binary_data = base64.b64decode(binary_attack_json["content"])
    assert binary_data == b"Binary malicious payload"

    deserialized_bin_attacks = InjectionAttacksDictTypeAdapter.validate_json(json_bin_attacks)

    # Verify binary structure is preserved
    assert len(deserialized_bin_attacks) == 1
    assert isinstance(deserialized_bin_attacks["vector3"], BinaryContentAttack)
    assert deserialized_bin_attacks["vector3"].content == b"Binary malicious payload"


def test_injection_attacks_dict_adapter_mixed_roundtrip():
    """Test serialization and deserialization of InjectionAttacksDict with mixed types."""
    # Create a mixed dictionary with both types (using the base InjectionAttack type)
    mixed_attacks: InjectionAttacksDict = {
        "vector1": StrContentAttack(content="String payload"),
        "vector2": BinaryContentAttack(content=b"Binary payload", media_type="image/png"),
    }

    # Serialize to JSON and deserialize
    # Use a properly configured adapter for mixed attacks (with binary content)
    json_mixed = InjectionAttacksDictTypeAdapter.dump_json(mixed_attacks)

    # Verify binary content is correctly encoded as base64 in the JSON
    parsed_mixed = json.loads(json_mixed)
    binary_attack_json = parsed_mixed["vector2"]
    assert binary_attack_json["kind"] == "binary"
    # Check that content is a valid base64 string
    binary_data = base64.b64decode(binary_attack_json["content"])
    assert binary_data == b"Binary payload"

    deserialized_mixed = InjectionAttacksDictTypeAdapter.validate_json(json_mixed)

    # Verify discriminated types are preserved
    assert len(deserialized_mixed) == 2
    assert isinstance(deserialized_mixed["vector1"], StrContentAttack)
    assert isinstance(deserialized_mixed["vector2"], BinaryContentAttack)
    assert deserialized_mixed["vector1"].content == "String payload"
    assert deserialized_mixed["vector2"].content == b"Binary payload"
