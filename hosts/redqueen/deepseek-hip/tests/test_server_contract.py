from __future__ import annotations

import sys
import unittest
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from cached_model import install_cached_moe as cached_model_installer
from deepseek_server import (
    _BYTE_LEVEL_ENCODER,
    DeepSeekEngine,
    RequestError,
    _IncrementalTextDecoder,
    _request_options,
    install_cached_moe,
)


class FakeTokenizer:
    def __init__(self):
        self.pieces = {1: b"hello", 2: b" ", 3: b"world", 4: b"", 5: "｜literal｜"}
        self.all_special_ids = set()
        self.added_tokens_decoder = {
            4: SimpleNamespace(special=True),
            5: SimpleNamespace(special=False),
        }
        self.backend_tokenizer = SimpleNamespace(decoder=type("ByteLevel", (), {})())

    def convert_ids_to_tokens(self, token_id):
        piece = self.pieces[token_id]
        if isinstance(piece, str):
            return piece
        return "".join(_BYTE_LEVEL_ENCODER[value] for value in piece)


class DeepSeekRequestContractTest(unittest.TestCase):
    def test_cached_moe_installer_is_bound_to_worker_entrypoint(self):
        self.assertIs(install_cached_moe, cached_model_installer)

    def test_malformed_thinking_mode_is_a_client_error(self):
        with self.assertRaises(RequestError):
            _request_options({"thinking_mode": []})

    def test_malformed_tool_schema_is_a_client_error(self):
        with self.assertRaises(RequestError):
            _request_options({"tools": [{}]})

    def test_malformed_tool_namespace_is_a_client_error(self):
        body = {
            "tools": [{"type": "function", "function": {"name": "lookup"}, "namespace": {"name": "a::b"}}]
        }
        with self.assertRaises(RequestError):
            _request_options(body)

    def test_malformed_tool_descriptions_are_client_errors(self):
        for tool in (
            {"function": {"name": "lookup", "description": []}},
            {"function": {"name": "lookup"}, "namespace": {"name": "search", "description": []}},
        ):
            with self.subTest(tool=tool), self.assertRaises(RequestError):
                _request_options({"tools": [tool]})

    def test_malformed_tool_parameter_schema_is_a_client_error(self):
        for parameters in ("oops", {"type": "string"}, {"type": "object", "properties": []},
                           {"type": "object", "required": "query"},
                           {"type": "object", "properties": {"query": {"type": 7}}},
                           {"type": "object", "required": ["query", "query"]},
                           {"type": "object", "properties": None},
                           {"type": "object", "properties": {"query": {"required": None}}},
                           {"type": "object", "properties": {"query": {"enum": None}}},
                           {"type": "object", "items": None},
                           {"type": "object", "allOf": None},
                           {"type": "object", "dependentRequired": None},
                           {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#",
                            "minimum": 0, "exclusiveMinimum": True},
                           {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#",
                            "properties": {"value": {"minimum": 0, "exclusiveMinimum": True,
                                                        "$schema": "http://json-schema.org/draft-04/schema#"}}},
                           {"type": "object", "$schema": "http://json-schema.org/draft-04/schema#",
                            "properties": {"value": {"exclusiveMinimum": 1,
                                                        "$schema": "http://json-schema.org/draft-07/schema#"}}},
                           None):
            tool = {"function": {"name": "lookup", "parameters": parameters}}
            with self.subTest(parameters=parameters), self.assertRaises(RequestError):
                _request_options({"tools": [tool]})

    def test_boolean_schemas_and_draft_seven_tuple_items_are_valid(self):
        parameters = {
            "type": "object",
            "properties": {"anything": True, "nothing": False},
            "items": [{"type": "string"}, False],
            "additionalItems": True,
        }
        _request_options({"tools": [{"function": {"name": "lookup", "parameters": parameters}}]})

    def test_draft_four_boolean_exclusive_bounds_require_matching_bound(self):
        for parameters in (
            {"type": "object", "$schema": "http://json-schema.org/draft-04/schema#",
             "minimum": 0, "exclusiveMinimum": True,
             "properties": {"value": {"minimum": 1, "exclusiveMinimum": False}}},
            {"type": "object", "maximum": 10, "exclusiveMaximum": False},
        ):
            with self.subTest(parameters=parameters):
                _request_options({"tools": [{"function": {"name": "lookup", "parameters": parameters}}]})

        invalid = {"type": "object", "exclusiveMinimum": True}
        with self.assertRaises(RequestError):
            _request_options({"tools": [{"function": {"name": "lookup", "parameters": invalid}}]})

    def test_streaming_rejects_modes_without_incremental_contracts(self):
        for body in (
            {"stream": True, "reasoning_effort": "high"},
            {"stream": True, "tools": [{"type": "function", "function": {"name": "lookup"}}]},
        ):
            with self.subTest(body=body), self.assertRaises(RequestError):
                _request_options(body)

    def test_incremental_decoder_preserves_output_and_bounds_pending_utf8(self):
        tokenizer = FakeTokenizer()
        decoder = _IncrementalTextDecoder(tokenizer)
        unicode_bytes = "漢🙂".encode("utf-8")
        tokenizer.pieces.update({10 + index: bytes([value]) for index, value in enumerate(unicode_bytes)})
        tokens = [1, 2, 3, 4] + [10 + index for index in range(len(unicode_bytes))]
        chunks, pending_lengths = [], []
        for token in tokens:
            chunks.append(decoder.put(token))
            pending_lengths.append(len(decoder.utf8.getstate()[0]))
        chunks.append(decoder.finish())

        self.assertEqual("".join(chunks), "hello world漢🙂")
        self.assertLessEqual(max(pending_lengths), 3)

    def test_added_literal_tokens_flush_incomplete_utf8_and_preserve_text(self):
        tokenizer = FakeTokenizer()
        tokenizer.pieces[10] = b"\xe6"
        decoder = _IncrementalTextDecoder(tokenizer)
        text = decoder.put(10) + decoder.put(5) + decoder.finish()

        self.assertEqual(text, "�｜literal｜")


class DeepSeekStreamBackpressureTest(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_stream_write_times_out(self):
        class StalledResponse:
            async def write(self, _payload):
                await asyncio.Event().wait()

        engine = object.__new__(DeepSeekEngine)
        with patch("deepseek_server.STREAM_WRITE_TIMEOUT_S", 0.01):
            with self.assertRaises(ConnectionResetError):
                await engine._write_stream_bytes(StalledResponse(), b"chunk")


if __name__ == "__main__":
    unittest.main()
