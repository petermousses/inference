from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from redqueen_adapters.image_api import (
    MAX_IMAGE_BYTES,
    ComfyBackend,
    ImageSettings,
    create_image_app,
)
from redqueen_adapters.jev_api import JevSettings, create_jev_app


TOKEN = "adapter-test-token"


def png(size: tuple[int, int] = (1024, 1024), alpha: int = 0) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, (255, 0, 0, alpha)).save(output, format="PNG")
    return output.getvalue()


def jpeg(size: tuple[int, int] = (64, 64)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (0, 255, 0)).save(output, format="JPEG")
    return output.getvalue()


class FakeImageBackend:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.release = asyncio.Event()
        self.block = False

    async def generate(self, request: dict) -> bytes:
        self.requests.append(request)
        if self.block:
            await self.release.wait()
        return png()

    async def edit(self, request: dict, images: list[tuple[str, bytes, str]]) -> bytes:
        self.requests.append({**request, "images": images})
        return png((512, 512))


class FakeJevBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    async def decide(self, state: object, question: dict) -> dict:
        self.calls.append((state, question))
        if question["type"] == "noul":
            return {"type": "noul", "confidence": 0.8, "noul": 0.8, "input_tokens": 9}
        probabilities = {
            key: 1 / len(question["criteria"]) for key in question["criteria"]
        }
        return {
            "type": question["type"],
            "confidence": max(probabilities.values()),
            "choice": next(iter(probabilities)),
            "probabilities": probabilities,
            "input_tokens": 11,
        }


class FakeClefBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def evaluate(self, model: str, payload: dict) -> dict:
        self.calls.append((model, payload))
        answers = {}
        for question_id, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[question_id] = {"type": "noul", "noul": 0.9}
            elif question["type"] == "choice":
                probabilities = {
                    key: 1 / len(question["criteria"])
                    for key in question["criteria"]
                }
                answers[question_id] = {
                    "type": "choice",
                    "choice": next(iter(probabilities)),
                    "probabilities": probabilities,
                    "confidence": max(probabilities.values()),
                }
            else:
                probabilities = {
                    str(index): 1 / len(question["criteria"])
                    for index in range(len(question["criteria"]))
                }
                answers[question_id] = {
                    "type": "score",
                    "score": sum(
                        int(key) * probability
                        for key, probability in probabilities.items()
                    ),
                    "probabilities": probabilities,
                    "confidence": max(probabilities.values()),
                }
        return {
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": 17, "output_tokens": 0},
        }


class RecordingComfyBackend(ComfyBackend):
    def __init__(self) -> None:
        workflow_dir = Path(__file__).resolve().parents[2] / "comfyui"
        super().__init__(ImageSettings(api_key=TOKEN, workflow_dir=workflow_dir))
        self.workflows: list[dict] = []
        self.removed: list[tuple[str, str, str]] = []

    async def _run(self, workflow: dict) -> bytes:
        self.workflows.append(workflow)
        return png()

    async def _upload(
        self, request_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        suffix = ".png" if content_type == "image/png" else ".jpg"
        return f"api/{request_id}/{len(self.removed)}{suffix}"

    def _remove_file(self, area: str, subfolder: str, filename: str) -> None:
        self.removed.append((area, subfolder, filename))


class AdapterTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._clients: list[TestClient] = []

    async def asyncTearDown(self) -> None:
        for client in reversed(self._clients):
            await client.close()

    async def client(self, app) -> TestClient:
        client = TestClient(TestServer(app))
        await client.start_server()
        self._clients.append(client)
        return client

    @staticmethod
    def auth(token: str = TOKEN) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def test_image_generation_requires_bearer_auth_and_returns_b64(self) -> None:
        backend = FakeImageBackend()
        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), backend)
        )

        missing = await client.post(
            "/v1/images/generations", json={"prompt": "a red cube"}
        )
        self.assertEqual(missing.status, 401)
        self.assertEqual(missing.headers["Cache-Control"], "no-store")

        response = await client.post(
            "/v1/images/generations",
            headers=self.auth(),
            json={
                "model": "qwen-image-2.1",
                "prompt": "a red cube",
                "size": "1024x1024",
                "quality": "high",
                "background": "transparent",
                "seed": 42,
                "n": 1,
                "response_format": "b64_json",
            },
        )
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), png())
        self.assertEqual(
            backend.requests,
            [
                {
                    "model": "qwen-image-2.1",
                    "prompt": "a red cube",
                    "size": "1024x1024",
                    "quality": "high",
                    "background": "transparent",
                    "seed": 42,
                    "n": 1,
                    "response_format": "b64_json",
                }
            ],
        )

    async def test_image_generation_rejects_unsupported_or_oversized_input(
        self,
    ) -> None:
        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), FakeImageBackend())
        )
        for payload in (
            {"prompt": "x", "n": 2},
            {"prompt": "x", "size": "640x640"},
            {"prompt": "x", "response_format": "url"},
            {"prompt": "x", "model": "arbitrary"},
            {"prompt": "x", "seed": 1.5},
            {"prompt": "x" * 10_001},
        ):
            with self.subTest(payload=payload):
                response = await client.post(
                    "/v1/images/generations", headers=self.auth(), json=payload
                )
                self.assertEqual(response.status, 400)

    async def test_image_model_catalog_contains_both_fixed_backends(self) -> None:
        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), FakeImageBackend())
        )
        response = await client.get("/v1/models", headers=self.auth())
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(
            [item["id"] for item in body["data"]],
            ["qwen-image-2.1", "qwen-image-2.1-uncensored"],
        )

    async def test_image_edit_accepts_two_bounded_references_and_rejects_mask(
        self,
    ) -> None:
        backend = FakeImageBackend()
        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), backend)
        )
        form = {
            "model": "qwen-image-2.1",
            "prompt": "make it green",
            "size": "512x512",
            "response_format": "b64_json",
        }
        data = __import__("aiohttp").FormData()
        for key, value in form.items():
            data.add_field(key, value)
        data.add_field(
            "image", png((64, 64)), filename="one.png", content_type="image/png"
        )
        data.add_field("image", jpeg(), filename="two.jpg", content_type="image/jpeg")
        response = await client.post("/v1/images/edits", headers=self.auth(), data=data)
        self.assertEqual(response.status, 200)
        self.assertEqual(len(backend.requests[0]["images"]), 2)

        masked = __import__("aiohttp").FormData()
        masked.add_field("prompt", "x")
        masked.add_field(
            "image", png((64, 64)), filename="one.png", content_type="image/png"
        )
        masked.add_field(
            "mask", png((64, 64)), filename="mask.png", content_type="image/png"
        )
        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=masked
        )
        self.assertEqual(response.status, 400)
        self.assertIn("mask", (await response.json())["error"]["message"])

        corrupt = __import__("aiohttp").FormData()
        corrupt.add_field("prompt", "x")
        corrupt.add_field(
            "image",
            b"\x89PNG\r\n\x1a\nnot-a-png",
            filename="bad.png",
            content_type="image/png",
        )
        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=corrupt
        )
        self.assertEqual(response.status, 400)

        oversized = __import__("aiohttp").FormData()
        oversized.add_field("prompt", "x")
        oversized.add_field(
            "image",
            io.BytesIO(b"x" * (MAX_IMAGE_BYTES + 1)),
            filename="oversized.png",
            content_type="image/png",
        )
        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=oversized
        )
        self.assertEqual(response.status, 400)
        self.assertIn("exceeds", (await response.json())["error"]["message"])

    async def test_image_edit_accepts_openai_array_field_name(self) -> None:
        backend = FakeImageBackend()
        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), backend)
        )
        data = __import__("aiohttp").FormData()
        data.add_field("prompt", "make it blue")
        data.add_field("size", "512x512")
        data.add_field(
            "image[]", png((64, 64)), filename="one.png", content_type="image/png"
        )
        data.add_field(
            "image[]", jpeg(), filename="two.jpg", content_type="image/jpeg"
        )

        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=data
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(
            [
                (name, content_type)
                for name, _, content_type in backend.requests[0]["images"]
            ],
            [("one.png", "image/png"), ("two.jpg", "image/jpeg")],
        )

        too_many = __import__("aiohttp").FormData()
        too_many.add_field("prompt", "make it blue")
        for field_name in ("image", "image[]", "image[]"):
            too_many.add_field(
                field_name,
                png((64, 64)),
                filename="reference.png",
                content_type="image/png",
            )
        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=too_many
        )
        self.assertEqual(response.status, 400)
        self.assertIn("at most two", (await response.json())["error"]["message"])

        unknown = __import__("aiohttp").FormData()
        unknown.add_field("prompt", "make it blue")
        unknown.add_field(
            "image[0]", png((64, 64)), filename="one.png", content_type="image/png"
        )
        response = await client.post(
            "/v1/images/edits", headers=self.auth(), data=unknown
        )
        self.assertEqual(response.status, 400)
        self.assertIn("unsupported multipart field", (await response.json())["error"]["message"])
        self.assertEqual(len(backend.requests), 1)

    async def test_image_output_contract_rejects_wrong_size_or_alpha(self) -> None:
        class BadBackend(FakeImageBackend):
            async def generate(self, request: dict) -> bytes:
                if request["background"] == "transparent":
                    return png(alpha=255)
                return png((512, 512), alpha=0)

        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), BadBackend())
        )
        transparent = await client.post(
            "/v1/images/generations",
            headers=self.auth(),
            json={"prompt": "x", "background": "transparent"},
        )
        self.assertEqual(transparent.status, 502)

        wrong_size = await client.post(
            "/v1/images/generations",
            headers=self.auth(),
            json={"prompt": "x", "background": "opaque"},
        )
        self.assertEqual(wrong_size.status, 502)

    async def test_opaque_image_output_is_flattened_to_rgb(self) -> None:
        class NearlyOpaqueBackend(FakeImageBackend):
            async def generate(self, request: dict) -> bytes:
                return png(alpha=254)

        client = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), NearlyOpaqueBackend())
        )
        response = await client.post(
            "/v1/images/generations",
            headers=self.auth(),
            json={"prompt": "x", "background": "opaque"},
        )
        self.assertEqual(response.status, 200)
        body = await response.json()
        with Image.open(
            io.BytesIO(base64.b64decode(body["data"][0]["b64_json"]))
        ) as image:
            image.load()
            self.assertEqual(image.size, (1024, 1024))
            self.assertEqual(image.mode, "RGB")

    async def test_comfy_workflow_mapping_is_fixed_and_request_scoped(self) -> None:
        backend = RecordingComfyBackend()
        generation = {
            "model": "qwen-image-2.1",
            "prompt": "glass sphere",
            "size": "512x512",
            "quality": "low",
            "background": "transparent",
            "seed": 44,
            "n": 1,
            "response_format": "b64_json",
        }
        await backend.generate(generation)
        workflow = backend.workflows[-1]
        self.assertEqual(workflow["5"]["inputs"]["width"], 512)
        self.assertEqual(workflow["5"]["inputs"]["height"], 512)
        self.assertEqual(workflow["7"]["inputs"]["seed"], 44)
        self.assertEqual(workflow["7"]["inputs"]["steps"], 4)
        self.assertIn("RGBA format", workflow["4"]["inputs"]["prompt"])
        self.assertRegex(
            workflow["9"]["inputs"]["filename_prefix"], r"^api/[0-9a-f]{32}/"
        )

        edit = {
            **generation,
            "prompt": "recolor",
            "background": "auto",
            "quality": "high",
        }
        images = [
            ("one.png", b"one", "image/png"),
            ("two.jpg", b"two", "image/jpeg"),
        ]
        await backend.edit(edit, images)
        workflow = backend.workflows[-1]
        self.assertEqual(workflow["8"]["inputs"]["seed"], 44)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 25)
        self.assertRegex(workflow["1"]["inputs"]["image"], r"^api/[0-9a-f]{32}/0\.png$")
        self.assertRegex(workflow["2"]["inputs"]["image"], r"^api/[0-9a-f]{32}/0\.jpg$")
        self.assertEqual([item[0] for item in backend.removed], ["input", "input"])

    async def test_uncensored_alias_uses_only_fixed_gguf_loader_mapping(self) -> None:
        backend = RecordingComfyBackend()
        request = {
            "model": "qwen-image-2.1-uncensored",
            "prompt": "a red cube",
            "size": "512x512",
            "quality": "low",
            "background": "opaque",
            "seed": 13,
            "n": 1,
            "response_format": "b64_json",
        }

        await backend.generate(request)
        workflow = backend.workflows[-1]
        diffusion_loaders = [
            node for node in workflow.values()
            if node.get("class_type") in {"UNETLoader", "UnetLoaderGGUF"}
        ]
        clip_loaders = [
            node for node in workflow.values() if node.get("class_type") == "CLIPLoader"
        ]
        self.assertEqual(len(diffusion_loaders), 1)
        self.assertEqual(diffusion_loaders[0]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(
            diffusion_loaders[0]["inputs"],
            {"unet_name": "qwen-image-2.1-UC-Q4_K_M.gguf"},
        )
        self.assertEqual(len(clip_loaders), 1)
        self.assertEqual(
            clip_loaders[0]["inputs"]["clip_name"],
            "qwen3vl_8b_int8_convrot.safetensors",
        )
        self.assertIn(
            "qwen-image-2.1-uncensored",
            workflow["9"]["inputs"]["filename_prefix"],
        )

        edit = {**request, "background": "auto"}
        await backend.edit(edit, [("reference.png", b"x", "image/png")])
        edit_workflow = backend.workflows[-1]
        edit_loader = next(
            node for node in edit_workflow.values()
            if node.get("class_type") == "UnetLoaderGGUF"
        )
        edit_clip_loaders = [
            node for node in edit_workflow.values()
            if node.get("class_type") == "CLIPLoader"
        ]
        self.assertEqual(
            edit_loader["inputs"]["unet_name"], "qwen-image-2.1-UC-Q4_K_M.gguf"
        )
        self.assertEqual(len(edit_clip_loaders), 1)
        self.assertEqual(
            edit_clip_loaders[0]["inputs"]["clip_name"],
            "qwen3vl_8b_int8_convrot.safetensors",
        )

    async def test_uncensored_workflow_fails_closed_on_unexpected_loader_graph(self) -> None:
        backend = RecordingComfyBackend()
        backend._workflows["generate"]["1"]["class_type"] = "UnexpectedLoader"
        request = {
            "model": "qwen-image-2.1-uncensored",
            "prompt": "a red cube",
            "size": "512x512",
            "quality": "low",
            "background": "opaque",
            "seed": 13,
            "n": 1,
            "response_format": "b64_json",
        }
        with self.assertRaisesRegex(RuntimeError, "expected exactly one fixed UNETLoader"):
            await backend.generate(request)

    async def test_comfy_cleanup_refuses_paths_outside_its_state_area(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "state" / "output" / "api" / "job"
            output.mkdir(parents=True)
            inside = output / "image.png"
            outside = root / "outside.png"
            inside.write_bytes(b"inside")
            outside.write_bytes(b"outside")
            settings = ImageSettings(
                api_key=TOKEN,
                workflow_dir=Path(__file__).resolve().parents[2] / "comfyui",
                comfy_state_dir=root / "state",
            )
            backend = ComfyBackend(settings)
            backend._remove_file("output", "api/job", "image.png")
            backend._remove_file("output", "../../", "outside.png")
            self.assertFalse(inside.exists())
            self.assertTrue(outside.exists())

    async def test_image_queue_rejects_excess_work_instead_of_growing_unbounded(
        self,
    ) -> None:
        backend = FakeImageBackend()
        backend.block = True
        settings = ImageSettings(api_key=TOKEN, max_active=1, max_waiting=0)
        client = await self.client(create_image_app(settings, backend))
        first = asyncio.create_task(
            client.post(
                "/v1/images/generations", headers=self.auth(), json={"prompt": "first"}
            )
        )
        while not backend.requests:
            await asyncio.sleep(0)
        second = await client.post(
            "/v1/images/generations", headers=self.auth(), json={"prompt": "second"}
        )
        self.assertEqual(second.status, 429)
        self.assertEqual(second.headers["Retry-After"], "5")
        backend.release.set()
        self.assertEqual((await first).status, 200)

    async def test_jev_contract_auth_validation_and_usage(self) -> None:
        backend = FakeJevBackend()
        client = await self.client(create_jev_app(JevSettings(api_key=TOKEN), backend))
        payload = {
            "model": "jev-latest",
            "state": {"ticket": "customer asks for refund"},
            "questions": {
                "refund": {"type": "noul", "instructions": "is this a refund request?"},
                "route": {
                    "type": "choice",
                    "instructions": "where should it go?",
                    "criteria": {"billing": "billing", "support": "technical support"},
                },
            },
        }
        unauthorized = await client.post("/v1/systemone", json=payload)
        self.assertEqual(unauthorized.status, 401)

        response = await client.post("/v1/systemone", headers=self.auth(), json=payload)
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["model"], "jevk5-4b-v0.3")
        self.assertEqual(body["usage"], {"input_tokens": 20, "output_tokens": 0})
        self.assertEqual(set(body["answers"]), {"refund", "route"})

        bad = await client.post(
            "/v1/systemone",
            headers=self.auth(),
            json={
                "state": "x",
                "questions": {
                    "q": {"type": "choice", "instructions": "x", "criteria": ["only"]}
                },
            },
        )
        self.assertEqual(bad.status, 400)

    async def test_jev_rejects_unknown_models_and_too_many_questions(self) -> None:
        client = await self.client(
            create_jev_app(JevSettings(api_key=TOKEN), FakeJevBackend())
        )
        for payload in (
            {
                "model": "wrong",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            },
            {
                "state": "x",
                "questions": {
                    str(index): {"type": "noul", "instructions": "x"}
                    for index in range(33)
                },
            },
        ):
            response = await client.post(
                "/v1/systemone", headers=self.auth(), json=payload
            )
            self.assertEqual(response.status, 400)

    async def test_shared_systemone_dispatches_clef_family_and_images(self) -> None:
        backend = FakeClefBackend()
        client = await self.client(
            create_jev_app(
                JevSettings(api_key=TOKEN), FakeJevBackend(), clef_backend=backend
            )
        )
        image_url = "data:image/png;base64," + base64.b64encode(
            png((64, 64))
        ).decode()
        payload = {
            "model": "clef-flash",
            "state": {"ticket": "check this image"},
            "questions": {
                "has_item": {
                    "type": "noul",
                    "instructions": "Is there an item in the image?",
                }
            },
            "images": [image_url],
        }
        path = "/v1/systemone"
        self.assertEqual((await client.post(path, json=payload)).status, 401)

        response = await client.post(path, headers=self.auth(), json=payload)
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["model"], "clef-flash-bf16")
        self.assertEqual(body["usage"]["input_tokens"], 17)
        self.assertEqual(backend.calls[0][0], "clef-flash-bf16")
        self.assertEqual(backend.calls[0][1]["model"], "clef-flash-bf16")
        self.assertEqual(backend.calls[0][1]["images"], [image_url])

        invalid = await client.post(
            path,
            headers=self.auth(),
            json={**payload, "images": ["https://example.com/image.png"]},
        )
        self.assertEqual(invalid.status, 400)

        mismatched_model = await client.post(
            path, headers=self.auth(), json={**payload, "model": "clef-not-real"}
        )
        self.assertEqual(mismatched_model.status, 400)

    async def test_openai_decisions_accepts_bounded_clef_image_input(self) -> None:
        backend = FakeClefBackend()
        client = await self.client(
            create_jev_app(
                JevSettings(api_key=TOKEN), FakeJevBackend(), clef_backend=backend
            )
        )
        image_url = "data:image/png;base64," + base64.b64encode(
            png((64, 64))
        ).decode()
        payload = {
            "model": "clef-flash-q8",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "classify this image"},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "original",
                        },
                    ],
                }
            ],
            "questions": [
                {
                    "type": "predicate",
                    "name": "is_red",
                    "instructions": "Is the main object red?",
                },
                {
                    "type": "choice",
                    "name": "object",
                    "instructions": "What is shown?",
                    "choices": [
                        {"value": "red_cube", "description": "a red cube"},
                        {"value": "other"},
                    ],
                },
                {
                    "type": "score",
                    "name": "visibility",
                    "instructions": "How visible is the main object?",
                    "levels": [
                        {"label": "obscured", "description": "hard to see"},
                        {"label": "clear", "description": "easy to see"},
                    ],
                },
            ],
        }

        response = await client.post(
            "/v1/decisions", headers=self.auth(), json=payload
        )

        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["model"], "clef-flash-q8")
        self.assertEqual(body["answers"][0]["probability"], 0.9)
        self.assertEqual(body["answers"][0]["name"], "is_red")
        self.assertEqual(body["answers"][1]["choice"], "red_cube")
        self.assertEqual(body["answers"][1]["name"], "object")
        self.assertEqual(body["answers"][2]["score"], 0.5)
        self.assertEqual(
            [item["label"] for item in body["answers"][2]["probabilities"]],
            ["obscured", "clear"],
        )
        self.assertEqual(body["usage"]["total_tokens"], 17)
        self.assertEqual(backend.calls[0][0], "clef-flash-q8")
        self.assertEqual(backend.calls[0][1]["images"], [image_url])
        self.assertEqual(
            backend.calls[0][1]["state"],
            [{"role": "user", "content": "classify this image"}],
        )
        unsupported_remote_image = await client.post(
            "/v1/decisions",
            headers=self.auth(),
            json={
                **payload,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.test/image.png",
                            }
                        ],
                    }
                ],
            },
        )
        self.assertEqual(unsupported_remote_image.status, 400)

    async def test_openai_decisions_maps_boolean_choices_and_rejects_jev_images(
        self,
    ) -> None:
        jev_backend = FakeJevBackend()
        client = await self.client(
            create_jev_app(JevSettings(api_key=TOKEN), jev_backend)
        )
        payload = {
            "model": "jevk5-4b-v0.3",
            "input": "Does this need a human?",
            "questions": [
                {
                    "type": "choice",
                    "instructions": "Choose whether a human is needed.",
                    "choices": [
                        {"value": True},
                        {"value": "true"},
                        {"value": False},
                    ],
                }
            ],
        }
        response = await client.post(
            "/v1/decisions", headers=self.auth(), json=payload
        )
        self.assertEqual(response.status, 200)
        answer = (await response.json())["answers"][0]
        self.assertIsNone(answer["name"])
        self.assertIs(answer["choice"], True)
        self.assertEqual(
            [item["value"] for item in answer["probabilities"]],
            [True, "true", False],
        )
        self.assertEqual(
            jev_backend.calls[0][1]["criteria"],
            {
                "openai_choice_0": "boolean true",
                "openai_choice_1": "true",
                "openai_choice_2": "boolean false",
            },
        )

        unsupported_image = await client.post(
            "/v1/decisions",
            headers=self.auth(),
            json={
                **payload,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": (
                                    "data:image/png;base64,"
                                    + base64.b64encode(png((64, 64))).decode()
                                ),
                            }
                        ],
                    }
                ],
            },
        )
        self.assertEqual(unsupported_image.status, 400)
        self.assertIn(
            "local Clef models",
            (await unsupported_image.json())["error"]["message"],
        )

    async def test_jev_backend_failure_is_redacted_as_502(self) -> None:
        class BrokenJevBackend(FakeJevBackend):
            async def decide(self, state: object, question: dict) -> dict:
                raise RuntimeError("secret upstream detail")

        client = await self.client(
            create_jev_app(JevSettings(api_key=TOKEN), BrokenJevBackend())
        )
        response = await client.post(
            "/v1/systemone",
            headers=self.auth(),
            json={
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            },
        )
        self.assertEqual(response.status, 502)
        self.assertNotIn("secret", await response.text())

    async def test_health_is_local_probe_only_and_models_requires_auth(self) -> None:
        image = await self.client(
            create_image_app(ImageSettings(api_key=TOKEN), FakeImageBackend())
        )
        self.assertEqual((await image.get("/healthz")).status, 200)
        self.assertEqual((await image.get("/v1/models")).status, 401)
        self.assertEqual(
            (await image.get("/v1/models", headers=self.auth())).status, 200
        )

        jev = await self.client(
            create_jev_app(JevSettings(api_key=TOKEN), FakeJevBackend())
        )
        self.assertEqual((await jev.get("/healthz")).status, 200)
        self.assertEqual((await jev.get("/v1/models", headers=self.auth())).status, 200)


if __name__ == "__main__":
    unittest.main()
