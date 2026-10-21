#!/usr/bin/env python3
"""Generate the deterministic Redqueen workload-regression suite v1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path


SUITE_ID = "redqueen-workload-v2"
MODEL_ALIASES = (
    "qwen3.8-27b",
    "gemma-4-e4b-it",
    "gemma-4-12b-it",
    "gemma-4-26b-a4b-it",
    "qwen3.6-35b-a3b",
    "deepseek-v4.1-flash-1017-unsloth",
    "glm-5.3-flash-abliterated",
    "embeddinggemma-2",
    "clef-flash-bf16",
    "clef-flash-q8",
    "clef-flash-q4",
    "clef-q4",
)
OPENAI_SERVICE_MODELS = ("deepseek",)
TEXT_MODELS = tuple(model for model in MODEL_ALIASES if model != "embeddinggemma-2") + OPENAI_SERVICE_MODELS
MULTIMODAL_MODELS = (
    "qwen3.6-35b-a3b",
    "deepseek-v4.1-flash-1017-unsloth",
    "glm-5.3-flash-abliterated",
    "deepseek",
)
EMBEDDING_MODEL = "embeddinggemma-2"
DECISION_MODEL = "jevk5-4b-v0.3"
CLM_MODEL = "clm-v0.1-8b"
CLEF_MODELS = ("clef-flash-bf16", "clef-flash-q8", "clef-flash-q4", "clef-q4")
IMAGE_MODELS = ("qwen-image-2.1", "qwen-image-2.1-uncensored")
SOURCE_IMAGE_NAME = "edit-source.png"

RETRIEVAL_CASES = [
    ("How do I reset my account password?", "A password reset link is sent to the email address on the account.", ["Cancel a subscription from billing settings.", "Track a parcel with its shipment number.", "Download a receipt from the invoices page."]),
    ("Where can I see the next invoice date?", "The billing page lists the next invoice date and plan renewal.", ["Change the password in account security.", "The courier page shows the estimated delivery date.", "Refunds are returned to the original payment method."]),
    ("My package has not arrived yet.", "Use the tracking number to check the shipment status and expected delivery.", ["A failed charge appears in billing history.", "Update a forgotten password from the sign-in screen.", "Return an item within the stated return period."]),
    ("I need to send back a jacket that does not fit.", "Start a return from the order page and select the clothing item.", ["Reset account credentials in security settings.", "Invoices are available in the billing portal.", "Contact the carrier about a delayed parcel."]),
    ("How can I turn off automatic renewal?", "Disable auto-renewal in subscription settings before the next billing date.", ["Change the account password using the reset email.", "Use the shipment page to find a delivery estimate.", "Request a replacement for a damaged product."]),
    ("The app closes when I open the reports screen.", "Collect the app version and crash details, then send them to technical support.", ["Download invoices from billing history.", "Return the item from the order page.", "Check the delivery status with the carrier."]),
    ("I was billed twice for the same order.", "Report the duplicate transaction to billing support with both charge identifiers.", ["Use account settings to reset a password.", "Ask the courier to locate the shipment.", "Request a return label for an unwanted item."]),
    ("The delivery address on my order is wrong.", "Contact shipping support quickly to check whether the parcel address can still be changed.", ["Open billing history to dispute a duplicate charge.", "Update the password from account security.", "Turn off renewal in subscription settings."]),
    ("The export button does nothing in the dashboard.", "Send the browser version and dashboard steps to technical support.", ["The receipt can be downloaded from invoices.", "A parcel can be tracked using its tracking number.", "Returns are started from the order page."]),
    ("Can I get a copy of my payment receipt?", "Download the payment receipt from the invoice history page.", ["Ask technical support to investigate an app crash.", "Use the carrier tracking page for a parcel.", "Choose a return reason on the order page."]),
    ("The tracking page says delivery was attempted, but I was home.", "Ask shipping support or the carrier to investigate the delivery attempt.", ["Billing can provide an invoice copy.", "Change the password from account security.", "Disable recurring billing in subscription settings."]),
    ("I want to change the email used to sign in.", "Update the sign-in email in account profile settings.", ["Dispute the charge with billing support.", "Ask shipping support to update a parcel address.", "Return a product through order history."]),
    ("A feature stopped working after the latest update.", "Report the affected feature and app version to technical support.", ["See the next charge date on the billing page.", "Use the order page to create a return.", "Check a delivery with the carrier."]),
    ("My refund has not appeared on the card yet.", "Check the refund status with billing support; card refunds can take several business days.", ["Use account settings to change a sign-in email.", "Ask technical support for help with the app.", "Use the tracking number to find a parcel."]),
    ("The parcel is marked delivered but is missing.", "Ask shipping support to investigate the delivery scan and parcel location.", ["Find a receipt in invoice history.", "Reset the password with the account email.", "Disable renewal before the next billing date."]),
    ("I need to replace an item that arrived cracked.", "Contact returns support and provide the order details and damage description.", ["Ask billing for an invoice copy.", "Update sign-in email in profile settings.", "Send the browser and app version to technical support."]),
    ("I cannot sign in because my password is forgotten.", "Use the password reset flow and follow the link sent to the account email.", ["Ask billing why the invoice is late.", "Track the parcel with the courier.", "Return the item from order history."]),
    ("The monthly plan renewed and I did not want another month.", "Contact billing support about the renewal charge and cancellation options.", ["Report the dashboard error to technical support.", "Ask shipping to locate the missing parcel.", "Request a replacement for damaged goods."]),
    ("I entered the wrong apartment number for delivery.", "Contact shipping support to ask whether the parcel address can be corrected.", ["Change the password through account security.", "Download the receipt from billing history.", "Cancel the subscription in plan settings."]),
    ("The CSV download contains garbled characters.", "Send a sample export and browser details to technical support.", ["Use the invoice page to view the next billing date.", "Start a return from order history.", "Check parcel status with its tracking number."]),
    ("Where do I find the charge for last month?", "Open billing history and select last month's invoice.", ["Use the account profile to change the sign-in email.", "Contact the courier about a delivery scan.", "Ask technical support to investigate a crash."]),
    ("The carrier has had my parcel at the depot for a week.", "Ask shipping support to contact the carrier about the delayed parcel.", ["Request an invoice copy from billing.", "Reset account credentials from the sign-in screen.", "Disable the subscription's automatic renewal."]),
    ("I was charged after cancelling my subscription.", "Give billing support the cancellation confirmation and charge date.", ["Ask technical support to repair the export.", "Request a return label for the jacket.", "Update the parcel's apartment number."]),
    ("I want to return the wrong color that was delivered.", "Start a return for the order and select the incorrect-color reason.", ["Ask billing to explain a renewal charge.", "Use the password reset link to sign in.", "Contact the carrier about a delayed shipment."]),
]

IMAGE_PROMPTS = [
    "A studio photograph of one red ceramic teapot centered on a pale yellow table, soft daylight.",
    "Three ripe yellow lemons in a blue ceramic bowl on a plain white counter, overhead view.",
    "A small green bicycle leaning against a brick wall on a rainy city street, realistic photograph.",
    "A white sailboat with a single red sail on a calm turquoise lake, distant mountains, clear morning.",
    "A close-up photograph of a single purple orchid in a small glass vase against a neutral background.",
    "A wooden chessboard with exactly two pieces, one black knight and one white pawn, viewed from above.",
    "A friendly orange cat sitting beside a blue suitcase in a sunlit train station.",
    "A glass of iced tea with a lemon slice on a round wooden table beside an open window.",
    "A tiny red cabin among tall evergreen trees under a clear starry night sky, wide landscape.",
    "A silver spoon holding one green olive above a white plate, clean product photography.",
    "A yellow school bus crossing a small bridge over a stream in an autumn forest.",
    "A ceramic blue bowl filled with strawberries on a pink tablecloth, bright studio lighting.",
]

EDIT_PROMPTS = [
    "Change the blue square to a green circle and keep the white background.",
    "Replace the blue square with a red triangle, keeping the background plain white.",
    "Turn the blue square into a yellow star and preserve the white background.",
    "Make the square green, leaving its shape and the white background unchanged.",
    "Change the square to orange and keep it centered on the white background.",
    "Replace the blue shape with a purple heart; leave the background white.",
    "Make the square smaller and red while keeping it in the center of the white canvas.",
    "Change the blue square to a black circle and keep the background white.",
]

DECISION_EXPECTED = [
    "technical", "billing", "shipping", "returns", "billing", "technical",
    "billing", "shipping", "technical", "billing", "shipping", "technical",
    "technical", "billing", "shipping", "returns", "technical", "billing",
    "shipping", "technical", "billing", "shipping", "billing", "returns",
]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def source_png() -> bytes:
    width = height = 128
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            if 40 <= x < 88 and 40 <= y < 88:
                row.extend((32, 96, 224))
            else:
                row.extend((255, 255, 255))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )


def make_text_cases(model: str) -> list[dict]:
    cases = []
    operations = (
        ("add", 17, 26, 43),
        ("subtract", 40, 17, 23),
        ("multiply", 11, 8, 88),
        ("divide", 144, 12, 12),
        ("add", 258, 167, 425),
        ("subtract", 903, 478, 425),
        ("multiply", 13, 17, 221),
        ("divide", 154, 11, 14),
    )
    symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
    routes = (
        (("chat", "/v1/chat/completions"),)
        if model in OPENAI_SERVICE_MODELS
        else (
            ("chat", "/v1/chat/completions"),
            ("completions", "/v1/completions"),
            ("responses", "/v1/responses"),
        )
    )
    for route_name, route in routes:
        for index, (operation, left, right, answer) in enumerate(operations, 1):
            prompt = f"Calculate {left} {symbols[operation]} {right}. Reply with the integer only: no explanation, punctuation, or markdown."
            if route == "/v1/chat/completions":
                request = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            elif route == "/v1/completions":
                request = {
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": 16,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            else:
                request = {
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 16,
                    "stream": True,
                }
            cases.append({
                "schema_version": 1,
                "suite_id": SUITE_ID,
                "id": f"text-{model}-{route_name}-{index:02d}",
                "workload": "text",
                "path": route,
                "request": request,
                "scoring": {"kind": "exact", "pointer": "/text_output", "expected": str(answer)},
            })
    return cases


def make_prefill_cases(model: str) -> list[dict]:
    topics = (
        "renewable power forecasts and battery dispatch",
        "coastal weather observations and vessel schedules",
        "public library inventory and inter-branch transfers",
        "laboratory calibration records and instrument maintenance",
        "municipal water sampling and treatment operations",
        "rail freight manifests and depot maintenance windows",
        "agricultural soil surveys and irrigation planning",
        "regional clinic appointment and equipment schedules",
    )
    cases = []
    for index, expected in enumerate((
        "K7M4Q2", "R9C3W8", "P5D2N6", "T4B8H1",
        "M3J7V2", "C8W5F9", "H2N6D4", "B9P1G7",
    ), 1):
        target_record = (index * 19) % 160
        records = []
        for row in range(160):
            topic = topics[(row * 5 + index) % len(topics)]
            if row == target_record:
                records.append(
                    f"archive record {row:03d}: access code {expected}; topic {topic}; "
                    "the access code is the authoritative value for this record."
                )
            else:
                records.append(
                    f"archive record {row:03d}: topic {topic}; status reviewed; "
                    f"reference {((row * 29 + index * 13) % 997):03d}; follow-up next cycle."
                )
        prompt = (
            "Read the archive carefully. Each numbered record is independent. "
            f"\n\n{chr(10).join(records)}\n\n"
            f"What is the access code in archive record {target_record:03d}? "
            "Reply with the code only."
        )
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"prefill-{model}-{index:02d}",
            "workload": "prefill",
            "path": "/v1/chat/completions",
            "request": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "scoring": {"kind": "exact", "pointer": "/text_output", "expected": expected},
        })
    return cases


def make_embedding_cases() -> list[dict]:
    cases = []
    for index, (query, relevant, distractors) in enumerate(RETRIEVAL_CASES, 1):
        position = (index * 3) % 4
        documents = list(distractors)
        documents.insert(position, relevant)
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"embedding-search-{index:02d}",
            "workload": "embeddings",
            "path": "/v1/embeddings",
            "request": {
                "model": EMBEDDING_MODEL,
                "input": [f"task: search result | query: {query}"] + [f"title: none | text: {document}" for document in documents],
                "encoding_format": "float",
            },
            "scoring": {"kind": "retrieval", "relevant": [position], "metric": "mrr"},
        })
    return cases


def make_decision_cases(model: str, workload: str, prefix: str, *, limit: int | None = None) -> list[dict]:
    options = {
        "billing": "charges, invoices, subscriptions, or refunds",
        "technical": "bugs, crashes, account access, or product operation",
        "shipping": "delivery, tracking, address changes, or missing parcels",
        "returns": "returning, replacing, or exchanging a delivered product",
    }
    cases = []
    input_cases = list(zip(RETRIEVAL_CASES, DECISION_EXPECTED, strict=True))
    if limit is not None:
        input_cases = input_cases[:limit]
    for index, ((ticket, _, _), expected) in enumerate(input_cases, 1):
        criteria = {name: options[name] for name in options}
        rotation = index % len(criteria)
        ordered = list(criteria.items())
        criteria = dict(ordered[rotation:] + ordered[:rotation])
        if index % 2 == 0:
            route = "/v1/systemone"
            request = {
                "model": model,
                "state": {"ticket": ticket},
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Choose the single best support queue for the customer's request.",
                        "criteria": criteria,
                    }
                },
            }
            pointer = "/answers/route/choice"
        else:
            route = "/v1/decisions"
            request = {
                "model": model,
                "input": ticket,
                "questions": [{
                    "type": "choice",
                    "name": "route",
                    "instructions": "Choose the single best support queue for the customer's request.",
                    "choices": [{"value": name, "description": description} for name, description in criteria.items()],
                }],
            }
            pointer = "/answers/0/choice"
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"{prefix}-route-{index:02d}",
            "workload": workload,
            "path": route,
            "request": request,
            "scoring": {"kind": "exact", "pointer": pointer, "expected": expected},
        })
    return cases


def make_multimodal_cases(data_url: str, model: str) -> list[dict]:
    questions = [
        ("What color is the square? Reply with exactly one lowercase word.", "blue"),
        ("What shape is the colored object? Reply with exactly one lowercase word.", "square"),
        ("What color is the background? Reply with exactly one lowercase word.", "white"),
        ("Is the square on the left, center, or right? Reply with one lowercase word.", "center"),
        ("How many blue squares are visible? Reply with an integer only.", "1"),
        ("Is there a red object? Reply with only yes or no.", "no"),
        ("Is most of the image background white? Reply with only yes or no.", "yes"),
        ("Name the color of the central shape. Reply with one lowercase word.", "blue"),
    ]
    cases = []
    for index, (prompt, expected) in enumerate(questions, 1):
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"multimodal-{model}-{index:02d}",
            "workload": "multimodal",
            "path": "/v1/chat/completions",
            "request": {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "scoring": {"kind": "exact", "pointer": "/text_output", "expected": expected},
        })
    return cases


def make_image_cases(source_sha256: str, model: str) -> list[dict]:
    cases = []
    for index, prompt in enumerate(IMAGE_PROMPTS, 1):
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"image-generation-{model}-{index:02d}",
            "workload": "images",
            "path": "/v1/images/generations",
            "request": {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": "512x512",
                "quality": "medium",
                "background": "opaque",
                "response_format": "b64_json",
                "seed": 61000 + index,
            },
        })
    for index, prompt in enumerate(EDIT_PROMPTS, 1):
        cases.append({
            "schema_version": 1,
            "suite_id": SUITE_ID,
            "id": f"image-edit-{model}-{index:02d}",
            "workload": "images",
            "path": "/v1/images/edits",
            "request": {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": "512x512",
                "quality": "medium",
                "background": "opaque",
                "response_format": "b64_json",
                "seed": 62000 + index,
            },
            "files": [{
                "field": "image[]",
                "asset": SOURCE_IMAGE_NAME,
                "filename": "source.png",
                "content_type": "image/png",
                "sha256": source_sha256,
            }],
        })
    return cases


def write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def make_suite(output_dir: Path) -> tuple[Path, str, int]:
    repository = Path(__file__).resolve().parents[4]
    destination = output_dir.resolve()
    if destination.is_relative_to(repository):
        raise ValueError("generated suite and assets must stay outside the repository")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(destination, 0o700)
    image = source_png()
    image_hash = hashlib.sha256(image).hexdigest()
    data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    cases = (
        [case for model in TEXT_MODELS for case in make_text_cases(model)]
        + [case for model in TEXT_MODELS for case in make_prefill_cases(model)]
        + make_embedding_cases()
        + make_decision_cases(DECISION_MODEL, "decisions", "decision")
        + [
            case
            for model in CLEF_MODELS
            for case in make_decision_cases(model, "decisions", f"decision-{model}", limit=16)
        ]
        + make_decision_cases(CLM_MODEL, "clm", "clm")
        + [case for model in MULTIMODAL_MODELS for case in make_multimodal_cases(data_url, model)]
        + [case for model in IMAGE_MODELS for case in make_image_cases(image_hash, model)]
    )
    suite_path = destination / f"{SUITE_ID}.jsonl"
    suite_bytes = ("".join(json.dumps(case, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for case in cases)).encode("utf-8")
    write_private(destination / SOURCE_IMAGE_NAME, image)
    write_private(suite_path, suite_bytes)
    return suite_path, hashlib.sha256(suite_bytes).hexdigest(), len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new private directory outside the repository")
    args = parser.parse_args()
    try:
        suite_path, digest, count = make_suite(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"cannot create workload suite: {error}", file=sys.stderr)
        return 2
    print(f"suite_id={SUITE_ID} cases={count} sha256={digest} path={suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
