import types
import unittest
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from redqueen_adapters.clm_runtime import configure_embedder


class CLMRuntimeTest(unittest.TestCase):
    def test_native_encoder_has_no_read_deadline_and_keeps_health_bounded(self):
        class Embedder:
            def __init__(self, timeout=300.0):
                self.timeout = timeout
                self.session = Mock()

            def fetch(self):
                return self.session.post("http://encoder/v1/embeddings", timeout=self.timeout)

            def healthy(self):
                return self.session.get("http://encoder/v1/models", timeout=5)

        module = types.SimpleNamespace(Embedder=Embedder)
        configure_embedder(module)
        for kwargs in ({}, {"timeout": 0.001}):
            encoder = module.Embedder(**kwargs)
            encoder.fetch()
            encoder.session.post.assert_called_once_with("http://encoder/v1/embeddings", timeout=(10.0, None))
            encoder.healthy()
            encoder.session.get.assert_called_once_with("http://encoder/v1/models", timeout=5)
        self.assertTrue(issubclass(module.Embedder, Embedder))

    @unittest.skipUnless(importlib.util.find_spec("clm"), "run in the pinned CLM venv for native encoder coverage")
    def test_pinned_encoder_survives_former_shortened_read_cutoff(self):
        import clm.embedder
        import numpy as np

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                time.sleep(0.05)
                payload = json.dumps({"data": [{"index": 0, "embedding": [3.0, 4.0]}], "usage": {"prompt_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass  # The negative-control request deliberately closed.

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        original = clm.embedder.Embedder
        url = f"http://127.0.0.1:{server.server_port}/v1/embeddings"
        try:
            encoder = original(url, timeout=0.01)
            try:
                with self.assertRaises(clm.embedder.EmbedderError):
                    encoder._fetch(["hello"])
            finally:
                encoder.session.close()
            configure_embedder(clm.embedder)
            encoder = clm.embedder.Embedder(url, timeout=0.01)
            try:
                vectors, tokens = encoder._fetch(["hello"])
                np.testing.assert_allclose(vectors[0], [0.6, 0.8])
                self.assertEqual(tokens, 1)
            finally:
                encoder.session.close()
        finally:
            clm.embedder.Embedder = original
            server.shutdown()
            server.server_close()
            worker.join()
