import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ResearchCitationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_docstore = main.DOCSTORE
        main.DOCSTORE = Path(self.temp_dir.name) / "documents.json"
        main.write_json(main.DOCSTORE, [])
        self.client = TestClient(main.app)

    def tearDown(self):
        main.DOCSTORE = self.original_docstore
        self.temp_dir.cleanup()

    def test_catalogue_answers_link_sources_without_claiming_page_citations(self):
        questions = {
            "What does BASS-II study?": "bass-pmma-2015",
            "What did SAFFIRE investigate?": "saffire-iv-v-2021",
            "What does FLEX research cover?": "flex-program-overview",
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            for question, document_id in questions.items():
                with self.subTest(question=question):
                    response = self.client.post("/api/chat", json={"question": question})
                    self.assertEqual(response.status_code, 200)
                    result = response.json()
                    self.assertEqual(result["mode"], "catalogue-only")
                    self.assertIn("No page-cited scientific answer", result["answer"])
                    source = next(item for item in result["sources"] if item["document_id"] == document_id)
                    self.assertIsNone(source["page"])
                    self.assertTrue(source["url"].startswith("https://ntrs.nasa.gov/citations/"))

    def test_retrieval_cites_the_exact_uploaded_page(self):
        main.write_json(main.DOCSTORE, [{
            "id": "test-pdf",
            "title": "BASS-II source test.pdf",
            "program": "BASS-II",
            "summary": "Uploaded test document",
            "source": "User-provided document",
            "pages": 2,
            "text": "BASS-II oxygen flame spread page 2 polymer samples",
            "page_records": [
                {"page": 1, "text": "Introduction to the BASS-II study."},
                {"page": 2, "text": "BASS-II oxygen flame spread tests included polymer samples."},
            ],
            "tags": [],
        }])

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/api/chat",
                json={"question": "What did BASS-II test about oxygen flame spread?"},
            )

        result = response.json()
        self.assertEqual(result["mode"], "retrieval-only")
        self.assertIn("[BASS-II source test.pdf, p. 2]", result["answer"])
        self.assertEqual(result["sources"][0]["page"], 2)

    def test_gemini_claims_must_cite_a_retrieved_document_page(self):
        main.write_json(main.DOCSTORE, [{
            "id": "test-pdf",
            "title": "BASS-II source test.pdf",
            "program": "BASS-II",
            "summary": "Uploaded test document",
            "source": "User-provided document",
            "pages": 2,
            "text": "BASS-II oxygen flame spread page 2 polymer samples",
            "page_records": [
                {"page": 1, "text": "Introduction to the BASS-II study."},
                {"page": 2, "text": "BASS-II oxygen flame spread tests included polymer samples."},
            ],
            "tags": [],
        }])
        generated = {
            "source_findings": [{
                "text": "The tests included polymer samples.",
                "citations": [{"document_id": "test-pdf", "page": 2}],
            }],
            "interpretations": [{
                "text": "Oxygen may be relevant to flame spread.",
                "citations": [{"document_id": "test-pdf", "page": 99}],
            }],
        }
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(
                    text=json.dumps(generated) if "contents" in kwargs else ""
                )
            )
        )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-key"}):
            with patch("google.genai.Client", return_value=client):
                result = self.client.post(
                    "/api/chat",
                    json={"question": "What did BASS-II test about oxygen flame spread?"},
                ).json()

        self.assertEqual(result["mode"], "retrieval-fallback")
        self.assertIn("retrieved page excerpts", result["answer"])
        self.assertNotIn("Oxygen may be relevant", result["answer"])

        generated["interpretations"][0]["citations"][0]["page"] = 2
        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-key"}):
            with patch("google.genai.Client", return_value=client):
                result = self.client.post(
                    "/api/chat",
                    json={"question": "What did BASS-II test about oxygen flame spread?"},
                ).json()

        self.assertEqual(result["mode"], "gemini-rag")
        self.assertIn("NASA source findings (AI summary)", result["answer"])
        self.assertIn("AI-generated interpretation (not a NASA finding)", result["answer"])
        self.assertIn("BASS-II source test.pdf, p. 2", result["answer"])

    def test_empty_and_malformed_gemini_responses_are_reported_without_synthesis(self):
        main.write_json(main.DOCSTORE, [{
            "id": "test-pdf",
            "title": "BASS-II source test.pdf",
            "program": "BASS-II",
            "summary": "Uploaded test document",
            "source": "NASA Technical Reports Server (NTRS)",
            "url": "https://ntrs.nasa.gov/citations/20150008961",
            "pages": 1,
            "text": "BASS-II PMMA flame spread",
            "page_records": [
                {"page": 1, "text": "BASS-II PMMA flame spread experiments."},
            ],
            "tags": [],
        }])
        responses = [
            (
                SimpleNamespace(
                    text="",
                    candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
                ),
                "Gemini returned no text (finish reason: MAX_TOKENS).",
            ),
            (
                SimpleNamespace(
                    text="{malformed",
                    candidates=[SimpleNamespace(finish_reason="STOP")],
                ),
                "Gemini returned malformed JSON.",
            ),
        ]

        for generated_response, expected_error in responses:
            client = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **kwargs: generated_response
                )
            )
            with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-key"}):
                with patch("google.genai.Client", return_value=client):
                    result = self.client.post(
                        "/api/chat",
                        json={"question": "What did BASS-II study about PMMA flame spread?"},
                    ).json()

            self.assertEqual(result["mode"], "retrieval-fallback")
            self.assertIn(expected_error, result["answer"])
            self.assertIn("[BASS-II source test.pdf, p. 1]", result["answer"])
            self.assertIn("not an AI-generated synthesis", result["answer"])
            self.assertEqual(result["sources"][0]["url"], "https://ntrs.nasa.gov/citations/20150008961")


if __name__ == "__main__":
    unittest.main()
