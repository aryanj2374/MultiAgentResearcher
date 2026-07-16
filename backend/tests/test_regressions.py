from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.critic import critique_all
from backend.agents.extractor import _detect_effect_direction, extract_all
from backend.agents.planner import _heuristic_plan
from backend.agents.referee import verify_synthesis
from backend.agents.retriever import retrieve_papers
from backend.agents.synthesizer import _fallback_synthesis
from backend.main import generate_sse_events
from backend.orchestrator import _merge_sub_results, run_question_with_progress
from backend.schemas import (
    Critique,
    Paper,
    ResearchPlan,
    StudyExtraction,
    SubQuestionResult,
    Verification,
    AskRequest,
)
from backend.semantic_scholar import SemanticScholarError
from backend.storage import load_run
from backend.utils import build_citation_map, safe_json_loads


def make_paper(paper_id: str, *, title: str = "Study") -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Alex Smith"],
        year=2024,
        abstract="A randomized trial found improved outcomes in 120 participants.",
    )


def make_extraction(paper_id: str, direction: str = "positive") -> StudyExtraction:
    return StudyExtraction(
        paper_id=paper_id,
        claim_summary="The intervention improved the measured outcome.",
        study_type="RCT",
        population="adults",
        sample_size=120,
        effect_direction=direction,
        key_snippet="The intervention improved outcomes.",
        limitations=["Abstract-only extraction."],
        apa_citation="Smith et al. (2024). Study.",
    )


def make_critique(paper_id: str) -> Critique:
    return Critique(
        paper_id=paper_id,
        risk_of_bias="medium",
        rationale=["Abstract-only assessment."],
        red_flags=[],
    )


class FakeLLM:
    available = True

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    async def chat(self, *args, **kwargs) -> str:
        return next(self._responses)


class UtilityTests(unittest.TestCase):
    def test_json_fragment_ignores_brackets_inside_strings(self) -> None:
        payload = {"message": "a closing brace } and bracket ]", "items": [1, 2]}
        wrapped = f"Here is JSON:\n{json.dumps(payload)}\nThanks"
        self.assertEqual(safe_json_loads(wrapped), payload)

    def test_duplicate_author_year_labels_are_unique(self) -> None:
        mapping = build_citation_map([make_paper("p1"), make_paper("p2")])
        self.assertEqual(mapping, {"p1": "Smith2024a", "p2": "Smith2024b"})

    def test_null_language_is_not_classified_as_positive(self) -> None:
        self.assertEqual(
            _detect_effect_direction("There was no significant improvement in outcomes."),
            "null",
        )
        self.assertEqual(
            _detect_effect_direction(
                "There was no significant improvement in the primary outcome. "
                "A secondary outcome improved."
            ),
            "mixed",
        )

    def test_storage_rejects_non_uuid_paths(self) -> None:
        self.assertIsNone(load_run("../../secrets"))

    def test_question_validation_strips_before_checking_length(self) -> None:
        self.assertEqual(AskRequest(question="  valid question  ").question, "valid question")
        with self.assertRaises(ValueError):
            AskRequest(question="   ")

    def test_heuristic_planner_does_not_split_noun_phrases(self) -> None:
        direct = _heuristic_plan("Do diet and exercise improve health?")
        decomposed = _heuristic_plan(
            "Does exercise improve mood and does meditation improve sleep?"
        )
        self.assertEqual(direct.strategy, "direct")
        self.assertEqual(decomposed.strategy, "decompose")
        self.assertEqual(len(decomposed.sub_questions), 2)


class AgentIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_keeps_pipeline_paper_identity(self) -> None:
        paper = make_paper("actual")
        response = make_extraction("hallucinated").model_dump_json()
        extraction = (await extract_all([paper], FakeLLM([response])))[0]
        self.assertEqual(extraction.paper_id, "actual")

    async def test_critic_keeps_extraction_identity(self) -> None:
        extraction = make_extraction("actual")
        response = make_critique("hallucinated").model_dump_json()
        critique = (await critique_all([make_paper("actual")], [extraction], FakeLLM([response])))[0]
        self.assertEqual(critique.paper_id, "actual")

    async def test_extractions_run_concurrently_and_keep_order(self) -> None:
        class SlowLLM:
            available = True

            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0

            async def chat(self, *args, **kwargs) -> str:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0.01)
                self.active -= 1
                return make_extraction("generated").model_dump_json()

        llm = SlowLLM()
        papers = [make_paper(f"p{index}") for index in range(3)]
        extractions = await extract_all(papers, llm)

        self.assertEqual(llm.max_active, 3)
        self.assertEqual(
            [extraction.paper_id for extraction in extractions],
            [paper.paper_id for paper in papers],
        )


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_success_is_not_retried(self) -> None:
        client = AsyncMock()
        client.search_papers.return_value = []
        with patch("backend.agents.retriever._CLIENT", client):
            _, metadata = await retrieve_papers("plain terms", max_retries=3)

        tried = metadata["queries_tried"]
        self.assertEqual(len(tried), len(set(tried)))
        self.assertEqual(client.search_papers.await_count, len(set(tried)))

    async def test_transient_search_errors_are_retried(self) -> None:
        client = AsyncMock()
        client.search_papers.side_effect = [
            SemanticScholarError("temporary"),
            SemanticScholarError("temporary"),
            [make_paper("p1")],
        ]
        with (
            patch("backend.agents.retriever._CLIENT", client),
            patch("backend.agents.retriever.asyncio.sleep", new=AsyncMock()),
        ):
            papers, metadata = await retrieve_papers("plain terms", max_retries=3)

        self.assertEqual([paper.paper_id for paper in papers], ["p1"])
        self.assertEqual(metadata["total_attempts"], 3)


class SynthesisTests(unittest.TestCase):
    def test_fallback_synthesis_passes_its_referee(self) -> None:
        papers = [make_paper("p1"), make_paper("p2")]
        extractions = [make_extraction("p1"), make_extraction("p2", "null")]
        critiques = [make_critique("p1"), make_critique("p2")]

        synthesis = _fallback_synthesis("Does it work?", papers, extractions, critiques)
        verification = verify_synthesis(synthesis, papers, critiques)

        self.assertTrue(verification.passed, verification.issues)
        self.assertIn("Smith2024a", " ".join(synthesis.final_answer))
        self.assertIn("Smith2024b", " ".join(synthesis.final_answer))

    def test_deep_results_do_not_double_count_papers(self) -> None:
        shared = make_paper("shared")
        result_a = SubQuestionResult(
            sub_question="a",
            papers=[shared],
            extractions=[make_extraction("shared")],
            critiques=[make_critique("shared")],
        )
        result_b = SubQuestionResult(
            sub_question="b",
            papers=[shared],
            extractions=[make_extraction("shared", "null")],
            critiques=[make_critique("shared")],
        )

        papers, extractions, critiques = _merge_sub_results([result_a, result_b])
        self.assertEqual((len(papers), len(extractions), len(critiques)), (1, 1, 1))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_unhandled_stream_errors_become_error_events(self) -> None:
        async def broken_stream(question: str):
            if False:
                yield {}
            raise RuntimeError("internal details")

        with patch("backend.main.run_question_with_progress", broken_stream):
            events = [event async for event in generate_sse_events("question")]

        self.assertEqual(len(events), 1)
        self.assertIn('"type": "error"', events[0])
        self.assertNotIn("internal details", events[0])

    async def test_deep_stream_branches_overlap(self) -> None:
        active = 0
        max_active = 0

        async def process(question, llm, logs, log_prefix=""):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [], [], [], logs

        plan = ResearchPlan(
            is_complex=True,
            original_question="q",
            sub_questions=["a", "b"],
            strategy="decompose",
            reasoning="test",
        )
        empty_synthesis = _fallback_synthesis("q", [], [], [])

        with (
            patch("backend.orchestrator.ChatLLM", return_value=Mock(available=False)),
            patch("backend.orchestrator.plan_research", new=AsyncMock(return_value=plan)),
            patch("backend.orchestrator._process_single_question", new=process),
            patch("backend.orchestrator.synthesize", new=AsyncMock(return_value=empty_synthesis)),
            patch(
                "backend.orchestrator.verify_synthesis",
                return_value=Verification(passed=True, issues=[]),
            ),
            patch("backend.orchestrator.save_run"),
        ):
            events = [event async for event in run_question_with_progress("q")]

        self.assertEqual(max_active, 2)
        self.assertEqual(events[-1]["type"], "result")


if __name__ == "__main__":
    unittest.main()
