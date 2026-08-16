from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.critic import critique_all
from backend.agents.extractor import (
    _detect_effect_direction,
    _extract_key_findings_from_abstract,
    extract_all,
)
from backend.agents.planner import _heuristic_plan
from backend.agents.referee import verify_synthesis
from backend.agents.retriever import filter_relevant, retrieve_papers
from backend.agents.synthesizer import (
    _describe_direction,
    _fallback_synthesis,
    _no_evidence_synthesis,
    _normalize_synthesis_payload,
    _reconcile_citations,
    synthesize,
)
from backend.main import generate_sse_events
from backend.orchestrator import _merge_sub_results, run_question_with_progress
from backend.schemas import (
    Critique,
    Paper,
    ResearchPlan,
    StudyExtraction,
    SubQuestionResult,
    Synthesis,
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


class ExtractionQualityTests(unittest.TestCase):
    def test_findings_are_preferred_over_background_framing(self) -> None:
        abstract = (
            "Background: Creatine supplementation is well-established for enhancing "
            "physical performance. However, its effects on cognitive function have "
            "not been thoroughly investigated. Methods: Fourteen men ingested 20 g/day "
            "of creatine or placebo for 7 days. Results: Creatine resulted in "
            "significantly lower muscle soreness scores (d = -0.59; p = 0.046) and "
            "improved cognitive performance."
        )
        summary = _extract_key_findings_from_abstract(abstract)

        self.assertIsNotNone(summary)
        self.assertIn("significantly lower muscle soreness", summary)
        self.assertNotIn("have not been thoroughly investigated", summary)

    def test_objective_sentence_is_not_reported_as_a_finding(self) -> None:
        abstract = (
            "Objective: This study aimed to evaluate the effects of cold-water "
            "immersion on post-training recovery in athletes. Results: Immersion "
            "produced a greater reduction in creatine kinase than control "
            "24 h after the intervention (-21.32%; p < 0.001)."
        )
        summary = _extract_key_findings_from_abstract(abstract)

        self.assertIn("greater reduction in creatine kinase", summary)
        self.assertNotIn("aimed to evaluate", summary)

    def test_study_protocols_do_not_yield_future_tense_findings(self) -> None:
        # Registered protocols describe planned work; nothing there is a result.
        abstract = (
            "Background Resistance training improves strength in older adults. "
            "Methods Thirty-six older adults will be randomly assigned to one of "
            "three groups. Results We hypothesize that training plus creatine will "
            "result in greater improvements in body composition."
        )
        summary = _extract_key_findings_from_abstract(abstract)

        self.assertIsNotNone(summary)
        self.assertNotIn("will be randomly assigned", summary)
        self.assertNotIn("will result in greater improvements", summary)

    def test_unstructured_abstract_still_returns_a_result_sentence(self) -> None:
        abstract = (
            "Creatine is an organic compound found in muscle tissue and is widely "
            "used by athletes across many sporting disciplines worldwide today. "
            "We found that supplementation increased lean mass by 1.2 kg (p = 0.01) "
            "relative to placebo across the full trial period."
        )
        summary = _extract_key_findings_from_abstract(abstract)

        self.assertIn("increased lean mass", summary)


class RelevanceFilterTests(unittest.TestCase):
    def _paper(self, paper_id: str, title: str, abstract: str = "") -> Paper:
        return Paper(
            paper_id=paper_id,
            title=title,
            authors=["Alex Smith"],
            year=2024,
            abstract=abstract or title,
        )

    def test_papers_missing_the_subject_are_dropped(self) -> None:
        question = "What are the effects of creatine on cognition and physicals?"
        on_topic = self._paper(
            "keep", "Effects of creatine supplementation on cognitive performance"
        )
        off_topic = self._paper(
            "drop", "Relationships between cognition, function, and quality of life in HIV+ men"
        )

        kept, dropped = filter_relevant([on_topic, off_topic], question)

        self.assertEqual([p.paper_id for p in kept], ["keep"])
        self.assertEqual([p.paper_id for p in dropped], ["drop"])

    def test_subject_papers_survive_without_every_outcome_term(self) -> None:
        # A creatine safety paper is relevant even if it never says "cognition".
        question = "What are the effects of creatine on cognition and physicals?"
        safety = self._paper("safety", "Adverse effects of creatine supplementation")

        kept, _ = filter_relevant([safety], question)

        self.assertEqual([p.paper_id for p in kept], ["safety"])

    def test_filter_never_returns_an_empty_set(self) -> None:
        question = "What are the effects of creatine on cognition?"
        unrelated = [self._paper("a", "Allergic rhinitis in schoolchildren")]

        kept, _ = filter_relevant(unrelated, question)

        self.assertEqual(len(kept), 1)


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

    def test_object_shaped_bullets_are_not_discarded(self) -> None:
        # The model often returns {"text": ..., "citation": ...} objects. Those
        # used to fail schema validation and silently drop a good synthesis.
        papers = [make_paper("p1")]
        payload = {
            "final_answer": [
                {"text": "Creatine improved strength.", "citation": "Smith2024"},
                {"text": "No effect on memory [Smith2024]."},
            ],
            "evidence_consensus": ["Studies broadly agree.", "Effect sizes are small."],
            "top_limitations_overall": "Abstract-only synthesis.",
            "confidence_score": "72%",
            "confidence_rationale": ["Consistent direction."],
            "citations_used": ["Smith2024"],
        }
        normalized = _normalize_synthesis_payload(
            payload, papers, build_citation_map(papers)
        )
        synthesis = Synthesis.model_validate(normalized)

        self.assertEqual(
            synthesis.final_answer,
            ["Creatine improved strength. [Smith2024]", "No effect on memory [Smith2024]."],
        )
        self.assertEqual(synthesis.confidence_score, 72)
        # Display labels are resolved back to pipeline paper ids.
        self.assertEqual(synthesis.citations_used, ["p1"])
        self.assertEqual(
            synthesis.evidence_consensus, "Studies broadly agree. Effect sizes are small."
        )

    def test_missing_consensus_citation_is_backfilled(self) -> None:
        papers = [make_paper("p1")]
        synthesis = Synthesis(
            final_answer=["Creatine improved strength [Smith2024]."],
            evidence_consensus="Studies broadly agree.",
            top_limitations_overall=["Abstract-only."],
            confidence_score=60,
            confidence_rationale=["Consistent."],
            citations_used=[],
        )
        reconciled = _reconcile_citations(synthesis, papers, build_citation_map(papers))

        self.assertIn("[Smith2024]", reconciled.evidence_consensus)
        self.assertEqual(reconciled.citations_used, ["p1"])
        self.assertTrue(verify_synthesis(reconciled, papers, []).passed)

    def test_fallback_bullets_quote_findings_not_counts(self) -> None:
        papers = [make_paper("p1"), make_paper("p2")]
        extractions = [make_extraction("p1"), make_extraction("p2", "null")]
        critiques = [make_critique("p1"), make_critique("p2")]

        synthesis = _fallback_synthesis("Does it work?", papers, extractions, critiques)
        themed = [b for b in synthesis.final_answer if b.startswith("**")]

        self.assertTrue(themed, synthesis.final_answer)
        self.assertIn("The intervention improved the measured outcome.", " ".join(themed))

    def test_minority_positive_is_not_reported_as_consensus(self) -> None:
        # 2 positive out of 6 must never read as "most studies report benefits".
        self.assertIn("2 of 6", _describe_direction(2, 0, 6))
        self.assertIn("most studies report benefits", _describe_direction(4, 1, 6))

    def test_reviews_are_excluded_from_pooled_sample_size(self) -> None:
        papers = [make_paper("p1"), make_paper("p2")]
        primary = make_extraction("p1").model_copy(update={"sample_size": 600})
        review = make_extraction("p2").model_copy(
            update={"sample_size": 50000, "study_type": "meta_analysis"}
        )
        synthesis = _fallback_synthesis(
            "Does it work?", papers, [primary, review], [make_critique("p1")]
        )
        coverage = [b for b in synthesis.final_answer if "Sample coverage" in b]

        self.assertTrue(coverage)
        self.assertIn("600 participants", coverage[0])
        self.assertNotIn("50,600", coverage[0])

    def test_fabricated_citations_never_reach_the_user(self) -> None:
        # A small model will cite papers that were never retrieved. Presenting
        # that as evidence is worse than presenting the heuristic fallback.
        papers = [make_paper("p1")]
        extractions = [make_extraction("p1")]
        critiques = [make_critique("p1")]
        invented = json.dumps(
            {
                "final_answer": ["Creatine boosts strength by 8% [Kreider2003]."],
                "evidence_consensus": "Studies agree [Kreider2003].",
                "top_limitations_overall": ["Abstract-only."],
                "confidence_score": 90,
                "confidence_rationale": ["Strong evidence."],
                "citations_used": ["p1"],
            }
        )

        synthesis = asyncio.run(
            synthesize("Does creatine work?", papers, extractions, critiques, FakeLLM([invented]))
        )

        joined = " ".join(synthesis.final_answer)
        self.assertNotIn("Kreider2003", joined)
        self.assertTrue(verify_synthesis(synthesis, papers, critiques).passed)

    def test_rate_limited_search_is_not_reported_as_no_research(self) -> None:
        synthesis = _no_evidence_synthesis("Does creatine work?", rate_limited=True)

        joined = " ".join(synthesis.final_answer).lower()
        self.assertIn("rate limited", joined)
        self.assertNotIn("no academic papers were found", joined)
        self.assertEqual(synthesis.confidence_score, 0)

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
