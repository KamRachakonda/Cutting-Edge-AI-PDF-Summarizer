import unittest

from career_intelligence import (
    DocumentChunk,
    ExplainableResumeScorer,
    QueryRoute,
    QueryRouter,
    SourceAwareRetriever,
    career_report_markdown,
)


class CareerIntelligenceTests(unittest.TestCase):
    def test_router_is_explicit_about_source(self):
        router = QueryRouter()
        self.assertEqual(router.route("Summarize my resume"),
                         QueryRoute.DOCUMENT)
        self.assertEqual(router.route(
            "What is the latest company news?"), QueryRoute.HYBRID)
        self.assertEqual(router.route(
            "What is the latest company news?", False), QueryRoute.WEB)

    def test_retrieval_preserves_page_provenance(self):
        retriever = SourceAwareRetriever([
            DocumentChunk("Python and AWS delivery", "resume.pdf", page=2),
            DocumentChunk("Marketing strategy", "jd.pdf", page=4),
        ])
        evidence = retriever.retrieve("AWS", limit=1)
        self.assertEqual(evidence[0].citation, "resume.pdf, page 2")

    def test_score_is_deterministic_and_explainable(self):
        resume = "Skills: Python, AWS\nExperience: 5 years building APIs"
        jd = "Skills: Python, AWS, Docker\nExperience: 5 years\nEducation: degree"
        scorer = ExplainableResumeScorer()
        first = scorer.score(resume, jd)
        second = scorer.score(resume, jd)
        self.assertEqual(first, second)
        skills = next(
            item for item in first.dimensions if item.name == "skills")
        self.assertIn("python", skills.matched)
        self.assertTrue(skills.evidence)
        self.assertIn("docker", skills.missing)

    def test_report_contains_caveat_and_evidence(self):
        score = ExplainableResumeScorer().score("Skills: Python", "Skills: Python, AWS")
        report = career_report_markdown(
            score, ["Confirm missing AWS before adding it."])
        self.assertIn("Resume evidence for 'python'", report)
        self.assertIn("does not infer experience", report)


if __name__ == "__main__":
    unittest.main()
