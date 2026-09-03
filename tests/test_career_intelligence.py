import unittest

from career_intelligence import DocumentChunk, ExplainableResumeScorer, QueryRoute, QueryRouter, SourceAwareRetriever, career_report_markdown


class CareerIntelligenceTests(unittest.TestCase):
    def test_router(self):
        router = QueryRouter()
        self.assertEqual(router.route("Summarize my resume"), QueryRoute.DOCUMENT)
        self.assertEqual(router.route("What is the latest company news?"), QueryRoute.HYBRID)
        self.assertEqual(router.route("What is the latest company news?", False), QueryRoute.WEB)

    def test_retrieval_preserves_provenance(self):
        evidence = SourceAwareRetriever([DocumentChunk("Python and AWS delivery", "resume.pdf", page=2)]).retrieve("AWS", 1)
        self.assertEqual(evidence[0].citation, "resume.pdf, page 2")

    def test_score_is_deterministic(self):
        resume = "Solution architecture, APIs, cloud, customer workshops, strategy and revenue growth."
        jd = "Solution architecture, APIs, cloud, customer consulting, strategy and revenue growth."
        scorer = ExplainableResumeScorer()
        first = scorer.score(resume, jd)
        self.assertEqual(first, scorer.score(resume, jd))
        self.assertTrue(0 <= first.overall <= 100)
        self.assertAlmostEqual(sum(scorer.WEIGHTS.values()), 1.0)

    def test_report_contains_score_and_caveat(self):
        score = ExplainableResumeScorer().score("Skills: Python", "Skills: Python, AWS")
        report = career_report_markdown(score, ["Confirm missing evidence before adding AWS."])
        self.assertIn("Resume-JD score", report)
        self.assertIn("deterministic", report)


if __name__ == "__main__":
    unittest.main()
