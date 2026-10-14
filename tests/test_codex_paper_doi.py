"""Offline regression coverage for accepted ChemRxiv DOI document links."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "src/browsergym/knows/eval/tasks/docs_5_influential_papers/utils.py"
spec = importlib.util.spec_from_file_location("paper_doi_test_utils", SOURCE)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class ChemRxivDoiTests(unittest.TestCase):
    def test_saved_retry_link_forms_are_retained_as_typed_dois(self):
        # Exact DOI suffixes from the saved model output; syntax recognition
        # does not assert these papers exist or satisfy citation criteria.
        suffixes = ["chemrxiv-2023-sx2hz", "chemrxiv-2023-tkcd9", "chemrxiv-2023-frr79",
                    "chemrxiv-2024-bdp43", "chemrxiv-2023-8s8zw"]
        text = "\n\n".join(f"Paper {i}\nhttps://chemrxiv.org/doi/10.26434/{suffix}"
                            for i, suffix in enumerate(suffixes, 1))
        links = utils.extract_paper_links_from_text(text, ["chemrxiv.org", "arxiv.org"])
        self.assertEqual(len(links), 5)
        self.assertEqual({utils.extract_paper_id(link) for link in links},
                         {("DOI", f"10.26434/{suffix}") for suffix in suffixes})

    def test_direct_url_variants_preserve_doi_without_query_or_fragment(self):
        for url in ("https://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz",
                    "http://www.chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz",
                    "https://CHEMRXIV.ORG/doi/10.26434/chemrxiv-2023-sx2hz?tracking=test#abstract"):
            with self.subTest(url=url):
                self.assertEqual(utils.extract_paper_id(url), ("DOI", "10.26434/chemrxiv-2023-sx2hz"))

    def test_new_form_rejects_wrong_origin_and_malformed_doi(self):
        for url in ("https://chemrxiv.org.example/doi/10.26434/chemrxiv-2023-sx2hz",
                    "https://[malformed-host/doi/10.26434/chemrxiv-2023-sx2hz",
                    "https://example.org/chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz",
                    "ftp://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz",
                    "https://chemrxiv.org/doi/10.12/invalid",
                    "https://chemrxiv.org/doi/11.26434/invalid",
                    "https://chemrxiv.org/doi/10.26434/",
                    "https://chemrxiv.org/doi/not-a-doi",
                    "https://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz/extra"):
            with self.subTest(url=url):
                self.assertIsNone(utils.extract_paper_id(url))

    def test_existing_platform_identifiers_are_unchanged(self):
        cases = {
            "https://doi.org/10.26434/chemrxiv-2023-sx2hz": ("DOI", "10.26434/chemrxiv-2023-sx2hz"),
            "https://arxiv.org/abs/2408.03175v2": ("ARXIV", "2408.03175"),
            "https://arxiv.org/pdf/2408.03175v2.pdf": ("ARXIV", "2408.03175"),
            "https://chemrxiv.org/engage/chemrxiv/article-details/6519a1f5ade1178b247a3b71":
                ("CHEMRXIV_ID", "6519a1f5ade1178b247a3b71"),
            "https://nature.com/articles/s41560-024-01569-9": ("DOI", "10.1038/s41560-024-01569-9"),
            "https://biorxiv.org/content/10.1101/2023.01.01.123456v1": ("DOI", "10.1101/2023.01.01.123456"),
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(utils.extract_paper_id(url), expected)

    def test_link_matching_deduplicates_doi_alias_without_inventing_visits(self):
        direct = "https://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz"
        alias = "https://doi.org/10.26434/chemrxiv-2023-sx2hz"
        result, doc_ids, visited_ids, matched = utils.match_paper_links_with_browsing_history(
            direct + "\n" + alias, [alias], domains=["chemrxiv.org"], min_papers=1)
        self.assertTrue(result)
        self.assertEqual(len(doc_ids), 1)
        self.assertEqual(doc_ids, visited_ids)
        self.assertEqual(matched, 1)
        result, doc_ids, visited_ids, matched = utils.match_paper_links_with_browsing_history(
            direct, [], domains=["chemrxiv.org"], min_papers=1)
        self.assertFalse(result)
        self.assertEqual(len(doc_ids), 1)
        self.assertEqual(visited_ids, [])
        self.assertEqual(matched, 0)

    def test_recognized_doi_uses_exact_existing_metadata_lookup(self):
        pid = utils.extract_paper_id("https://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz")
        paper = {"title": "Fixture paper", "citationCount": 17}
        response = Mock(status_code=200)
        response.json.return_value = [paper]
        with patch.object(utils, "_s2_request_with_backoff", return_value=response) as request, \
                patch.object(utils, "search_s2_by_title") as search:
            self.assertEqual(utils.fetch_papers_from_semantic_scholar([pid], fields="title,citationCount"), [paper])
            request.assert_called_once_with(
                "post", "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "title,citationCount"},
                json={"ids": ["DOI:10.26434/chemrxiv-2023-sx2hz"]})
            search.assert_not_called()

    def test_unindexed_or_invalid_candidate_does_not_gain_metadata(self):
        pid = utils.extract_paper_id("https://chemrxiv.org/doi/10.26434/chemrxiv-2023-sx2hz")
        response = Mock(status_code=200)
        response.json.return_value = [None]
        with patch.object(utils, "_s2_request_with_backoff", return_value=response), \
                patch.object(utils, "search_s2_by_title") as search:
            self.assertEqual(utils.fetch_papers_from_semantic_scholar([pid], fallback_titles=["A guess"]), [None])
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
