from datetime import datetime, timezone
import unittest

from publish_research_snapshot import publish_research_snapshot
import test_export_research_snapshot as snapshot_fixtures


class _PutResult:
    url = "https://store.private.blob.vercel-storage.com/research/latest.json"
    pathname = "research/latest.json"


class _GetResult:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


class _FakeClient:
    def __init__(self, token: str, *, mismatch: bool = False) -> None:
        self.token = token
        self.mismatch = mismatch
        self.payload = b""
        self.put_kwargs = {}
        self.get_kwargs = {}

    def put(self, pathname: str, payload: bytes, **kwargs):
        self.pathname = pathname
        self.payload = payload
        self.put_kwargs = kwargs
        return _PutResult()

    def get(self, url: str, **kwargs):
        self.url = url
        self.get_kwargs = kwargs
        return _GetResult(b"different" if self.mismatch else self.payload)


class PublishResearchSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = snapshot_fixtures.ResearchSnapshotTests(
            methodName="test_writes_round_trip_json"
        )
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _write_sources(self, *args, **kwargs):
        return self.fixture._write_sources(*args, **kwargs)

    def test_publish_uploads_private_blob_and_reads_it_back(self) -> None:
        run_card, watchlist = self._write_sources()
        clients = []

        def factory(**kwargs):
            client = _FakeClient(**kwargs)
            clients.append(client)
            return client

        result = publish_research_snapshot(
            run_card,
            watchlist,
            token="test-token",
            client_factory=factory,
            published_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "published")
        self.assertTrue(result["verified"])
        self.assertEqual(result["priority_rows"], 2)
        self.assertEqual(clients[0].pathname, "research/latest.json")
        self.assertEqual(clients[0].put_kwargs["access"], "private")
        self.assertTrue(clients[0].put_kwargs["overwrite"])
        self.assertEqual(clients[0].put_kwargs["cache_control_max_age"], 60)
        self.assertFalse(clients[0].get_kwargs["use_cache"])

    def test_publish_rejects_failed_read_back(self) -> None:
        run_card, watchlist = self._write_sources()

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            publish_research_snapshot(
                run_card,
                watchlist,
                token="test-token",
                client_factory=lambda **kwargs: _FakeClient(
                    **kwargs, mismatch=True
                ),
            )

    def test_publish_requires_token(self) -> None:
        run_card, watchlist = self._write_sources()

        with self.assertRaisesRegex(RuntimeError, "BLOB_READ_WRITE_TOKEN"):
            publish_research_snapshot(
                run_card,
                watchlist,
                token="",
                client_factory=_FakeClient,
            )


if __name__ == "__main__":
    unittest.main()
