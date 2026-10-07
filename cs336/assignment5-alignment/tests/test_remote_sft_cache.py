import hashlib
import io

from scripts import prepare_sft_data as cache


def expected(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_remote_cache_reuses_existing_copy_and_never_redownloads(tmp_path, monkeypatch):
    content = b"already uploaded dataset"
    monkeypatch.setattr(cache, "FILES", {"train.jsonl.gz": expected(content)})
    source = tmp_path / "runs/old/workspace/data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)

    def forbidden_download(*args):
        raise AssertionError("A verified local copy must not trigger a network download")

    monkeypatch.setattr(cache, "download", forbidden_download)
    target = tmp_path / "shared"
    cache.prepare(target, tmp_path / "runs")
    source.write_bytes(b"source was changed later")
    cache.prepare(target, tmp_path / "runs")
    assert (target / "train.jsonl.gz").read_bytes() == content
    assert (target / "manifest.json").is_file()


def test_download_resumes_partial_file_and_verifies_hash(tmp_path, monkeypatch):
    content = b"0123456789"
    partial = tmp_path / "train.part"
    partial.write_bytes(content[:4])

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 4-9/10"}

    def open_url(request, timeout):
        assert request.get_header("Range") == "bytes=4-"
        return Response(content[4:])

    monkeypatch.setattr(cache.urllib.request, "urlopen", open_url)
    cache.download("https://example.invalid/train", partial, expected(content))
    assert partial.read_bytes() == content


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    content = b"0123456789"
    partial = tmp_path / "train.part"
    partial.write_bytes(content[:4])

    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr(cache.urllib.request, "urlopen", lambda *a, **kw: Response(content))
    cache.download("https://example.invalid/train", partial, expected(content))
    assert partial.read_bytes() == content
