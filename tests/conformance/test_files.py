"""Conformance: filesystem tool calls across all backends."""

from __future__ import annotations

import pytest

from rath.backend import (
    Backend,
    FileContent,
    FileEntries,
    FileWriteResult,
    FilesExists,
    FilesList,
    FilesRead,
    FilesWrite,
)

pytestmark = pytest.mark.anyio


async def test_write_then_read_text(backend: Backend) -> None:
    async with await backend.open() as sb:
        write = await sb.dispatch(FilesWrite(path="hello.txt", data="hello"))
        assert isinstance(write, FileWriteResult)
        assert write.bytes_written == 5
        read = await sb.dispatch(FilesRead(path="hello.txt"))
        assert isinstance(read, FileContent)
        assert read.data == "hello"


async def test_write_then_read_bytes(backend: Backend) -> None:
    payload = b"\x00\x01\x02binary\xffdata"
    async with await backend.open() as sb:
        await sb.dispatch(FilesWrite(path="blob.bin", data=payload))
        result = await sb.dispatch(FilesRead(path="blob.bin", encoding=None))
        assert isinstance(result, FileContent)
        assert result.data == payload


async def test_exists_true_and_false(backend: Backend) -> None:
    async with await backend.open() as sb:
        await sb.dispatch(FilesWrite(path="present.txt", data=""))
        assert await sb.dispatch(FilesExists(path="present.txt")) is True
        assert await sb.dispatch(FilesExists(path="nope.txt")) is False


async def test_list_returns_sorted_entries(backend: Backend) -> None:
    async with await backend.open() as sb:
        for name in ("c.txt", "a.txt", "b.txt"):
            await sb.dispatch(FilesWrite(path=name, data=name))
        result = await sb.dispatch(FilesList(path="."))
        assert isinstance(result, FileEntries)
        names = [e.name for e in result.entries]
        assert names == sorted(names)
        assert {"a.txt", "b.txt", "c.txt"}.issubset(set(names))


async def test_read_missing_path_raises(backend: Backend) -> None:
    async with await backend.open() as sb:
        with pytest.raises(FileNotFoundError):
            await sb.dispatch(FilesRead(path="does-not-exist.txt"))


async def test_write_creates_parent_dirs(backend: Backend) -> None:
    async with await backend.open() as sb:
        await sb.dispatch(
            FilesWrite(path="a/b/c/deep.txt", data="deep")
        )
        result = await sb.dispatch(FilesRead(path="a/b/c/deep.txt"))
        assert isinstance(result, FileContent)
        assert result.data == "deep"
