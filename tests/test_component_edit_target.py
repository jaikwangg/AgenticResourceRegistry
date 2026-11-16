# SPDX-FileCopyrightText: 2026 Worakij <worakij.w@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for services/component_edit_target.py.

The edit target is the version an author's edits land on: the approved default, or the
newest reviewed successor of it. The database is mocked; these tests cover how the
service picks among the versions the query returns, and which listing fields it refuses
to change while a successor is being edited.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from models.mcp import ListingStatus
from models.prompt import PromptVersion
from services.component_edit_target import (
    editable_version,
    editable_versions,
    reject_listing_edits_on_successor,
)

# ── Helpers ──────────────────────────────────────────────


def _version(listing_id, version, status=ListingStatus.pending):
    return SimpleNamespace(id=uuid.uuid4(), listing_id=listing_id, version=version, status=status)


def _listing(version="1.0.0", status=ListingStatus.approved, **fields):
    listing = SimpleNamespace(id=uuid.uuid4(), latest_version=None, latest_version_id=None, **fields)
    if version is not None:
        listing.latest_version = _version(listing.id, version, status)
        listing.latest_version_id = listing.latest_version.id
    return listing


def _db(candidates=()):
    """A session whose one query returns ``candidates``."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(candidates)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


# ── editable_versions ────────────────────────────────────


class TestEditableVersions:
    @pytest.mark.asyncio
    async def test_no_approved_default_skips_the_query(self):
        draft = _listing(status=ListingStatus.draft)
        unversioned = _listing(version=None)
        db = _db()

        result = await editable_versions(db, "prompt", [draft, unversioned], PromptVersion)

        assert result == {draft.id: draft.latest_version, unversioned.id: None}
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_listings(self):
        db = _db()

        assert await editable_versions(db, "prompt", [], PromptVersion) == {}
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approved_default_without_successor_stays(self):
        listing = _listing()
        db = _db()

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result == {listing.id: listing.latest_version}
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_picks_the_newest_successor(self):
        listing = _listing("1.0.0")
        patch = _version(listing.id, "1.0.1")
        minor = _version(listing.id, "1.2.0")
        db = _db([patch, minor])

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result[listing.id] is minor

    @pytest.mark.asyncio
    async def test_compares_versions_numerically(self):
        listing = _listing("1.9.0")
        successor = _version(listing.id, "1.10.0")
        db = _db([_version(listing.id, "1.9.1"), successor])

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result[listing.id] is successor

    @pytest.mark.asyncio
    async def test_ignores_candidates_not_newer_than_the_default(self):
        listing = _listing("2.0.0")
        db = _db([_version(listing.id, "1.5.0"), _version(listing.id, "2.0.0")])

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result[listing.id] is listing.latest_version

    @pytest.mark.asyncio
    async def test_unorderable_default_keeps_the_default(self):
        listing = _listing("not-a-version")
        db = _db([_version(listing.id, "9.0.0")])

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result[listing.id] is listing.latest_version

    @pytest.mark.asyncio
    async def test_unorderable_candidate_keeps_the_default(self):
        listing = _listing("1.0.0")
        db = _db([_version(listing.id, "2.0.0"), _version(listing.id, "next")])

        result = await editable_versions(db, "prompt", [listing], PromptVersion)

        assert result[listing.id] is listing.latest_version

    @pytest.mark.asyncio
    async def test_batches_listings_into_one_query(self):
        first, second, draft = _listing("1.0.0"), _listing("3.0.0"), _listing(status=ListingStatus.draft)
        successor = _version(first.id, "1.1.0")
        db = _db([successor, _version(second.id, "2.0.0")])

        result = await editable_versions(db, "prompt", [first, second, draft], PromptVersion)

        assert result == {
            first.id: successor,
            second.id: second.latest_version,
            draft.id: draft.latest_version,
        }
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_query_is_scoped_to_kind_and_approved_listings(self):
        approved, draft = _listing(), _listing(status=ListingStatus.draft)
        db = _db()

        await editable_versions(db, "prompt", [approved, draft], PromptVersion)

        sql = str(db.execute.await_args.args[0].compile(compile_kwargs={"literal_binds": True}))
        assert "reviews.subject_type = 'prompt'" in sql
        assert approved.id.hex in sql
        assert draft.id.hex not in sql


# ── editable_version ─────────────────────────────────────


class TestEditableVersion:
    @pytest.mark.asyncio
    async def test_returns_the_successor(self):
        listing = _listing("1.0.0")
        successor = _version(listing.id, "1.1.0")

        assert await editable_version(_db([successor]), "prompt", listing, PromptVersion) is successor

    @pytest.mark.asyncio
    async def test_returns_the_default_when_not_approved(self):
        listing = _listing(status=ListingStatus.pending)

        assert await editable_version(_db(), "prompt", listing, PromptVersion) is listing.latest_version

    @pytest.mark.asyncio
    async def test_returns_none_without_versions(self):
        assert await editable_version(_db(), "prompt", _listing(version=None), PromptVersion) is None


# ── reject_listing_edits_on_successor ────────────────────


class TestRejectListingEditsOnSuccessor:
    FIELDS = ("name", "owner")

    def _req(self, **fields):
        return SimpleNamespace(**{"name": None, "owner": None, **fields})

    def test_allows_edits_on_the_default(self):
        listing = _listing(name="old", owner="team-a")

        reject_listing_edits_on_successor(listing, listing.latest_version, self._req(name="new"), self.FIELDS)

    def test_allows_edits_without_a_version(self):
        listing = _listing(name="old", owner="team-a")

        reject_listing_edits_on_successor(listing, None, self._req(name="new"), self.FIELDS)

    def test_allows_unset_and_unchanged_fields_on_a_successor(self):
        listing = _listing(name="old", owner="team-a")
        successor = _version(listing.id, "1.1.0")

        reject_listing_edits_on_successor(listing, successor, self._req(name="old"), self.FIELDS)

    def test_allows_fields_missing_from_the_request(self):
        listing = _listing(name="old", owner="team-a")
        successor = _version(listing.id, "1.1.0")

        reject_listing_edits_on_successor(listing, successor, SimpleNamespace(), self.FIELDS)

    def test_rejects_changes_on_a_successor(self):
        listing = _listing(name="old", owner="team-a")
        successor = _version(listing.id, "1.1.0")

        with pytest.raises(HTTPException) as exc:
            reject_listing_edits_on_successor(listing, successor, self._req(name="new", owner="team-b"), self.FIELDS)

        assert exc.value.status_code == 409
        assert exc.value.detail.startswith("name, owner cannot change while editing pending version 1.1.0")

    def test_only_checks_the_given_fields(self):
        listing = _listing(name="old", owner="team-a")
        successor = _version(listing.id, "1.1.0")

        reject_listing_edits_on_successor(listing, successor, self._req(owner="team-b"), ("name",))
