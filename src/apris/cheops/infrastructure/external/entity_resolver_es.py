from __future__ import annotations

import os
from typing import Any, Sequence

from apris.cheops.domain.entity_resolution import LegalEntityRecord, group_entity_records


class ElasticsearchEntityResolver:
    """Client for resolving entity identifiers via Elasticsearch."""

    def __init__(self, es_url: str | None = None) -> None:
        if es_url is None:
            es_url = os.environ.get("ES_URL", "http://localhost:9200")
        
        # In a real implementation this would initialize the AsyncElasticsearch client:
        # from elasticsearch import AsyncElasticsearch
        # self.es = AsyncElasticsearch([es_url])
        self.es_url = es_url

    async def resolve_entities(
        self, records: Sequence[LegalEntityRecord | dict[str, Any]]
    ) -> dict[str, list[LegalEntityRecord]]:
        """
        Groups entity records.
        """
        return group_entity_records(records)
