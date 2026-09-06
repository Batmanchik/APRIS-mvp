import logging
from typing import Any, Sequence
from apris.cheops.domain.entity_resolution import (
    LegalEntityRecord,
    normalize_entity_record,
    resolve_entity_key,
)

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.exceptions import ConnectionError as ESConnectionError

    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    AsyncElasticsearch = None  # type: ignore
    ESConnectionError = Exception  # type: ignore

logger = logging.getLogger(__name__)


class ElasticsearchEntityResolver:
    """
    Elasticsearch-backed Entity Resolver for <50ms processing times per transaction.
    Gracefully falls back to deterministic local resolution if ES is unavailable.
    """

    def __init__(self, es_url: str = "http://elasticsearch:9200", index_name: str = "entities"):
        self.es_url = es_url
        self.index_name = index_name
        self._es = AsyncElasticsearch([es_url]) if ES_AVAILABLE else None
        self._connected = False

    async def initialize(self) -> None:
        if not self._es:
            logger.warning("Elasticsearch client not available. Using local fallback.")
            return
        try:
            if not await self._es.indices.exists(index=self.index_name):
                await self._es.indices.create(
                    index=self.index_name,
                    settings={
                        "analysis": {
                            "analyzer": {
                                "fuzzy_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "asciifolding"],
                                }
                            }
                        }
                    },
                    mappings={
                        "properties": {
                            "entity_key": {"type": "keyword"},
                            "entity_type": {"type": "keyword"},
                            "jurisdiction": {"type": "keyword"},
                            "source_entity_id": {"type": "keyword"},
                            "name": {"type": "text", "analyzer": "fuzzy_analyzer"},
                            "bin": {"type": "keyword"},
                            "iin": {"type": "keyword"},
                            "registration_no": {"type": "keyword"},
                            "tax_id": {"type": "keyword"},
                            "aliases": {"type": "text", "analyzer": "fuzzy_analyzer"},
                        }
                    },
                )
            self._connected = True
            logger.info("Successfully connected to Elasticsearch for entity resolution.")
        except ESConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}. Falling back to basic resolution.")
            self._connected = False
        except Exception as e:
            logger.error(f"Unexpected error initializing Elasticsearch: {e}")
            self._connected = False

    async def index_entity(self, raw: LegalEntityRecord | dict[str, Any]) -> str:
        record = normalize_entity_record(raw)
        entity_key = resolve_entity_key(record)

        if self._connected and self._es:
            try:
                await self._es.index(
                    index=self.index_name,
                    id=entity_key,
                    document={
                        "entity_key": entity_key,
                        "entity_type": record.entity_type,
                        "jurisdiction": record.jurisdiction,
                        "source_entity_id": record.source_entity_id,
                        "name": record.name,
                        "bin": record.bin,
                        "iin": record.iin,
                        "registration_no": record.registration_no,
                        "tax_id": record.tax_id,
                        "aliases": list(record.aliases),
                    },
                    refresh=True,
                )
            except Exception as e:
                logger.warning(f"Failed to index entity in ES: {e}")

        return entity_key

    async def resolve_entity(self, raw: LegalEntityRecord | dict[str, Any]) -> str:
        """
        Resolves an entity using Elasticsearch fuzzy matching and anchor matching.
        If no match is found, indexes the entity and returns a deterministic key.
        Falls back to pure deterministic resolution if ES is unavailable.
        """
        record = normalize_entity_record(raw)
        deterministic_key = resolve_entity_key(record)

        if not self._connected or not self._es:
            return deterministic_key

        try:
            should_clauses: list[dict[str, Any]] = []
            if record.bin:
                should_clauses.append({"term": {"bin": record.bin}})
            if record.iin:
                should_clauses.append({"term": {"iin": record.iin}})
            if record.registration_no:
                should_clauses.append({"term": {"registration_no": record.registration_no}})
            if record.tax_id:
                should_clauses.append({"term": {"tax_id": record.tax_id}})
            if record.source_entity_id:
                should_clauses.append({"term": {"source_entity_id": record.source_entity_id}})

            if record.name:
                should_clauses.append(
                    {
                        "match": {
                            "name": {
                                "query": record.name,
                                "fuzziness": "AUTO",
                            }
                        }
                    }
                )

            if not should_clauses:
                await self.index_entity(record)
                return deterministic_key

            response = await self._es.search(
                index=self.index_name,
                query={
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1,
                    }
                },
                size=1,
            )

            hits = response.get("hits", {}).get("hits", [])
            if hits:
                return str(hits[0]["_id"])

            await self.index_entity(record)
            return deterministic_key

        except Exception as e:
            logger.warning(f"ES resolve failed: {e}. Falling back to deterministic key.")
            return deterministic_key

    async def resolve_and_group(
        self, records: Sequence[LegalEntityRecord | dict[str, Any]]
    ) -> dict[str, list[LegalEntityRecord]]:
        """
        Groups entities together by resolving them asynchronously via ES or fallback.
        Handles intra-batch linkage correctly by locally resolving before querying ES.
        """
        import asyncio

        normalized_records = [normalize_entity_record(raw) for raw in records]

        # Intra-batch local grouping
        local_groups: dict[str, list[LegalEntityRecord]] = {}
        for rec in normalized_records:
            det_key = resolve_entity_key(rec)
            local_groups.setdefault(det_key, []).append(rec)

        async def resolve_leader(det_key: str, leader: LegalEntityRecord) -> tuple[str, str]:
            es_key = await self.resolve_entity(leader)
            return det_key, es_key

        tasks = [resolve_leader(dk, recs[0]) for dk, recs in local_groups.items()]
        resolved_pairs = await asyncio.gather(*tasks)

        key_mapping = {dk: es_key for dk, es_key in resolved_pairs}

        grouped: dict[str, list[LegalEntityRecord]] = {}
        for rec in normalized_records:
            dk = resolve_entity_key(rec)
            final_key = key_mapping[dk]
            grouped.setdefault(final_key, []).append(rec)

        return grouped
