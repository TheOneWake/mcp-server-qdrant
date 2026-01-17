import logging
import uuid
from datetime import datetime
from typing import Any, Iterable

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        try:
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            message = str(exc).lower()
            if "doesn't exist" in message or "not found" in message:
                # Collection may have been deleted between exists check and upsert.
                await self._ensure_collection_exists(collection_name)
                await self._client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=uuid.uuid4().hex,
                            vector={vector_name: embeddings[0]},
                            payload=payload,
                        )
                    ],
                )
            else:
                raise

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        search_results = await self._query_points(
            query,
            collection_name=collection_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results
        ]

    async def delete_by_ids(
        self,
        ids: list[str],
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Delete points in the Qdrant collection by IDs.
        :param ids: List of point IDs to delete.
        :param collection_name: The name of the collection to delete from, optional.

        :return: Number of points scheduled for deletion.
        """
        if not ids:
            return 0
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=ids),
        )
        return len(ids)

    async def delete_by_filter(
        self,
        query_filter: models.Filter,
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Delete points in the Qdrant collection by filter.
        :param query_filter: Filter to select points to delete.
        :param collection_name: The name of the collection to delete from, optional.

        :return: Number of points scheduled for deletion.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        count_result = await self._client.count(
            collection_name=collection_name,
            count_filter=query_filter,
        )

        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=query_filter),
        )
        return count_result.count if count_result else 0

    async def delete_by_query(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> int:
        """
        Delete points in the Qdrant collection by semantic query.
        :param query: Query to match for deletion.
        :param collection_name: The name of the collection to delete from, optional.
        :param limit: Maximum number of points to delete.
        :param query_filter: Optional filter to apply to the query.

        :return: Number of points scheduled for deletion.
        """
        collection_name = collection_name or self._default_collection_name
        search_results = await self._query_points(
            query,
            collection_name=collection_name,
            limit=limit,
            query_filter=query_filter,
        )
        if not search_results:
            return 0

        ids = [str(result.id) for result in search_results]
        return await self._delete_ids(ids, collection_name=collection_name)

    async def dedupe_by_payload(
        self,
        dedupe_key: str,
        *,
        timestamp_field: str | None = None,
        keep: str = "newest",
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        batch_size: int = 256,
    ) -> int:
        """
        Remove duplicates by payload key, keeping newest/oldest based on timestamp field.
        :param dedupe_key: Payload key used for grouping (e.g., "metadata.file_path").
        :param timestamp_field: Payload key for ordering (e.g., "metadata.last_updated").
        :param keep: "newest" or "oldest".
        :param collection_name: Collection to dedupe.
        :param query_filter: Optional filter to scope dedupe.
        :param batch_size: Scroll batch size.

        :return: Number of points deleted.
        """
        collection_name = collection_name or self._default_collection_name
        records = await self._scroll_points(
            collection_name=collection_name,
            query_filter=query_filter,
            batch_size=batch_size,
        )
        if not records:
            return 0

        groups: dict[str, list[tuple[models.Record, float | None]]] = {}
        for record in records:
            payload = record.payload or {}
            group_value = self._get_payload_value(payload, dedupe_key)
            if group_value is None:
                continue
            ts_value = (
                self._get_payload_value(payload, timestamp_field)
                if timestamp_field
                else None
            )
            ts = self._parse_timestamp(ts_value)
            groups.setdefault(str(group_value), []).append((record, ts))

        ids_to_delete: list[str] = []
        reverse = keep == "newest"

        for _, items in groups.items():
            if len(items) <= 1:
                continue
            items.sort(
                key=lambda item: item[1]
                if item[1] is not None
                else float("-inf"),
                reverse=reverse,
            )
            for record, _ in items[1:]:
                ids_to_delete.append(str(record.id))

        if not ids_to_delete:
            return 0

        return await self._delete_ids(ids_to_delete, collection_name=collection_name)

    async def _query_points(
        self,
        query: str,
        *,
        collection_name: str | None,
        limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return list(search_results.points)

    async def _scroll_points(
        self,
        *,
        collection_name: str | None,
        query_filter: models.Filter | None,
        batch_size: int,
    ) -> list[models.Record]:
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        records: list[models.Record] = []
        offset = None

        while True:
            points, next_offset = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            records.extend(points)
            if not next_offset or not points:
                break
            offset = next_offset

        return records

    async def _delete_ids(
        self,
        ids: Iterable[str],
        *,
        collection_name: str | None = None,
        batch_size: int = 256,
    ) -> int:
        ids_list = list(ids)
        if not ids_list:
            return 0
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return 0

        deleted = 0
        for batch in self._chunk(ids_list, batch_size):
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=list(batch)),
            )
            deleted += len(batch)
        return deleted

    @staticmethod
    def _chunk(values: list[str], size: int) -> Iterable[list[str]]:
        for i in range(0, len(values), size):
            yield values[i : i + size]

    @staticmethod
    def _get_payload_value(payload: dict[str, Any], key: str) -> Any:
        if not key:
            return None
        current: Any = payload
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _parse_timestamp(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            try:
                if cleaned.endswith("Z"):
                    cleaned = cleaned[:-1] + "+00:00"
                return datetime.fromisoformat(cleaned).timestamp()
            except ValueError:
                try:
                    return datetime.strptime(cleaned, "%Y%m%d").timestamp()
                except ValueError:
                    return None
        return None

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            try:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            except Exception as exc:
                message = str(exc)
                if "already exists" not in message.lower():
                    raise
                logger.info(
                    "Collection %s already exists; continuing without create.",
                    collection_name,
                )

        # Create payload indexes if configured
        if self._field_indexes:
            for field_name, field_type in self._field_indexes.items():
                try:
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                except Exception as exc:
                    message = str(exc)
                    if "already exists" in message.lower():
                        continue
                    raise
