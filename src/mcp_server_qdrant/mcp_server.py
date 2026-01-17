import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str | None,
                Field(
                    description="The collection to store the information in (optional).",
                    default=None,
                ),
            ] = None,
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Keep the memory for later use, when you are asked to remember something.
            """
            logger.debug("Storing information in Qdrant: %s", information)

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str | None,
                Field(
                    description="The collection to search in (optional).",
                    default=None,
                ),
            ] = None,
            query_filter: Annotated[
               ArbitraryFilter | None,
               Field(
                   description="The filter to apply to the query (optional). Only available if configured.",
                   default=None
               )
            ] = None,
        ) -> list[str] | None:
            """
            Look up memories in Qdrant. Use this tool when you need to:
             - Find memories by their content
             - Access memories for further analysis
             - Get some personal information about the user
            """
            
            # Use provided filter if allowed
            final_filter = None
            if self.qdrant_settings.allow_arbitrary_filter:
                 final_filter = query_filter

            if final_filter:
                 # Convert dict to models.Filter if needed, but signature says ArbitraryFilter which is Dict[str, Any]
                 # QdrantConnector.search expects models.Filter | None.
                 # Need to convert dictionary to Filter model
                 try:
                    final_filter = models.Filter(**final_filter)
                 except Exception as e:
                     logger.error(f"Failed to parse query_filter: {e}")
                     final_filter = None

            logger.debug("Finding results for query: %s", query)

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=final_filter,
            )
            if not entries:
                return [f"No results for query '{query}'."]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def delete(
            ctx: Context,
            ids: Annotated[
                list[str] | None,
                Field(
                    description="Point IDs to delete. If provided, query/filter are ignored.",
                    default=None,
                ),
            ] = None,
            dedupe_key: Annotated[
                str | None,
                Field(
                    description=(
                        "Payload key to dedupe by (e.g., metadata.file_path). Keeps newest/oldest by timestamp."
                    ),
                    default=None,
                ),
            ] = None,
            timestamp_field: Annotated[
                str | None,
                Field(
                    description=(
                        "Payload timestamp field used for ordering (e.g., metadata.last_updated)."
                    ),
                    default="metadata.last_updated",
                ),
            ] = None,
            keep: Annotated[
                str | None,
                Field(
                    description="Which entry to keep per group: newest or oldest.",
                    default="newest",
                ),
            ] = None,
            query: Annotated[
                str | None,
                Field(
                    description=(
                        "Semantic query to match for deletion. Use with optional query_filter."
                    ),
                    default=None,
                ),
            ] = None,
            collection_name: Annotated[
                str | None,
                Field(
                    description="The collection to delete from (optional).",
                    default=None,
                ),
            ] = None,
            limit: Annotated[
                int | None,
                Field(
                    description="Maximum number of points to delete when using query.",
                    default=None,
                ),
            ] = None,
            query_filter: Annotated[
                ArbitraryFilter | None,
                Field(
                    description="The filter to apply to deletion (optional). Only available if configured.",
                    default=None,
                ),
            ] = None,
        ) -> str:
            """
            Delete information from Qdrant by ID, query, or filter.
            """
            if ids:
                deleted = await self.qdrant_connector.delete_by_ids(
                    ids, collection_name=collection_name
                )
                return f"Delete requested for {deleted} points."

            final_filter = None
            if query_filter is not None:
                if not self.qdrant_settings.allow_arbitrary_filter:
                    raise ValueError("Arbitrary filters are disabled for this server.")
                try:
                    final_filter = models.Filter(**query_filter)
                except Exception as exc:
                    logger.error("Failed to parse query_filter: %s", exc)
                    final_filter = None

            if dedupe_key:
                delete_keep = keep or "newest"
                deleted = await self.qdrant_connector.dedupe_by_payload(
                    dedupe_key,
                    timestamp_field=timestamp_field,
                    keep=delete_keep,
                    collection_name=collection_name,
                    query_filter=final_filter,
                )
                return (
                    f"Deduped {deleted} points by '{dedupe_key}' (keep {delete_keep})."
                )

            if query:
                delete_limit = limit or self.qdrant_settings.search_limit
                deleted = await self.qdrant_connector.delete_by_query(
                    query,
                    collection_name=collection_name,
                    limit=delete_limit,
                    query_filter=final_filter,
                )
                return f"Delete requested for {deleted} points."

            if final_filter:
                deleted = await self.qdrant_connector.delete_by_filter(
                    final_filter, collection_name=collection_name
                )
                return f"Delete requested for {deleted} points."

            raise ValueError(
                "Delete requires ids, query, or query_filter to be provided."
            )

        # Register tools directly with clearer descriptions
        self.tool(
            find,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            self.tool(
                store,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                delete,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
            )
