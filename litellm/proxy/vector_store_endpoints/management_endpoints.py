"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import copy
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

from fastapi import APIRouter, Depends, HTTPException

if TYPE_CHECKING:
    from prisma.models import LiteLLM_ManagedVectorStoresTable as _VectorStoreRow

    from litellm.models.model import LiteLLM_ProxyModelTable
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import (
    LiteLLM_ManagedVectorStoresTable,
    ResponseLiteLLM_ManagedVectorStore,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.rbac_utils import check_feature_access_for_user
from litellm.proxy.vector_store_endpoints.utils import can_user_access_vector_store
from litellm.repositories.model_repository import ModelRepository
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ManagedVectorStoresRepository
from litellm.secret_managers.main import get_secret
from litellm.types.router import CredentialLiteLLMParams
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
    VectorStoreInfoRequest,
    VectorStoreUpdateRequest,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

router: Final = APIRouter()


def _vector_store_table(prisma_client: "PrismaClient") -> "TableActions[_VectorStoreRow]":
    return ManagedVectorStoresRepository(prisma_client).table


def _row_to_vector_store(row: "_VectorStoreRow") -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(**row.model_dump())


_LITELLM_PARAMS_MASKER: Final = SensitiveDataMasker()


_REDACT_LITELLM_PARAMS_MAX_DEPTH: Final = 10
_EMBEDDING_CONFIG_EXCLUDED_KEYS: Final = frozenset({"custom_llm_provider", "litellm_credential_name", "model"})

EmbeddingResolution: TypeAlias = tuple[
    str,
    dict[str, object],  # mutable-ok: downstream embedding APIs accept keyword dictionaries
]


def _as_string_object_dict(  # mutable-ok: normalizes external credential mappings for provider calls
    value: object,
) -> dict[str, object] | None:  # mutable-ok: downstream provider API requires a keyword dictionary
    if not isinstance(value, Mapping):
        return None
    mapping: Final = cast(  # cast-ok: isinstance validated the otherwise unparameterized Mapping
        Mapping[object, object], value
    )
    result: Final[dict[str, object]] = {}  # mutable-ok: built locally and returned to the provider boundary
    for key, item in mapping.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _redact_sensitive_litellm_params(litellm_params: Any, _depth: int = 0) -> Any:
    """
    Replace credential-bearing values in ``litellm_params`` with
    ``REDACTED_BY_LITELM`` while preserving non-secret keys (``api_base``,
    ``region``, ``model``, ``api_version``).

    Handles three input shapes:

    * ``dict`` — recurse into nested dicts (e.g. ``litellm_embedding_config``
      which itself carries ``api_key`` / ``aws_*`` / ``vertex_credentials``).
    * ``str`` — the in-memory registry occasionally holds the params as a
      JSON-serialized string. Parse, redact, re-serialize. If parsing
      fails, return the redaction sentinel rather than echo the value
      back verbatim.
    * Anything else, or ``None`` — passed through.

    Recursion depth is bounded by ``_REDACT_LITELLM_PARAMS_MAX_DEPTH`` —
    matching the convention of other allowlisted recursive helpers in the
    repo (see ``tests/code_coverage_tests/recursive_detector.py``).
    """
    if _depth >= _REDACT_LITELLM_PARAMS_MAX_DEPTH:
        return REDACTED_BY_LITELM_STRING
    if litellm_params is None:
        return None
    if isinstance(litellm_params, str):
        try:
            parsed: Final = json.loads(litellm_params)
        except (TypeError, ValueError):
            return REDACTED_BY_LITELM_STRING
        return json.dumps(_redact_sensitive_litellm_params(parsed, _depth + 1))
    if not isinstance(litellm_params, dict):
        return litellm_params
    out: Final[dict[str, Any]] = {}
    for k, v in litellm_params.items():
        if _LITELLM_PARAMS_MASKER.is_sensitive_key(k):
            out[k] = REDACTED_BY_LITELM_STRING
        elif isinstance(v, dict):
            out[k] = _redact_sensitive_litellm_params(v, _depth + 1)
        else:
            out[k] = v
    return out


async def _fetch_and_authorize_vector_store(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: "PrismaClient",
) -> "LiteLLM_ManagedVectorStore":
    """
    Look up a vector store by id and confirm the caller can access it.
    Raises HTTPException(404) on miss and HTTPException(403) on access
    denial.
    """
    row: Final = await _vector_store_table(prisma_client).find_unique(where={"vector_store_id": vector_store_id})
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store with ID {vector_store_id} not found",
        )
    typed: Final = _row_to_vector_store(row)
    if not await _check_vector_store_access(typed, user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You do not have permission to access this vector store",
        )
    return typed


def _resolve_secret_references(  # mutable-ok: downstream embedding APIs accept keyword dictionaries
    config: Mapping[str, object],
) -> dict[str, object] | None:  # mutable-ok: downstream provider API requires a keyword dictionary
    resolved_config: Final[dict[str, object]] = {}  # mutable-ok: built locally for the provider boundary
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("os.environ/"):
            resolved_value: object | None = cast(  # cast-ok: secret backends expose a deliberately broad return type
                object | None, get_secret(value)
            )
            if resolved_value is None:
                return None
            resolved_config[key] = resolved_value
        elif value is not None:
            resolved_config[key] = value
    return resolved_config


def _embedding_resolution_from_credentials(credentials: Mapping[str, object]) -> EmbeddingResolution | None:
    model: Final = credentials.get("model")
    provider: Final = credentials.get("custom_llm_provider")
    if not isinstance(model, str) or not model:
        return None

    provider_name: Final = provider if isinstance(provider, str) and provider else "openai"
    provider_qualified_model: Final = model if "/" in model else f"{provider_name}/{model}"
    if "*" in provider_qualified_model:
        return None

    config: Final = _resolve_secret_references(
        {  # mutable-ok: provider resolver consumes keyword dictionaries
            key: value
            for key, value in credentials.items()
            if key not in _EMBEDDING_CONFIG_EXCLUDED_KEYS and value is not None
        }
    )
    if not config:
        return None
    return provider_qualified_model, config


def _resolve_embedding_model_config_from_router(
    embedding_model: str,
    llm_router: "Router | None",
    team_id: str | None,
) -> EmbeddingResolution | None:
    if not embedding_model or llm_router is None:
        return None
    try:
        credentials: Final = llm_router.get_deployment_credentials_with_provider(
            model_id=embedding_model,
            team_id=team_id,
        )
        typed_credentials: Final = _as_string_object_dict(credentials)
        if typed_credentials is None:
            return None
        return _embedding_resolution_from_credentials(typed_credentials)
    except Exception as e:
        verbose_proxy_logger.debug("Error resolving embedding model %s from Router: %s", embedding_model, e)
        return None


def _select_team_safe_db_model(
    models: Sequence["LiteLLM_ProxyModelTable"],
    team_id: str | None,
) -> "LiteLLM_ProxyModelTable | None":
    team_models: Final = tuple(model for model in models if team_id is not None and model.team_id == team_id)
    shared_models: Final = tuple(model for model in models if model.team_id is None)
    return next(iter(team_models), None) or next(iter(shared_models), None)


def _embedding_resolution_from_db_model(
    model: "LiteLLM_ProxyModelTable",
) -> EmbeddingResolution | None:
    if model.blocked:
        return None
    params: Final = _as_string_object_dict(
        cast(object, model.litellm_params)  # cast-ok: Prisma model params are legacy unparameterized dictionaries
    )
    if params is None:
        return None
    underlying_model: Final = params.get("model")
    if not isinstance(underlying_model, str) or not underlying_model:
        return None

    credential_name: Final = params.get("litellm_credential_name")
    if credential_name is not None and not isinstance(credential_name, str):
        return None
    named_credentials: Final = (
        _as_string_object_dict(
            cast(  # cast-ok: credential storage exposes a legacy unparameterized dictionary
                object, CredentialAccessor.get_credential_values(credential_name)
            )
        )
        if credential_name is not None
        else {}  # mutable-ok: local empty credential overlay
    )
    if named_credentials is None or (credential_name is not None and not named_credentials):
        return None

    deployment_credentials: Final = _as_string_object_dict(
        CredentialLiteLLMParams.model_validate(params).model_dump(exclude_none=True)
    )
    if deployment_credentials is None:
        return None
    provider: Final = params.get("custom_llm_provider")
    credentials: Final = {  # mutable-ok: provider resolver consumes keyword dictionaries
        **deployment_credentials,
        **named_credentials,
        "model": underlying_model,
        **(
            {"custom_llm_provider": provider}  # mutable-ok: conditional keyword overlay
            if isinstance(provider, str) and provider
            else {}  # mutable-ok: conditional keyword overlay
        ),
    }
    return _embedding_resolution_from_credentials(credentials)


async def _resolve_embedding_model_config_from_db(
    embedding_model: str,
    prisma_client: "PrismaClient",
    team_id: str | None,
) -> EmbeddingResolution | None:
    if not embedding_model:
        return None
    where: Final[dict[str, object]] = {  # mutable-ok: Prisma requires a JSON-shaped filter dictionary
        "blocked": False,
        "OR": [  # mutable-ok: Prisma requires a JSON-shaped OR list
            {"model_name": embedding_model},  # mutable-ok: Prisma filter clause
            {"model_id": embedding_model},  # mutable-ok: Prisma filter clause
            {  # mutable-ok: Prisma filter clause
                "model_info": {  # mutable-ok: Prisma JSON filter
                    "path": ["team_public_model_name"],  # mutable-ok: Prisma JSON path list
                    "equals": json.dumps(embedding_model),
                }
            },
        ],
    }
    try:
        candidates: Final = await ModelRepository(prisma_client).find_many(where=where)
        selected: Final = _select_team_safe_db_model(models=candidates, team_id=team_id)
        return _embedding_resolution_from_db_model(selected) if selected is not None else None
    except Exception as e:
        verbose_proxy_logger.debug("Error resolving embedding model %s from database: %s", embedding_model, e)
        return None


async def _resolve_embedding_model_config(
    embedding_model: str,
    prisma_client: "PrismaClient | None",
    llm_router: "Router | None",
    team_id: str | None,
) -> EmbeddingResolution | None:
    router_resolution: Final = _resolve_embedding_model_config_from_router(
        embedding_model=embedding_model,
        llm_router=llm_router,
        team_id=team_id,
    )
    if router_resolution is not None:
        return router_resolution
    if prisma_client is None:
        return None
    return await _resolve_embedding_model_config_from_db(
        embedding_model=embedding_model,
        prisma_client=prisma_client,
        team_id=team_id,
    )


########################################################
# Helper Functions
########################################################
async def _check_vector_store_access(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Check if the user has access to the vector store.

    Delegates to :func:`can_user_access_vector_store`, which honors:
    - PROXY_ADMIN bypass
    - legacy vector stores with no team_id
    - key-level and team-level ``object_permission.vector_stores`` allowlists
    - team_id match between key and store
    """
    return await can_user_access_vector_store(vector_store=vector_store, user_api_key_dict=user_api_key_dict)


async def create_vector_store_in_db(
    vector_store_id: str,
    custom_llm_provider: str,
    prisma_client: "PrismaClient | None",
    vector_store_name: str | None = None,
    vector_store_description: str | None = None,
    vector_store_metadata: dict | None = None,
    litellm_params: dict | None = None,
    litellm_credential_name: str | None = None,
    team_id: str | None = None,
    user_id: str | None = None,
) -> LiteLLM_ManagedVectorStore:
    """
    Helper function to create a vector store in the database.

    This function handles:
    - Checking if vector store already exists
    - Creating the vector store in the database
    - Adding it to the vector store registry

    Returns:
        LiteLLM_ManagedVectorStore: The created vector store object

    Raises:
        HTTPException: If vector store already exists or database error occurs
    """
    from litellm.types.router import GenericLiteLLMParams

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Check if vector store already exists
    existing_vector_store: Final = await _vector_store_table(prisma_client).find_unique(
        where={"vector_store_id": vector_store_id}
    )
    if existing_vector_store is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store with ID {vector_store_id} already exists",
        )

    # Prepare data for database
    data_to_create: Final[dict[str, object]] = {
        "vector_store_id": vector_store_id,
        "custom_llm_provider": custom_llm_provider,
    }

    if vector_store_name is not None:
        data_to_create["vector_store_name"] = vector_store_name
    if vector_store_description is not None:
        data_to_create["vector_store_description"] = vector_store_description
    if vector_store_metadata is not None:
        data_to_create["vector_store_metadata"] = safe_dumps(vector_store_metadata)
    if litellm_credential_name is not None:
        data_to_create["litellm_credential_name"] = litellm_credential_name
    if team_id is not None:
        data_to_create["team_id"] = team_id
    if user_id is not None:
        data_to_create["user_id"] = user_id

    # Handle litellm_params - always provide at least an empty dict.
    # The earlier behaviour resolved ``litellm_embedding_config`` from the
    # admin-configured router/DB model and persisted the cleartext result
    # (``api_key``, ``api_base``, ``api_version``) into this row. That
    # exposed every env-stored embedding-model credential on the
    # ``/vector_store/{new,info,update,list}`` responses. Keep the user's
    # raw ``litellm_embedding_model`` reference; resolution now happens in
    # ``_update_request_data_with_litellm_managed_vector_store_registry``
    # at request-handling time so the cleartext config exists only in
    # per-request memory and never reaches the database.
    if litellm_params:
        litellm_params_dict: Final = GenericLiteLLMParams(**litellm_params).model_dump(exclude_none=True)
        data_to_create["litellm_params"] = safe_dumps(litellm_params_dict)
    else:
        # Provide empty dict if no litellm_params provided
        data_to_create["litellm_params"] = safe_dumps({})

    # Create in database
    _new_vector_store: Final = await _vector_store_table(prisma_client).create(data=data_to_create)

    new_vector_store: Final[LiteLLM_ManagedVectorStore] = _row_to_vector_store(_new_vector_store)

    # Add vector store to registry
    if litellm.vector_store_registry is not None:
        litellm.vector_store_registry.add_vector_store_to_registry(vector_store=new_vector_store)

    verbose_proxy_logger.info("Vector store %s created in database successfully", vector_store_id)

    return new_vector_store


########################################################
# Management Endpoints
########################################################
@router.post(
    "/vector_store/new",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new vector store.

    Parameters:
    - vector_store_id: str - Unique identifier for the vector store
    - custom_llm_provider: str - Provider of the vector store
    - vector_store_name: Optional[str] - Name of the vector store
    - vector_store_description: Optional[str] - Description of the vector store
    - vector_store_metadata: Optional[Dict] - Additional metadata for the vector store
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    try:
        vector_store_id: Final = vector_store.get("vector_store_id")
        custom_llm_provider: Final = vector_store.get("custom_llm_provider")

        if not vector_store_id or not custom_llm_provider:
            raise HTTPException(
                status_code=400,
                detail="vector_store_id and custom_llm_provider are required",
            )

        # Extract and validate metadata
        metadata: Final = vector_store.get("vector_store_metadata")
        validated_metadata: dict | None = None
        if metadata is not None and isinstance(metadata, dict):
            validated_metadata = metadata

        new_vector_store: Final = await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider=custom_llm_provider,
            prisma_client=prisma_client,
            vector_store_name=vector_store.get("vector_store_name"),
            vector_store_description=vector_store.get("vector_store_description"),
            vector_store_metadata=validated_metadata,
            litellm_params=vector_store.get("litellm_params"),
            litellm_credential_name=vector_store.get("litellm_credential_name"),
            team_id=user_api_key_dict.team_id,
            user_id=user_api_key_dict.user_id,
        )

        # Apply the same litellm_params redaction the list / info / update
        # endpoints already use, so a caller-supplied credential or a
        # cleartext value persisted by an earlier proxy version doesn't
        # come back in the response.
        response_vs: Final = LiteLLM_ManagedVectorStore(**new_vector_store)
        response_vs["litellm_params"] = _redact_sensitive_litellm_params(new_vector_store.get("litellm_params"))

        return {
            "status": "success",
            "message": f"Vector store {vector_store.get('vector_store_id')} created successfully",
            "vector_store": response_vs,
        }
    except Exception as e:
        verbose_proxy_logger.exception("Error creating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
@router.get(
    "/v1/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
async def list_vector_stores(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = 1,
    page_size: int = 100,
):
    """
    List all available vector stores with optional filtering and pagination.
    Combines both in-memory vector stores and those stored in the database.
    Database is the source of truth - deleted stores are removed from memory, updated stores sync to memory.

    Parameters:
    - page: int - Page number for pagination (default: 1)
    - page_size: int - Number of items per page (default: 100)
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    vector_store_map: Final[dict[str, LiteLLM_ManagedVectorStore]] = {}
    db_vector_store_ids: Final[set] = set()

    try:
        # Get vector stores from database first (source of truth)
        vector_stores_from_db: Final = await VectorStoreRegistry._get_vector_stores_from_db(prisma_client=prisma_client)

        # Build map from database vector stores
        for vector_store in vector_stores_from_db:
            vector_store_id = vector_store.get("vector_store_id", None)
            if vector_store_id:
                vector_store_map[vector_store_id] = vector_store
                db_vector_store_ids.add(vector_store_id)

        # Process in-memory vector stores
        if litellm.vector_store_registry is not None:
            in_memory_vector_stores: Final = copy.deepcopy(litellm.vector_store_registry.vector_stores)

            vector_stores_to_delete_from_memory: Final[list[str]] = []

            for vector_store in in_memory_vector_stores:
                vector_store_id = vector_store.get("vector_store_id", None)
                if not vector_store_id:
                    continue

                # If vector store is in memory but NOT in database, it was deleted
                if vector_store_id not in db_vector_store_ids:
                    verbose_proxy_logger.info(
                        "Vector store %s exists in memory but not in database - marking for deletion from cache",
                        vector_store_id,
                    )
                    vector_stores_to_delete_from_memory.append(vector_store_id)
                # If not in our map yet, add it (only in-memory, not in DB)
                elif vector_store_id not in vector_store_map:
                    vector_store_map[vector_store_id] = vector_store

            # Synchronize in-memory registry with database
            # 1. Remove deleted vector stores from memory
            for vs_id in vector_stores_to_delete_from_memory:
                litellm.vector_store_registry.delete_vector_store_from_registry(vector_store_id=vs_id)
                verbose_proxy_logger.debug("Removed deleted vector store %s from in-memory registry", vs_id)

            # 2. Update in-memory registry with database versions (for updates)
            for vector_store in vector_stores_from_db:
                vector_store_id = vector_store.get("vector_store_id", None)
                if vector_store_id:
                    litellm.vector_store_registry.update_vector_store_in_registry(
                        vector_store_id=vector_store_id, updated_data=vector_store
                    )

        # Filter vector stores based on access control
        accessible_vector_stores: Final = []
        for vs in vector_store_map.values():
            if await _check_vector_store_access(vs, user_api_key_dict):
                redacted = LiteLLM_ManagedVectorStore(**vs)
                redacted["litellm_params"] = _redact_sensitive_litellm_params(vs.get("litellm_params"))
                accessible_vector_stores.append(redacted)

        total_count: Final = len(accessible_vector_stores)
        total_pages: Final = (total_count + page_size - 1) // page_size

        # Format response using LiteLLM_ManagedVectorStoreListResponse
        response: Final = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=accessible_vector_stores,
            total_count=total_count,
            current_page=page,
            total_pages=total_pages,
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception("Error listing vector stores: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/delete",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_vector_store(
    data: VectorStoreDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a vector store from both database and in-memory registry.

    Parameters:
    - vector_store_id: str - ID of the vector store to delete
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if vector store exists in database or in-memory registry
        db_vector_store_exists = False
        memory_vector_store_exists = False
        vector_store_to_check = None

        existing_vector_store: Final = await _vector_store_table(prisma_client).find_unique(
            where={"vector_store_id": data.vector_store_id}
        )
        if existing_vector_store is not None:
            db_vector_store_exists = True
            vector_store_to_check = _row_to_vector_store(existing_vector_store)

        # Check in-memory registry
        if litellm.vector_store_registry is not None:
            memory_vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if memory_vector_store is not None:
                memory_vector_store_exists = True
                if vector_store_to_check is None:
                    vector_store_to_check = memory_vector_store

        # If not found in either location, raise 404
        if not db_vector_store_exists and not memory_vector_store_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )

        # Check access control
        if vector_store_to_check and not await _check_vector_store_access(vector_store_to_check, user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to delete this vector store",
            )

        # Delete from database if exists
        if db_vector_store_exists:
            await _vector_store_table(prisma_client).delete(where={"vector_store_id": data.vector_store_id})

        # Delete from in-memory registry if exists
        if memory_vector_store_exists and litellm.vector_store_registry is not None:
            litellm.vector_store_registry.delete_vector_store_from_registry(vector_store_id=data.vector_store_id)

        return {
            "status": "success",
            "message": f"Vector store {data.vector_store_id} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/info",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ResponseLiteLLM_ManagedVectorStore,
)
async def get_vector_store_info(
    data: VectorStoreInfoRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return a single vector store's details"""
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        if litellm.vector_store_registry is not None:
            vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if vector_store is not None:
                # Check access control
                if not await _check_vector_store_access(vector_store, user_api_key_dict):
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: You do not have permission to access this vector store",
                    )

                vector_store_metadata: Final = vector_store.get("vector_store_metadata")
                # Parse metadata if it's a JSON string
                parsed_metadata: dict | None = None
                if isinstance(vector_store_metadata, str):
                    parsed_metadata = json.loads(vector_store_metadata)
                elif isinstance(vector_store_metadata, dict):
                    parsed_metadata = vector_store_metadata

                vector_store_pydantic_obj: Final = LiteLLM_ManagedVectorStoresTable(
                    vector_store_id=vector_store.get("vector_store_id") or "",
                    custom_llm_provider=vector_store.get("custom_llm_provider") or "",
                    vector_store_name=vector_store.get("vector_store_name") or None,
                    vector_store_description=vector_store.get("vector_store_description") or None,
                    vector_store_metadata=parsed_metadata,
                    created_at=vector_store.get("created_at") or None,
                    updated_at=vector_store.get("updated_at") or None,
                    litellm_credential_name=vector_store.get("litellm_credential_name"),
                    litellm_params=_redact_sensitive_litellm_params(vector_store.get("litellm_params")),
                    team_id=vector_store.get("team_id") or None,
                    user_id=vector_store.get("user_id") or None,
                )
                return {"vector_store": vector_store_pydantic_obj}

        vector_store_typed: Final = await _fetch_and_authorize_vector_store(
            vector_store_id=data.vector_store_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )
        vector_store_dict: Final = dict(vector_store_typed)
        if "litellm_params" in vector_store_dict:
            vector_store_dict["litellm_params"] = _redact_sensitive_litellm_params(vector_store_dict["litellm_params"])
        return {"vector_store": vector_store_dict}
    except HTTPException:
        # Preserve 403/404 from the access-control / not-found checks above;
        # the catch-all below would otherwise rewrite them as 500.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting vector store info: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/update",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_vector_store(
    data: VectorStoreUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update vector store details in both database and in-memory registry.
    The updated data is immediately synchronized to the in-memory registry.
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client
    from litellm.types.router import GenericLiteLLMParams

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        update_data: Final = data.model_dump(exclude_unset=True)
        vector_store_id: Final[str] = update_data.pop("vector_store_id")

        # Per-store access control: anyone authenticated who passes the
        # premium-feature gate could otherwise update *any* vector store —
        # including stores belonging to other teams.
        await _fetch_and_authorize_vector_store(
            vector_store_id=vector_store_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

        # Handle metadata serialization
        if update_data.get("vector_store_metadata") is not None:
            update_data["vector_store_metadata"] = safe_dumps(update_data["vector_store_metadata"])

        # Handle litellm_params if provided. As with the create path, the
        # embedding-config auto-resolve previously persisted cleartext
        # credentials into the row; resolution now happens at request-
        # handling time in
        # ``_update_request_data_with_litellm_managed_vector_store_registry``
        # so this row only ever stores the user-supplied
        # ``litellm_embedding_model`` reference.
        if "litellm_params" in update_data:
            _input_litellm_params: Final[dict] = update_data.get("litellm_params", {}) or {}
            litellm_params_dict: Final = GenericLiteLLMParams(**_input_litellm_params).model_dump(exclude_none=True)
            update_data["litellm_params"] = safe_dumps(litellm_params_dict)

        # Update in database
        updated: Final = await _vector_store_table(prisma_client).update(
            where={"vector_store_id": vector_store_id},
            data=update_data,
        )

        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {vector_store_id} not found",
            )

        updated_vs: Final = _row_to_vector_store(updated)

        # Immediately update in-memory registry to keep it in sync
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=updated_vs,
            )
            verbose_proxy_logger.debug(
                "Updated vector store %s in both database and in-memory registry", vector_store_id
            )

        # The DB row is returned in full, so the response would otherwise
        # echo the persisted ``litellm_params`` (including provider
        # credentials) back to the caller — even when the caller only
        # changed unrelated fields like ``vector_store_description``.
        response_vs: Final = LiteLLM_ManagedVectorStore(**updated_vs)
        response_vs["litellm_params"] = _redact_sensitive_litellm_params(updated_vs.get("litellm_params"))
        return {
            "status": "success",
            "message": f"Vector store {vector_store_id} updated successfully",
            "vector_store": response_vs,
        }
    except HTTPException:
        # Preserve 403/404 responses from the access-control / not-found
        # checks above; the catch-all below would otherwise rewrite them
        # as 500 with the original status code embedded in the detail.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
