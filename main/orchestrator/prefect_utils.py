# orchestrator/prefect_utils.py

from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import (
    ConcurrencyLimitCreate,
    ConcurrencyLimitUpdate,
)

async def ensure_concurrency_limit(name: str, limit: int):
    """
    Prefect 2.x–correct concurrency limit create/update.
    Idempotent and safe.
    """

    async with get_client() as client:
        # Prefect 2.x requires pagination args
        limits = await client.read_concurrency_limits(
            limit=100,
            offset=0,
        )

        existing = next((l for l in limits if l.name == name), None)

        if existing:
            if existing.limit != limit:
                await client.update_concurrency_limit(
                    concurrency_limit_id=existing.id,
                    concurrency_limit=ConcurrencyLimitUpdate(
                        limit=limit
                    ),
                )
        else:
            await client.create_concurrency_limit(
                concurrency_limit=ConcurrencyLimitCreate(
                    name=name,
                    limit=limit,
                )
            )
