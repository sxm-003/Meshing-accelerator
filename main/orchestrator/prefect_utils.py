# orchestrator/prefect_utils.py

from prefect.client.orchestration import get_client

async def ensure_concurrency_limit(name: str, limit: int):
    """
    Create or update a Prefect concurrency limit.
    Safe to call multiple times.
    """

    async with get_client() as client:
        limits = await client.read_concurrency_limits()
        existing = next((l for l in limits if l.name == name), None)

        if existing:
            if existing.limit != limit:
                await client.update_concurrency_limit(
                    concurrency_limit_id=existing.id,
                    limit=limit,
                )
        else:
            await client.create_concurrency_limit(
                name=name,
                limit=limit,
            )

