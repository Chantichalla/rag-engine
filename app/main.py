# app/main.py
import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Ultimate RAG Project API",
    description="An advanced, multi-retriever RAG system."
)

# readiness & chain holder
app.state.ready = False
app.state.chain = None

@app.middleware("http")
async def readiness_middleware(request: Request, call_next):
    if not app.state.ready and request.url.path not in ("/health", "/ready"):
        return JSONResponse({"detail": "service starting, try again"}, status_code=503)
    return await call_next(request)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    return {"ready": app.state.ready}


async def _maybe_await(maybe):
    """If maybe is awaitable, await it; otherwise return it."""
    if asyncio.iscoroutine(maybe) or asyncio.isfuture(maybe):
        return await maybe
    return maybe


@app.on_event("startup")
async def startup_load():
    """
    Resolve final_rag_chain in whatever form it's exported:
      - async def final_rag_chain(...)  -> await it
      - coroutine object                 -> await it
      - sync callable returning chain   -> call/await if needed
      - already-built chain object      -> use as-is
    Store the resolved chain on app.state.chain and set app.state.ready=True.
    """
    logger.info("Starting component load...")
    try:
        # import inside startup so heavy imports happen here
        import app.chains as chains_module  # keep module import local

        maybe = getattr(chains_module, "final_rag_chain", None)

        chain_obj = None

        # 1) if it's an async function (callable coroutine function), call & await it
        if asyncio.iscoroutinefunction(maybe):
            chain_obj = await maybe()
        else:
            # 2) if it's a coroutine object or future, await it
            if asyncio.iscoroutine(maybe) or asyncio.isfuture(maybe):
                chain_obj = await maybe
            else:
                # 3) if it's callable (sync factory), call it and await result if needed
                if callable(maybe):
                    got = maybe()
                    chain_obj = await _maybe_await(got)
                else:
                    # 4) assume it's already the chain object
                    chain_obj = maybe

        # final safety: if still awaitable, await it
        if asyncio.iscoroutine(chain_obj) or asyncio.isfuture(chain_obj):
            chain_obj = await _maybe_await(chain_obj)

        app.state.chain = chain_obj
        app.state.ready = True
        logger.info("Components loaded successfully; app is ready.")
    except Exception as e:
        logger.exception("Failed to load components at startup: %s", e)
        app.state.chain = None
        app.state.ready = False


@app.get("/")
def read_root():
    return {"message": "Welcome to the Ultimate RAG API!"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a query and returns a RAG-generated answer.
    Uses the resolved chain stored at app.state.chain.
    """
    logger.debug("Received query: %s", request.query)

    if not app.state.ready or app.state.chain is None:
        raise HTTPException(status_code=503, detail="service not ready")

    chain = app.state.chain

    # If chain is accidentally still an awaitable, resolve it now
    if asyncio.iscoroutine(chain) or asyncio.isfuture(chain):
        chain = await _maybe_await(chain)
        app.state.chain = chain  # cache resolved chain

    # Try async API first
    try:
        if hasattr(chain, "ainvoke"):
            result = await chain.ainvoke({"query": request.query})
            return ChatResponse(answer=result)
    except Exception as e:
        logger.exception("Async chain invocation failed: %s", e)
        # fall through to other attempts

    # Try sync API via thread
    if hasattr(chain, "invoke"):
        try:
            result = await asyncio.to_thread(lambda: chain.invoke({"query": request.query}))
            return ChatResponse(answer=result)
        except Exception as e:
            logger.exception("Synchronous chain invocation failed: %s", e)
            raise HTTPException(status_code=500, detail="chain invocation failed")

    # If chain is callable (maybe __call__), try that
    if callable(chain):
        try:
            maybe = chain({"query": request.query})
            result = await _maybe_await(maybe)
            return ChatResponse(answer=result)
        except Exception as e:
            logger.exception("Callable chain invocation failed: %s", e)
            raise HTTPException(status_code=500, detail="chain invocation failed")

    logger.error("Chain object has no invocation method (ainvoke/invoke/__call__)")
    raise HTTPException(status_code=500, detail="invalid chain object")
