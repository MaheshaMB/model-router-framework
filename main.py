from src import ModelRouter

router = ModelRouter()

def handler(event, context):
    user_query = event["question"]

    handle = router.select_model(
        text=user_query,
        task_type="chat",
        context_tokens=event.get("context_tokens"),
        tenant_id=event.get("tenant_id"),
        tenant_tier=event.get("tenant_tier", "standard"),
    )

    answer = handle.chat(messages=[{"role": "user", "content": user_query}])
    return {"answer": str(answer)}

if __name__ == "__main__":
    sample_event = {
        "question": "What is the capital of France?",
        "context_tokens": None,
        "tenant_id": "tenant_123",
        "tenant_tier": "premium"
    }
    print(handler(sample_event, None))
    
    sample_event_1 = {
        "question": "Explain in detail how these 20 pages of architecture docs affect our deployment pipeline.",
    }
    print(handler(sample_event_1, None))
    
    
    chunk = "Serverless RAG architecture details..."
    handle = router.select_model(text=chunk, task_type="embedding")
    print(handle.embed(chunk))