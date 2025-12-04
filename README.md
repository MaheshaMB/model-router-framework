🚀 **Model Router Framework**


A smart decision-engine for automated LLM selection — solving the challenge of manual model choice, cost inefficiency, throttling failures, and inconsistent behavior across multi-provider AI systems .


🧭 **It dynamically selects the right LLM or embedding model based on:**

* Task Type ➝ Chat / Embedding
* Fallback Policy ➝ Automatic retry & switch when **throttled exception**
* Query Size ➝ Token length + context window
* Complexity ➝ Shallow answer vs deep reasoning
* Language ➝ English / Multilingual
* Cost Tier ➝ Low / Medium / High budget governance
* Tenant Tier ➝ Free / Standard / Premium multi-tenant control
* Fallback Policy ➝ Automatic retry & switch when throttled

⚙️ **Router Behaviors**

✔ Automatically estimates token size

✔ Detects language (English vs multilingual)

✔ Evaluates complexity heuristically

✔ Validates model context/embedding limits

✔ Applies cost/tenant policy rules

✔ Retry w/ exponential backoff on throttling

✔ Failover to backup model if primary is throttled

✔ Transparent to calling app

🏗 **Architecture**

        Application / Agent / RAG
                │
                ▼
        ModelRouter.select_model()   ←  Tenant + metadata
                │
                ▼
        Feature Extraction (task/size/lang/complexity/context)
                │
                ▼
        Routing Rules (fetch rule.json from S3 / local)
                │
                ▼
        Model Config Selection (fetch modle.json from S3 / local)
                │
                ▼
        Provider Client (Bedrock / Claude / Gemini) + Retries + Backup Failover
                │
                ▼
        ModelHandle.chat() / ModelHandle.embed()


🧩 **Installation**

1. Clone the repository
   
        git clone https://github.com/MaheshaMB/model-router-framework.git
        cd model-router-framework

3. Install dependencies
  
        pip install -r requirements.txt

⚙️ **Runtime Configuration**

For local JSON mode: 

        export FETCH_DRIVE=false

Or enable S3 configuration in production:

        export FETCH_DRIVE=true
        export MODEL_ROUTER_CONFIG_BUCKET=my-router-config
        export MODEL_ROUTER_MODELS_KEY=models.json
        export MODEL_ROUTER_RULES_KEY=routing_rules.json

Also set required LLM SDK credentials:

        export AWS_REGION=us-east-1
        export AWS_ACCESS_KEY_ID=your_key
        export AWS_SECRET_ACCESS_KEY=your_key
        export ANTHROPIC_API_KEY=your_key
        export GOOGLE_API_KEY=your_key


🚀 **Usage Example**

Chat

        from model_router_framework import ModelRouter
        
        router = ModelRouter()

        # Simple user query
        query = "What is the llm model router ?"
        handle = router.select_model(text=query)
        response = handle.chat([{"role": "user", "content": query}])
        print(response)

        # Complex user query
        query = "Explain in detail of software architecture design in 20 pages"
        handle = router.select_model(text=query, task_type="chat")
        response = handle.chat([{"role": "user", "content": query}])
        print(response)


Embedding

        from model_router_framework import ModelRouter
        
        router = ModelRouter()
        chunk = "Explain in detail of software architecture design in 20 pages"
        handle = router.select_model(text=chunk, task_type="embedding")
        response = handle.embed(chunk)
        print(response)


🥇 **Status**

* This is currently a POC implementation, and it can be evolved into an MVP and further to a production-ready solution by applying the necessary configuration enhancements and environment-specific adjustments.

* This solution is designed specifically for your application’s LLM call workflow, including RAG and Agentic AI flows. 

* It can be packaged as a reusable library and integrated directly into your application wherever LLM initialization and call execution are performed.
  
 
🙌 **Contributions Welcome**
Issues, improvements, additional provider support, and real-world routing rules are encouraged!
 
 
💡 **Author Note**
This project aims to simplify llm-call, Agentic-AI, Enterprise RAG adoption by removing the complexity of selecting and managing multiple LLMs behind the scenes.
If you are working in AI Platform Engineering, Multi-LLM Systems, or Serverless AI, let’s collaborate! 
