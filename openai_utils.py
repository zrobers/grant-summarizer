from openai import OpenAI
import os
from typing import Optional, Dict, Any, Union, List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Message = Union[str, Dict[str, str]]

def chat(
    messages: List[Message],
    model: str = "gpt-4.1",
    output_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Sends a chat completion request.  

    • If you pass a list of strings, each string is turned into
      a {"role":"user", "content":...} message.  
    • If you pass dicts with "role"/"content", they’re used as-is.  
    • If output_schema is provided, it’s sent as json_schema=… and
      the parsed JSON is returned. Otherwise you get back plain text.
    """

    # 1) normalize any bare strings into user messages
    normalized: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, str):
            normalized.append({"role": "user", "content": m})
        else:
            # assume it's already {"role":…,"content":…}
            normalized.append(m)
    params: Dict[str, Any] = {
        "model": model,
        "messages": normalized,
    }

    # 2) optional JSON schema enforcement
    if output_schema:
        params["json_schema"] = output_schema

    resp = client.chat.completions.create(**params)
    choice = resp.choices[0].message

    # 3) return the right thing
    if output_schema:
        return choice.json     # parsed object
    else:
        return choice.content  # raw string

    
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Returns the embedding vector for a given piece of text.
    Uses the new `client.embeddings.create` endpoint.
    """
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    # response.data is a list of objects; each has an .embedding attribute
    return response.data[0].embedding
