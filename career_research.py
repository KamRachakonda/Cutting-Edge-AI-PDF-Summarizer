import requests
from career_config import TAVILY_API_KEY

def search_web(query: str, max_results: int = 5):
    if not TAVILY_API_KEY: return []
    response=requests.post("https://api.tavily.com/search",json={"api_key":TAVILY_API_KEY,"query":query,"search_depth":"advanced","max_results":max_results,"include_answer":False},timeout=30)
    response.raise_for_status()
    return [{"title":x.get("title",""),"url":x.get("url",""),"content":x.get("content","")} for x in response.json().get("results",[])]

def research_context(queries):
    results=[]; seen=set()
    for q in queries[:5]:
        for item in search_web(q,4):
            if item["url"] not in seen: seen.add(item["url"]); results.append({**item,"query":q})
    return results

def format_research(results):
    return "\n\n".join(f"[{i+1}] {r['title']}\nURL: {r['url']}\n{r['content']}" for i,r in enumerate(results))
