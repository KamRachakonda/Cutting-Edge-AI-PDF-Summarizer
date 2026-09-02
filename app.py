import json
import streamlit as st
import pandas as pd

from career_documents import extract_document, classify_document
from career_rag import CareerRAG
from career_ai import CareerAI
from career_config import GROQ_API_KEY, TAVILY_API_KEY, WEIGHTS
from career_research import research_context, format_research

st.set_page_config(page_title="AI Career Intelligence", page_icon="🚀", layout="wide")
st.markdown("""<style>.main .block-container{max-width:1400px;padding-top:1.5rem}.hero{padding:2rem 2.5rem;border-radius:18px;background:linear-gradient(135deg,#0f172a,#312e81);color:white;margin-bottom:1.5rem}.hero h1{font-size:2.6rem;margin:0}.hero p{color:#dbeafe;font-size:1.05rem}</style>""", unsafe_allow_html=True)
st.markdown("""<div class='hero'><h1>🚀 AI Career Intelligence Platform</h1><p>Resume + Position Description + semantic RAG + optional external research.</p></div>""", unsafe_allow_html=True)

if not GROQ_API_KEY: st.warning("GROQ_API_KEY is not configured. Add it to .env.")
if "docs" not in st.session_state: st.session_state.docs=[]
if "rag" not in st.session_state: st.session_state.rag=None
if "resume_analysis" not in st.session_state: st.session_state.resume_analysis=None
if "pd_analysis" not in st.session_state: st.session_state.pd_analysis=None
if "match" not in st.session_state: st.session_state.match=None
if "research" not in st.session_state: st.session_state.research=[]

with st.sidebar:
    st.header("📂 Document Intelligence")
    files=st.file_uploader("Upload Resume, PD/JD and supporting documents", type=["pdf","docx","txt","md","csv","json","log"], accept_multiple_files=True)
    web_enabled=st.checkbox("🌐 Enable external research", value=bool(TAVILY_API_KEY), disabled=not bool(TAVILY_API_KEY))
    if not TAVILY_API_KEY: st.caption("Add TAVILY_API_KEY to enable web research.")
    if files and st.button("📥 Index Documents", use_container_width=True):
        docs=[]
        for f in files:
            try:
                text,meta=extract_document(f.name,f.getvalue())
                docs.append({"filename":f.name,"text":text,"type":classify_document(f.name,text),"meta":meta})
            except Exception as e: st.error(f"{f.name}: {e}")
        if docs:
            rag=CareerRAG(); rag.add_documents(docs)
            st.session_state.docs=docs; st.session_state.rag=rag
            st.session_state.resume_analysis=None; st.session_state.pd_analysis=None; st.session_state.match=None
            st.success(f"Indexed {len(docs)} document(s)")
    if st.session_state.docs:
        st.subheader("Indexed Documents")
        for d in st.session_state.docs: st.write(f"**{d['type']}** — {d['filename']}")

if not st.session_state.docs:
    st.info("Upload your resume and target PD/JD, then click Index Documents.")
    st.stop()

resume_docs=[d for d in st.session_state.docs if d["type"]=="Resume"]
pd_docs=[d for d in st.session_state.docs if d["type"]=="Position Description"]
if not resume_docs or not pd_docs:
    st.warning("Automatic classification needs help. Select the documents manually.")
    names=[d["filename"] for d in st.session_state.docs]
    c1,c2=st.columns(2)
    with c1: rn=st.selectbox("Resume",names)
    with c2: pn=st.selectbox("Position Description / JD",names)
    resume_docs=[d for d in st.session_state.docs if d["filename"]==rn]
    pd_docs=[d for d in st.session_state.docs if d["filename"]==pn]
resume_text="\n\n".join(d["text"] for d in resume_docs)
pd_text="\n\n".join(d["text"] for d in pd_docs)
ai=CareerAI()

if st.button("🧠 Analyze Career Fit", type="primary", use_container_width=True):
    with st.spinner("Analyzing candidate evidence and role requirements..."):
        st.session_state.resume_analysis=ai.analyze_resume(resume_text)
        st.session_state.pd_analysis=ai.analyze_pd(pd_text)
        st.session_state.match=ai.match(st.session_state.resume_analysis,st.session_state.pd_analysis)
if not st.session_state.match:
    st.info("Click Analyze Career Fit to generate the scorecard."); st.stop()

match=st.session_state.match; dims=match.get("dimensions",{})
def score(name):
    try:return float(dims.get(name,{}).get("score",0))
    except:return 0
overall=round(sum(score(k)*v for k,v in WEIGHTS.items()))

c=st.columns(4); c[0].metric("Overall Fit",f"{overall}/100"); c[1].metric("Resume Docs",len(resume_docs)); c[2].metric("Role Docs",len(pd_docs)); c[3].metric("Research Sources",len(st.session_state.research))
score_df=pd.DataFrame([{"Dimension":k,"Score":score(k),"Weight":f"{v:.0%}","Status":dims.get(k,{}).get("status","Unknown")} for k,v in WEIGHTS.items()])
st.dataframe(score_df,use_container_width=True,hide_index=True)

if overall>=85: st.success("Strong alignment — position yourself as a high-confidence candidate.")
elif overall>=70: st.info("Good alignment — targeted positioning and gap mitigation should improve competitiveness.")
else: st.warning("Material gaps exist — focus on evidence and gap-closure strategy.")

t1,t2,t3,t4,t5,t6=st.tabs(["🎯 Strengths & Gaps","🔎 Evidence","🤖 Career Advisor","✍️ Resume Optimiser","🎤 Interview Prep","📄 Summarise"])
with t1:
    for name in WEIGHTS:
        x=dims.get(name,{})
        with st.expander(f"{name} — {x.get('score',0)}/100 · {x.get('status','Unknown')}"):
            st.write("**Evidence:**",x.get("evidence","Not provided")); st.write("**Missing:**",x.get("missing","None identified")); st.write("**Recommendation:**",x.get("recommendation",""))
with t2:
    st.dataframe(pd.DataFrame([{"Requirement":k,"Status":dims.get(k,{}).get("status","Unknown"),"Evidence":dims.get(k,{}).get("evidence",""),"Gap":dims.get(k,{}).get("missing","")} for k in WEIGHTS]),use_container_width=True,hide_index=True)
    st.markdown("### Why this score?"); st.write(match.get("overall_rationale",""))
with t3:
    q=st.text_area("Ask a career question",placeholder="Should I apply? How should I position myself? What are my biggest gaps?")
    if q and st.button("Ask Advisor"):
        context=st.session_state.rag.context(q,8) if st.session_state.rag else ""
        if web_enabled:
            role=st.session_state.pd_analysis.get("role_title",""); company=st.session_state.pd_analysis.get("company","")
            st.session_state.research=research_context([q,f"{company} {role} skills priorities 2026"])
        with st.spinner("Reasoning over your documents and research..."):
            st.markdown(ai.advisor(q,context,format_research(st.session_state.research)))
        if st.session_state.research:
            with st.expander("External Sources"):
                for i,r in enumerate(st.session_state.research,1): st.markdown(f"**[{i}] {r['title']}**  \\n{r['url']}")
with t4:
    task=st.text_area("Optimisation goal","Rewrite my executive summary and the most relevant experience bullets for this role. Preserve every fact and metric.")
    if st.button("Generate Optimised Positioning"):
        with st.spinner("Generating..."): st.markdown(ai.generate(task,f"RESUME:\n{resume_text}\n\nROLE:\n{pd_text}"))
with t5:
    task=st.text_area("Interview preparation","Create 10 likely questions, answer frameworks and STAR stories mapped to the role.")
    if st.button("Generate Interview Pack"):
        with st.spinner("Generating..."): st.markdown(ai.generate(task,f"RESUME:\n{resume_text}\n\nROLE:\n{pd_text}\n\nMATCH:\n{json.dumps(match)}"))
with t6:
    typ=st.selectbox("Summary type",["Executive","Detailed","Bullet points","Brief"])
    if st.button("Summarise Documents"):
        with st.spinner("Summarising..."): st.markdown(ai.generate(f"Create a {typ.lower()} summary highlighting purpose, themes, requirements and actionable takeaways.",f"{resume_text}\n\n{pd_text}"))

st.divider(); st.caption("AI Career Intelligence • Streamlit + Groq + semantic RAG + optional web research")
