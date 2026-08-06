import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "slack_sdk", "aiohttp", "nest_asyncio",
    "transformers", "accelerate", "bitsandbytes",
    "sentencepiece", "protobuf"])

import getpass
SLACK_BOT_TOKEN = getpass.getpass("1  Slack Bot Token  (xoxb-...): ").strip()
SLACK_CHANNEL   = input(          "2  Slack Channel ID (C...):     ").strip()
print()

TIMEZONE             = "America/Chicago"
RUN_HOUR             = 7
RUN_MINUTE           = 0
DAYS_LOOKBACK        = 30
DAYS_LOOKBACK_AUTHOR = 30
MAX_PAPERS_PER_TOPIC = 10
MAX_AUTHOR_PAPERS    = 10
MAX_SCHOLAR_PAPERS   = 20

TRACKED_AUTHORS = [
    "Serdar Bozdag",
    "Faisal Mahmood",

]

LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

import asyncio, aiohttp, re, logging, torch
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import nest_asyncio
nest_asyncio.apply()

from slack_sdk.web.async_client import AsyncWebClient
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ResearchBot")

print(f"Loading {LLM_MODEL} in 4-bit on GPU...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model loaded on {device.upper()}")
if device == "cpu":
    print("No GPU - generation will be very slow.")
    print("   Go to Runtime -> Change runtime type -> T4 GPU and re-run.\n")
else:
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {mem:.1f} GB\n")

TOPICS = [
    {
        "name": "Multi-Omics Integration with GNNs",
        "arxiv_queries": [
            'abs:"multi-omics" OR abs:"multi-modal omics"',
            'abs:"graph neural network" OR abs:"GNN" OR abs:"graph convolutional"',
            'ti:"omics integration" OR ti:"multi-omics"',
        ],
        "biorxiv_terms": [
            "multi-omics graph neural network",
            "GNN omics integration",
            "multi-omics graph convolutional",
        ],
        "scholar_queries": [
            "multi-omics GNN integration graph neural network",
        ],
        "must_contain_any": ["multi-omics", "multiomics", "multi-modal omics",
                             "omics integration", "genomics", "transcriptomics",
                             "proteomics", "metabolomics", "single cell", "spatial transcriptomics", "computational pathology", 
                             "whole-slide images"],
        "must_also_contain_any": ["graph neural", "gnn", "graph convolutional",
                                  "graph attention", "graph transformer"],
    },
    {
        "name": "Cancer Biomarker Discovery - Multi-Omics ML",
        "arxiv_queries": [
            'abs:"multi-omics" OR abs:"multi-modal omics"',
            'abs:"cancer" OR abs:"tumor" OR abs:"oncology"',
            'abs:"biomarker" OR abs:"prognosis" OR abs:"survival prediction"',
        ],
        "biorxiv_terms": [
            "multi-omics cancer biomarker",
            "cancer multi-omics prognosis survival",
            "pan-cancer omics machine learning",
        ],
        "scholar_queries": [
            "multi-omics cancer biomarker machine learning prognosis",
        ],
        "must_contain_any": ["multi-omics", "multiomics", "omics integration",
                             "genomics", "transcriptomics", "proteomics"],
        "must_also_contain_any": ["cancer", "tumor", "tumour", "oncology",
                                  "carcinoma", "leukemia", "glioma"],
    },
    {
        "name": "Alzheimer's Disease Progression - Multi-Omics ML",
        "arxiv_queries": [
            "abs:\"Alzheimer\" OR abs:\"Alzheimer's disease\" OR ti:\"Alzheimer\"",
            'abs:"omics" OR abs:"genomics" OR abs:"proteomics" OR abs:"transcriptomics"',
            'abs:"machine learning" OR abs:"deep learning" OR abs:"neural network"',
        ],
        "biorxiv_terms": [
            "Alzheimer multi-omics machine learning",
            "Alzheimer genomics deep learning",
            "Alzheimer proteomics transcriptomics",
        ],
        "scholar_queries": [
            "Alzheimer disease multi-omics deep learning progression",
        ],
        "must_contain_any": ["alzheimer", "alzheimer's", "ad progression",
                             "dementia", "neurodegeneration"],
        "must_also_contain_any": ["omics", "genomics", "proteomics", "transcriptomics",
                                  "metabolomics", "multi-omics", "machine learning",
                                  "deep learning"],
    },
    {
        "name": "Spatial Transcriptomics from Histopathology",
        "arxiv_queries": [
            'abs:"spatial transcriptomics" OR ti:"spatial transcriptomics"',
            'abs:"histopathology" OR abs:"whole slide image" OR abs:"H&E"',
            'abs:"spatial gene expression" OR abs:"visium" OR abs:"10x spatial"',
        ],
        "biorxiv_terms": [
            "spatial transcriptomics histopathology deep learning",
            "spatial gene expression histology neural network",
            "visium spatial transcriptomics transformer",
        ],
        "scholar_queries": [
            "spatial transcriptomics histopathology deep learning whole slide",
        ],
        "must_contain_any": ["spatial transcriptomics", "spatial gene expression",
                             "visium", "spatial omics", "10x spatial"],
        "must_also_contain_any": ["histopathology", "histology", "whole slide",
                                  "h&e", "deep learning", "neural network",
                                  "transformer", "convolutional"],
    },
]

@dataclass
class Paper:
    arxiv_id:      str
    title:         str
    authors:       list
    abstract:      str
    published:     str
    url:           str
    pdf_url:       str
    categories:    list
    source:        str = "arXiv"
    lecture_notes: str = ""


def is_relevant(paper, must_contain_any, must_also_contain_any):
    text = (paper.title + " " + paper.abstract).lower()
    return (any(kw.lower() in text for kw in must_contain_any) and
            any(kw.lower() in text for kw in must_also_contain_any))


class ArxivFetcher:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, session):
        self.session = session

    async def search(self, query, days_back, max_results=15):
        since      = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
        full_query = f"({query}) AND submittedDate:[{since} TO 99991231235959]"
        params     = {"search_query": full_query, "start": 0,
                      "max_results": max_results,
                      "sortBy": "submittedDate", "sortOrder": "descending"}
        try:
            async with self.session.get(
                self.BASE_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status != 200:
                    log.warning(f"arXiv {r.status}")
                    return []
                return self._parse(await r.text())
        except Exception as e:
            log.error(f"arXiv error: {e}")
            return []

    async def search_by_author(self, author_name, days_back, max_results=20):
        parts      = author_name.strip().split()
        last_name  = parts[-1] if parts else author_name
        since      = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
        query      = f'au:"{author_name}" AND submittedDate:[{since} TO 99991231235959]'
        params     = {"search_query": query, "start": 0,
                      "max_results": max_results,
                      "sortBy": "submittedDate", "sortOrder": "descending"}
        papers = []
        try:
            async with self.session.get(
                self.BASE_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status == 200:
                    papers = self._parse(await r.text())
        except Exception as e:
            log.error(f"arXiv author search error: {e}")

        if not papers:
            query  = f'au:{last_name} AND submittedDate:[{since} TO 99991231235959]'
            params["search_query"] = query
            try:
                async with self.session.get(
                    self.BASE_URL, params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as r:
                    if r.status == 200:
                        candidates = self._parse(await r.text())
                        for p in candidates:
                            if any(author_name.lower() in a.lower() for a in p.authors):
                                papers.append(p)
            except Exception as e:
                log.error(f"arXiv author fallback error: {e}")

        return papers

    def _parse(self, xml_text):
        papers, ns = [], {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_text)
            for e in root.findall("atom:entry", ns):
                try:
                    aid = e.find("atom:id", ns).text.split("/abs/")[-1].strip()
                    papers.append(Paper(
                        arxiv_id   = aid,
                        title      = re.sub(r"\s+", " ", e.find("atom:title", ns).text).strip(),
                        abstract   = re.sub(r"\s+", " ", e.find("atom:summary", ns).text).strip(),
                        published  = e.find("atom:published", ns).text[:10],
                        authors    = [a.find("atom:name", ns).text
                                      for a in e.findall("atom:author", ns)],
                        url        = f"https://arxiv.org/abs/{aid}",
                        pdf_url    = f"https://arxiv.org/pdf/{aid}",
                        categories = [c.get("term", "")
                                      for c in e.findall("atom:category", ns)],
                        source     = "arXiv",
                    ))
                except Exception as ex:
                    log.warning(f"arXiv entry parse: {ex}")
        except Exception as ex:
            log.error(f"arXiv XML parse: {ex}")
        return papers


class BiorxivFetcher:
    SEARCH_URL  = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    AUTHOR_URL  = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, session):
        self.session = session

    def _date_filter(self, days_back: int) -> str:
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"FIRST_PDATE:[{since} TO {today}]"

    def _item_to_paper(self, item: dict) -> Paper:
        pmcid    = item.get("id", "")
        doi      = item.get("doi", "")
        title    = item.get("title", "").strip().rstrip(".")
        abstract = item.get("abstractText", "") or item.get("abstract", "")
        abstract = abstract.strip()
        date     = item.get("firstPublicationDate", "") or item.get("pubYear", "")
        author_list = []
        for a in item.get("authorList", {}).get("author", []):
            full = (a.get("fullName") or
                    f"{a.get('firstName','')} {a.get('lastName','')}").strip()
            if full:
                author_list.append(full)
        if not author_list:
            raw = item.get("authorString", "")
            author_list = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
        url     = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PPR/{pmcid}"
        pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi and "biorxiv" in doi.lower() else ""
        return Paper(
            arxiv_id   = f"epmc_{pmcid or doi.replace('/','_')}",
            title      = title,
            abstract   = abstract,
            published  = date,
            authors    = author_list,
            url        = url,
            pdf_url    = pdf_url,
            categories = ["bioRxiv/Preprint"],
            source     = "bioRxiv",
        )

    async def _epmc_search(self, query: str, page_size: int = 25) -> list:
        params = {
            "query":      query,
            "source":     "PPR",
            "resultType": "core",
            "pageSize":   page_size,
            "format":     "json",
            "cursorMark": "*",
        }
        papers = []
        try:
            async with self.session.get(
                self.SEARCH_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status != 200:
                    log.warning(f"Europe PMC status {r.status} for query: {query[:60]}")
                    return []
                data    = await r.json(content_type=None)
                results = data.get("resultList", {}).get("result", [])
                for item in results:
                    try:
                        papers.append(self._item_to_paper(item))
                    except Exception as ex:
                        log.warning(f"Europe PMC parse error: {ex}")
        except Exception as e:
            log.error(f"Europe PMC search error: {e}")
        return papers

    async def search(self, terms: list, days_back: int, max_results: int = 15) -> list:
        date_flt = self._date_filter(days_back)
        seen, all_papers = set(), []
        for term in terms:
            query   = f'({term}) AND {date_flt}'
            papers  = await self._epmc_search(query, page_size=max_results)
            log.info(f"  [Europe PMC/bioRxiv] '{term[:50]}' -> {len(papers)} results")
            for p in papers:
                if p.arxiv_id not in seen and p.title:
                    seen.add(p.arxiv_id)
                    all_papers.append(p)
            if len(all_papers) >= max_results * 2:
                break
            await asyncio.sleep(1)
        return all_papers[:max_results * 2]

    async def search_by_author(self, author_name: str, days_back: int,
                               max_results: int = 20) -> list:
        date_flt  = self._date_filter(days_back)
        last_name = author_name.strip().split()[-1]
        for name_q in [f'AUTH:"{author_name}"', f'AUTH:{last_name}']:
            query  = f'{name_q} AND {date_flt}'
            papers = await self._epmc_search(query, page_size=max_results)
            verified = []
            for p in papers:
                if any(author_name.lower() in a.lower() or last_name.lower() in a.lower()
                       for a in p.authors):
                    verified.append(p)
            if verified:
                log.info(f"  [Europe PMC/bioRxiv] author '{author_name}' -> {len(verified)} papers")
                return verified[:max_results]
            await asyncio.sleep(1)
        return []


class GoogleScholarFetcher:
    SEARCH_URL        = "https://api.semanticscholar.org/graph/v1/paper/search"
    AUTHOR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/author/search"
    AUTHOR_PAPERS_URL = "https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"

    PAPER_FIELDS  = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,publicationDate"
    AUTHOR_FIELDS = "authorId,name,paperCount"

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "ResearchBot/1.0 (academic use)"}
            )
        return self._session

    @staticmethod
    def _item_to_paper(item: dict) -> Paper | None:
        title    = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title:
            return None
        authors  = [a.get("name", "").strip() for a in item.get("authors", []) if a.get("name")]
        year     = str(item.get("year") or item.get("publicationDate", "")[:4] if item.get("publicationDate") else "")
        pid      = item.get("paperId", title[:30].replace(" ", "_"))

        ext      = item.get("externalIds") or {}
        doi      = ext.get("DOI", "")
        url      = (f"https://doi.org/{doi}" if doi
                    else f"https://www.semanticscholar.org/paper/{pid}")
        pdf_info = item.get("openAccessPdf") or {}
        pdf_url  = pdf_info.get("url", "")

        return Paper(
            arxiv_id   = f"s2_{pid}",
            title      = title,
            abstract   = abstract,
            published  = year,
            authors    = authors if authors else ["Unknown"],
            url        = url,
            pdf_url    = pdf_url,
            categories = ["Semantic Scholar"],
            source     = "Semantic Scholar",
        )

    async def _s2_search(self, query: str, year_start: int,
                         max_results: int = 10) -> list:
        session = await self._get_session()
        params  = {
            "query":  query,
            "fields": self.PAPER_FIELDS,
            "limit":  min(max_results, 100),
            "year":   f"{year_start}-",
        }
        papers = []
        try:
            async with session.get(
                self.SEARCH_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status == 429:
                    log.warning("Semantic Scholar rate-limited (429) — skipping query.")
                    return []
                if r.status != 200:
                    log.warning(f"Semantic Scholar status {r.status} for: {query[:60]}")
                    return []
                data = await r.json(content_type=None)
                for item in data.get("data", []):
                    p = self._item_to_paper(item)
                    if p:
                        papers.append(p)
        except asyncio.TimeoutError:
            log.warning(f"Semantic Scholar timeout for: {query[:50]}")
        except Exception as e:
            log.error(f"Semantic Scholar search error: {e}")
        return papers

    async def search(self, queries: list, days_back: int,
                     max_results: int = 5) -> list:
        since_year = (datetime.now(timezone.utc) - timedelta(days=days_back)).year
        seen, all_papers = set(), []

        for query in queries:
            log.info(f"  [Semantic Scholar] '{query[:60]}'")
            papers = await self._s2_search(query, since_year, max_results=max_results * 2)
            log.info(f"    {len(papers)} results")
            for p in papers:
                if p.arxiv_id not in seen and p.title:
                    seen.add(p.arxiv_id)
                    all_papers.append(p)
            await asyncio.sleep(1.5)

        return all_papers[:max_results]

    async def search_by_author(self, author_name: str, days_back: int,
                               max_results: int = 10) -> list:
        since_year = (datetime.now(timezone.utc) - timedelta(days=days_back)).year
        session    = await self._get_session()

        author_id = None
        try:
            async with session.get(
                self.AUTHOR_SEARCH_URL,
                params={"query": author_name, "fields": self.AUTHOR_FIELDS, "limit": 5},
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    for candidate in data.get("data", []):
                        name = candidate.get("name", "")
                        last = author_name.strip().split()[-1].lower()
                        if author_name.lower() in name.lower() or last in name.lower():
                            author_id = candidate.get("authorId")
                            log.info(f"  [S2 author] found '{name}' id={author_id}")
                            break
        except Exception as e:
            log.error(f"S2 author search error: {e}")

        if not author_id:
            log.warning(f"  [S2 author] no profile found for '{author_name}'")
            return []

        await asyncio.sleep(1)
        papers = []
        try:
            url = self.AUTHOR_PAPERS_URL.format(author_id=author_id)
            async with session.get(
                url,
                params={
                    "fields": self.PAPER_FIELDS,
                    "limit":  min(max_results * 3, 100),
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    for item in data.get("data", []):
                        p = self._item_to_paper(item)
                        if p:
                            year_str = p.published
                            year_ok  = (not year_str or
                                        not year_str.isdigit() or
                                        int(year_str) >= since_year)
                            if year_ok:
                                papers.append(p)
                else:
                    log.warning(f"S2 author papers status {r.status}")
        except Exception as e:
            log.error(f"S2 author papers error: {e}")

        log.info(f"  [S2 author] '{author_name}' -> {len(papers)} papers since {since_year}")
        return papers[:max_results]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


def generate_lecture_notes(paper: Paper) -> str:
    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += f" et al. ({len(paper.authors)} total)"

    messages = [
        {"role": "system", "content": "You are an expert biomedical AI researcher and educator."},
        {"role": "user", "content": (
            f"Generate comprehensive, detailed lecture notes for a graduate-level "
            f"audience based on this research paper.\n\n"
            f"Title: {paper.title}\n"
            f"Authors: {authors_str}\n"
            f"Published: {paper.published}\n"
            f"Source: {paper.source}\n"
            f"Categories: {', '.join(paper.categories[:4])}\n\n"
            f"Abstract:\n{paper.abstract}\n\n"
            f"Write detailed lecture notes covering ALL of these sections:\n\n"
            f"1. CORE CONTRIBUTION\n"
            f"What is the primary scientific contribution and why does it matter?\n\n"
            f"2. BIOLOGICAL CONTEXT AND MOTIVATION\n"
            f"Explain the biological problem, disease context, and clinical need.\n\n"
            f"3. METHODOLOGY DEEP-DIVE\n"
            f"Detail the model architecture, algorithms, and technical innovations "
            f"with mathematical intuitions where relevant.\n\n"
            f"4. KEY RESULTS AND FINDINGS\n"
            f"Experimental results, datasets used, benchmarks, and performance metrics.\n\n"
            f"5. CRITICAL ANALYSIS\n"
            f"Strengths, limitations, potential biases, and open questions.\n\n"
            f"6. IMPACT AND FUTURE DIRECTIONS\n"
            f"Clinical and translational implications, how this advances the field.\n\n"
            f"7. KEY TAKEAWAYS\n"
            f"3-5 bullet points of the most important concepts to remember.\n\n"
            f"Be thorough, technically precise, and pedagogically clear."
        )},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            f"<|system|>You are an expert biomedical AI researcher and educator.<|end|>\n"
            f"<|user|>{messages[1]['content']}<|end|>\n<|assistant|>"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5048,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        log.error("GPU OOM - clearing cache, using fallback.")
        return _fallback_notes(paper)
    except Exception as ex:
        log.error(f"Generation error: {ex}")
        return _fallback_notes(paper)


def _fallback_notes(paper: Paper) -> str:
    sents   = [s.strip() for s in paper.abstract.split(". ") if s.strip()]
    intro   = ". ".join(sents[:2]) + "."  if len(sents) >= 2 else paper.abstract
    methods = ". ".join(sents[2:5]) + "." if len(sents) >  4 else "See full abstract."
    results = ". ".join(sents[5:])  + "." if len(sents) >  5 else "See full abstract."
    return (
        f"*1. CORE CONTRIBUTION*\n{intro}\n\n"
        f"*2. BIOLOGICAL CONTEXT*\n"
        f"Addresses a key challenge in computational biology and precision medicine.\n\n"
        f"*3. METHODOLOGY*\n{methods}\n\n"
        f"*4. KEY RESULTS*\n{results}\n\n"
        f"*5. CRITICAL ANALYSIS*\n"
        f"Strengths: Novel methodological contribution with clear biological motivation.\n"
        f"Limitations: Independent cohort validation needed.\n\n"
        f"*6. FUTURE DIRECTIONS*\n"
        f"Patient stratification, biomarker discovery, and clinical translation.\n\n"
        f"*7. KEY TAKEAWAYS*\n"
        f"- Multi-modal data integration requires careful architectural design\n"
        f"- ML reveals non-linear patterns invisible to traditional statistics\n"
        f"- Interpretability and explainability remain active frontiers\n"
        f"- Reproducibility depends on open data and code sharing\n\n"
        f"_(Fallback notes - GPU generation failed)_"
    )


def blk_header(text):
    return {"type": "header",
            "text": {"type": "plain_text", "text": text[:150], "emoji": True}}

def blk_section(text):
    return {"type": "section",
            "text": {"type": "mrkdwn", "text": text[:2900]}}

def blk_divider():
    return {"type": "divider"}

def blk_context(text):
    return {"type": "context",
            "elements": [{"type": "mrkdwn", "text": text[:2900]}]}

def chunk_text(text, max_len=2800):
    if len(text) <= max_len:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > max_len:
            if cur:
                chunks.append(cur.strip())
            cur = line + "\n"
        else:
            cur += line + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text[:max_len]]

async def slack_post(client, text, blocks):
    for i in range(0, max(len(blocks), 1), 50):
        await client.chat_postMessage(
            channel=SLACK_CHANNEL,
            text=text,
            blocks=blocks[i:i + 50] or None,
            unfurl_links=False,
        )
        await asyncio.sleep(0.5)

async def post_paper_with_notes(client, paper, idx, total, generate_notes=True):
    source_badge = paper.source

    if generate_notes:
        print(f"    [{idx}/{total}] {source_badge}  {paper.title[:200]}...")
        paper.lecture_notes = generate_lecture_notes(paper)
        if device == "cuda":
            torch.cuda.empty_cache()

    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += f" +{len(paper.authors)-3} more"
    abstract_preview = (paper.abstract[:450] + "..."
                        if len(paper.abstract) > 450 else paper.abstract)

    link_parts = [f"<{paper.url}|View Paper>"]
    if paper.pdf_url:
        link_parts.append(f"<{paper.pdf_url}|Download PDF>")

    await slack_post(client,
        text=paper.title,
        blocks=[
            blk_divider(),
            blk_header(f"Paper {idx} of {total}  [{paper.source}]"),
            blk_section(
                f"*<{paper.url}|{paper.title}>*\n\n"
                f"*Authors:* {authors_str}\n"
                f"*Published:* {paper.published}   "
                f"*Source:* {source_badge}   "
                f"*Categories:* {', '.join(paper.categories[:3])}"
            ),
            blk_section(f"*Abstract:*\n{abstract_preview}"),
            blk_section("  |  ".join(link_parts)),
        ],
    )

    if generate_notes and paper.lecture_notes:
        note_blocks = [blk_header(f"Notes: {paper.title[:200]}")]
        for chunk in chunk_text(paper.lecture_notes, 5000):
            note_blocks.append(blk_section(chunk))
        await slack_post(client, text="Notes", blocks=note_blocks)

    print(f"    Posted.")
    await asyncio.sleep(2)


async def post_scholar_paper(client, paper, idx, total):
    print(f"    [{idx}/{total}] Semantic Scholar  {paper.title[:200]}...")

    paper.lecture_notes = generate_lecture_notes(paper)
    if device == "cuda":
        torch.cuda.empty_cache()

    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += f" +{len(paper.authors)-3} more"
    abstract_preview = (paper.abstract[:450] + "..."
                        if len(paper.abstract) > 450 else paper.abstract) or "_No abstract available._"

    url_text = f"<{paper.url}|View Paper>" if paper.url else "_No URL available_"

    await slack_post(client,
        text=paper.title,
        blocks=[
            blk_divider(),
            blk_header(f"Scholar {idx} of {total}  [Semantic Scholar]"),
            blk_section(
                f"*{paper.title}*\n\n"
                f"*Authors:* {authors_str}\n"
                f"*Year:* {paper.published or 'N/A'}   *Source:* Semantic Scholar"
            ),
            blk_section(f"*Abstract:*\n{abstract_preview}"),
            blk_section(url_text),
        ],
    )

    if paper.lecture_notes:
        note_blocks = [blk_header(f"Notes: {paper.title[:200]}")]
        for chunk in chunk_text(paper.lecture_notes, 5000):
            note_blocks.append(blk_section(chunk))
        await slack_post(client, text="Notes", blocks=note_blocks)

    print(f"    Posted (with LLM notes).")
    await asyncio.sleep(2)


async def run_author_section(client, arxiv_fetcher, biorxiv_fetcher, scholar_fetcher):
    print(f"  Author Tracking")

    await slack_post(client,
        text="Author Tracking",
        blocks=[
            blk_divider(),
            blk_header("Author Tracking"),
            blk_section(
                f"Recent papers by tracked authors "
                f"(arXiv/bioRxiv: last {DAYS_LOOKBACK_AUTHOR} days, "
                f"Semantic Scholar: last {DAYS_LOOKBACK_AUTHOR} days):\n"
                + "\n".join(f"  • {a}" for a in TRACKED_AUTHORS)
            ),
        ],
    )

    total_author_papers = 0

    for author in TRACKED_AUTHORS:
        print(f"\n  Searching for: {author}")

        await slack_post(client,
            text=f"Papers by {author}",
            blocks=[blk_section(f"*Papers by {author}*")],
        )

        seen_ids, preprint_papers = set(), []

        print(f"    arXiv...")
        arxiv_papers = await arxiv_fetcher.search_by_author(
            author, DAYS_LOOKBACK_AUTHOR, max_results=MAX_AUTHOR_PAPERS
        )
        for p in arxiv_papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                preprint_papers.append(p)
        await asyncio.sleep(2)

        print(f"    bioRxiv (Europe PMC)...")
        biorxiv_papers = await biorxiv_fetcher.search_by_author(
            author, DAYS_LOOKBACK_AUTHOR, max_results=MAX_AUTHOR_PAPERS
        )
        for p in biorxiv_papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                preprint_papers.append(p)

        print(f"    Found {len(preprint_papers)} preprint paper(s) "
              f"({len(arxiv_papers)} arXiv, {len(biorxiv_papers)} bioRxiv)")

        if preprint_papers:
            await slack_post(client,
                text=f"arXiv / bioRxiv papers by {author}",
                blocks=[blk_section(
                    f"*arXiv / bioRxiv papers by {author}* "
                    f"({len(preprint_papers)} found — LLM notes included)"
                )],
            )
            for idx, paper in enumerate(preprint_papers[:MAX_AUTHOR_PAPERS], 1):
                for tracked in TRACKED_AUTHORS:
                    paper.authors = [
                        f"*{a}*" if tracked.lower() in a.lower() else a
                        for a in paper.authors
                    ]
                print(f"    Generating LLM notes for author paper [{idx}]...")
                await post_paper_with_notes(client, paper, idx, len(preprint_papers), generate_notes=True)
                total_author_papers += 1
        else:
            await slack_post(client,
                text="No recent preprint papers found.",
                blocks=[blk_section(
                    f"_No papers found for *{author}* "
                    f"on arXiv or bioRxiv in the last {DAYS_LOOKBACK_AUTHOR} days._\n"
                    f"_(Note: arXiv author search may miss papers if the name format varies)_"
                )],
            )

        print(f"    Semantic Scholar (author)...")
        scholar_papers = await scholar_fetcher.search_by_author(
            author, DAYS_LOOKBACK_AUTHOR, max_results=MAX_AUTHOR_PAPERS
        )
        scholar_unique = []
        preprint_titles = {p.title.lower()[:60] for p in preprint_papers}
        for p in scholar_papers:
            if p.arxiv_id not in seen_ids and p.title.lower()[:60] not in preprint_titles:
                seen_ids.add(p.arxiv_id)
                scholar_unique.append(p)

        print(f"    Found {len(scholar_unique)} additional Semantic Scholar paper(s)")

        if scholar_unique:
            await slack_post(client,
                text=f"Semantic Scholar papers by {author}",
                blocks=[blk_section(
                    f"*Semantic Scholar papers by {author}* "
                    f"({len(scholar_unique)} found — with LLM notes)"
                )],
            )
            for idx, paper in enumerate(scholar_unique, 1):
                await post_scholar_paper(client, paper, idx, len(scholar_unique))
                total_author_papers += 1
        else:
            await slack_post(client,
                text="No additional Semantic Scholar papers found.",
                blocks=[blk_section(
                    f"_No additional Semantic Scholar papers found for *{author}*._"
                )],
            )

    return total_author_papers


async def run_digest(client):
    now   = datetime.now(ZoneInfo(TIMEZONE))
    since = (now - timedelta(days=DAYS_LOOKBACK)).strftime("%B %d, %Y")
    today = now.strftime("%A, %B %d, %Y  %H:%M Istanbul")

    print(f"  Starting digest - last {DAYS_LOOKBACK} days (since {since})")

    await slack_post(client,
        text="Research Digest starting...",
        blocks=[
            blk_header("Research Digest - Multi-Omics & ML"),
            blk_section(
                f"*Date:* {today}\n"
                f"*Topic Coverage:* Last {DAYS_LOOKBACK} days (since {since})\n"
                f"*Author Coverage:* Last {DAYS_LOOKBACK_AUTHOR} days\n"
                f"*Sources:* arXiv + bioRxiv (Europe PMC) + Semantic Scholar\n\n"
                f"*Topics:*\n"
                f"  - Multi-Omics Integration with GNNs\n"
                f"  - Cancer Biomarker Discovery\n"
                f"  - Alzheimer's Disease Progression\n"
                f"  - Spatial Transcriptomics from Histology\n\n"
                f"*Tracked Authors:*\n"
                + "\n".join(f"  - {a}" for a in TRACKED_AUTHORS) + "\n\n"
                f"_Running `{LLM_MODEL}` locally on {device.upper()}..._\n"
                f"_All sources (arXiv, bioRxiv, Semantic Scholar) include LLM notes._"
            ),
        ],
    )

    total_topic   = 0
    total_arxiv   = 0
    total_biorxiv = 0
    total_scholar = 0

    scholar_fetcher = GoogleScholarFetcher()

    async with aiohttp.ClientSession() as session:
        arxiv_fetcher   = ArxivFetcher(session)
        biorxiv_fetcher = BiorxivFetcher(session)

        for topic in TOPICS:
            print(f"\n  Topic: {topic['name']}")

            await slack_post(client,
                text=topic["name"],
                blocks=[blk_divider(), blk_header(topic["name"])],
            )
            await asyncio.sleep(1)

            seen_ids, arxiv_candidates = set(), []
            for query in topic["arxiv_queries"]:
                print(f"  [arXiv] {query[:70]}")
                for p in await arxiv_fetcher.search(query, DAYS_LOOKBACK, max_results=15):
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        arxiv_candidates.append(p)
                await asyncio.sleep(3)

            arxiv_papers = [
                p for p in arxiv_candidates
                if is_relevant(p, topic["must_contain_any"], topic["must_also_contain_any"])
            ][:MAX_PAPERS_PER_TOPIC]
            print(f"  arXiv: {len(arxiv_candidates)} fetched -> {len(arxiv_papers)} relevant")

            biorxiv_candidates = []
            for term in topic["biorxiv_terms"]:
                print(f"  [bioRxiv/EuropePMC] {term}")
                for p in await biorxiv_fetcher.search([term], DAYS_LOOKBACK, max_results=15):
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        biorxiv_candidates.append(p)
                await asyncio.sleep(2)

            biorxiv_papers = [
                p for p in biorxiv_candidates
                if is_relevant(p, topic["must_contain_any"], topic["must_also_contain_any"])
            ][:MAX_PAPERS_PER_TOPIC]
            print(f"  bioRxiv: {len(biorxiv_candidates)} fetched -> {len(biorxiv_papers)} relevant")

            all_preprint_papers = arxiv_papers + biorxiv_papers

            if not all_preprint_papers:
                await slack_post(client,
                    text="No relevant preprint papers found.",
                    blocks=[blk_section(
                        f"_No relevant arXiv/bioRxiv papers for *{topic['name']}* "
                        f"in the last {DAYS_LOOKBACK} days._"
                    )],
                )
            else:
                source_parts = []
                if arxiv_papers:   source_parts.append(f"{len(arxiv_papers)} arXiv")
                if biorxiv_papers: source_parts.append(f"{len(biorxiv_papers)} bioRxiv")
                await slack_post(client,
                    text=f"Found {len(all_preprint_papers)} preprint papers",
                    blocks=[blk_context(
                        f"Found {len(all_preprint_papers)} preprint paper(s): "
                        f"{', '.join(source_parts)} — with LLM notes"
                    )],
                )
                for idx, paper in enumerate(all_preprint_papers, 1):
                    await post_paper_with_notes(client, paper, idx,
                                                len(all_preprint_papers), generate_notes=True)
                    total_topic += 1
                    if paper.source == "arXiv":  total_arxiv   += 1
                    else:                         total_biorxiv += 1

            print(f"  [Semantic Scholar] searching topic...")
            scholar_papers_raw = await scholar_fetcher.search(
                topic.get("scholar_queries", [topic["name"]]),
                DAYS_LOOKBACK,
                max_results=MAX_SCHOLAR_PAPERS,
            )

            preprint_titles = {p.title.lower()[:200] for p in all_preprint_papers}
            scholar_papers  = [
                p for p in scholar_papers_raw
                if p.title.lower()[:200] not in preprint_titles
            ][:MAX_SCHOLAR_PAPERS]
            print(f"  Google Scholar: {len(scholar_papers_raw)} fetched "
                  f"{len(scholar_papers)} unique")

            if scholar_papers:
                await slack_post(client,
                    text=f"Semantic Scholar results for {topic['name']}",
                    blocks=[
                        blk_context(
                            f"*Semantic Scholar:* {len(scholar_papers)} additional paper(s) "
                            f"— with LLM notes"
                        )
                    ],
                )
                for idx, paper in enumerate(scholar_papers, 1):
                    await post_scholar_paper(client, paper, idx, len(scholar_papers))
                    total_scholar += 1
            else:
                await slack_post(client,
                    text="No additional Semantic Scholar results.",
                    blocks=[blk_context(
                        "_No additional Semantic Scholar results for this topic._"
                    )],
                )

        total_author = await run_author_section(
            client, arxiv_fetcher, biorxiv_fetcher, scholar_fetcher
        )

    await scholar_fetcher.close()

    tz  = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    nxt = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now >= nxt:
        nxt += timedelta(days=1)
    h, rem = divmod(int((nxt - now).total_seconds()), 3600)
    m = rem // 60

    grand_total = total_topic + total_scholar + total_author

    await slack_post(client,
        text=f"Done - {grand_total} papers processed.",
        blocks=[
            blk_divider(),
            blk_header("Digest Complete"),
            blk_section(
                f"*Topic preprints (with LLM notes):* {total_topic}  "
                f"(arXiv: {total_arxiv}, bioRxiv: {total_biorxiv})\n"
                f"*Topic Semantic Scholar (with LLM notes):* {total_scholar}\n"
                f"*Author papers:* {total_author}  "
                f"({', '.join(TRACKED_AUTHORS)})\n\n"
                f"Next automatic digest in *{h}h {m}m* "
                f"({RUN_HOUR:02d}:{RUN_MINUTE:02d} America/Chicago)."
            ),
            blk_context(
                f"ResearchBot  |  arXiv + bioRxiv (Europe PMC) + Semantic Scholar  |  "
                f"Local LLM: {LLM_MODEL}  |  Device: {device.upper()}"
            ),
        ],
    )

    print(f"\n  Done")
    print(f"     Topic preprints (LLM): {total_topic}  ({total_arxiv} arXiv, {total_biorxiv} bioRxiv)")
    print(f"     Topic Semantic Scholar (with LLM): {total_scholar}")
    print(f"     Author papers: {total_author}")
    print(f"     Next digest  : {nxt.strftime('%Y-%m-%d %H:%M %Z')} (in {h}h {m}m)")


async def scheduler(client):
    while True:
        tz  = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)
        nxt = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
        if now >= nxt:
            nxt += timedelta(days=1)
        wait = (nxt - now).total_seconds()
        h, rem = divmod(int(wait), 3600)
        m = rem // 60
        print(f"Next digest: {nxt.strftime('%Y-%m-%d %H:%M %Z')}  (in {h}h {m}m)")
        await asyncio.sleep(wait)
        try:
            await run_digest(client)
        except Exception as e:
            log.error(f"Scheduled digest error: {e}", exc_info=True)


async def main():
    client = AsyncWebClient(token=SLACK_BOT_TOKEN)
    try:
        auth = await client.auth_test()
        print(f"Slack connected: @{auth['user']} in '{auth['team']}'\n")
    except Exception as e:
        print(f"Slack auth failed: {e}")
        return

    await run_digest(client)
    await scheduler(client)


asyncio.run(main())
