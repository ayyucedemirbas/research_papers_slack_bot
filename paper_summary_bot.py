import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "slack_sdk", "aiohttp", "nest_asyncio",
    "transformers", "accelerate", "bitsandbytes",
    "sentencepiece", "protobuf"])

import getpass
SLACK_BOT_TOKEN = getpass.getpass("1  Slack Bot Token  (xoxb-...): ").strip()
SLACK_CHANNEL   = input(          "2  Slack Channel ID (C...):     ").strip()
print()

TIMEZONE             = "Europe/Istanbul"
RUN_HOUR             = 7
RUN_MINUTE           = 0
DAYS_LOOKBACK        = 120
DAYS_LOOKBACK_AUTHOR = 300
MAX_PAPERS_PER_TOPIC = 5
MAX_AUTHOR_PAPERS    = 10

TRACKED_AUTHORS = [
    "Serdar Bozdag",
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

TOPICS = [
    {
        "name": "Multi-Omics Integration with GNNs",
        "arxiv_queries": [
            'abs:"multi-omics" OR abs:"multi-modal omics"',
            'abs:"graph neural network" OR abs:"GNN" OR abs:"graph convolutional"',
            'ti:"omics integration" OR ti:"multi-omics"',
        ],
        "biorxiv_terms": ["multi-omics graph neural", "GNN omics integration",
                          "multi-omics GNN", "omics graph convolutional"],
        "must_contain_any": ["multi-omics", "multiomics", "multi-modal omics",
                             "omics integration", "genomics", "transcriptomics",
                             "proteomics", "metabolomics"],
        "must_also_contain_any": ["graph neural", "gnn", "graph convolutional",
                                  "graph attention", "graph transformer"],
    },
    {
        "name": "Cancer Biomarker Discovery — Multi-Omics ML",
        "arxiv_queries": [
            'abs:"multi-omics" OR abs:"multi-modal omics"',
            'abs:"cancer" OR abs:"tumor" OR abs:"oncology"',
            'abs:"biomarker" OR abs:"prognosis" OR abs:"survival prediction"',
        ],
        "biorxiv_terms": ["multi-omics cancer biomarker", "cancer multi-omics prognosis",
                          "pan-cancer omics", "multi-omics survival"],
        "must_contain_any": ["multi-omics", "multiomics", "omics integration",
                             "genomics", "transcriptomics", "proteomics"],
        "must_also_contain_any": ["cancer", "tumor", "tumour", "oncology",
                                  "carcinoma", "leukemia", "glioma"],
    },
    {
        "name": "Alzheimer's Disease Progression — Multi-Omics ML",
        "arxiv_queries": [
            "abs:\"Alzheimer\" OR abs:\"Alzheimer's disease\" OR ti:\"Alzheimer\"",
            'abs:"omics" OR abs:"genomics" OR abs:"proteomics" OR abs:"transcriptomics"',
            'abs:"machine learning" OR abs:"deep learning" OR abs:"neural network"',
        ],
        "biorxiv_terms": ["Alzheimer multi-omics", "Alzheimer genomics deep learning",
                          "Alzheimer proteomics machine learning",
                          "Alzheimer transcriptomics"],
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
        "biorxiv_terms": ["spatial transcriptomics histopathology",
                          "spatial transcriptomics deep learning",
                          "spatial gene expression histology",
                          "visium deep learning"],
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
    notes: str = ""

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
                        # Filter: at least one author must match the full name
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
    BASE_URL = "https://api.biorxiv.org/details/biorxiv/{start}/{end}/0/json"

    def __init__(self, session):
        self.session = session

    async def _fetch_window(self, days_back):
        end_date   = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        url        = self.BASE_URL.format(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        all_items = []
        cursor    = 0
        while True:
            paged_url = url.replace("/0/json", f"/{cursor}/json")
            try:
                async with self.session.get(paged_url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status != 200:
                        break
                    data  = await r.json(content_type=None)
                    items = data.get("collection", [])
                    if not items:
                        break
                    all_items.extend(items)
                    total = int(data.get("messages", [{}])[0].get("total", 0))
                    cursor += len(items)
                    if cursor >= total or cursor >= 200:   # cap at 200 to stay fast
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                log.error(f"bioRxiv pagination error: {e}")
                break
        return all_items

    def _item_to_paper(self, item):
        doi      = item.get("doi", "")
        bxid     = doi.replace("/", "_") if doi else item.get("title", "")[:30].replace(" ", "_")
        authors  = [a.strip() for a in re.split(r"[;,]", item.get("authors", "")) if a.strip()]
        category = item.get("category", "")
        return Paper(
            arxiv_id   = f"biorxiv_{bxid}",
            title      = item.get("title", "").strip(),
            abstract   = item.get("abstract", "").strip(),
            published  = item.get("date", ""),
            authors    = authors,
            url        = f"https://doi.org/{doi}" if doi else "https://www.biorxiv.org",
            pdf_url    = f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi else "",
            categories = [category] if category else ["bioRxiv"],
            source     = "bioRxiv",
        )

    async def search(self, terms: list, days_back: int, max_results: int = 15) -> list:
        items   = await self._fetch_window(days_back)
        papers  = []
        seen    = set()
        for item in items:
            text = (item.get("title", "") + " " + item.get("abstract", "")).lower()
            if any(t.lower() in text for t in terms):
                p = self._item_to_paper(item)
                if p.arxiv_id not in seen:
                    seen.add(p.arxiv_id)
                    papers.append(p)
            if len(papers) >= max_results:
                break
        return papers

    async def search_by_author(self, author_name: str, days_back: int,
                               max_results: int = 20) -> list:
        items   = await self._fetch_window(days_back)
        papers  = []
        seen    = set()
        name_lc = author_name.lower()
        last    = author_name.strip().split()[-1].lower()
        for item in items:
            authors_raw = item.get("authors", "").lower()
            if name_lc in authors_raw or last in authors_raw:
                p = self._item_to_paper(item)
                if p.arxiv_id not in seen:
                    seen.add(p.arxiv_id)
                    papers.append(p)
            if len(papers) >= max_results:
                break
        return papers


def generate_notes(paper: Paper) -> str:
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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
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
        log.error("GPU OOM — clearing cache, using fallback.")
        return _fallback_notes(paper)
    except Exception as ex:
        log.error(f"Generation error: {ex}")
        return _fallback_notes(paper)


def _fallback_notes(paper: Paper):
    pass


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
    source_badge = "arXiv" if paper.source == "arXiv" else "bioRxiv"

    if generate_notes:
        print(f"    [{idx}/{total}] {source_badge}  {paper.title[:60]}...")
        print(f"    Generating notes on GPU...")
        paper.notes = generate_notes(paper)
        if device == "cuda":
            torch.cuda.empty_cache()

    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += f" +{len(paper.authors)-3} more"
    abstract_preview = (paper.abstract[:450] + "..."
                        if len(paper.abstract) > 450 else paper.abstract)

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
            blk_section(f"<{paper.url}|View Paper>  |  <{paper.pdf_url}|Download PDF>"),
        ],
    )

    if generate_notes and paper.notes:
        note_blocks = [blk_header(f"Notes for {paper.title[:80]}")]
        for chunk in chunk_text(paper.notes, 2800):
            note_blocks.append(blk_section(chunk))
        await slack_post(client, text="Notes", blocks=note_blocks)

    print(f"Posted.")
    await asyncio.sleep(2)


async def run_author_section(client, arxiv_fetcher, biorxiv_fetcher):
    await slack_post(client,
        text="Author Tracking",
        blocks=[
            blk_divider(),
            blk_header("Author Tracking"),
            blk_section(
                f"Recent papers by tracked authors "
                f"(last {DAYS_LOOKBACK_AUTHOR} days):\n"
                + "\n".join(f"  • {a}" for a in TRACKED_AUTHORS)
            ),
        ],
    )

    total_author_papers = 0

    for author in TRACKED_AUTHORS:
        print(f"\n  Searching for: {author}")

        await slack_post(client,
            text=f"Papers by {author}",
            blocks=[blk_section(f"*🔎 Papers by {author}*")],
        )

        seen_ids = set()
        all_papers = []
        print(f"    arXiv...")
        arxiv_papers = await arxiv_fetcher.search_by_author(
            author, DAYS_LOOKBACK_AUTHOR, max_results=MAX_AUTHOR_PAPERS
        )
        for p in arxiv_papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                all_papers.append(p)
        await asyncio.sleep(2)

        print(f"    bioRxiv...")
        biorxiv_papers = await biorxiv_fetcher.search_by_author(
            author, DAYS_LOOKBACK_AUTHOR, max_results=MAX_AUTHOR_PAPERS
        )
        for p in biorxiv_papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                all_papers.append(p)

        print(f"    Found {len(all_papers)} paper(s) "
              f"({len(arxiv_papers)} arXiv, {len(biorxiv_papers)} bioRxiv)")

        if not all_papers:
            await slack_post(client,
                text="No recent papers found.",
                blocks=[blk_section(
                    f"_No papers found for *{author}* "
                    f"on arXiv or bioRxiv in the last {DAYS_LOOKBACK_AUTHOR} days._\n"
                    f"_(Note: arXiv author search may miss papers if the name format varies)_"
                )],
            )
            continue

        for idx, paper in enumerate(all_papers[:MAX_AUTHOR_PAPERS], 1):
            source_badge = "arXiv" if paper.source == "arXiv" else "bioRxiv"
            authors_str  = ", ".join(paper.authors[:4])
            if len(paper.authors) > 4:
                authors_str += f" +{len(paper.authors)-4} more"
            abstract_preview = (paper.abstract[:350] + "..."
                                if len(paper.abstract) > 350 else paper.abstract)

            for tracked in TRACKED_AUTHORS:
                authors_str = authors_str.replace(
                    tracked, f"*{tracked}*"
                )

            await slack_post(client,
                text=paper.title,
                blocks=[
                    blk_divider(),
                    blk_header(f"{source_badge}  Paper {idx} of {len(all_papers)}"),
                    blk_section(
                        f"*<{paper.url}|{paper.title}>*\n\n"
                        f"*Authors:* {authors_str}\n"
                        f"*Published:* {paper.published}   "
                        f"*Categories:* {', '.join(paper.categories[:3])}"
                    ),
                    blk_section(f"*Abstract:*\n{abstract_preview}"),
                    blk_section(f"<{paper.url}|View Paper>  |  <{paper.pdf_url}|Download PDF>"),
                ],
            )
            total_author_papers += 1
            await asyncio.sleep(1)

    return total_author_papers


async def run_digest(client):
    now   = datetime.now(ZoneInfo(TIMEZONE))
    since = (now - timedelta(days=DAYS_LOOKBACK)).strftime("%B %d, %Y")
    today = now.strftime("%A, %B %d, %Y  %H:%M Istanbul")


    await slack_post(client,
        text="Starting...",
        blocks=[
            blk_header("Research Digest — Multi-Omics & ML"),
            blk_section(
                f"*Date:* {today}\n"
                f"*Topic Coverage:* Last {DAYS_LOOKBACK} days (since {since})\n"
                f"*Author Coverage:* Last {DAYS_LOOKBACK_AUTHOR} days\n"
                f"*Sources:* arXiv + bioRxiv\n\n"
                f"*Topics:*\n"
                f"  - Multi-Omics Integration with GNNs\n"
                f"  - Cancer Biomarker Discovery\n"
                f"  - Alzheimer's Disease Progression\n"
                f"  - Spatial Transcriptomics from Histology\n\n"
                f"*Tracked Authors:*\n"
                + "\n".join(f"  - {a}" for a in TRACKED_AUTHORS) + "\n\n"
                f"_Running `{LLM_MODEL}` locally on {device.upper()}..._"
            ),
        ],
    )

    total_topic   = 0
    total_arxiv   = 0
    total_biorxiv = 0

    async with aiohttp.ClientSession() as session:
        arxiv_fetcher   = ArxivFetcher(session)
        biorxiv_fetcher = BiorxivFetcher(session)

        for topic in TOPICS:
            print(f"  {topic['name']}")

            await slack_post(client,
                text=topic["name"],
                blocks=[blk_divider(), blk_header(topic["name"])],
            )
            await asyncio.sleep(1)

            seen_ids, arxiv_candidates = set(), []
            for query in topic["arxiv_queries"]:
                print(f"  [arXiv] {query}")
                for p in await arxiv_fetcher.search(query, DAYS_LOOKBACK, max_results=15):
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        arxiv_candidates.append(p)
                await asyncio.sleep(3)

            arxiv_papers = [
                p for p in arxiv_candidates
                if is_relevant(p, topic["must_contain_any"], topic["must_also_contain_any"])
            ][:MAX_PAPERS_PER_TOPIC]
            print(f"  arXiv: {len(arxiv_candidates)} fetched → {len(arxiv_papers)} relevant")

            biorxiv_candidates = []
            for term in topic["biorxiv_terms"]:
                print(f"  [bioRxiv] {term}")
                for p in await biorxiv_fetcher.search([term], DAYS_LOOKBACK, max_results=15):
                    if p.arxiv_id not in seen_ids:
                        seen_ids.add(p.arxiv_id)
                        biorxiv_candidates.append(p)
                await asyncio.sleep(2)

            biorxiv_papers = [
                p for p in biorxiv_candidates
                if is_relevant(p, topic["must_contain_any"], topic["must_also_contain_any"])
            ][:MAX_PAPERS_PER_TOPIC]
            print(f"  bioRxiv: {len(biorxiv_candidates)} fetched → {len(biorxiv_papers)} relevant")

            all_papers = arxiv_papers + biorxiv_papers

            if not all_papers:
                await slack_post(client,
                    text="No relevant papers found.",
                    blocks=[blk_section(
                        f"_No relevant papers for *{topic['name']}* "
                        f"in the last {DAYS_LOOKBACK} days on arXiv or bioRxiv._"
                    )],
                )
                continue

            source_parts = []
            if arxiv_papers:   source_parts.append(f"{len(arxiv_papers)} arXiv")
            if biorxiv_papers: source_parts.append(f"{len(biorxiv_papers)} bioRxiv")
            await slack_post(client,
                text=f"Found {len(all_papers)} papers",
                blocks=[blk_context(f"Found {len(all_papers)} paper(s): {', '.join(source_parts)}")],
            )

            for idx, paper in enumerate(all_papers, 1):
                await post_paper_with_notes(client, paper, idx, len(all_papers), generate_notes=True)
                total_topic += 1
                if paper.source == "arXiv":   total_arxiv   += 1
                else:                          total_biorxiv += 1

        total_author = await run_author_section(client, arxiv_fetcher, biorxiv_fetcher)

    tz  = ZoneInfo(TIMEZONE)
    now = datetime.now(tz)
    nxt = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now >= nxt:
        nxt += timedelta(days=1)
    h, rem = divmod(int((nxt - now).total_seconds()), 3600)
    m = rem // 60

    await slack_post(client,
        text=f"Done, {total_topic + total_author} papers processed.",
        blocks=[
            blk_divider(),
            blk_header("Complete"),
            blk_section(
                f"*Topic papers:* {total_topic}  "
                f"(arXiv: {total_arxiv}, bioRxiv: {total_biorxiv})\n"
                f"*Author papers:* {total_author}  "
                f"({', '.join(TRACKED_AUTHORS)})\n\n"
                f"Next automatic digest in *{h}h {m}m* "
                f"(19:00 Istanbul time, UTC+3)."
            ),
            blk_context(
                f"ResearchBot  |  arXiv + bioRxiv  |  "
                f"Local LLM: {LLM_MODEL}  |  Device: {device.upper()}"
            ),
        ],
    )

    print(f"     Topic papers : {total_topic}  ({total_arxiv} arXiv, {total_biorxiv} bioRxiv)")
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

    await run_digest(client)      # run immediately
    await scheduler(client)       # then daily at 07:00 Istanbul


asyncio.run(main())
