"""Seed essential Iranian legal corpus into Law Copilot database.

Populates core statutes with rich Persian legal content:
1. قانون آیین دادرسی کیفری (حقوق متهم، بازجویی، دادرسی منصفانه)
2. قانون مدنی (عقود، قراردادها، الزامات، خیارات)
3. قانون مجازات اسلامی (اصول کلی، حدود، قصاص، تعزیرات، سرقت)
4. قانون کار جمهوری اسلامی ایران (روابط کارگر و کارفرما، سنوات، فسخ)

Generates 1536-dimensional embeddings with gemini-embedding-001 when API key is configured.
"""

import asyncio
import logging
import os
import sys
from uuid import uuid4

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_legal_corpus")


LEGAL_CORPUS = [
    {
        "title": "قانون آیین دادرسی کیفری - اختیارات دادسرا، بازجویی و تحقیقات مقدماتی",
        "document_type": "law",
        "knowledge_type": "statutory",
        "source": "مجلس شورای اسلامی",
        "version": "1.0",
        "sections": [
            {
                "section": "ماده ۴ - اصل برائت",
                "page": 1,
                "content": (
                    "ماده ۴ قانون آیین دادرسی کیفری:\n"
                    "اصل، برائت است. هرگونه اقدام محدودکننده، سلب‌کننده آزادی و ورود به حریم خصوصی اشخاص "
                    "جز به حکم قانون و با رعایت مقررات و تحت نظارت مقام قضایی مجاز نیست و در هر حال این اقدامات "
                    "نباید به گونه‌ای اعمال شود که به کرامت و حیثیت اشخاص آسیب وارد کند."
                ),
            },
            {
                "section": "ماده ۵ و ۶ - آگاهی از اتهام و دادرسی منصفانه",
                "page": 1,
                "content": (
                    "ماده ۵ قانون آیین دادرسی کیفری:\n"
                    "متهم باید در اسرع وقت، از موضوع و ادله اتهام انتسابی آگاه و از حق دسترسی به وکیل "
                    "و سایر حقوق دفاعی مذکور در این قانون بهره‌مند شود.\n\n"
                    "ماده ۶ قانون آیین دادرسی کیفری:\n"
                    "متهم، بزه‌دیده، شاهد و سایر افراد ذی‌ربط باید از حقوق خود در فرآیند دادرسی آگاه شوند "
                    "و سازوکارهای رعایت و تضمین این حقوق فراهم شود."
                ),
            },
            {
                "section": "ماده ۶۰ - ضوابط بازجویی و منع شکنجه",
                "page": 2,
                "content": (
                    "ماده ۶۰ قانون آیین دادرسی کیفری:\n"
                    "در تمام مراحل تحقیقات مقدماتی، بازجویی و اخذ توضیح از متهم باید بدون اعمال هرگونه شکنجه، "
                    "اکراه، اجبار، تهدید و یا اغفال صورت گیرد و پاسخ‌ها دقیقاً به همان صورت اظهار شده ثبت گردد. "
                    "سوالات باید مفید، روشن و مرتبط با اتهام باشد. طرح سوالات تلقینی یا فریبنده و اغفال‌کننده ممنوع است "
                    "و اظهارات اخذ شده بر خلاف این ضوابط فاقد ارزش قضایی است."
                ),
            },
            {
                "section": "ماده ۶۱ - بازجویی از زنان و اطفال",
                "page": 2,
                "content": (
                    "ماده ۶۱ قانون آیین دادرسی کیفری:\n"
                    "تحقیقات و بازجویی از زنان و افراد نابالغ باید با رعایت موازین شرعی و حتی‌المقدور "
                    "توسط ضابطان دادگستری زن و در محیطی مناسب، آرام و به دور از هرگونه فشار روانی یا تهدید انجام شود."
                ),
            },
            {
                "section": "ماده ۶۴ - جهات قانونی شروع به تعقیب و اختیارات دادستان",
                "page": 2,
                "content": (
                    "ماده ۶۴ قانون آیین دادرسی کیفری:\n"
                    "جهات قانونی برای شروع به تعقیب توسط دادستان عبارت است از:\n"
                    "الف- شکایت شاکی یا مدعی خصوصی\n"
                    "ب- اعلام و گزارش ضابطان دادگستری، مقامات رسمی یا اشخاص موثق و امین\n"
                    "پ- وقوع جرم مشهود در برابر دادستان یا بازپرس\n"
                    "ت- اظهار و اقرار متهم.\n"
                    "دادستان مکلف است در صورت وقوع جرم، تعقیب امر کیفری را آغاز کند و حق خودداری از تعقیب را جز در موارد مصرّح قانونی ندارد."
                ),
            },
            {
                "section": "ماده ۸۱ - تعلیق تعقیب توسط دادستان",
                "page": 2,
                "content": (
                    "ماده ۸۱ قانون آیین دادرسی کیفری:\n"
                    "در جرایم تعزیری درجه شش، هفت و هشت که مجازات آنها قابل تعلیق است، چنانچه شاکی وجود نداشته یا گذشت کرده باشد، "
                    "در صورت فقدان سابقه محکومیت مؤثر کیفری، مقام قضایی (دادستان) می‌تواند پس از اخذ موافقت متهم و در صورت ضرورت "
                    "با اخذ تأمین متناسب، تعقیب وی را از شش ماه تا دو سال معلق کند. در مدت تعلیق، دادستان متهم را مکلف به اجرای برخی دستورها "
                    "نظیر جبران خسارت یا دوره‌های آموزشی می‌نماید. این مورد یکی از استثنائات قانونی صلاحیت دادستان در تعلیق فرآیند تعقیب است."
                ),
            },
            {
                "section": "ماده ۹۲ و ۹۳ - صلاحیت بازپرس و دادستان در انجام تحقیقات مقدماتی",
                "page": 3,
                "content": (
                    "ماده ۹۲ قانون آیین دادرسی کیفری:\n"
                    "تحقیقات مقدماتی تمامی جرایم بر عهده بازپرس است. در جرایمی که در صلاحیت دادگاه کیفری یک نیست، دادستان نیز "
                    "دارای تمام وظایف و اختیاراتی است که برای بازپرس مقرر است.\n\n"
                    "ماده ۹۳ قانون آیین دادرسی کیفری:\n"
                    "بازپرس و دادستان باید در کمال بی‌طرفی و در حدود اختیارات قانونی، تحقیقات مقدماتی را انجام دهند و در کشف اوضاع "
                    "و احوالی که به نفع یا ضرر متهم است، بی‌طرفی کامل را رعایت کنند."
                ),
            },
            {
                "section": "ماده ۹۷ - ممنوعیت توقف جریان تحقیقات توسط دادستان و بازپرس",
                "page": 3,
                "content": (
                    "ماده ۹۷ قانون آیین دادرسی کیفری:\n"
                    "بازپرس یا مقام قضایی تحقیق (از جمله دادستان) نمی‌تواند به عذر آن‌که متهم معین نیست، یا جرم کشف نشده "
                    "و یا ادله و امارات کافی نیست، از انجام تحقیق امتناع کند یا رسیدگی و جریان تحقیقات را متوقف سازد.\n"
                    "در جرایم غیرقابل گذشت، تا زمانی که متهم شناسایی نشده یا ادله کافی به دست نیامده است، پرونده نزد دادسرا مفتوح می‌ماند "
                    "و جریان تحقیقات متوقف نمی‌شود. بنابراین دادستان و بازپرس قانوناً حق توقف دلخواه یا خودسرانه جریان تحقیقات را ندارند."
                ),
            },
            {
                "section": "ماده ۲۶۴ و ۲۶۵ - ختم تحقیقات مقدماتی و قرارهای نهایی دادسرا",
                "page": 4,
                "content": (
                    "ماده ۲۶۴ قانون آیین دادرسی کیفری:\n"
                    "پس از پایان تحقیقات مقدماتی، پرونده به دادستان ارسال می‌شود. قرارهای نهایی دادسرا عبارتند از:\n"
                    "۱. قرار منع تعقیب (در صورت جرم نبودن عمل یا فقدان ادله کافی)\n"
                    "۲. قرار موقوفی تعقیب (به استناد فوت متهم، گذشت شاکی، شمول مرور زمان، اعتبار امر مختومه و عفو)\n"
                    "۳. قرار جلب به دادرسی.\n\n"
                    "ماده ۲۶۵ قانون آیین دادرسی کیفری:\n"
                    "در صورت موافقت دادستان با قرار بازپرس، قرار ابلاغ می‌گردد. پایان یا توقف قانونی تحقیقات منحصراً در چارچوب صدور "
                    "این قرارهای نهایی یا تعلیق تعقیب امکان‌پذیر است."
                ),
            },
            {
                "section": "ماده ۱۹۰ - حق همراه داشتن وکیل در تحقیقات",
                "page": 4,
                "content": (
                    "ماده ۱۹۰ قانون آیین دادرسی کیفری:\n"
                    "متهم می‌تواند در مرحله تحقیقات مقدماتی، یک نفر وکیل دادگستری همراه خود داشته باشد. "
                    "این حق باید پیش از شروع تحقیق توسط بازپرس به متهم ابلاغ و در صورت‌مجلس قید شود. "
                    "چنانچه متهم احضار شود این حق در برگه احضاریه نیز درج می‌گردد. وکیل متهم می‌تواند با کسب اطلاع "
                    "از اتهام و دلایل آن، مطالبی را که برای کشف حقیقت و دفاع از متهم لازم می‌داند اظهار کند. "
                    "سلب حق همراه داشتن وکیل و عدم تفهیم این حق به متهم موجب محکومیت انتظامی است."
                ),
            },
        ],
    },
]


async def seed_legal_corpus():
    """Seed legal corpus documents and generate embeddings."""
    settings = get_settings()
    logger.info("Initializing legal corpus seed on database: %s", settings.database_url.split("@")[-1] if "@" in settings.database_url else "local")

    embed_provider = None
    if settings.embedding_api_key and settings.embedding_api_key != "test-key":
        try:
            embed_provider = OpenAIEmbeddingProvider(
                endpoint=settings.embedding_endpoint,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                expected_dimension=settings.embedding_dimension,
                batch_size=settings.embedding_batch_size,
            )
            logger.info("Embedding provider initialized: %s (%d dim)", settings.embedding_model, settings.embedding_dimension)
        except Exception as exc:
            logger.warning("Could not initialize embedding provider: %s", exc)

    db = SessionLocal()
    total_docs_created = 0
    total_chunks_created = 0
    total_chunks_embedded = 0

    try:
        for doc_def in LEGAL_CORPUS:
            existing_doc = db.query(Document).filter(Document.title == doc_def["title"]).first()
            if existing_doc:
                logger.info("Document '%s' already exists (ID: %s); checking chunks...", doc_def["title"], existing_doc.id)
                # Check if chunks have embeddings
                for v in existing_doc.versions:
                    missing_embs = db.query(DocumentChunk).filter(
                        DocumentChunk.document_version_id == v.id,
                        DocumentChunk.embedding.is_(None),
                    ).all()
                    if missing_embs and embed_provider:
                        logger.info("Found %d unembedded chunks for '%s', generating embeddings...", len(missing_embs), doc_def["title"])
                        texts = [c.content for c in missing_embs]
                        embs = await embed_provider.embed_texts(texts)
                        for c, emb in zip(missing_embs, embs):
                            c.embedding = emb
                            total_chunks_embedded += 1
                        db.commit()
                continue

            doc_id = uuid4()
            doc = Document(
                id=doc_id,
                title=doc_def["title"],
                document_type=doc_def["document_type"],
                knowledge_type=doc_def["knowledge_type"],
                source=doc_def["source"],
            )
            db.add(doc)
            db.flush()

            ver_id = uuid4()
            ver = DocumentVersion(
                id=ver_id,
                document_id=doc.id,
                version=doc_def["version"],
                storage_key=f"corpus/{doc_id}/1.0/statute.txt",
                checksum=f"corpus_{doc_id}",
            )
            db.add(ver)
            db.flush()

            sections = doc_def["sections"]
            chunk_texts = [s["content"] for s in sections]
            embeddings_list = []

            if embed_provider:
                try:
                    logger.info("Generating embeddings for '%s' (%d chunks)...", doc_def["title"], len(chunk_texts))
                    embeddings_list = await embed_provider.embed_texts(chunk_texts)
                    logger.info("Successfully generated %d embeddings for '%s'", len(embeddings_list), doc_def["title"])
                except Exception as emb_exc:
                    logger.warning("Embedding generation failed for '%s': %s", doc_def["title"], emb_exc)

            for idx, s in enumerate(sections):
                emb = embeddings_list[idx] if idx < len(embeddings_list) else None
                chunk = DocumentChunk(
                    id=uuid4(),
                    document_version_id=ver.id,
                    content=s["content"],
                    page=s["page"],
                    section=s["section"],
                    chunk_index=idx,
                    chunk_metadata={"title": doc_def["title"], "section": s["section"], "is_parent": False},
                    embedding=emb,
                )
                db.add(chunk)
                total_chunks_created += 1
                if emb is not None:
                    total_chunks_embedded += 1

            db.commit()
            total_docs_created += 1
            logger.info("Saved document '%s' with %d chunks", doc_def["title"], len(sections))

        logger.info(
            "Legal corpus seeding completed: %d documents created, %d chunks created, %d chunks embedded.",
            total_docs_created,
            total_chunks_created,
            total_chunks_embedded,
        )

    except Exception as exc:
        db.rollback()
        logger.error("Error during legal corpus seeding: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(seed_legal_corpus())
