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
        "title": "قانون آیین دادرسی کیفری - حقوق متهم و بازجویی",
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
                "section": "ماده ۱۹۰ - حق همراه داشتن وکیل در تحقیقات",
                "page": 3,
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
    {
        "title": "قانون مدنی - قواعد عمومی قراردادها و الزامات",
        "document_type": "law",
        "knowledge_type": "statutory",
        "source": "مجلس شورای اسلامی",
        "version": "1.0",
        "sections": [
            {
                "section": "ماده ۱۰ - آزادی قراردادها و نفوذ اراده",
                "page": 1,
                "content": (
                    "ماده ۱۰ قانون مدنی جمهوری اسلامی ایران:\n"
                    "قراردادهای خصوصی نسبت به کسانی که آن را منعقد نموده‌اند، در صورتی که مخالف صریح قانون "
                    "و نظم عمومی یا اخلاق حسنه نباشد، نافذ و معتبر است."
                ),
            },
            {
                "section": "ماده ۱۹۰ - شرایط اساسی صحت معاملات",
                "page": 1,
                "content": (
                    "ماده ۱۹۰ قانون مدنی:\n"
                    "برای صحت هر معامله شرایط ذیل اساسی است:\n"
                    "۱- قصد طرفین و رضای آن‌ها\n"
                    "۲- اهلیت طرفین (بلوغ، عقل، رشد)\n"
                    "۳- موضوع معین که مورد معامله باشد (مالیت و قابلیت نقل و انتقال داشته باشد)\n"
                    "۴- مشروعیت جهت معامله (در صورتی که جهت تصریح شده باشد)."
                ),
            },
            {
                "section": "ماده ۲۱۹ و ۲۲۰ - لزوم عقد و آثار تعهدات",
                "page": 2,
                "content": (
                    "ماده ۲۱۹ قانون مدنی:\n"
                    "عقودی که بر طبق قانون واقع شده باشد، بین متعاملین و قائم‌مقام آن‌ها لازم‌الاتباع است "
                    "مگر اینکه به رضای طرفین اقاله یا به علت قانونی فسخ شود.\n\n"
                    "ماده ۲۲۰ قانون مدنی:\n"
                    "عقود نه فقط متعاملین را به اجرای چیزی که در آن تصریح شده است ملزم می‌نماید بلکه به تمام نتایجی "
                    "نیز که به موجب عرف و عادت یا به موجب قانون از عقد حاصل می‌شود ملزم می‌کند."
                ),
            },
            {
                "section": "ماده ۳۹۶ - انواع خیارات قانونی فسخ قرارداد",
                "page": 3,
                "content": (
                    "ماده ۳۹۶ قانون مدنی:\n"
                    "خیاراتی که به موجب آن می‌توان معامله را فسخ کرد از قرار ذیل است:\n"
                    "۱- خیار مجلس (تا زمانی که طرفین از جلسه قرارداد جدا نشده‌اند)\n"
                    "۲- خیار حیوان (در بیع حیوان تا ۳ روز)\n"
                    "۳- خیار شرط (حق فسخ در مدت معین برای یکی یا هر دو طرف)\n"
                    "۴- خیار تاخیر ثمن\n"
                    "۵- خیار رویت و تخلف وصف (عدم مطابقت مال با اوصاف مقرر)\n"
                    "۶- خیار غبن (عدم تعادل فاحش قیمت و ارزش)\n"
                    "۷- خیار عیب (وجود نقص یا عیب مخفی در مبیع)\n"
                    "۸- خیار تدلیس (فریب و پنهان‌کاری یکی از طرفین)\n"
                    "۹- خیار تبعض صفقه\n"
                    "۱۰- خیار تخلف شرط (عدم انجام شرط ضمن عقد)."
                ),
            },
        ],
    },
    {
        "title": "قانون کار - روابط کار، سنوات و فسخ قرارداد",
        "document_type": "law",
        "knowledge_type": "statutory",
        "source": "وزارت تعاون، کار و رفاه اجتماعی",
        "version": "1.0",
        "sections": [
            {
                "section": "ماده ۲ و ۳ و ۷ - تعریف کارگر، کارفرما و قرارداد کار",
                "page": 1,
                "content": (
                    "ماده ۲ و ۳ قانون کار جمهوری اسلامی ایران:\n"
                    "کارگر از لحاظ این قانون کسی است که به هر عنوان در مقابل دریافت حق‌السعی اعم از مزد، حقوق، "
                    "سهم سود و سایر مزایا به درخواست کارفرما کار می‌کند.\n"
                    "کارفرما شخصی است حقیقی یا حقوقی که کارگر به درخواست و به حساب او در مقابل دریافت حق‌السعی کار می‌کند.\n\n"
                    "ماده ۷ قانون کار:\n"
                    "قرارداد کار عبارت است از قرارداد کتبی یا شفاهی که به موجب آن کارگر در قبال دریافت حق‌السعی "
                    "کاری را برای مدت موقت یا مدت غیرموقت برای کارفرما انجام می‌دهد."
                ),
            },
            {
                "section": "ماده ۲۴ - پاداش پایان کار و محاسبه سنوات خدمت",
                "page": 2,
                "content": (
                    "ماده ۲۴ قانون کار:\n"
                    "در صورت خاتمه قرارداد کار، کار موقت یا مدت معین، کارفرما مکلف است به کارگری که مطابق قرارداد "
                    "یک سال یا بیشتر به کار اشتغال داشته است برای هر سال سابقه، اعم از متوالی یا متناوب، بر اساس آخرین حقوق "
                    "مبلغی معادل یک ماه حقوق به عنوان مزایای پایان کار (حق سنوات) به وی پرداخت نماید."
                ),
            },
            {
                "section": "ماده ۲۷ - شرایط اخراج و قصور کارگر",
                "page": 2,
                "content": (
                    "ماده ۲۷ قانون کار:\n"
                    "هر گاه کارگر در انجام وظایف محوله قصور ورزد و یا آیین‌نامه‌های انضباطی کارگاه را پس از تذکرات کتبی "
                    "نقض نماید، کارفرما در صورت اعلام نظر مثبت شورای اسلامی کار علاوه بر مطالبات و دیون معوقه به نسبت هر سال سابقه کار "
                    "معادل یک ماه آخرین حقوق کارگر را به عنوان حق سنوات به وی پرداخته و قرارداد کار را فسخ نماید."
                ),
            },
        ],
    },
    {
        "title": "قانون مجازات اسلامی - کلیات جرایم و مجازات‌ها",
        "document_type": "law",
        "knowledge_type": "statutory",
        "source": "قوه قضائیه",
        "version": "1.0",
        "sections": [
            {
                "section": "ماده ۲ و ۱۴ - تعریف جرم و انواع مجازات‌ها",
                "page": 1,
                "content": (
                    "ماده ۲ قانون مجازات اسلامی:\n"
                    "هر رفتاری اعم از فعل یا ترک فعل که در قانون برای آن مجازات تعیین شده است جرم محسوب می‌شود.\n\n"
                    "ماده ۱۴ قانون مجازات اسلامی:\n"
                    "مجازات‌های مقرر در این قانون چهار قسم است:\n"
                    "الف- حد (مجازاتی که موجب، نوع، میزان و کیفیت اجرای آن در شرع مقدس تعیین شده است)\n"
                    "ب- قصاص (مجازات اصلی جنایات عمدی بر نفس، اعضا و منافع)\n"
                    "پ- دیه (مالی که در شرع برای جبران جنایات غیرعمدی یا خطای محض مقرر است)\n"
                    "ت- تعزیر (مجازاتی که در شرع مقدار آن معین نشده و به حکم دادگاه تعیین می‌شود)."
                ),
            },
            {
                "section": "ماده ۱۹ - درجات هشت‌گانه مجازات‌های تعزیری",
                "page": 2,
                "content": (
                    "ماده ۱۹ قانون مجازات اسلامی:\n"
                    "مجازات‌های تعزیری به هشت درجه تقسیم می‌شود:\n"
                    "درجه ۱: حبس بیش از ۲۵ سال، جزای نقدی بیش از ۲ میلیارد و ۸۰۰ میلیون ریال، مصادره کل اموال.\n"
                    "درجه ۲: حبس بیش از ۱۵ تا ۲۵ سال، جزای نقدی بیش از ۱ میلیارد و ۵۰۰ میلیون تا ۲ میلیارد و ۸۰۰ میلیون ریال.\n"
                    "درجه ۳: حبس بیش از ۱۰ تا ۱۵ سال.\n"
                    "درجه ۴: حبس بیش از ۵ تا ۱۰ سال.\n"
                    "درجه ۵: حبس بیش از ۲ تا ۵ سال، محرومیت از حقوق اجتماعی ۵ تا ۱۵ سال.\n"
                    "درجه ۶: حبس بیش از ۶ ماه تا ۲ سال، شلاق از ۳۱ تا ۷۴ ضربه.\n"
                    "درجه ۷: حبس از ۹۱ روز تا ۶ ماه، شلاق از ۱۱ تا ۳۰ ضربه.\n"
                    "درجه ۸: حبس تا ۳ ماه، جزای نقدی تا ۱۰ میلیون ریال، شلاق تا ۱۰ ضربه."
                ),
            },
            {
                "section": "ماده ۲۶۷ و ۲۶۸ - تعریف و شرایط سرقت حدی و تعزیری",
                "page": 3,
                "content": (
                    "ماده ۲۶۷ قانون مجازات اسلامی:\n"
                    "سرقت عبارت از ربودن مال متعلق به غیر است.\n\n"
                    "ماده ۲۶۸ قانون مجازات اسلامی:\n"
                    "سرقت در صورتی که دارای تمام شرایط چهارده‌گانه ذیل باشد موجب حد است:\n"
                    "۱- مال در حرز باشد (حرز مکانی است که مال برای حفظ از دستبرد در آن نگهداری می‌شود)\n"
                    "۲- سارق حرز را بشکند یا از آن عبور کند\n"
                    "۳- سارق مال را از حرز خارج کند\n"
                    "۴- هتک حرز و سرقت مخفیانه باشد\n"
                    "۵- سارق پدر یا جد پدری صاحب مال نباشد\n"
                    "۶- ارزش مال مسروق در زمان اخراج از حرز معادل چهار و نیم نخود طلای مسکوک باشد\n"
                    "۷- مال مسروق از اموال دولتی یا عمومی یا وقفی نباشد.\n\n"
                    "در صورت فقدان هر یک از این شرایط، سرقت تعزیری محسوب شده و طبق مواد ۶۵۱ تا ۶۶۷ قانون مجازات بخش تعزیرات مجازات می‌شود."
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
