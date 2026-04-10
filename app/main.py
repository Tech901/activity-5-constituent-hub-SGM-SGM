"""
Activity 5 - Constituent Services Hub
AI-102: Azure AI Language services for citizen complaint processing

Your task:
  1. PII detection + redaction -- scan citizen complaints, detect names/SSNs/
     addresses, produce sanitized versions
  2. Sentiment + key phrase extraction -- analyze complaint tone and extract
     actionable topics
  3. Multilingual support -- detect language, translate non-English to English
  4. Intent recognition (CLU) -- classify citizen messages into intents
     (report-issue, check-status, ask-question) with entity extraction

Output: result.json with required fields (task, status, outputs, metadata)
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_sdk_version() -> str:
    try:
        from importlib.metadata import version
        return version("azure-ai-textanalytics")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------
_language_client = None
_translator_client = None
_clu_client = None


def _get_language_client():
    """Lazily initialize the Azure AI Language client."""
    global _language_client
    if _language_client is None:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        _language_client = TextAnalyticsClient(
            endpoint=os.environ["AZURE_AI_LANGUAGE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_AI_LANGUAGE_KEY"]),
        )
    return _language_client


def _get_translator_client():
    """Lazily initialize the Azure Translator client."""
    global _translator_client
    if _translator_client is None:
        from azure.ai.translation.text import TextTranslationClient
        from azure.core.credentials import AzureKeyCredential
        _translator_client = TextTranslationClient(
            credential=AzureKeyCredential(os.environ["AZURE_TRANSLATOR_KEY"]),
            region=os.environ.get("AZURE_TRANSLATOR_REGION", "eastus"),
        )
    return _translator_client


def _get_clu_client():
    """Lazily initialize the Conversational Language Understanding client."""
    global _clu_client
    if _clu_client is None:
        from azure.ai.language.conversations import ConversationAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        _clu_client = ConversationAnalysisClient(
            endpoint=os.environ["AZURE_AI_LANGUAGE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_AI_LANGUAGE_KEY"]),
        )
    return _clu_client


# ---------------------------------------------------------------------------
# CLU intent name mapping (PascalCase CLU -> kebab-case pipeline)
# ---------------------------------------------------------------------------
_CLU_INTENT_MAP = {
    "ReportIssue": "report-issue",
    "CheckStatus": "check-status",
    "GetInformation": "ask-question",
}


def _keyword_intent_fallback(text: str) -> dict:
    """Simple keyword-based intent classifier used as CLU fallback."""
    lower = text.lower()

    report_keywords = [
        "report", "broken", "break", "pothole", "trash", "graffiti",
        "noise", "dumped", "leak", "flood", "damaged", "water",
        "sewer", "fire", "emergency", "crack", "collapse",
    ]
    status_keywords = [
        "status", "update", "case", "follow", "submitted",
        "tracking", "assigned", "scheduled", "reference",
    ]
    question_keywords = [
        "where", "how", "information", "schedule", "what",
        "hours", "fee", "apply", "sign up", "find",
    ]

    report_score = sum(1 for kw in report_keywords if kw in lower)
    status_score = sum(1 for kw in status_keywords if kw in lower)
    question_score = sum(1 for kw in question_keywords if kw in lower)

    scores = {
        "report-issue": report_score,
        "check-status": status_score,
        "ask-question": question_score,
    }

    total = sum(scores.values())
    if total == 0:
        return {
            "top_intent": "ask-question",
            "confidence": 0.0,
            "entities": [],
        }

    top_intent = max(scores, key=scores.get)
    return {
        "top_intent": top_intent,
        "confidence": round(scores[top_intent] / total, 2),
        "entities": [],
    }


def load_complaints() -> list[dict]:
    """Load citizen complaints from data/complaints.json."""
    data_path = Path(__file__).parent.parent / "data" / "complaints.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1 - PII Detection and Redaction
# ---------------------------------------------------------------------------
def detect_and_redact_pii(text: str) -> dict:
    """Scan citizen complaint text for PII and produce a redacted version.

    Args:
        text: Raw citizen complaint text.

    Returns:
        dict with keys: original_text, redacted_text, entities
    """
    client = _get_language_client()
    response = client.recognize_pii_entities([text])
    doc = response[0]

    entities = [
        {
            "text": entity.text,
            "category": entity.category,
            "confidence_score": entity.confidence_score,
        }
        for entity in doc.entities
    ]

    return {
        "original_text": text,
        "redacted_text": doc.redacted_text,
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# Step 2 - Sentiment Analysis and Key Phrase Extraction
# ---------------------------------------------------------------------------
def analyze_sentiment_and_phrases(text: str) -> dict:
    """Analyze complaint tone and extract actionable topics.

    Args:
        text: Citizen complaint text (use redacted version).

    Returns:
        dict with keys: sentiment, confidence_scores, key_phrases
    """
    client = _get_language_client()

    sentiment_response = client.analyze_sentiment([text])
    sentiment_doc = sentiment_response[0]
    sentiment = sentiment_doc.sentiment
    scores = sentiment_doc.confidence_scores
    confidence_scores = {
        "positive": scores.positive,
        "negative": scores.negative,
        "neutral": scores.neutral,
    }

    phrases_response = client.extract_key_phrases([text])
    key_phrases = list(phrases_response[0].key_phrases)

    return {
        "sentiment": sentiment,
        "confidence_scores": confidence_scores,
        "key_phrases": key_phrases,
    }


# ---------------------------------------------------------------------------
# Step 3 - Language Detection and Translation
# ---------------------------------------------------------------------------
def detect_and_translate(text: str) -> dict:
    """Detect the language of text and translate to English if needed.

    Args:
        text: Input text in any language.

    Returns:
        dict with keys: detected_language, confidence, original_text,
        translated_text, was_translated
    """
    client = _get_language_client()

    detection_response = client.detect_language([text])
    detection_doc = detection_response[0]
    detected_language = detection_doc.primary_language.iso6391_name
    confidence = detection_doc.primary_language.confidence_score

    if detected_language != "en":
        translator = _get_translator_client()
        translation_response = translator.translate(
            body=[text],
            to_language=["en"],
        )
        translated_text = translation_response[0].translations[0].text
        was_translated = True
    else:
        translated_text = text
        was_translated = False

    return {
        "detected_language": detected_language,
        "confidence": confidence,
        "original_text": text,
        "translated_text": translated_text,
        "was_translated": was_translated,
    }


# ---------------------------------------------------------------------------
# Step 4 - Intent Recognition with CLU
# ---------------------------------------------------------------------------
def recognize_intent(text: str) -> dict:
    """Classify citizen message intent using Conversational Language
    Understanding.

    Falls back to keyword matching if CLU is not configured.

    Args:
        text: Citizen message text.

    Returns:
        dict with keys: top_intent, confidence, entities
    """
    clu_project = os.environ.get("CLU_PROJECT_NAME", "")
    clu_deployment = os.environ.get("CLU_DEPLOYMENT_NAME", "")
    if not clu_project or not clu_deployment:
        return _keyword_intent_fallback(text)

    try:
        client = _get_clu_client()

        task = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "text": text,
                    "participantId": "user",
                }
            },
            "parameters": {
                "projectName": clu_project,
                "deploymentName": clu_deployment,
                "stringIndexType": "TextElement_V8",
            },
        }

        response = client.analyze_conversation(task)
        prediction = response.result.prediction

        raw_intent = prediction.top_intent
        top_intent = _CLU_INTENT_MAP.get(raw_intent, raw_intent)

        confidence = 0.0
        for intent in prediction.intents:
            if intent.category == raw_intent:
                confidence = intent.confidence_score
                break

        entities = [
            {
                "entity": entity.text,
                "category": entity.category,
                "text": entity.text,
            }
            for entity in prediction.entities
        ]

        return {
            "top_intent": top_intent,
            "confidence": confidence,
            "entities": entities,
        }

    except NotImplementedError:
        raise
    except Exception:
        return _keyword_intent_fallback(text)


def main():
    """Main function -- run the constituent services pipeline."""

    complaints_data = load_complaints()
    complaint_texts = [c["text"] for c in complaints_data]

    steps_completed = []
    pii_entities_found = 0
    languages_detected = set()
    translations_performed = 0

    # Step 1: PII detection and redaction
    print("\n--- Step 1: PII Detection and Redaction ---")
    pii_results = []
    for i, text in enumerate(complaint_texts):
        try:
            pii_result = detect_and_redact_pii(text)
            pii_results.append(pii_result)
            pii_entities_found += len(pii_result.get("entities", []))
            print(f"  ✓ Complaint {i + 1}: {len(pii_result.get('entities', []))} PII entities redacted")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 1 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if pii_results:
        steps_completed.append("pii_detection")

    # Step 2: Sentiment and key phrases (on redacted text)
    print("\n--- Step 2: Sentiment Analysis and Key Phrases ---")
    sentiment_results = []
    texts_for_sentiment = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_sentiment):
        try:
            sentiment = analyze_sentiment_and_phrases(text)
            sentiment_results.append(sentiment)
            print(f"  ✓ Complaint {i + 1}: {sentiment.get('sentiment', '?')} sentiment")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 2 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if sentiment_results:
        steps_completed.append("sentiment_analysis")

    # Step 3: Language detection and translation
    print("\n--- Step 3: Language Detection and Translation ---")
    translation_results = []
    texts_for_translation = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_translation):
        try:
            translation = detect_and_translate(text)
            translation_results.append(translation)
            lang = translation.get("detected_language", "?")
            languages_detected.add(lang)
            if translation.get("was_translated"):
                translations_performed += 1
                print(f"  ✓ Complaint {i + 1}: {lang} → en (translated)")
            else:
                print(f"  ✓ Complaint {i + 1}: {lang} (no translation needed)")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 3 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if translation_results:
        steps_completed.append("translation")

    # Step 4: Intent recognition (on redacted text)
    print("\n--- Step 4: Intent Recognition ---")
    intent_results = []
    texts_for_intent = (
        [r.get("redacted_text", "") for r in pii_results]
        if pii_results
        else complaint_texts
    )
    for i, text in enumerate(texts_for_intent):
        try:
            intent = recognize_intent(text)
            intent_results.append(intent)
            print(f"  ✓ Complaint {i + 1}: {intent.get('top_intent', '?')} ({intent.get('confidence', 0):.0%})")
        except NotImplementedError:
            print(f"  ⏭ Complaint {i + 1}: Step 4 not implemented yet")
            break
        except Exception as e:
            print(f"  ✗ Complaint {i + 1}: Error - {e}")
            break
    if intent_results:
        steps_completed.append("intent_recognition")

    # Determine status
    if len(steps_completed) == 4:
        status = "success"
    elif len(steps_completed) > 0:
        status = "partial"
    else:
        status = "error"

    # Build pipeline summary
    pipeline_summary = {
        "total_complaints": len(complaint_texts),
        "steps_completed": steps_completed,
        "pii_entities_found": pii_entities_found,
        "languages_detected": sorted(languages_detected),
        "translations_performed": translations_performed,
    }

    result = {
        "task": "constituent_hub",
        "status": status,
        "outputs": {
            "pii_results": pii_results,
            "sentiment_results": sentiment_results,
            "translation_results": translation_results,
            "intent_results": intent_results,
            "pipeline_summary": pipeline_summary,
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "azure-ai-language",
            "sdk_version": _get_sdk_version(),
        },
    }

    output_path = Path(__file__).parent.parent / "result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete: {len(steps_completed)}/4 steps ({status})")
    print(f"Result written to result.json")
    if steps_completed:
        print(f"Steps completed: {', '.join(steps_completed)}")
    else:
        print("No steps completed — implement the TODO functions!")


if __name__ == "__main__":
    main()