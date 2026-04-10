---
title: "Activity 5 Reflection - Constituent Services Hub"
type: reflection
version: "1.0.0"
---

# Activity 5 Reflection

Answer each question in 3-5 sentences. Thoughtful, specific responses earn full credit.

## 1. PII Redaction in Practice

What PII categories did the Azure AI Language service detect in the Memphis 311 complaints? Did you notice any false positives (non-PII flagged as sensitive) or false negatives (PII that was missed)? How might PII detection requirements differ between a government agency processing citizen complaints and a commercial customer service system?

The service detected Person names, phone numbers, addresses, and email addresses across the complaints. Most redactions were accurate, though one complaint flagged a street intersection as an address when it was really just a location reference (a minor false positive). A government agency like Memphis 311 has stricter obligations around data handling than a commercial system, meaning PII should be redacted before storage and not just before transmission. A commercial customer service system might intentionally retain some PII for personalization, whereas a government system should treat all citizen data as sensitive by default.

## 2. Sentiment as a Routing Signal

How could sentiment analysis be used to prioritize Memphis 311 complaints — for example, routing highly negative complaints to senior staff? What are the risks of relying solely on sentiment for prioritization (consider complaints that are neutral in tone but urgent in nature)? How would you combine sentiment with other signals for a more robust routing system?

Complaints with strong negative sentiment could be automatically escalated to senior staff or flagged for same-day response (especially those mentioning safety hazards or flooding). The risk is that a neutral-toned complaint about a water main leak is just as urgent as an angry one about a pothole, but sentiment alone would deprioritize it. A more robust system would combine sentiment with key phrase matching and route based on both tone and topic. For example: neutral sentiment plus phrases like "sewer" or "flooding" should still trigger a high-priority queue regardless of how politely the citizen wrote.

## 3. Multilingual Challenges

How accurately did the Language service detect the language of short versus long text samples? What challenges might arise with code-switching (mixing languages in a single message), which is common in multilingual communities? How would you handle a complaint where language detection confidence is low?

Language detection was accurate and confident on the longer Spanish and Vietnamese complaints in the dataset. Shorter texts would likely produce lower confidence scores since there are fewer linguistic signals to work with. Code-switching is a real challenge in diverse communities (a message mixing English and Spanish could confuse detection or produce an unreliable result). For low-confidence detections, I would default to treating the text as English and passing it through the pipeline unchanged rather than risking a bad translation, then flag it for human review in the output.

## 4. CLU vs. Keyword Matching

Compare the CLU model's intent classification with the keyword-based fallback. In what scenarios did CLU perform better, and where did keyword matching suffice? What are the trade-offs of training and maintaining a custom CLU model versus using a simpler rule-based approach for a city's 311 system? *(If CLU was not configured in your environment, discuss how you would expect it to differ based on the training data in `data/intent_examples.json`.)*

CLU returned 100% confidence on clear report-issue complaints, which the keyword fallback matched correctly too, so the gap was not visible on straightforward cases. Where CLU would likely pull ahead is on ambiguous phrasing: something like "I need to know if my neighbor's tree is a city responsibility" would confuse keyword matching but CLU's trained model could classify it as ask-question. The trade-off is real in that CLU requires labeled training data, retraining when new complaint patterns emerge, and a deployed endpoint to maintain. For a city 311 system with limited IT staff, the keyword fallback is a reasonable production safety net that keeps the pipeline running during model updates or outages.

## 5. Pipeline Design

Which step in your NLP pipeline was most critical to get right first, and why? If you were deploying this pipeline for production use handling thousands of complaints daily, what monitoring, fallback mechanisms, or error handling would you add? How would you measure pipeline health over time?

PII redaction had to be first and correct since everything downstream depends on sanitized text. A misredaction at Step 1 means sensitive data flows through sentiment analysis, translation, and intent classification (potentially appearing in logs). For production at scale, I would add per-step latency tracking, dead-letter queuing for complaints that fail processing so nothing is silently dropped, and a retry mechanism with exponential backoff for transient API errors. Pipeline health over time would be measured by tracking key metrics: redaction rate, translation failure rate, and intent confidence distribution. A sudden drop in average confidence on intent classification would signal that citizen complaint patterns have shifted and the CLU model needs retraining.