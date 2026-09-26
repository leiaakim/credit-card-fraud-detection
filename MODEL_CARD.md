# Model Card: Credit Card Fraud Detector (LightGBM)

| | |
|---|---|
| **Model** | LightGBM gradient-boosted trees (200 trees, `reg_lambda=1.0`, otherwise default), inside a scikit-learn pipeline that standardises `Time` and `Amount` |
| **Decision rule** | Flag a transaction for review if fraud score ≥ **0.11** (chosen to minimise the expected cost of errors) |
| **Version** | 1.0, portfolio / research prototype, September 2026 |
| **Tracking** | MLflow experiment `credit-card-fraud-detection`, run `FINAL - LightGBM (Before SMOTE) - test` |
| **Author** | Personal portfolio project |
| **Status** | **Not approved for production use.** See *Failure modes* and *Recommendations*. |

---

## 1. Intended use

**Primary use.** Score card-not-present and card-present payment transactions in near real time and **route high-risk transactions to a human fraud analyst** for review, step-up authentication or a customer check.

**Intended users.** Fraud-operations teams at a card issuer or payment processor, working inside a wider fraud-management system (rules, analyst review, customer-contact processes).

**Out of scope.**
- **Fully automated blocking** of transactions or cards without human review or a fast customer-override path.
- Any decision about a person beyond the single transaction: **credit limits, creditworthiness, account closure, insurance pricing or law-enforcement referral.** Using the model for these purposes would change its legal risk category (Section 7).
- Data from other issuers, countries, time periods or payment channels without re-validation.

## 2. Training data

| | |
|---|---|
| Source | [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Worldline and ULB Machine Learning Group) |
| Coverage | 284,807 transactions by European cardholders over **two days in September 2013** |
| Label | `Class` = 1 for fraud: **492 cases (0.173%)** |
| Features | `Time` (seconds since the first transaction), `Amount` (EUR), and V1–V28, anonymised PCA components of confidential original features |
| Split | 80% training (227,845 rows, 394 fraud) / 20% held-out test (56,962 rows, 98 fraud), stratified, random seed 42 |

### Limitations of the training data
1. **Very old and very short.** Two days from 2013. Fraud tactics, payment channels (mobile wallets, instant payments) and spending patterns have changed substantially since. The model has never seen a weekend, a holiday season, a month-end or a real fraud campaign.
2. **Few positive examples.** 394 training frauds. Metrics carry wide uncertainty: the 95% bootstrap interval for validation recall is **0.69–0.88**.
3. **Anonymised features.** V1–V28 cannot be interpreted, audited for proxies of protected characteristics, or explained to a customer.
4. **No customer, merchant or device context.** No cardholder history, merchant category, geography, device or channel, which production fraud models rely on heavily.
5. **Label quality unknown.** Fraud labels usually come from chargebacks and investigations; undetected fraud is labelled "legitimate", so true recall is likely **overstated**.
6. **Random, not time-based, split.** Train and test come from the same two days, so the reported results are an *optimistic* estimate of performance on future data.

## 3. Evaluation results

All figures come from the held-out test set, which was used **once**, after the model, SMOTE decision and threshold were fixed on cross-validation.

| Metric (test, 98 fraud / 56,864 legitimate) | Threshold 0.11 (chosen) | Threshold 0.5 (reference) |
|---|---|---|
| Recall (fraud caught) | **0.796** (78 / 98) | 0.776 (76 / 98) |
| Precision (alerts that are fraud) | **0.867** | 0.916 |
| F1 | 0.830 | 0.840 |
| False alarms | 12 | 7 |
| AUC-ROC | 0.974 | 0.974 |
| Average precision (PR-AUC) | 0.860 | 0.860 |
| Fraud **value** caught | **57%** (€6,057 of €10,645) | |
| Business cost (missed fraud value + €5 per false alarm) | €4,648 | |

**Consistency.** Five-fold cross-validation on the training set gave recall 0.817 and precision 0.897 at the same threshold, in line with the test result.

**Model comparison** (5-fold CV recall at 0.5, 394 fraud): LightGBM 0.787, Random Forest 0.777, XGBoost 0.774, Logistic Regression 0.642. The three tree ensembles are **statistically indistinguishable** on recall. At their own cost-optimal thresholds, Random Forest (at a €1–€5 false-alarm cost) or XGBoost (at €10–€50) is cheaper than LightGBM by up to 8%.

**SMOTE.** Oversampling raised recall at 0.5 from 0.787 to 0.843 but quadrupled false alarms (26 → 111), and gave no meaningful gain in average precision (0.842 → 0.853). After threshold tuning, the cost was identical (€12,931 vs €12,949), so the simpler model without SMOTE was kept.

**Most influential features (SHAP):** V14, V4, V26, V12 and `Time`.

## 4. Failure modes and known weaknesses

| Failure mode | Evidence | Consequence | Mitigation |
|---|---|---|---|
| **Misses high-value fraud** | 5 missed frauds (€520–€1,810) = 94% of missed value; only 57% of fraud value caught | Largest losses come from the cases the model misses | Weight training by `Amount`; route high-value transactions with moderate scores to review |
| **`Time` used as a feature** | 5th most important SHAP feature; counts seconds from an arbitrary 2013 start point | Learned a two-day artefact; will behave unpredictably in production | Replace with hour-of-day / day-of-week, or drop; retrain |
| **Concept drift / adversarial adaptation** | Fraudsters change tactics once blocked | Recall decays silently; labels arrive weeks late | Monitoring signals (Section 6); scheduled retraining |
| **False alarms rise under data shift** | Simulated shift doubled false alarms (12 → 23) with unchanged recall | Customer friction, analyst overload | Alert-rate and precision monitoring; threshold recalibration |
| **Over-sensitive drift alarms** | Day 1 → day 2: 73% of columns flagged as drifted, but recall unchanged | Alarm fatigue, unnecessary retraining | Weight drift by feature importance; act on performance signals |
| **Small-sample uncertainty** | 98 test frauds; recall CI roughly ±0.09 | Reported numbers could be several points off | Re-validate on larger, recent data |
| **Poor calibration if resampled** | SMOTE inflates scores | Scores can't be read as probabilities | Keep the model without SMOTE, or recalibrate |
| **Unseen fraud types** | Only fraud patterns from two days in 2013 | New schemes (e.g. account takeover, authorised push-payment fraud) are not covered | Combine with rules, anomaly detection and analyst feedback |

## 5. Ethical considerations

- **Who bears the errors.** A **false negative** costs the issuer (and, until reimbursed, the cardholder) money. A **false positive** is a declined or delayed payment for a legitimate customer: embarrassment at a till, a failed urgent payment, or a blocked card while travelling. These harms are not captured by the €5 cost assumption and are larger for customers with fewer alternatives (a single card, no online banking).
- **Fairness cannot be assessed on this data.** No demographic or geographic attributes are available, and the PCA features may encode proxies (location, merchant type, spending level) for protected characteristics such as age, nationality or disability. A production version must test false-alarm rates across customer groups (e.g. by age band, region, card type) before deployment, and monitor them afterwards.
- **Human oversight.** The model is designed to *prioritise* transactions for human review, not to make final decisions. Analysts should see the score, the main SHAP drivers and the transaction context, and be able to overrule the model; overrides should feed back into training.
- **Transparency and contestability.** Customers affected by a block need a fast way to confirm the transaction and to contest a decision. Under GDPR Article 22, decisions based solely on automated processing that significantly affect a person require safeguards including human intervention; a fully automated card block could fall within this.
- **Privacy.** The public dataset is already anonymised. A production system would process personal data under GDPR (lawful basis: fraud prevention is typically a legitimate interest), with data minimisation and retention limits.
- **Feedback loops.** Flagged transactions are investigated and therefore labelled more reliably than unflagged ones. Retraining on these labels can reinforce the model's existing blind spots. Random sampling of unflagged transactions for review reduces this.

## 6. Monitoring and retraining

| Signal | Baseline (test) | Trigger |
|---|---|---|
| Alert rate | 0.16% | 7-day average outside 0.08%–0.32% |
| Score-tail drift (share of scores ≥ 0.01) | 0.19% | Week-over-week change > 25% |
| Drift in top SHAP features (V14, V4, V12, V26, `Amount`) | Training distribution | Drift on any of them for 3 consecutive days |
| Recall on confirmed fraud | 0.80 | Rolling 30 days < 0.72 |
| Alert precision | 0.87 | Rolling 30 days < 0.75 |
| Business cost per 10,000 transactions | €816 | 30% above baseline for a month |

Retrain monthly on recent labelled data and whenever a performance trigger fires; validate out-of-time; deploy as a shadow or challenger model first.

## 7. Regulatory context: the EU AI Act

**Is fraud detection "high-risk" under the EU AI Act?** A common assumption is that it is, because it is used in financial services. **The text of the Act says the opposite for the core use case.** Annex III, point 5(b) lists as high-risk:

> "AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score, **with the exception of AI systems used for the purpose of detecting financial fraud**."

Recital 58 adds that "AI systems provided for by Union law for the purpose of detecting fraud in the offering of financial services … should not be considered to be high-risk under this Regulation." So **a transaction-level fraud-detection model, used as intended in Section 1, is not listed as a high-risk AI system.**

It still sits close to the high-risk boundary, which is why this card treats it with high-risk-grade discipline:

1. **Function creep into credit decisions.** If fraud scores are used to set credit limits, refuse credit or close accounts, the system is evaluating creditworthiness of natural persons, which *is* high-risk under Annex III 5(b).
2. **Access to essential services.** Other Annex III 5 uses, e.g. (a) eligibility for essential public benefits and services, or (c) risk assessment and pricing for life and health insurance, are high-risk. Reusing fraud scores in those processes would make that use high-risk.
3. **Profiling.** Under Article 6(3), a system listed in Annex III is always high-risk when it profiles natural persons. A future version that builds customer-level risk profiles for any listed purpose would be caught.
4. **Other law still applies regardless:** GDPR (including Article 22 on automated decisions), consumer-protection law, PSD2 strong-customer-authentication rules, and the European Banking Authority's guidelines on internal governance and ICT risk. Supervisors expect **model risk management** for fraud models whether or not the AI Act classes them as high-risk.

**If a use did fall into the high-risk category,** the provider would need, among other things: a risk-management system (Art. 9), data governance and bias examination (Art. 10), technical documentation (Art. 11), logging (Art. 12), transparency to deployers (Art. 13), human oversight (Art. 14), and accuracy, robustness and cybersecurity (Art. 15). This project already produces building blocks for several of these: this card and the README (documentation), MLflow run logs (record-keeping), SHAP explanations (transparency), the drift reports (robustness), and the human-review design (oversight).

**Timing.** Under the Digital Omnibus agreement reached in 2026, obligations for stand-alone Annex III high-risk systems apply from **2 December 2027** (previously 2 August 2026). *This section is a summary for a portfolio project, not legal advice.*

## 8. Recommendations before any real use

1. Retrain on recent, multi-month data with customer, merchant and device features, and validate **out-of-time**.
2. Replace `Time` with cyclical time features; weight training by `Amount`.
3. Confirm the false-alarm cost with the fraud-operations team and re-run the model comparison (XGBoost is the challenger).
4. Run a fairness assessment of false-alarm rates across customer groups.
5. Deploy in **shadow mode** alongside existing rules, with the monitoring in Section 6.
6. Document every retrained version in MLflow and update this card.

---

*Sources: [EU AI Act, Annex III](https://artificialintelligenceact.eu/annex/3/) · [EU AI Act, Recital 58](https://artificialintelligenceact.eu/recital/58/) · [Gibson Dunn: EU AI Act Omnibus agreement, postponed high-risk deadlines](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/) · Dataset: A. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE CIDM 2015.*
