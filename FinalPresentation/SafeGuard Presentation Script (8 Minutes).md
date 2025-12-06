**Speaker 1: The Context & The Problem (Slides 1-3)**  
(Time: \~2:00)

* **Slide 1: Introduction**  
  * "Good morning, Instructors and TAs. We are Group 47, and we are presenting **SafeGuard**: our collaborative multi-agent system project completed during the Fall 2025 term at the University of Waterloo."
  * This project focuses on one of the most important challenges in modern AI systems: how to ensure that Large Language Models behave safely and reliably in real-world settings.
* **Slide 2: Problem Statement**  
  * “As LLMs become embedded into search engines, productivity apps, and customer support systems, they’re increasingly being treated as mission-critical tools. But despite their impressive abilities, they still remain fundamentally unreliable.”
  * “They hallucinate facts, generate biased or toxic language, and are especially vulnerable to adversarial prompting or jailbreak attacks. These weaknesses make it difficult to trust LLMs in any high-stakes environment.”
  * “Existing safeguards—like rule-based filters or single-model approaches such as Llama Guard—often over-block harmless content or fail to detect subtle, high-risk prompts. And in many real deployments, whether in finance, healthcare, or public search, even one unsafe output can break user trust and cause significant harm.”  
* **Slide 3: Why it matters (Why Current Solutions Fail)**  
  * “The core weakness of current safety systems is that they are monolithic. They rely on a single safety classifier or rule-based filter to detect every possible category of risk.”
  * “As we described in our report, this creates a single point of failure: if an adversarial prompt exposes a weakness in that one model, the entire safeguard collapses.”
  * “Another issue is that these models are usually optimized for a narrow label space—such as toxicity only, or jailbreak detection only. They don’t jointly reason about factuality, groundedness, bias, or context-specific harms, which often overlap in real LLM interactions.”
  * “So our hypothesis is that safety shouldn’t be centralized in one model — it should be distributed. Instead of relying on one generalist filter, we propose a system of specialized critic agents that collaborate, each analyzing the output from a different perspective.”
  * “In short, we need a safeguard that doesn’t just filter, but works together.”
  
**Speaker 2: The Solution & Architecture (Slides 4-6)**  
(Time: \~2:15)

* **Slide 4: Solution Introduction**  
  * "This led us to design SafeGuard, a Collaborative Multi-Agent Framework. Our solution moves from relying on a single generalist model to using a specialized team of experts to perform safeguarding."  
* **Slide 5: The Workflow**  
  * "Here is the blueprint of our system. The LLM response is simultaneously routed to four distinct **Parallel Critic Agents**. Each agent independently assesses the risk, providing a label and a confidence score concurrently under multi-threading system. This parallel nature is key to maintaining low latency and high throughput, especially when we are scaling up to more agents or larger parameter critics. Their outputs are then sent to a central **Aggregator Module** which synthesizes these signals. The final decision is determined by a simple threshold: if the collective danger confidence score is greater than **0.7**, the response is refused; otherwise, it is released."  
* **Slide 6: Agent Specialization**  
  * "To maximize coverage across diverse LLM failure modes, the system uses four domain-specialized safeguard agents, each trained on targeted datasets and optimized for a specific risk category:: **Factuality Agent**, which detects misinformation and hallucinations using evidence-based datasets such as FEVER (Fact Extraction and VERification), TruthfulQA; **Toxicity Agent**, identifies hate speech, harassment, and subtle toxic phrasing. Trained on ToxiGen opensourced dataset; **Sexual Content Agent**, it flags explicit, sensitive, or policy-violating material, ensuring compliance with content-moderation requirements involving sexual content, and drug references, etc; and finally, a dedicated **Jailbreak Agent**, built on datasets like In-the-Wild Jailbreak Prompts, is used to detect sophisticated prompt injection attacks, role-play–based attacks, etc. By separating these safety concerns, we achieve fault isolation, domain-optimized accuracy, and a modular architecture where each safeguard can evolve independently. This ensures that a weakness in one safety dimension does not compromise the integrity of the entire system."

**Speaker 3: Implementation & Results (Slides 7-9)**  
(Time: \~2:30)
* **Slide 7: Decision Logic**  
  * "The decision-making process is summarized here. After the **Execution** and **Scoring** phases, the Aggregator performs **Synthesis**. This is where we account for the signal heterogeneity. For example, if a user attempts a clever jailbreak, the Jailbreak agent might flag it with 95% confidence, while the Toxicity agent might return a low confidence score because the language itself is neutral. The Aggregator detects and prioritizes the high-confidence danger signal from the specialized agent, triggering an action."  
  * "Our evaluation methodology assessed SafeGuard at two levels: first, we evaluated each individual safeguard module on their respective benchmark datasets—FEVER for factuality, ToxiGen for toxicity, CardiffNLP for sexual content, and JailbreakBench for jailbreak detection. Second, we evaluated the full integrated system using multi-domain benchmarks to understand how effectively the aggregator integrates signals from all safeguards."  
* **Slide 8: Technical Stack**  
  * "Every critic agent is built on the lightweight **DeBERTa-v3-small** architecture with under 500 million parameters, optimized for speed and parallel processing. This is vital for real-time deployment."  
  * "All modules were developed using PyTorch and Hugging Face Transformers. Each critic agent was fine-tuned using consistent hyperparameters across all safeguards: training for three epochs, with a batch size of 16, and early stopping based on validation loss. These settings ensured efficient fine-tuning while maintaining comparability across critic agents."  
  * "This modular approach grants us a major advantage in **Interpretability**: unlike a monolithic black box, we know *exactly* which agent—whether Factuality, Toxicity, Sexual Content, or Jailbreak—blocked the response, making debugging transparent and enabling more targeted improvements."  
* **Slide 9: Key Results**  
  * "We benchmarked SafeGuard against leading baselines — Granite Guardian and ShieldGemma — across three benchmarks: JailBreak Bench, HarmBench, and WildGuardMix."
  * "On HarmBench, SafeGuard achieved an 87.0% F1-score, outperforming ShieldGemma at 86% and Granite Guardian at 100% given a smaple size of 1000, ran over 10,000 different examples."
  * "While Granite Guardian scored higher on JailBreak and WildGuardMix, SafeGuard showed a balanced performance profile across diverse risks and almost compared to those SOTA models."


    
**Speaker 4: Conclusion & Future Directions (Slides 10)**  
(Time: \~1:15)

* **Slide 10: Summary & Next Steps**  
  * "To summarize, this term we built a fully functioning multi-agent safeguard pipeline—from model fine-tuning, to critic specialization, to aggregation—and demonstrated that specialization fundamentally improves robustness. Our experiments showed that no single critic can reliably capture all failure modes, but their coordinated signals form a far more resilient safety layer. This validates our core hypothesis: safety is stronger when distributed."
  * "While our project concluded with the Fall term, this framework is designed for extension. The most immediate future direction for this research would be replacing our current rule-based aggregation threshold (the fixed 0.7) with **Learned Aggregation strategies**—essentially training a smaller router model to make the final decision. Additionally, the modular structure allows for easy expansion to new critics like PII or Bias Detection.
  *
 * ** **Slide 11: Thank you**  
  * Thank you so much for listening and hope you were as intrigued as us!"
 
