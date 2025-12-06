**Speaker 1: The Context & The Problem (Slides 1-3)**  
(Time: \~2:00)

* **Slide 1: Introduction**  
  * "Good morning, Instructors and TAs. We are Group 47, and we are presenting **SafeGuard**: our collaborative multi-agent system project completed during the Fall 2025 term at the University of Waterloo."  
* **Slide 2: Problem Statement**  
  * "Our research addresses a fundamental issue: LLMs are now mission-critical, but they remain fundamentally unreliable. They suffer from hallucinations, bias, and, crucially, adversarial vulnerability. Existing safeguards, like simple filters or single models like Llama Guard, struggle to balance safety and utility."  
* **Slide 3: Why it matters (Why Current Solutions Fail)**  
  * "This lack of reliability creates a massive trust gap. In any real-world deployment—whether for finance, health, or public search—you cannot afford outputs that are factually incorrect or toxic, especially when under attack."
  * "The core weakness of prior approaches is that they are **monolithic**. They rely on a single, large safety model or a simple rule-based filter to catch everything. As we noted in our report, this creates a **single point of failure**: if an adversarial prompt, or 'jailbreak,' finds a weakness in that one shield, the whole system fails. We hypothesized that safety must be distributed, not centralized. We need a system that doesn't just filter, but **collaborates**."
  
**Speaker 2: The Solution & Architecture (Slides 4-6)**  
(Time: \~2:15)

* **Slide 4: Solution Introduction**  
  * "This led us to design SafeGuard, a Collaborative Multi-Agent Framework. Our solution moves from relying on a single generalist model to using a specialized team of experts."  
* **Slide 5: The Workflow**  
  * "Here is the blueprint of our system. The LLM response is simultaneously routed to four distinct **Parallel Critic Agents**. Each agent independently assesses the risk, providing a label and a confidence score. This parallel nature is key to maintaining low latency. Their outputs are then sent to a central **Aggregator Module** which synthesizes these signals. The final decision is determined by a simple threshold: if the collective danger confidence is greater than **0.7**, the response is refused; otherwise, it is released."  
* **Slide 6: Agent Specialization**  
  * "To ensure maximum coverage across diverse failure modes, we defined four specialized agents: **Factuality**, trained on datasets like FEVER; **Toxicity** to handle hate speech; **Sexual Content** for policy compliance; and critically, a dedicated **Jailbreak** agent to detect sophisticated prompt injection attacks. By separating these concerns, we ensure that a failure in one dimension does not compromise the others."

**Speaker 3: Implementation & Results (Slides 7-9)**  
(Time: \~2:30)

* **Slide 7: Decision Logic**  
  * "The decision-making process is summarized here. After the **Execution** and **Scoring** phases, the Aggregator performs **Synthesis**. This is where we account for the signal heterogeneity. For example, if a user attempts a clever jailbreak, the Jailbreak agent might flag it with 95% confidence, while the Toxicity agent might return a low confidence score because the language itself is neutral. The Aggregator detects and prioritizes the high-confidence danger signal from the specialized agent, triggering an action."  
* **Slide 8: Technical Stack**  
  * "To address the constraint of efficiency, every critic agent is built on the lightweight **DeBERTa-v3-small** architecture. With under 500 million parameters, these models are optimized for speed and parallel processing. This is vital for real-time deployment. Furthermore, this modular approach grants us a major advantage in **Interpretability**: unlike a monolithic black box, we know *exactly* which agent (Factuality, Jailbreak, etc.) blocked the response, making debugging transparent."  
* **Slide 9: Key Results**  
  * "Finally, the evaluation. We benchmarked SafeGuard against state-of-the-art single-model baselines, including Granite Guardian and ShieldGemma, using multi-domain datasets like HarmBench. The results validated our core hypothesis: SafeGuard achieved an **87.0% F1-score on HarmBench**. Crucially, this demonstrates that specialized collaboration provides comparable coverage compared to monolithic models, which often showed higher accuracy on narrow, individual benchmarks but often introduced a higher inference cost & slower performance."

**Speaker 4: Conclusion & Future Directions (Slides 10)**  
(Time: \~1:15)

* **Slide 10: Summary & Next Steps**  
  * "To summarize, this term we built a fully functioning multi-agent safeguard pipeline—from model fine-tuning, to critic specialization, to aggregation—and demonstrated that specialization fundamentally improves robustness. Our experiments showed that no single critic can reliably capture all failure modes, but their coordinated signals form a far more resilient safety layer. This validates our core hypothesis: safety is stronger when distributed."
  * "While our project concluded with the Fall term, this framework is designed for extension. The most immediate future direction for this research would be replacing our current rule-based aggregation threshold (the fixed 0.7) with **Learned Aggregation strategies**—essentially training a smaller router model to make the final decision. Additionally, the modular structure allows for easy expansion to new critics like PII or Bias Detection. Thank you, for listening!"
 
