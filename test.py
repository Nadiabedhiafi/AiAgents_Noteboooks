I’m happy to walk you through the work we’ve done over the past few weeks. I’ll start with a summary of the business exploration, then move on to the technical aspects. I know the technical side is what interests you most, but this structure is important to show that we’re not just complaining about business needs—we’re actually backing them up with concrete evidence.

After that, we’ll cover the framework overview and benchmark, along with our proposed agentic architecture, and finally the current status and next steps.

We identified the most important and highest-value components for credit analysis, and we also highlighted a clear need for “chat with my document” capabilities, initially for the E&I stream and very likely for other streams as well. That’s why this capability has been integrated into the architecture we’ll present later.

⸻

Early approaches (first weeks)

I’ll briefly go through the approaches we tested during the first weeks, mainly to help guide the business team:
	1.	Approach 1 (pre-June PoC)
The goal was to extract the 9 indicators from the CRF and enrich them with the Annual Report to explain the variation of these indicators.
The output was indeed explanatory, but it contained a lot of redundancy and was far from the actual business needs.
	2.	Approach 2
This was similar to the first approach, but with a stronger prompt formatting layer to give more context to the LLM. In the first approach, prompts were almost empty and didn’t clearly specify what we were looking for.
The results were better, but still limited.
	3.	Approach 3
This approach combined the previous year’s Annual Review, the 9 indicators, and the financial statements.
The results were much closer to actual business needs.
	4.	Approach 4
I tested this mainly to prove a point to Gabor: giving the LLM last year’s Annual Review and asking it to just update values and enrich the text makes no sense.
I could have argued this without running tests because the outcome was predictable, but I strongly believe it’s better to show one result once than explain the same thing a hundred times.

Based on that, I proposed a modular breakdown of the 2024 CRODA credit memo, where each subsection of the financial analysis is handled independently.
At the moment, we’re still missing the input documents and page mappings, but Gabor and Pascal are working on this for each credit memo (there are six now). We also pushed Pascal to define a proper, granular division of the financial analysis, with a clearer and revised ordering of source documents.

⸻

Framework benchmark (the part you were waiting for)

Now to the most anticipated part for you: the benchmark.

We carefully reviewed the presentation you shared with us, and based on our own experience, we agreed on the shortlist:
	•	LangGraph
	•	CrewAI
	•	Google Agent SDK

Our evaluation criteria included:
	•	Level of abstraction
	•	Multi-agent orchestration
	•	Tool integration
	•	State and memory management
	•	Support for long-running workflows

To keep it short:
I was personally convinced by LangGraph because of its high modularity, its ability to support complex long-running workflows, and its strong state management. However, a major drawback is the lack of native evaluation and testing frameworks, especially for RAG and agent evaluation.

Rafal, on the other hand, was more convinced by the Google Agent SDK, mainly because of its strong evaluation, tracking, and observability features, as well as its support for long-running workflows and robust memory handling.

As a result, we decided that:
	•	I would test LangGraph
	•	Rafal would test Google Agent SDK

We did not spend time testing CrewAI, as it has already received very negative feedback within the BNP ecosystem, particularly around limitations in long-running memory, evaluation, and tracking.

⸻

LangGraph experiments

As I mentioned earlier, I strongly believe that modularity is key to tackling financial analysis properly. We deal with multiple prompts, and each one needs very precise context, without exceeding token limits—especially since the source documents are already large.

I started with a parallel design pattern:
	•	Inputs:
	•	FY2024 CRODA results (chunked)
	•	CRF with 11 indicators
	•	Current year and previous year values
	•	These inputs were fed into three parallel tasks, each executed by the agent.
	•	Each task made a separate LLM call with a different prompt, targeting a specific sub-analysis.
	•	The results were then combined into a single output.

The execution time was around 12 seconds using GPT-OSS-120B. In my opinion, the output was quite solid, but it still needs to be validated with Gabor.

I then tested a loop-based agent design:
	•	Same inputs as before
	•	The profitability section was generated through three sequential LLM calls, then aggregated
	•	The result was passed to a second agent acting as a critic
	•	Based on the critic’s feedback and verdict, the system either stopped or iterated again, with a maximum of three iterations

In this case, all three iterations were used, resulting in a total execution time of approximately 125 seconds.
