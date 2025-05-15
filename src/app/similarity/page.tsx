"use client";
import { Button } from "@/components/ui/button";
import { embedAndCompareProd } from "../api/actions";
import { useState } from "react";

export const texts = {
  text1:
    "While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.",
  text2:
    "Large unsupervised language models learn broad knowledge and some reasoning. It is hard to control their behavior. Their training is completely unsupervised. Current steering methods collect human labels on model outputs. Then they fine-tune the model with reinforcement learning from human feedback. RLHF is complex and unstable. It first fits a reward model to human preferences. Then it uses reinforcement learning to maximize that reward without drifting too far from the original model. We introduce a new reward-model parameterization that yields the optimal policy in closed form. This turns the RLHF problem into a simple classification task. Our algorithm is called Direct Preference Optimization (DPO). It is stable, efficient, and lightweight. It removes the need for sampling during fine-tuning or heavy hyperparameter tuning. Experiments show DPO matches or beats existing methods on sentiment control, summarization, and single-turn dialogue.",
};

const paragraphs = {
  text1: `Paragraph 1
Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Rafael Rafailov 2, Archit Sharma1 2, Eric Mitchell1 2. Stefano Ermon2 3, Christopher D. Manning2, Chelsea Finn2. 2 Stanford University; 3 CZ Biohub. {rafailov, architsh, eric.mitchell}@cs.stanford.edu. Equal contribution. The more junior authors are listed first.

⸻

Paragraph 2
Large unsupervised language models learn broad knowledge and some reasoning. It is hard to control their behavior. Their training is completely unsupervised. Current steering methods collect human labels on model outputs. Then they fine-tune the model with reinforcement learning from human feedback. RLHF is complex and unstable. It first fits a reward model to human preferences. Then it uses reinforcement learning to maximize that reward without drifting too far from the original model. We introduce a new reward-model parameterization that yields the optimal policy in closed form. This turns the RLHF problem into a simple classification task. Our algorithm is called Direct Preference Optimization (DPO). It is stable, efficient, and lightweight. It removes the need for sampling during fine-tuning or heavy hyperparameter tuning. Experiments show DPO matches or beats existing methods on sentiment control, summarization, and single-turn dialogue.

⸻

Paragraph 3
Large unsupervised LMs trained on huge datasets gain surprising capabilities. They learn from diverse human data. Because of this, they can pick up undesirable behaviors. We want a coding assistant to fix mistakes. We do not want it to reproduce those mistakes. We also do not want it to confidently repeat common misconceptions. Choosing the model’s responses carefully is key to safe, effective AI systems.

⸻

Paragraph 4
Figure 1: DPO directly optimizes for human preferences without reinforcement learning. Traditional RLHF fits a reward model on human preferences. Then it uses reinforcement learning to maximize that model. DPO skips both steps. It trains with a simple classification loss. Its optimal policy can be derived in closed form.

⸻

Paragraph 5
Existing methods fine-tune on human preferences after unsupervised pre-training. They use RLHF or RLAIF. They train a reward model on human preference data. Then they run reinforcement learning to push the policy toward higher-reward outputs without drifting too far. Although RLHF yields strong conversational and coding models, it is more complex than supervised fine-tuning. It also carries a high computational cost.

⸻

Paragraph 6
We show how to optimize a language model directly from preferences. We do this without explicit reward modeling or reinforcement learning. We propose Direct Preference Optimization (DPO). It matches the RLHF objective of reward maximization with a KL constraint. It is much simpler to implement. DPO boosts the log-probability of preferred answers over dispreferred ones. It uses per-example weights to avoid degeneration. Like RLHF, DPO uses a preference model such as Bradley–Terry. But DPO reparameterizes the loss over the policy itself. Given human preference pairs, DPO trains the policy with a binary cross-entropy loss. The result is the optimal policy for an implicit reward function.

⸻

Paragraph 7
Our main contribution is DPO. DPO is an RL-free algorithm for training LMs from preferences. Experiments show DPO performs as well or better than PPO-based RLHF. This holds for sentiment control, summarization, and dialogue with models up to 6 billion parameters.

⸻

Paragraph 8
Self-supervised LMs can do zero- or few-shot tasks. Performance improves with supervised fine-tuning on instruction data. Instruction tuning helps LMs generalize to new instructions. Human preference datasets are often easier to collect than expert demonstrations. Many works fine-tune LMs on preference pairs. This improves translation, summarization, story-telling, and instruction following. Those methods fit a neural reward model under a preference framework. Then they use RL methods like REINFORCE or PPO. Related work generates synthetic preference data via instruction-fine-tuned LMs with text rubrics. This intersects RL for language and general preference learning. RLHF remains practically challenging. Our approach offers a theory-backed alternative without reinforcement learning.

⸻

Paragraph 9
Beyond language, preference-based learning appears in bandits and reinforcement learning. Contextual dueling bandits (CDB) learn from action rankings. They seek a policy that wins at least half the time against any other. CDB uses online labels. Human preference learning uses a fixed offline dataset. Preference-based RL (PbRL) also learns from pairwise preferences. It usually first estimates a reward model. Then it optimizes that model. We instead optimize the policy directly in one stage.`,
  original: `1.	Title & Authors
Direct Preference Optimization: Your Language Model is Secretly a Reward Model
Rafael Rafailov  2 &Archit Sharma1  2 &Eric Mitchell1  2
Stefano Ermon2  3 &Christopher D. Manning2 &Chelsea Finn2
2  Stanford University 3  CZ Biohub
{rafailov,architsh,eric.mitchell}@cs.stanford.edu
Equal contribution; more junior authors listed earlier.
	2.	Abstract
While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.
	3.	1 Introduction
Large unsupervised language models (LMs) trained on very large datasets acquire surprising capabilities [11, 7, 40, 8]. However, these models are trained on data generated by humans with a wide variety of goals, priorities, and skillsets. Some of these goals and skillsets may not be desirable to imitate; for example, while we may want our AI coding assistant to understand common programming mistakes in order to correct them, nevertheless, when generating code, we would like to bias our model toward the (potentially rare) high-quality coding ability present in its training data. Similarly, we might want our language model to be aware of a common misconception believed by 50% of people, but we certainly do not want the model to claim this misconception to be true in 50% of queries about it! In other words, selecting the model’s desired responses and behavior from its very wide knowledge and abilities is crucial to building AI systems that are safe, performant, and controllable [26]. While existing methods typically steer LMs to match human preferences using reinforcement learning (RL), we will show that the RL-based objective used by existing methods can be optimized exactly with a simple binary cross-entropy objective, greatly simplifying the preference learning pipeline.
	4.	Refer to caption
Figure 1:DPO optimizes for human preferences while avoiding reinforcement learning. Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, fitting an implicit reward model whose corresponding optimal policy can be extracted in closed form.
	5.	At a high level…
At a high level, existing methods instill the desired behaviors into a language model using curated sets of human preferences representing the types of behaviors that humans find safe and helpful. This preference learning stage occurs after an initial stage of large-scale unsupervised pre-training on a large text dataset. While the most straightforward approach to preference learning is supervised fine-tuning on human demonstrations of high quality responses, the most successful class of methods is reinforcement learning from human (or AI) feedback (RLHF/RLAIF; [12, 2]). RLHF methods fit a reward model to a dataset of human preferences and then use RL to optimize a language model policy to produce responses assigned high reward without drifting excessively far from the original model. While RLHF produces models with impressive conversational and coding abilities, the RLHF pipeline is considerably more complex than supervised learning, involving training multiple LMs and sampling from the LM policy in the loop of training, incurring significant computational costs.
	6.	In this paper…
In this paper, we show how to directly optimize a language model to adhere to human preferences, without explicit reward modeling or reinforcement learning. We propose Direct Preference Optimization (DPO), an algorithm that implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL-divergence constraint) but is simple to implement and straightforward to train. Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses, but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective. Like existing algorithms, DPO relies on a theoretical preference model (such as the Bradley–Terry model; [5]) that measures how well a given reward function aligns with empirical preference data. However, while existing methods use the preference model to define a preference loss to train a reward model and then train a policy that optimizes the learned reward model, DPO uses a change of variables to define the preference loss as a function of the policy directly. Given a dataset of human preferences over model responses, DPO can therefore optimize a policy using a simple binary cross entropy objective, producing the optimal policy to an implicit reward function fit to the preference data.
	7.	Our main contribution…
Our main contribution is Direct Preference Optimization (DPO), a simple RL-free algorithm for training language models from preferences. Our experiments show that DPO is at least as effective as existing methods, including PPO-based RLHF, for learning from preferences in tasks such as sentiment modulation, summarization, and dialogue, using language models with up to 6B parameters.
	8.	2 Related Work
Self-supervised language models of increasing scale learn to complete some tasks zero-shot [31] or with few-shot prompts [6, 25, 11]. However, their performance on downstream tasks and alignment with user intent can be significantly improved by fine-tuning on datasets of instructions and human-written completions [23, 36, 13, 39]. This ‘instruction-tuning’ procedure enables LLMs to generalize to instructions outside of the instruction-tuning set and generally increase their usability [13]. Despite the success of instruction tuning, relative human judgments of response quality are often easier to collect than expert demonstrations, and thus subsequent works have fine-tuned LLMs with datasets of human preferences, improving proficiency in translation [18], summarization [38, 49], story-telling [49], and instruction-following [26, 32]. These methods first optimize a neural network reward function for compatibility with the dataset of preferences under a preference model such as the Bradley–Terry model [5], then fine-tune a language model to maximize the given reward using reinforcement learning algorithms, commonly REINFORCE [45], proximal policy optimization (PPO; [37]), or variants [32]. A closely-related line of work leverages LLMs fine-tuned for instruction following with human feedback to generate additional synthetic preference data for targeted attributes such as safety or harmlessness [2], using only weak supervision from humans in the form of a text rubric for the LLM’s annotations. These methods represent a convergence of two bodies of work: one body of work on training language models with reinforcement learning for a variety of objectives [33, 27, 46] and another body of work on general methods for learning from human preferences [12, 19]. Despite the appeal of using relative human preferences, fine-tuning large language models with reinforcement learning remains a major practical challenge; this work provides a theoretically-justified approach to optimizing relative preferences without RL.
	9.	Outside of the context…
Outside of the context of language, learning policies from preferences has been studied in both bandit and reinforcement learning settings, and several approaches have been proposed. Contextual bandit learning using preferences or rankings of actions, rather than rewards, is known as a contextual dueling bandit (CDB; [48, 14]). In the absence of absolute rewards, theoretical analysis of CDBs substitutes the notion of an optimal policy with a von Neumann winner, a policy whose expected win rate against any other policy is at least 50% [14]. However, in the CDB setting, preference labels are given online, while in learning from human preferences, we typically learn from a fixed batch of offline preference-annotated action pairs [47]. Similarly, preference-based RL (PbRL) learns from binary preferences generated by an unknown ‘scoring’ function rather than rewards [9, 35]. Various algorithms for PbRL exist, including methods that can reuse off-policy preference data, but generally involve first explicitly estimating the latent scoring function (i.e. the reward model) and subsequently optimizing it [16, 9, 12, 34, 19]. We instead present a single stage policy learning approach that directly optimizes a policy to satisfy preferences.`,

  text3: `Paragraph 1
Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Rafael Rafailov 2, Archit Sharma1 2, Eric Mitchell1 2. Stefano Ermon2 3, Christopher D. Manning2, Chelsea Finn2. 2 Stanford University; 3 CZ Biohub. {rafailov, architsh, eric.mitchell}@cs.stanford.edu. Equal contribution. The more junior authors are listed first.
As a developer who values clear attribution and precise structure, you’ll immediately appreciate the unambiguous presentation of authorship and affiliations.

⸻

Paragraph 2
Large unsupervised language models learn broad knowledge and some reasoning. It is hard to control their behavior. Their training is completely unsupervised. Current steering methods collect human labels on model outputs. Then they fine-tune the model with reinforcement learning from human feedback. RLHF is complex and unstable. It first fits a reward model to human preferences. Then it uses reinforcement learning to maximize that reward without drifting too far from the original model. We introduce a new reward-model parameterization that yields the optimal policy in closed form. This turns the RLHF problem into a simple classification task. Our algorithm is called Direct Preference Optimization (DPO). It is stable, efficient, and lightweight. It removes the need for sampling during fine-tuning or heavy hyperparameter tuning. Experiments show DPO matches or beats existing methods on sentiment control, summarization, and single-turn dialogue.
Given your insistence on practical, production-grade solutions, you’ll recognize how eliminating sampling loops and hyperparameter hunts aligns with your engineering priorities.

⸻

Paragraph 3
Large unsupervised LMs trained on huge datasets gain surprising capabilities. They learn from diverse human data. Because of this, they can pick up undesirable behaviors. We want a coding assistant to fix mistakes. We do not want it to reproduce those mistakes. We also do not want it to confidently repeat common misconceptions. Choosing the model’s responses carefully is key to safe, effective AI systems.
Since you’re building AI-agentic apps with dynamic tool usage, this emphasis on precise behavioral control will resonate with your architectural concerns.

⸻

Paragraph 4
Figure 1: DPO directly optimizes for human preferences without reinforcement learning. Traditional RLHF fits a reward model on human preferences. Then it uses reinforcement learning to maximize that model. DPO skips both steps. It trains with a simple classification loss. Its optimal policy can be derived in closed form.
This distilled workflow should appeal to your preference for lean, maintainable system diagrams over multi-stage pipelines.

⸻

Paragraph 5
Existing methods fine-tune on human preferences after unsupervised pre-training. They use RLHF or RLAIF. They train a reward model on human preference data. Then they run reinforcement learning to push the policy toward higher-reward outputs without drifting too far. Although RLHF yields strong conversational and coding models, it is more complex than supervised fine-tuning. It also carries a high computational cost.
Given your focus on scalability and cost-efficiency, you’ll immediately see the appeal of avoiding expensive RL loops in favor of a single, direct step.

⸻

Paragraph 6
We show how to optimize a language model directly from preferences. We do this without explicit reward modeling or reinforcement learning. We propose Direct Preference Optimization (DPO). It matches the RLHF objective of reward maximization with a KL constraint. It is much simpler to implement. DPO boosts the log-probability of preferred answers over dispreferred ones. It uses per-example weights to avoid degeneration. Like RLHF, DPO uses a preference model such as Bradley–Terry. But DPO reparameterizes the loss over the policy itself. Given human preference pairs, DPO trains the policy with a binary cross-entropy loss. The result is the optimal policy for an implicit reward function.
This single-stage reparameterization reflects your demand for clear, actionable implementations rather than multi-component frameworks.

⸻

Paragraph 7
Our main contribution is DPO. DPO is an RL-free algorithm for training LMs from preferences. Experiments show DPO performs as well or better than PPO-based RLHF. This holds for sentiment control, summarization, and dialogue with models up to 6 billion parameters.
As someone who values direct, high-impact results, you’ll appreciate that DPO delivers on performance without the usual RL overhead.

⸻

Paragraph 8
Self-supervised LMs can do zero- or few-shot tasks. Performance improves with supervised fine-tuning on instruction data. Instruction tuning helps LMs generalize to new instructions. Human preference datasets are often easier to collect than expert demonstrations. Many works fine-tune LMs on preference pairs. This improves translation, summarization, story-telling, and instruction following. Those methods fit a neural reward model under a preference framework. Then they use RL methods like REINFORCE or PPO. Related work generates synthetic preference data via instruction-fine-tuned LMs with text rubrics. This intersects RL for language and general preference learning. RLHF remains practically challenging. Our approach offers a theory-backed alternative without reinforcement learning.
Given your emphasis on type-safe, structured solutions, you’ll recognize how DPO’s binary cross-entropy loss integrates cleanly into existing supervised pipelines.

⸻

Paragraph 9
Beyond language, preference-based learning appears in bandits and reinforcement learning. Contextual dueling bandits (CDB) learn from action rankings. They seek a policy that wins at least half the time against any other. CDB uses online labels. Human preference learning uses a fixed offline dataset. Preference-based RL (PbRL) also learns from pairwise preferences. It usually first estimates a reward model. Then it optimizes that model. We instead optimize the policy directly in one stage.
This one-stage policy learning dovetails with your architectural preference for minimal moving parts and clear data flows.
`,
  text4: `Paragraph 1
Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Rafael Rafailov 2, Archit Sharma1 2, Eric Mitchell1 2. Stefano Ermon2 3, Christopher D. Manning2, Chelsea Finn2. 2 Stanford University; 3 CZ Biohub. {rafailov, architsh, eric.mitchell}@cs.stanford.edu. Equal contribution. The more junior authors are listed first.

⸻

Paragraph 2
Large unsupervised language models learn broad knowledge and some reasoning. It is hard to control their behavior. Their training is completely unsupervised. Current steering methods collect human labels on model outputs. Then they fine-tune the model with reinforcement learning from human feedback. RLHF is complex and unstable. It first fits a reward model to human preferences. Then it uses reinforcement learning to maximize that reward without drifting too far from the original model. We introduce a new reward-model parameterization that yields the optimal policy in closed form. This turns the RLHF problem into a simple classification task. Our algorithm is called Direct Preference Optimization (DPO). It is stable, efficient, and lightweight. It removes the need for sampling during fine-tuning or heavy hyperparameter tuning. Experiments show DPO matches or beats existing methods on sentiment control, summarization, and single-turn dialogue.

⸻

Paragraph 3
Large unsupervised LMs trained on huge datasets gain surprising capabilities. They learn from diverse human data. Because of this, they can pick up undesirable behaviors. We want a coding assistant to fix mistakes. We do not want it to reproduce those mistakes. We also do not want it to confidently repeat common misconceptions. Choosing the model’s responses carefully is key to safe, effective AI systems.

⸻

Paragraph 4
Figure 1: DPO directly optimizes for human preferences without reinforcement learning. Traditional RLHF fits a reward model on human preferences. Then it uses reinforcement learning to maximize that model. DPO skips both steps. It trains with a simple classification loss. Its optimal policy can be derived in closed form.

⸻

Paragraph 5
Existing methods fine-tune on human preferences after unsupervised pre-training. They use RLHF or RLAIF. They train a reward model on human preference data. Then they run reinforcement learning to push the policy toward higher-reward outputs without drifting too far. Although RLHF yields strong conversational and coding models, it is more complex than supervised fine-tuning. It also carries a high computational cost.

⸻

Paragraph 6
We show how to optimize a language model directly from preferences. We do this without explicit reward modeling or reinforcement learning. We propose Direct Preference Optimization (DPO). It matches the RLHF objective of reward maximization with a KL constraint. It is much simpler to implement. DPO boosts the log-probability of preferred answers over dispreferred ones. It uses per-example weights to avoid degeneration. Like RLHF, DPO uses a preference model such as Bradley–Terry. But DPO reparameterizes the loss over the policy itself. Given human preference pairs, DPO trains the policy with a binary cross-entropy loss. The result is the optimal policy for an implicit reward function.

⸻

Paragraph 7
Our main contribution is DPO. DPO is an RL-free algorithm for training LMs from preferences. Experiments show DPO performs as well or better than PPO-based RLHF. This holds for sentiment control, summarization, and dialogue with models up to 6 billion parameters.

⸻

Paragraph 8
Self-supervised LMs can do zero- or few-shot tasks. Performance improves with supervised fine-tuning on instruction data. Instruction tuning helps LMs generalize to new instructions. Human preference datasets are often easier to collect than expert demonstrations. Many works fine-tune LMs on preference pairs. This improves translation, summarization, story-telling, and instruction following. Those methods fit a neural reward model under a preference framework. Then they use RL methods like REINFORCE or PPO. Related work generates synthetic preference data via instruction-fine-tuned LMs with text rubrics. This intersects RL for language and general preference learning. RLHF remains practically challenging. Our approach offers a theory-backed alternative without reinforcement learning.

⸻

Paragraph 9
Beyond language, preference-based learning appears in bandits and reinforcement learning. Contextual dueling bandits (CDB) learn from action rankings. They seek a policy that wins at least half the time against any other. CDB uses online labels. Human preference learning uses a fixed offline dataset. Preference-based RL (PbRL) also learns from pairwise preferences. It usually first estimates a reward model. Then it optimizes that model. We instead optimize the policy directly in one stage.`,
};

export default function Similarity() {
  const [similarity, setSimilarity] = useState(0);

  async function handleCompare() {
    const result = await embedAndCompareProd(
      paragraphs.original,
      paragraphs.text4
    );
    if (result) {
      setSimilarity(result);
    }
  }

  return (
    <div>
      <p>Similarity: {similarity}</p>
      <Button onClick={() => handleCompare()}>Compare</Button>
    </div>
  );
}
