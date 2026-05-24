## What are Artificial Intelligence (AI) Agents?

AI agents are software systems designed to autonomously perform tasks on behalf of users or other systems. They perceive their environment, process information, and make decisions to achieve specific goals, adapting as needed.
These agents use advanced techniques, such as machine learning and natural language processing, to interact with humans and other systems. They can be integrated into websites and apps to enhance the customer experience, acting as virtual assistants, providing mental health support, simulating interviews, and other related tasks.
AI agents are considered the great evolution of the moment in AI. The architecture of AI agents generally includes:
- **Data Collection**: They receive information from the environment through sensors or virtual sources;
- **Processing and Analysis**: They use LLMs (Large Language Models) or SLMs (Small Language Models) to interpret the collected data;
- **Decisions and Actions**: Based on the analysis, they execute actions to achieve their objectives;

There are different types of AI Agents including:
- **Reactive Agents**: Respond directly to environmental stimuli without storing information about past experiences;
- **Agents with memory**: Store and use information to improve their performance over time;
- **Autonomous and Social Agents**: Interact with other agents and systems to collaborate on more complex objectives;

The application of AI Agents is expanding rapidly and will revolutionize the software market.


### Principles Defining AI Agents

All software autonomously performs different tasks, as determined by the software developer. So what makes AI Agents special?

AI Agents are “rational” agents. They make “rational” decisions based on their perceptions and data to produce results. We simply give them a task, the main guidelines, and the AI ​​Agents find ways to deliver a solution.

Therefore:
- AI Agents should have maximum autonomy, but always with human supervision;
- AI Agents use their own intermediate responses to improve the final response;
- AI Agents can query databases or the web, thus using reliable data sources. Various tools (function calling) can be used;
- AI Agents appear to have intelligence, but they don't. Everything is still high-speed processing due to the training of models with large volumes of data;
- AI Agents are based on LLMs, Mathematics, and Statistics.


---

## Large Language Models (LLMs) vs. Small Language Models (SLMs)

Both are categories of AI models focused on natural language processing. The main difference between them lies in the scale of their parameters and processing capacity.

LLMs are large-scale models, trained with billions of parameters and enormous volumes of textual data. They have a high capacity for text generation, understanding complex context, and adapting to diverse tasks without the need for specific additional training. However, their use demands robust infrastructure, significant computational power, and high costs.

On the other hand, SLMs are smaller models, optimized to run with fewer computational resources. Although they have fewer parameters and a reduced capacity to understand and generate complex texts, they are more efficient for specific applications and can be used on local devices such as cell phones and personal computers. This makes them ideal for less demanding tasks, such as lightweight virtual assistants and business applications that need quick and economical responses.

The essential difference between the two lies in the balance between performance and efficiency. While LLMs offer more sophisticated and versatile results, SLMs are more accessible, faster, and better suited for scenarios where resource savings are a priority.


---

## Prompt Engineering - The Secret to Using Generative AI Effectively

Prompt engineering is the practice of formulating clear and specific instructions for Generative AI models, aiming to obtain answers aligned with user expectations. A "prompt" is the textual input provided to the AI, which can range from a simple question to a complex set of guidelines. How you structure these instructions directly impacts the quality of the generated responses.

To create effective prompts, it is crucial to be clear and specific, providing details that guide the AI ​​precisely. For example, when requesting a summary of an article, instead of simply saying "Summarize this article," you could specify: "Summarize this article in 100 words, highlighting the main arguments and conclusions." This specificity helps the AI ​​understand exactly what is expected, resulting in more relevant and useful answers.

Furthermore, providing adequate context is fundamental. When interacting with the AI, including additional information or a specific scenario can better guide the generation of answers. For example, when asking AI to write a professional email, you can provide details about the recipient and the purpose of the email, ensuring that the response is more aligned with your needs.

Prompt engineering also involves experimentation and continuous refinement of instructions. Testing different approaches and adjusting prompts based on the responses obtained allows you to improve the interaction with AI, ensuring increasingly satisfactory results. This practice is essential to maximize the potential of generative AI tools, making them more effective and aligned with user objectives.

Mastering prompt engineering is fundamental to using generative AI efficiently. By formulating clear, specific, and contextualized instructions, and by continuously refining these instructions based on the results obtained, it is possible to obtain more accurate and useful responses, optimizing the interaction with AI models.

### What is Prompt Engineering?
Prompt engineering is the practice of structuring, phrasing, and designing inputs (prompts) to get the most accurate, relevant, and useful responses out of Large Language Models (LLMs).

Think of it less like programming with rigid code and more like giving clear, strategic instructions to a highly capable but literal-minded assistant. Because LLMs work by predicting the next most likely words based on patterns in their training data, how you frame your request completely changes the context they use to generate an answer.

#### Core Elements of a Perfect Prompt
Before looking at specific techniques, a highly effective prompt usually combines these components:

* Role/Persona: Who the AI should act like (e.g., "Act as a senior data analyst").
* Instruction/Task: What you explicitly want it to do (e.g., "Summarize this report").
* Context: The background information, constraints, or data it needs to consider.
* Output Indicator: The desired format, length, or tone (e.g., "Give me a 3-bullet-point summary written for a non-technical manager").

#### The Most Popular Prompt Engineering Techniques
Depending on how complex your task is, you can use different structuring strategies to dramatically improve the AI's output.

**1. Few-Shot Prompting (Providing Examples)**
Instead of just describing what you want, you give the model a few examples of input-output pairs. This establishes a pattern for the AI to replicate. It is incredibly useful for formatting data, sentiment analysis, or tone matching.

Example:
* Input: "I loved the movie!" -> Sentiment: Positive
* Input: "The food arrived cold and late." -> Sentiment: Negative
* Input: "The package arrived on Tuesday." -> Sentiment:


**2. Chain-of-Thought (CoT) Prompting**
For complex logic, math, or multi-step reasoning, asking the AI to "think step-by-step" prevents it from rushing to an incorrect conclusion. By forcing the model to generate its reasoning path out loud, it naturally arrives at a more accurate final answer.

Example:
"A retail store starts with 50 shirts. They sell 12 in the morning and receive a shipment of 20 more in the afternoon. Let's think step-by-step to find how many shirts they have now."

**3. Role Prompting (Persona-Based)**
Assigning a specific persona gives the AI a frame of reference. It automatically adjusts its vocabulary, tone, depth of knowledge, and hidden assumptions to match that character.

Example:
"Act as an experienced Python tech lead. Review this Flask code snippet for potential security vulnerabilities and performance bottlenecks."


**4. System/User Separation**
Popularized by modern chat interfaces and APIs, this technique separates high-level operational rules from the actual request.
* System Prompt: Sets the core behavior, constraints, and boundaries (e.g., "You are a helpful assistant. Never use jargon. Always respond in Markdown tables.").
* User Prompt: The specific dynamic question or task at hand.


**5. Generated Knowledge Prompting**
For deep or niche topics, you first ask the AI to generate relevant facts or background information about a concept before asking it to write the final response or make a decision. This ensures the correct context is fresh in the model's short-term memory (context window) before it tackles the main task.


**Quick Tips for Daily Auditing**
* Be direct instead of polite: The AI doesn't mind if you skip "please" and "thank you." Use strong command verbs like Analyze, Rewrite, Compare, or Extract.
* Use Delimiters: Use triple quotes ("""), XML tags (<data></data>), or backticks (`````) to clearly separate your instructions from the text or data you want the AI to process.
* Tell it what TO do, not just what NOT to do: Negative constraints (e.g., "Don't make it boring") are harder for LLMs to process than positive actions (e.g., "Write it in an engaging, conversational tone").

---


##### Where I stoped: **Principles Defining AI Agents**
