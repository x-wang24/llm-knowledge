# Workflow & Agent

1.  **介绍**
    

workflow：预定义流程编排大模型和工具

agent: 大模型自主动态决定流程和调用工具

workflow/agent更像是Agentic的不同表现形式，在一个Agentic系统里，agent与workflow并非二元划分，有白无黑，在一个更多是两者结合,如果在一个Multi-Agent应用里，局部一个agent也可以是workflow模式,在一个复杂的workflow里面，局部也可以有一个agent ,agent也可以当作tool使用

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/027180d0-9c58-48b1-b12b-e3dd02e3e70d.png?x-oss-process=image/crop,x_0,y_34,w_512,h_431/ignore-error,1)

2.  **Workflow & Agent基础模块**
    

![7b622e206ca2fd4174b4d6988826742.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/183c532d-436e-485a-9d87-4283879df950.png)

agent基础模块是一个增强型的大模型，主要包括检索，工具，记忆等增强功能，都是补充外部知识给大模型,agent自己也作为tool，agent as tools

agent也可以从下面这个图理解，核心结构包括模型,工具,编排，模型model用来做核心决策，可以是LLM,也可以是VLM，例如在GUI agent里，VLM就更适合做决策，随着模型发展，moel更多是以多模态大模型为主导，工具tools补充外界知识，编排层orchestration负责信息接受，记忆，状态管理，推理，指导下一步行动决策等

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/923e28be-50cb-40f7-8b3e-2e9ae0a5be85.png)

model vs agent

![4cd1374bcc10710f399f9aa3a510f5e.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/7fe10a52-ab79-49bd-a9bd-c41be55568ef.png)

3.  **常见基本模式**
    

**3.1 prompt** **chain：**提示链

![57ac82b35dc001e23011243658c4b31.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/ac71ae07-a9e1-4f78-836d-1f6972a8f4ca.png)

提示链将任务分解为一系列顺序的子任务

*   每个 LLM call 处理前一个 LLM call 的输出；
    
*   可以在中间任何步骤添加检查点（图中的 “Gate”），以确保处理过程仍在正轨上
    

      **适用场景**：

试用于能够把大任务分解成多个步骤的子任务，拆解后任务更简单，更容易完成

**场景举例**：

 1）编写故事，先生成故事大纲再根据大纲写具体细节

 2）nl2sql，先根据查询文本和schema生成初版查询sql语句，再检查sql是否正确并有错误修改,另外任何生成需要检查修改的都可以，

 3）营销文案生成

```latex
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    trace,
    ItemHelpers,
    MessageOutputItem,
    ModelSettings
)
# 使用自定义模型
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)
CUSTOM_MODEL_PROVIDER = CustomModelProvider()

story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
    model_settings=ModelSettings(temperature=0.01, top_p=0.9, tool_choice='required'),
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=client,
    ))

class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool

outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions='Read the given story outline, and judge the quality. Also, determine if it is a scifi story,response json format：{"good_quality":1, "is_scifi":1}',
    output_type=OutlineCheckerOutput,
    model_settings=ModelSettings(temperature=0.01, top_p=0.9, tool_choice='required'),
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=client,
    ))

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,
    model_settings=ModelSettings(temperature=0.01, top_p=0.9, tool_choice='required'),
    model=OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=client,
    ))

async def main():
    input_prompt = input("What kind of story do you want? ")

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):
        # 1. Generate an outline
        outline_result = await Runner.run(
            story_outline_agent,
            input_prompt,
        )
        print("Outline generated")

        # 2. Check the outline
        outline_checker_result = await Runner.run(
            outline_checker_agent,
            outline_result.final_output,
        )

        # 3. Add a gate to stop if the outline is not good quality or not a scifi story
        assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
        if not outline_checker_result.final_output.good_quality:
            print("Outline is not good quality, so we stop here.")
            exit(0)
        if not outline_checker_result.final_output.is_scifi:
            print("Outline is not a scifi story, so we stop here.")
            exit(0)
        print("Outline is good quality and a scifi story, so we continue to write the story.")
        # 4. Write the story
        story_result = await Runner.run(
            story_agent,
            outline_result.final_output,
        )
        print(f"Story: {story_result.final_output}")
if __name__ == "__main__":
    asyncio.run(main())


    
```

![bc26e6c7d11d6d3574961ed009fec37.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/386ef4b6-863b-49b7-880d-4fc65afe49c8.png)

**3.2  Routing：路由**

 ![73b742271d986665e37252d11a51f62.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/7ecd1f8a-eeda-4241-aa45-c693f16b30ea.png)

通过路由对输入进行分类，并将其转发到专门的后续任务

**适用场景**：

1.适用于存在不同类别的复杂任务，而且这些类别分开处理时，都能得到更好的效果

2.能够进行准确分类的场景，至于用大模型分类还是传统模型关系不大

**场景举例**：

 1）智能客服，将不同类型的用户问题转发到不同的下游流程，如常规问答，购票，行程规划

 2）大小模型路由，简单常规问题路由到速度快，性价比高的模型，复杂困难问题路由到更强大模型

 3）翻译,翻译成不同语言，不同语言调用各自模型

```latex
import asyncio
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from pydantic import BaseModel
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import (
    Model,
    Agent,
    Runner,
    RunConfig,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    ModelProvider,
    OpenAIChatCompletionsModel,
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

# 使用自定义模型
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)
CUSTOM_MODEL_PROVIDER = CustomModelProvider()

# 法语agent
french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
)
# 西班牙语agent
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish",
)

# 英语agent
english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    msg = input("Hi! We speak French, Spanish and English. How can I help? ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        with trace("Routing example"):
            result = Runner.run_streamed(
                agent,
                input=inputs,
                run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent

if __name__ == "__main__":
    asyncio.run(main())

    
```

![e6b407d339ad4792c8f071e3a5121d8.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/45cc4e42-cde2-4509-9ebf-fa8c1770c2e4.png)

**3.3   Parallelization：并行**

![4beab6b0342298070e34fc34a41b486.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/164ea1af-3f5e-48c6-b2b9-2d39b98d0291.png)

多个任务同时进行，然后对输出进行聚合处理

1.分段（Sectioning）：类似 MapReduce，将任务分解为独立的子任务并行运行，最后对输出进行聚合

2.投票（Voting）：相同的任务并行执行多次，以获得多样化的输出

**适用场景**：

1.并行化可以提高任务的最终完成速度

2.需要多种视角或尝试，对所有结果进行对比，取最好的结果

**场景举例：**

 1）质量评估（Sectioning）,每个模型从不同方面评估对象能力，比如一个模型评价视频流畅度，一个评价画面美感，一个模型评估创意

 2）旁路安全检测（Sectioning），一个模型实例处理用户查询，另一个模型检测是否合规，这通常比让同一个模型实例同时请求响应和安全防护效果更好。

 3）Code review（voting），几个不同模型或者几个不同的提示审查并标记代码，寻找漏洞

 4）创意生成（voting），几个不同模型生成不同创意，选择最好创意

```latex
import asyncio
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    trace,
    ItemHelpers,
    MessageOutputItem,
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    ModelProvider,
    Model,
    OpenAIChatCompletionsModel,
    RunConfig
)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


# set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
"""
# 韩语agent
Korean_agent = Agent(
    name="Korean_agent",
    instructions="You translate the user's message to Korean",
    handoff_description="An chinese to Korean translator",

)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Korean translation from the given options.",
    model=MODEL_NAME,
)

async def main():
    msg = input('input:')
    with trace('parall'):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(Korean_agent, msg, run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)),
            Runner.run(Korean_agent, msg, run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)),
            Runner.run(Korean_agent, msg, run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)),

        )
        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
            ItemHelpers.text_message_outputs(res_3.new_items),
        ]
        translations = "\n\n".join(outputs)
        print(f"\n\nTranslations:\n\n{translations}")

        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
            run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER)
        )
        print("\n\n-----")

        print(f"Best translation: {best_translation.final_output}")

if __name__ == '__main__':
    asyncio.run(main())

    
```

![c1535a4d1919fdec3b70503b0851afd.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/80137666-a390-40d8-b4d3-4c368e9a657d.png)

**3.4**   **Orchestrator-workers：编排者-工作者**

![6d2e5c45994af9631e6bedc497fb5b5.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/239783cf-a0d3-4301-9f3c-261cdefe8479.png)

中心大模型动态地分解任务，将其委托给 worker大模型，并汇总它们的结果，类似于分布式架构的Master-Workers模式

**适用场景**：

1.需要灵活选择子任务的任务或者子任务提前无法确定，根据需要灵活安排子任务

**场景举例**：

 1）**搜索**：大模型根据输入分配到不同来源，每个来源收集信息，分析信息，汇总信息

```latex
import asyncio
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    trace,
    ItemHelpers,
    MessageOutputItem,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    ModelProvider,
    OpenAIChatCompletionsModel,
    Model,
    RunConfig
)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# set_tracing_disabled(disabled=True)

class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


CUSTOM_MODEL_PROVIDER = CustomModelProvider()

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
"""
spanish_agent = Agent(
    name="Spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An chinese to spanish translator",
    model=MODEL_NAME,
)
french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An chinese to french translator",
    model=MODEL_NAME,
)
Korean_agent = Agent(
    name="Korean_agent",
    instructions="You translate the user's message to Korean",
    handoff_description="An chinese to Korean translator",
    model=MODEL_NAME,
)
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[spanish_agent.as_tool(
        tool_name="translate_to_spanish",
        tool_description="Translate the user's message to Spanish",
    ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        Korean_agent.as_tool(
            tool_name="translate_to_Korean",
            tool_description="Translate the user's message to Korean",
        )]
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    # model=MODEL_NAME,
    instructions="You inspect translations, correct them if needed, and must produce a final concatenated response.",
)
async def main():
    msg = input("Hi! What would you like translated, and to which languages? ")
    with trace('Orchestrator evaluator'):
        orchestrator_result = await Runner.run(orchestrator_agent, msg,
                                               run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
                                               )

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")
        print(' orchestrator_result', orchestrator_result.to_input_list())
        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list(),
            run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
        )
        print(f"\n\nFinal response:\n{synthesizer_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())

    
```

![cb50a1bda92147d32c5917cca07b4f1.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/1a3366cd-0866-478f-93ab-e1254f03d3ff.png)

**3.5**   **Evaluator-optimizer：评估者-优化者**

![f8b14a896ea90721ccaedd4b74878e8.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/b23f0ea3-f35f-47b4-8db2-91a4a21918d4.png)

1个模型生成，1个模型评估反馈，形成闭环

**适用场景**：

1.有明确的评估标准，并且迭代式改进确实有效

**场景举例**：

 1）**文章润色**：先生成个初稿，不断评估反馈优化

 2）**复杂搜索任务，**需要多轮搜索和分析以收集全面信息，评估者决定是否需要进一步搜索

```latex

import asyncio
import os
from typing import Literal
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace
from dataclasses import dataclass
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "api_key"
MODEL_NAME = 'qwen-plus'

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

class CustomModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)
CUSTOM_MODEL_PROVIDER = CustomModelProvider()

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


story_out_gen = Agent(name='story_out_gen',
                      instructions = ("You generate a very short story outline based on the user's input."
        "If there is any feedback provided, use it to improve the outline.")
                      )

class EvaluationFeedback(BaseModel):
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent(name="evaluator",
        instructions=(
            "You evaluate a story outline and decide if it's good enough."
            "If it's not good enough, you provide feedback on what needs to be improved."
            'Never give it a pass on the first try. response example ,json format,must return score value {"feedback":"...", "score":"needs_improvement"}'

        ),
    output_type=EvaluationFeedback
        )

async def main():
    msg = input('input:')
    # msg ='写个爱情故事'
    input_items :list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline:str |None = None

    with trace('LLM as a judge'):
        while True:
            story_outline_result = await Runner.run(
                story_out_gen,
                input_items,
                run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),
            )
            input_items = story_outline_result.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(story_outline_result.new_items)
            print('Story outline generated')
            evaluator_result = await Runner.run(evaluator,input_items,
                                                 run_config=RunConfig(model_provider=CUSTOM_MODEL_PROVIDER),

                                                 )
            result:EvaluationFeedback = evaluator_result.final_output
            print(f"Evaluator score: {result.score}")
            if result.score == "pass":
                print("Story outline is good enough, exiting.")
                break

            print("Re-running with feedback")
            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})
        print(f"Final story outline: {latest_outline}")
if __name__ == "__main__":
    asyncio.run(main())

    
```

![2958593124be172b257dc38726b036b.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/5d3508f8-6750-4e5e-b5c0-5d2d1ccee466.png)

    **3.6**  **模式组合**

最成功的实现采用最简单，易组合模式，并非复杂的框架系统

上面这些模式为基础模式，可以根据需要灵活组合定制

关键原则：

1.  **尽可能定性定量评估性能，确定最优组合**
    
    2.  **仅在提升效果明显时增加复杂性**
        
    3.  **这些模式为可自由组合节点，并发固定模式**
        

              5种高效组合模式

                **1.路由 + 提示链** 

                  机制：根据问题分类不同，路由到不同提示链

     示例：客服问题先路由分类：订单/活动/技术，再由不同专业提示链处理

  **2.路由 + 并行化**

            机制：先分类任务，特点任务并行处理

     示例：内容审核分类后：文本/图片/视频/音频，复杂任务并行化评估投票

  **3.编排者-工作者 + 评估者-优化者**

                  机制：编排者分配任务，工作者执行，评估者反馈优化

     示例：内容生产过程中，编排者根据任务确定需要调用的服务，工作者执行，评估者优化迭代

  **4.提示链 +  评估者-优化者**

                  机制：在提示链关键节点使用评估-优化循环提升质量

     示例：剧本脚本创作过程中，先生成大纲，再细化大纲，再根据大纲进行创作，评估优化

  **5.混合模式系统**

                  机制：不同任务阶段使用合适模式

     示例：全功能客服问题先路由分类：简单问题提示链，复杂问题编排者-工作者，容错低环节评估者-优化者确保质量

4.  **agent架构与协议**
    

**single-agent：**单agent架构通过持续扩展工具集处理多样化任务，具有维护简单的优势。每个新增工具都扩展了代理能力边界，无需过早引入复杂编排逻辑

![f8f336e18b8adc3f0e432c0169d8572.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/a08ce8fa-9949-465f-bd3b-64389d29eed9.png)

**multi-agent主要有2类架构：**

**管理者模式**：中央管理者通过工具调用协调专业agent，保持执行上下文与用户体验统一，适合需要集中控制的场景,类似master-worker模式

**去中心化模式**：各agent平等协作，通过控制权转移处理专业任务，适合需要领域专家深度参与的场景

![c621bae30a12dee560ed5b6932a59d6.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/0283a2fb-ffa9-403e-b408-98a805a7449b.png)

  ![0e9b3d5c34011fb51a141f22ab34094.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/f75b2c98-25be-42b0-9923-28c0cba54a3a.png)

  **agent协议：**

![b1fec3f1ed6d288d90eeb656e2e402a.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/e95dcbc6-bc98-4c7d-8711-40f7b6862ad8.png)

按交互对象 (Object Orientation)分：分为 **Context-Oriented (面向上下文)** 和 **Inter-Agent (面向 Agent 间)** 

按应用场景 (Application Scenario)：分为 **General-Purpose (通用)** 和 **Domain-Specific** **(****特定领域****）**

![c104a8d2f02781913abdce20a718d85.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/ad1e76d9-1e99-45be-b211-e817347f4a89.png)

代表性协议：

*   **MCP:**​用于agent调用工具的场景，MCP在简单工作流中的高效性和控制力
    

![4c72ba169484eb22e803f7c17cf7deb.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/57de9e7c-e229-4d17-b734-6b12e784c67e.png)

*   **A2A:**​  
    用于企业内部复杂协作的场景，A2A在多代理系统中的灵活性和任务管理能力
    

![bd60b6c768e4e67102da2aabc6d323e.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/34328a4c-313f-43b3-bd01-7121bb347450.png)

*   **ANP:**​用于跨域代理通信的场景，ANP在标准化跨域交互中具有优势
    

![9c5bc2b57471fa5e8855a28e8943032.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/ed92d5bb-e475-495b-8439-ac4bf9e327d5.png)

*   **Agora:**​用于自然语言到协议生成的场景，具有将用户意图转换为结构化协议方面的能力
    

![2387aa128c78733a8d9e012cf0c3a55.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/e477b810-4e7f-4da8-a914-9f24828bee06.png)

4种协议方式例子对比

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/d6356d85-be63-4ff0-887a-7e4c83a33a39.png)

*   **MCP**：像个大总管。一个中央 Agent (MCP Travel Client) 负责调用所有外部服务（机票、酒店、天气），然后汇总信息生成计划。优点是简单可控，缺点是中心化依赖高，不易扩展。
    
*   **A2A**：像多部门协作。任务被分配给专门的 Agent（天气，交通、住宿、活动），这些 Agent 可以直接相互沟通（比如机票 Agent 直接问天气 Agent 获取信息），最后由一个协调者汇总。更灵活，适合企业内复杂协作。
    
*   **ANP**：像跨公司合作。不同领域的 Agent（航空公司、酒店、天气）通过标准化的协议进行跨域交互和协商。适合独立 Agent 之间基于明确接口的协作。
    
*   **Agora**：像个智能翻译官。先用自然语言理解用户需求，然后生成标准化的协议分发给各个专业 Agent（机票、酒店、天气、预算）。将自然语言处理与 Agent 执行分离，适应性强。
    

**多个协议之间可以协同，例如MCP+A2A，另外从另一个角度看，tool可以当作低自主性agent,agent也可以当作高自主性的tool，多类协议可以融合，相互也可以扩展其他协议功能，协议除基础功能外，还有一个需要注意点安全，隐私保护**

![20e8328781e05ebdfdc0c8741a526f2.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/f89dd397-a2d8-4cef-b168-6c4fd3c3dfc1.png)

**5.agent优化**

 为了agent准备知道何时调用tool,如何调用tool，以及上下文信息理解，需要进行agent优化，优化方法参考下图,优先使用无参数优化办法，无法得到预期效果再尝试参数优化

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/a/RMMqNel0gSLMJnxa/1d8745f8616a41d99defc62d30b6479f4071.png)

**基于参数优化的agent模型训练，训练整体以sft与rf方式结合使用，分阶段训练，除此之外，大部分大模型训练都经过多阶段训练，多策略训练，包括理解模型,编辑模型,生成模型**

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/mPdnpEk1mBRkbqw9/img/50d25073-136a-40df-bdb4-ff09b24e5463.png)

**6.应用框架**

对于简单,单点应用可以使用agent adk,对于复杂应用可以优先选择langgraph,langgraph定义图的方式和tensorflow类似，先预定义图再编译运行，agent sdk类似函数式

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/bd074456-91de-4991-83ec-943f026df451.png)

[Agent Framework comparison - Google 表格](https://docs.google.com/spreadsheets/d/1B37VxTBuGLeTSPVWtz7UMsCdtXrqV5hCjWkbHN8tfAo/edit?ref=blog.langchain.dev&gid=0#gid=0)

**7.总结**

 Workflow/ Agent更多是一种模式，而不是硬性规范， 具体到业务场景，可以自由灵活组合改造这些模式适配业务场景，不能生搬硬套,比如并行里面可以有路由,串行

对于构建应用，尽可能寻找简单方案，如有必要再增加复杂性，也就是奥卡姆剃刀原理：如无必要，勿增实体。可以从简单prompt开始,再rag, few-shot尝试，再workflow,agent，multi-agent，不断评估系统效果性能，不断改进优化迭代

对于追求确定性，可控性，优先可以考虑workflow,对于需要自主性，灵活性可以考虑agent，agent特别适合传统确定性方法难以应对的工作流，2种模式可以共同在一个系统里

大模型领域的成功并不是构建最复杂的系统，而是构建符合场景需求的系统。 只有在性能有明显改善时，才应该考虑增加复杂性。

![7bd5e35970945e0c78994e0171897e3.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557LPeQK7no8/img/90254188-1512-4a61-80b0-39c1ede5b882.png)

另外随着基础大模型能力越来越强，agent也很越来越强大，未来更多是多个agent协同分工合作构建强大系统，完成任务或者复杂任务，群体智能比单体智能更强大，A2A协议就更多是agent之间基础能力协同，除了原有基础能力之间协同，如果大模型之间，agent之间能够相互学习进化，就像人和人交流会学习对方知识并能启发自己有新的思考，能力，那agent能力就更到一个更高级别的层次

**参考资料**

1.[Building Effective AI Agents \ Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

2.[How to think about agent frameworks](https://blog.langchain.dev/how-to-think-about-agent-frameworks/)

3.[Newwhitepaper\_Agents.pdf](https://ia800601.us.archive.org/15/items/google-ai-agents-whitepaper/Newwhitepaper_Agents.pdf)

4.[GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows](https://github.com/openai/openai-agents-python)

5.[https://github.langchain.ac.cn/langgraph/tutorials/workflows](https://github.langchain.ac.cn/langgraph/tutorials/workflows)

6.[https://mp.weixin.qq.com/s/g93CF\_akhd7F-RmpuVoliw](https://mp.weixin.qq.com/s/g93CF_akhd7F-RmpuVoliw)

7.[A Survey on the Optimization of Large Language Model-based Agents](https://arxiv.org/pdf/2503.12434)

8.A Survey of AI Agent Protocols