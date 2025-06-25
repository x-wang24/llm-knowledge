# mcp服务封装

1.  **mcp概念**
    

mcp(Model Context Protocol)是一种标准化协议，用于将 AI 代理连接到各种外部工具和数据源。将其想象成一个 USB-C 端口 - 但用于 AI 应用程序

![fe0fd5f290fab5b450772177fe939cd.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/f4e62e2c-7b70-402e-b4b9-75ac770b606a.png)

**2.mcp对api对比**

传统api:每个api有自己的定义规则，集成比较繁琐

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/454db415-17b7-45b1-a74b-517d273f88af.png)

![99b0d6873e38577413f69f7652e8790.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/81f38dc9-a483-4901-a17a-aee1f9b99583.png)

### MCP 与传统 API 之间的主要区别：

*   **单一协议：**MCP 充当标准化的“连接器”，因此集成一个 MCP 意味着可能访问多个工具和服务，而不仅仅是一个
    
*   **动态发现：**MCP 允许 AI 模型动态发现可用工具并与之交互，而无需对每个集成进行硬编码知识
    
*   **双向通信：**MCP 支持持久的实时双向通信 - 类似于 WebSockets。AI 模型既可以检索信息，也可以动态触发作
    

**3.mcp架构**

   mcp遵循Client/Server架构，将 MCP 可用看作一座桥梁：MCP 本身不处理繁重的 logic;它只是协调 AI 模型和工具之间的数据和指令流。正如 USB-C 简化了将不同设备连接到计算机的方式一样，MCP 也简化了 AI 模型与数据、工具和服务的交互方式

 ![48b7ba0932f934786a19ef6afe649d9.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/4daf76de-5ede-4004-895a-25cea102c837.png)

**4.使用场景**

**mcp使用场景**：

1.  **AI 代理与智能助手**
    

    **场景示例**​：旅行规划助手需同时访问日历、航班系统、邮件服务，医疗诊断助手需整合患者病史、检验报告和影像数据

    **优势**​：MCP 通过统一协议动态发现工具，避免为每个服务单独开发接口，显著减少集成

2.  **需要实时双向交互的系统**
    

   **场景示例**​：智能IDE（如Cursor）需实时同步代码库变更、文档更新和API规范，金融风控系统需即时触发交易警报或自动止损

    **优势**​：MCP 支持类 WebSocket 的双向通信，允许AI主动推送通知或触发操作

3.  **工具频繁扩展的动态环境**
    

    **场景示例**​：业知识中枢需灵活接入新增数据库、CRM或ERP系统，数据分析平台需动态集成多个可视化工具

    **优势**​：通过 MCP 服务器即插即用新工具，无需重构核心代码

4.  **多源异构数据整合**
    

    **场景示例**​：跨平台销售助手需同时调取CRM客户数据、ERP库存信息和物流系统状态

    **优势**​：MCP 标准化数据访问，自动清洗多源数据生成结构化上下文

**使用传统api:**

传统API 更适合 **​确定性高、控制精细​**​ 的任务。MCP 提供了灵活动态功能，非常适合需要灵活性和上下文感知的场景，但可能不太适合高度受控的确定性应用程序

**1 .****需要精细的控制和高度特定的受限功****能**

    **场景示例**​：支付网关需精确校验金额、账户和风控规则、工业设备API需毫秒级响应控制指令

    **优势**​：传统API支持细粒度参数校验和错误处理，确保高可预测性

**2.性能敏感型应用**​

   **场景示例​**：高频交易系统、实时游戏引擎

   **优势**​：紧耦合设计减少协议转换开销，降低延迟

**5.mcp服务端**

   mcp服务建议优先使用fastmcp进行封装,优先使用sse通信

```latex
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```
```latex
import argparse
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, List
from dotenv import load_dotenv
import json
from fastmcp import FastMCP, Context
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

class MilvusConnector:
    def __init__(self, uri: str, token: Optional[str] = None, db_name: Optional[str] = "default"):
        self.uri = uri
        self.token = token
        self.client = MilvusClient(uri=uri, token=token, db_name=db_name)

    async def list_collections(self) -> list[str]:
        """List all collections in the database."""
        try:
            return self.client.list_collections()
        except Exception as e:
            raise ValueError(f"Failed to list collections: {str(e)}")

    async def search_collection(
            self,
            collection_name: str,
            query_text: str,
            limit: int = 5,
            output_fields: Optional[list[str]] = None,
            drop_ratio: float = 0.2,
    ) -> list[dict]:
        """
        Perform full text search on a collection.

        Args:
            collection_name: Name of collection to search
            query_text: Text to search for
            limit: Maximum number of results
            output_fields: Fields to return in results
            drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
        """
        try:
            search_params = {"params": {"drop_ratio_search": drop_ratio}}

            results = self.client.search(
                collection_name=collection_name,
                data=[query_text],
                anns_field="sparse",
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )
            return results
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}")

class MilvusContext:
    def __init__(self, connector: MilvusConnector):
        self.connector = connector


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MilvusContext]:
    """Manage application lifecycle for Milvus connector."""
    config = server.config

    connector = MilvusConnector(
        uri=config.get("milvus_uri", "http://localhost:19530"),
        token=config.get("milvus_token"),
        db_name=config.get("db_name", "default"),
    )

    try:
        yield MilvusContext(connector)
    finally:
        pass

mcp = FastMCP(name="Milvus", lifespan=server_lifespan)

@mcp.tool()
async def milvus_list_collections(ctx: Context) -> str:
    """List all collections in the database."""
    connector = ctx.request_context.lifespan_context.connector
    collections = await connector.list_collections()
    return f"Collections in database:\n{', '.join(collections)}"

@mcp.tool()
async def milvus_text_search(
    collection_name: str,
    query_text: str,
    limit: int = 5,
    output_fields: Optional[list[str]] = None,
    drop_ratio: float = 0.2,
    ctx: Context = None,
) -> str:
    """
    Search for documents using full text search in a Milvus collection.

    Args:
        collection_name: Name of the collection to search
        query_text: Text to search for
        limit: Maximum number of results to return
        output_fields: Fields to include in results
        drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.search_collection(
        collection_name=collection_name,
        query_text=query_text,
        limit=limit,
        output_fields=output_fields,
        drop_ratio=drop_ratio,
    )

    output = f"Search results for '{query_text}' in collection '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


def parse_arguments():
    parser = argparse.ArgumentParser(description="Milvus MCP Server")
    parser.add_argument(
        "--milvus-uri", type=str, default="http://127.0.0.1:19530", help="Milvus server URI"
    )
    parser.add_argument(
        "--milvus-token", type=str, default='root:Milvus', help="Milvus authentication token"
    )
    parser.add_argument("--milvus-db", type=str, default="default", help="Milvus database name")
    parser.add_argument("--sse", action="store_true", default=True, help="Enable SSE mode")
    parser.add_argument("--port", type=int, default=8003, help="Port number for SSE server")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()
    mcp.config = {
        "milvus_uri": os.environ.get("MILVUS_URI", args.milvus_uri),
        "milvus_token": os.environ.get("MILVUS_TOKEN", args.milvus_token),
        "db_name": os.environ.get("MILVUS_DB", args.milvus_db),
    }
    if args.sse:
        mcp.run(transport="sse", port=args.port, host="0.0.0.0")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
```

**6.mcp客户端**

```latex
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.output_beautify import typewriter_print
def init_agent_service():
    llm_cfg = {
        'model': 'qwen3-235b-a22b',
        'model_type': 'qwen_dashscope',
        'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # api_base
        'api_key': '',
        'generate_cfg': {
            # Add: When the content is `<think>this is the thought</think>this is the answer`
            # Do not add: When the response has been separated by reasoning_content and content
            # This parameter will affect the parsing strategy of tool call
            # 'thought_in_content': True,

            # When using the Dash Scope API, pass the parameter of whether to enable thinking mode in this way
            'enable_thinking': False,

            # When using OAI API, pass the parameter of whether to enable thinking mode in this way
            # 'extra_body': {
            #     'enable_thinking': False
            # }
        },
    }
    tools = [
        {
            'mcpServers': {  # You can specify the MCP configuration file
                'time': {
                    'command': 'uvx',
                    'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
                },
                'fetch': {
                    'command': 'uvx',
                    'args': ['mcp-server-fetch']
                },
                "milvus-mcp": {
                    "type": "sse",
                    "url": "http://127.0.0.1:8003/sse"
                },

            }
        },
        'code_interpreter',  # Built-in tools
    ]
    bot = Assistant(llm=llm_cfg,
                    function_list=tools,
                    name='Qwen3 Tool-calling Demo',
                    description="I'm a demo using the Qwen3 tool calling. Welcome to add and play with your own tools!")

    return bot

def app_tui():
    # Define the agent
    bot = init_agent_service()
    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        response_plain_text = ''
        for response in bot.run(messages=messages):
            response_plain_text = typewriter_print(response, response_plain_text)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            'What time is it?',
            'What are the collections I have in my Milvus DB?',
            'https://github.com/orgs/QwenLM/repositories Extract markdown content of this page, then draw a bar chart to display the number of stars.'
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run(server_name='0.0.0.0')


if __name__ == '__main__':
    app_gui()

```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/8a5ee397-a3a3-434b-8a08-16bc87515c05.png)

**7.mcp问题**

**问题1：mcp隐形要求**

mcp工具中有着自己的入参格式要求，如果参数传入不准，可能无结果或者结果不准，另外复杂工具需要前置条件

比如：调用高德周边搜索，智能体会直接调用 maps\_around\_search 函数，并基于自己的先验知识，给出“云谷园区”的经纬度坐标

![29bf51815b50557c0ce86f793de3b190.jpeg](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/ea938ecc-55b0-4019-8d2f-4c41b6c883ef.jpeg)

优化：先获取经纬度再调用周边搜索

![b72b8964d275b4b23cdf115845cce11.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jyaMr28lvD/img/41040201-c9d7-42d8-a1f3-5c6975fa8961.png)

优化sop：

1.先调用 maps\_text\_search 函数，将模糊的地点描述转化为详细的结构化地址

       2.再调用 maps\_geo 函数，获取该地点的经纬度坐标

       3.最后调用 maps\_around\_search 函数，搜索出 radius 半径范围的 POI 地点信息

**问题2：mcp精度问题**

mcp server大多数对开发者和模型来说属于黑盒,无法获取具体实现逻辑，也就无法知道工具精度

比如获取北京欢乐谷附近500m的餐厅,如果经纬度获取的误差在500以上，再推荐就不准了

**问题3：****多 MCP Server 组合使用**

当一个智能体潜在可能需要多种不同的 MCP Server 时,可能有名字重合相似（重命名,优化提示），功能相似（优化提示/few-shot/分组管理），工具过载(分组管理/工具检索/**按需加载**)

mcp集成方式：

1）将所有server添加到一个智能体, 如果少的话可以添加到一个，并且可以在系统提示词添加注意事项，如果太多就不优雅

2）每个server或几个server添加到一个智能体，比如地图相关的添加到一个智能体，搜索相关添加到一个智能体，aigc相关的添加到一个智能体，并且每个智能体都可以添加提示词去约束优化其行为

3）动态分配mcp server,按需加载,将mcp server看作一个工具集合，作为一个工具函数添加到智能体，允许智能体根据任务动态分配server,也可以工具检索添加到智能体

参考资料：

[What is Model Context Protocol (MCP)? How it simplifies AI integrations compared to APIs | AI Agents That Work](https://norahsakal.com/blog/mcp-vs-api-model-context-protocol-explained/)

[The FastMCP Server - FastMCP](https://gofastmcp.com/servers/fastmcp)

[Introduction - Model Context Protocol](https://modelcontextprotocol.io/introduction)

[https://github.com/zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus)

[Quickstart - mcp\_use](https://docs.mcp-use.io/quickstart)