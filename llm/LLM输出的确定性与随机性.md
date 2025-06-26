# LLM输出的确定性与随机性

1.  LLM回答的确定性与随机性
    

在同样的prompt输入条件下，LLM输出回答既可以是确定性的也可以是随机性的

LLM在输入prompt后，系统生成回答文本分为两个阶段，

（1）第一阶段：LLM处理输入token,生成输出token的score分布，在输入不变的情况下，输出的score分布也是不变的

（2）第二阶段：多种解码策略处理score分布，如果采用确定性策略，那回答就是确定性的，如果采用随机性策略，那回答就具有随机性

2.  LLM输入
    

基于next one token训练的模型属于因果语言模型(causal language modeling),LLM接受的只是文本，聊天消息,代码等只是一种特殊的文本形式,基于LLM的任何任务一定程度上都可以认为是对输入文本理解续写

（1）一般文本输入：一般输入的文本经过token化就可以输入给模型

（2）聊天消息输入：聊天消息经过聊天模板序列化为文本经token后输入给聊天模型,常用聊天模板ChatML格式,<l im\_start l>标记消息开始,<l im\_end l>标记消息结束

![66127e8bf07262adee5117903da2a74.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/bc6af7a7-ba5e-474b-9f1f-4beb5301194e.png "Jinja模板语言ChatML格式聊天模板")

![dca924c175047b2b5a608d3edb439a7.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/98ac7456-928a-4c90-a42e-140e0d789267.png "聊天消息经聊天模板转为text样例")

3.  LLM输出
    

LLM在输出token socre分布后需要解码策略确定采样的token,目前主要使用多策略联合采样(top-k,top-p,temperature)

LLM常用解码策略如下：

确定性策略:

1.greedy search(贪心搜索)：每一步都取概率最大的token,直接选择模型输出token分布值最大的值,不用做softmax

               优点：计算简单高效

  缺点：

        1.容易产生重复，文本不连贯,多样性不高

        2.每步选择局部最优,可能错过全局最优

  ![ed5f025ddd6784b53f2ac2a6b3a743b.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/bf7bdc87-4848-4011-bef1-e964ac28193c.png)

       2.beam search(束搜索):每一步保留topk个概率值大的输出,模型输出token分布值经过log\_softmax后保留,步数完成后最终选择概率和最大的路径序列,k=1就退化成贪心搜索

    优点：属于启发式搜索,适用于解空间较大情况，一定程度可以保证最终序列概率最优

    缺点:  1.每步选择局部最优,可能错过全局最优 2.有可能出现重复,前后矛盾情况 3.计算量随序列长度指数级增长

    ![f0bcdb6731d06c57e09b5c044ecb696.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/b6592a12-c672-4fe4-9c78-42d3c6383453.png)

随机性策略:

           3.随机采样:随机选择一个token

                  优点：简单,多样性高

                  缺点：容易产生重复，文本不连贯

    4.top-k采样：常用范围值5-50,每一步选取概率最高的k个token作为候选，再抽样选择一个，分布概率大的token采样到的概率越大

            优点：多样性会提升,有助于增强文本的连贯性，减少出现不常见或者与上下文无关的词,随机性会有助于提高生成质量

            缺点：

                   1.k值固定不动态，k值难确定

                   2.分布陡峭可能会采样到概率小的单词，分布平缓只能采样部分可用单词，

    k=1时就退化成贪心搜索，可能会导致文本不符合常识逻辑或者简单无聊

           ![792093b6564c730dcdc135d9520bbc7.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/cb6dc4dd-0b80-4f03-9e54-29f2d512c992.png)

相关代码：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/d0d768e3-60f1-4912-931c-dcd8c359b9db.png)

   5.top-p采样：常用范围值0.9-0.95,从累计概率和大于等于p的最小集合抽样选择一个，top-p常与top-k结合使用,如果k,p都启用,则p在k之后起作用,随机采样时分布概率大的token采样到的概率越大

    优点：

           1.动态设置token候选列表大小

           2.过滤掉低概率的token,

           3.top-p越大多样性越丰富

            缺点：p太小时模型输出越固定,低概率但有意义或创意的词被过滤掉

            ![427d84f2850c0d8abd28aa3974d6714.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/7ab86814-f8bf-4703-9393-491f13c26f48.png)

     相关代码：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/5ae65f80-afc8-4839-9906-d94a38b979c4.png)

        6.temperature采样:常用范围值(0,1\],通过温度调整token的score分布,温度越低，分布差距越大，越容易采样到概率大的token，温度越高，分布差距越小，低概率token采样机会增大,prompt越长越清晰，模型输出质量越好,可以适当提高温度值增加多样性,prompt越短越不清晰,高的温度值，输出就不稳定,随机采样时分布概率大的token采样到的概率越大

            优点：可用调整分布概率控制多样性和稳定性

            缺点：温度越高多样性,创意性越强,但也有可能产生错误或者不连贯，温度越低越保守稳定，容易出行重复

           ![5904db633dd20fbb02128968272f65d.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/6871db3b-cf08-48fe-ad11-2886cdcaf7c7.png)

相关代码：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/71b922a6-b3ab-48b8-80bd-02171142f37d.png)

       7.联合采样(top-k,top-p,temperature):多策略并行,使用先后顺序temp-topk-topp

            1.首先使用temperature调整模型输出的token score分布

            2.再选取score大的topk个token

            3.再从k个token中选择概率累计和达到top-p的token

![e68d490128680bec14c1ecd4664e7b5.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/b55abf49-e679-427f-9cbd-e45fa13d0cff.png)

    相关代码：

       ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/4c5bea28-b2f0-4739-b6c8-108826d5b917.png)

常用惩罚策略：

重复token惩罚：为了控制减少重复token出现，主要策略如下：

1.  repetition\_penalty：惩罚系数，对输出的token 已经在input里面token进行score惩罚
    

if  repetition\_penalty >1：减少重复词的生成概率

if  repetition\_penalty =1：保持原有生成概率

if  repetition\_penalty <1：增加重复词的生成概率

![0ddbd6dc9ce9d3f5ee2a6a2e02e42ab.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/ee4459f2-0f6e-432e-8071-88ab6e216da9.png)

2.  no\_repeat\_ngram\_size：生成文本时需要避免的ngram文本的大小，通过获取当前token的ngram，把模型输出scores对应的ngram token的值设置为无穷小，极大减低采样到ngram token的概率，所以基本ngram只能出现一次，谨慎使用
    

                   ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/149f6526-ef36-4205-b76e-57b4f75582ae.png)

长度惩罚：

1.  exponential\_decay\_length\_penalty ：当前长度超过设定的生成长度时,通过增加eos结束标记的score值，让采样到eos概率增大，达到生成结束的目的
    

                 ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1X3lE6YbVJDKnJbv/img/41c10c3b-ee65-4dfa-83fd-e63365865c75.png)

4. LLM生成停止策略

停止主要使用eos\_token特殊标记,由模型决定何时结束,max策略主要防止大模型停不下来

       1.eos\_token特殊标记：在生成过程中遇到停止特殊token,就结束生成

       2.max\_length:最大生成token长度

       3.max\_new\_tokens:除input token长度之外新生成的最大长度,max\_length= max\_new\_tokens + input\_ids\_length

       4.max\_time:生成token的最大时间限制范围内,比如设置为60s，那过了60s停止生成