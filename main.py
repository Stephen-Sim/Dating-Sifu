import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import uuid

import re

def is_relevant_question(question, guidelines):
    """
    Validate if the question is relevant to social skills and dating advice
    Returns (is_valid, message)
    """
    # Convert question to lowercase for better matching
    question = question.lower()
    
    # Check minimum length
    if len(question.strip()) < 5:
        return False, "请详细描述你的问题，这样我才能更好地帮助你。"
    
    # Keywords for relevant topics
    relevant_keywords = {
        "社交": ["聊天", "交往", "互动", "沟通", "朋友", "认识", "社交圈"],
        "约会": ["约会", "恋爱", "表白", "暗示", "追求", "相亲", "缘分"],
        "技巧": ["技巧", "方法", "建议", "怎么做", "如何", "策略", "经验"],
        "关系": ["关系", "感情", "喜欢", "爱情", "暧昧", "相处"],
        "心理": ["心理", "感受", "情绪", "自信", "压力", "焦虑", "紧张"]
    }
    
    # Count matches in each category
    matches = 0
    for category, keywords in relevant_keywords.items():
        if any(keyword in question for keyword in keywords):
            matches += 1
            
    # Check against negative patterns (questions that should be rejected)
    negative_patterns = [
        r'(广告|推广|营销|赚钱|理财)',
        r'(黄色|色情|约炮|一夜情)',
        r'(违法|犯罪|黑客|攻击)',
        r'(博彩|赌博|彩票)',
        r'(政治|宗教|种族)'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, question):
            return False, "抱歉，这个问题超出了社交技能辅导的范围。请咨询与社交、约会和人际关系相关的问题。"
            
    # Question must match at least one relevant category
    if matches == 0:
        suggestions = "\n".join([
            "- 如何自然开始对话",
            "- 约会时的话题选择",
            "- 如何展现自信",
            "- 理解对方的暗示",
            "- 建立良好的社交关系"
        ])
        return False, f"请问一些与社交技能、约会和人际关系相关的问题。例如：\n{suggestions}"
        
    return True, ""

def load_training_data(filename='./history.json'):
    default_system_prompt = """你现在是一个社交技能教练，帮助人们提升约会和社交能力。请注意:
    1. 给出实用、得体的建议
    2. 保持对话轻松自然
    3. 强调真诚和尊重
    4. 根据对方反应调整建议
    5. 提供具体的例子和场景"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # 构建增强的系统提示词
        system_prompt = default_system_prompt + "\n\n基于历史训练数据的指导原则:\n"
        
        # 添加核心原则
        system_prompt += "\n核心原则:\n"
        for principle in data["指导原则"]["核心原则"]:
            system_prompt += f"- {principle}\n"
            
        # 添加建议话题
        system_prompt += "\n建议话题:\n"
        for topic in data["指导原则"]["话题建议"]:
            system_prompt += f"- {topic}\n"
            
        return {
            "system_prompt": system_prompt,
            "training_history": data["conversations"],
            "full_data": data
        }
    except FileNotFoundError:
        print(f"提示: 文件 '{filename}' 未找到，使用默认系统提示词.")
        return {
            "system_prompt": default_system_prompt,
            "training_history": [],
            "full_data": {
                "version": "1.0",
                "description": "社交技能训练数据",
                "conversations": [],
                "指导原则": {
                    "核心原则": [
                        "尊重界限",
                        "展现真诚",
                        "保持自然",
                        "积极倾听",
                        "善于观察"
                    ],
                    "话题建议": [
                        "共同兴趣",
                        "人生目标",
                        "旅行经历",
                        "文化艺术"
                    ]
                }
            }
        }

def find_relevant_example(query, training_history):
    """根据用户输入找到相关的历史对话"""
    if not training_history:
        return None
        
    # 简单的关键词匹配
    relevant_conversations = []
    keywords = query.lower().split()
    
    for conv in training_history:
        score = 0
        # 检查元数据
        metadata_text = (
            conv["metadata"]["topic"].lower() + " " +
            conv["metadata"]["context"].lower() + " " +
            conv["metadata"]["learning_point"].lower()
        )
        for keyword in keywords:
            if keyword in metadata_text:
                score += 2
                
        # 检查对话内容
        for msg in conv["messages"]:
            if any(keyword in msg["content"].lower() for keyword in keywords):
                score += 1
                
        if score > 0:
            relevant_conversations.append((score, conv))
            
    if relevant_conversations:
        # 返回最相关的对话
        relevant_conversations.sort(key=lambda x: x[0], reverse=True)
        return relevant_conversations[0][1]
    return None

def format_example(example):
    """格式化对话示例"""
    if not example:
        return "没有直接相关的历史案例"
    
    formatted = f"场景: {example['metadata']['context']}\n"
    formatted += f"重点: {example['metadata']['learning_point']}\n\n"
    
    for msg in example['messages']:
        formatted += f"{msg['role'].upper()}: {msg['content']}\n"
    
    return formatted

def save_conversation(filename, data, new_conversation):
    """保存新的对话到历史文件"""
    data["conversations"].append(new_conversation)
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print("\n对话已成功保存到历史记录")
    except Exception as e:
        print(f"\n保存对话时出现错误: {str(e)}")

def analyze_conversation_topic(messages):
    """分析对话主题和上下文"""
    # 简单的主题分析逻辑
    common_topics = {
        "初次见面": ["打招呼", "开场", "认识", "第一次"],
        "展示兴趣": ["兴趣", "喜欢", "感兴趣"],
        "约会邀请": ["约会", "邀请", "一起"],
        "观察反馈": ["暗示", "反应", "表现"],
        "建立关系": ["关系", "互动", "联系"],
    }
    
    all_text = " ".join([msg["content"] for msg in messages]).lower()
    
    # 找出最匹配的主题
    max_matches = 0
    best_topic = "一般社交"
    for topic, keywords in common_topics.items():
        matches = sum(1 for keyword in keywords if keyword in all_text)
        if matches > max_matches:
            max_matches = matches
            best_topic = topic
            
    return {
        "topic": best_topic,
        "context": "一般场景",  # 可以根据需要扩展
        "learning_point": "社交技巧提升"
    }

# Create chat prompt template
prompt_template = """
你是一个专业的社交技能教练。基于以下系统设定和历史对话案例，为用户提供建议。

系统设定:
{system_prompt}

历史对话记录:
{chat_history}

相关历史案例:
{example}

用户问题:
{input}

请给出专业、实用且具体的建议。回答要:
1. 自然友好
2. 结合具体场景
3. 考虑实际情况
4. 注重互动技巧
5. 强调真诚尊重
"""

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages([ 
    ("system", prompt_template), 
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}") 
])

def handle_conversation():
    # Load training data
    data = load_training_data()
    training_history = data["training_history"]
    full_data = data["full_data"]
    
    # 初始化新的对话记录
    current_conversation = {
        "id": f"lesson_{str(uuid.uuid4())[:8]}",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "topic": "",
            "context": "",
            "learning_point": ""
        },
        "messages": []
    }
    
    # Initialize the model and chain
    try:
        model = OllamaLLM(model="qwen2.5")
        chain = prompt | model
    except Exception as e:
        print(f"设置失败: {str(e)}")
        return
    
    # Initialize chat history
    chat_history = [
        SystemMessage(content=data["system_prompt"])
    ]
    
    print("\n=== 智能社交教练 ===")
    print("情场军师: 你好！我是你的社交技能教练。我可以帮你提升社交能力，建立真诚的人际关系。")
    print("\n我可以帮你解答以下方面的问题:")
    print("1. 如何自然开始对话")
    print("2. 理解社交暗示")
    print("3. 建立有趣的互动")
    print("4. 约会技巧与礼仪")
    print("5. 提升自信与魅力")
    print("\n(输入'退出'或'exit'结束对话)")
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                print("情场军师: 请输入你的问题。")
                continue
                
            if user_input.lower() in ["退出", "exit", "quit"]:
                if current_conversation["messages"]:
                    metadata = analyze_conversation_topic(current_conversation["messages"])
                    current_conversation["metadata"].update(metadata)
                    save_conversation("./history.json", full_data, current_conversation)
                print("\n情场军师: 祝你社交顺利！记住保持真诚、自信和尊重。")
                break
                
            # Validate user input
            is_valid, validation_message = is_relevant_question(
                user_input, 
                full_data["指导原则"]
            )
            
            if not is_valid:
                print("\n情场军师:", validation_message)
                continue
            
            # 查找相关例子
            relevant_example = find_relevant_example(user_input, training_history)
            formatted_example = format_example(relevant_example)
            
            # 记录用户问题
            current_conversation["messages"].append({
                "role": "human",
                "content": user_input
            })
            
            # Generate response
            result = chain.invoke({
                "system_prompt": data["system_prompt"],
                "chat_history": chat_history,
                "example": formatted_example,
                "input": user_input
            })
            
            # 记录AI回答
            current_conversation["messages"].append({
                "role": "assistant",
                "content": str(result)
            })
            
            print("\n情场军师:", result)
            
            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=str(result)))
            
        except KeyboardInterrupt:
            if current_conversation["messages"]:
                metadata = analyze_conversation_topic(current_conversation["messages"])
                current_conversation["metadata"].update(metadata)
                save_conversation("./history.json", full_data, current_conversation)
            print("\n\n社交教练: 对话被中断。祝你社交愉快！")
            break
        except Exception as e:
            print(f"\n抱歉，处理你的问题时出现错误: {str(e)}")
            print("让我们继续对话。如果问题持续，你可以尝试重新启动程序。")

def main():
    try:
        handle_conversation()
    except KeyboardInterrupt:
        print("\n\n程序已终止。谢谢使用！")
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        print("请检查配置并重试。")

if __name__ == "__main__":
    main()