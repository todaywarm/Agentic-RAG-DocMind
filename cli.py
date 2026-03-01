#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocMind CLI - 命令行交互界面

使用方法：
    python cli.py                    # 启动交互式问答
    python cli.py --add <file>       # 添加文档
    python cli.py --eval             # 运行评估
    python cli.py --reset            # 重置知识库
"""

import os
import sys
import argparse

# 确保可以导入 docmind
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       ██████╗  ██████╗  ██████╗███╗   ███╗██╗███╗   ██╗██████╗║
║       ██╔══██╗██╔═══██╗██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
║       ██║  ██║██║   ██║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
║       ██║  ██║██║   ██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
║       ██████╔╝╚██████╔╝╚██████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
║       ╚═════╝  ╚═════╝  ╚═════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝║
║                                                               ║
║          Enterprise RAG System v2.0 - Powered by BGE          ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息"""
    print("""
📖 可用命令：
    /help       - 显示此帮助
    /add <file> - 添加文档到知识库
    /reset      - 重置知识库
    /clear      - 清除对话历史
    /logs       - 显示上次检索的日志
    /quit       - 退出程序
    
直接输入问题进行问答。
""")


def interactive_mode(agent):
    """交互式问答模式"""
    print_banner()
    print("\n🚀 DocMind 已就绪！输入 /help 查看可用命令。\n")
    
    last_logs = []
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith('/'):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                
                if cmd == '/quit' or cmd == '/exit':
                    print("👋 再见！")
                    break
                elif cmd == '/help':
                    print_help()
                elif cmd == '/add':
                    if not arg:
                        print("❌ 请指定文件路径：/add <file>")
                    else:
                        result = agent.add_document(arg)
                        print(f"📄 {result['message']}")
                elif cmd == '/reset':
                    confirm = input("⚠️ 确定要重置知识库吗？(y/n): ")
                    if confirm.lower() == 'y':
                        agent.reset_knowledge_base()
                        print("✅ 知识库已重置。")
                elif cmd == '/clear':
                    agent.clear_history()
                    print("✅ 对话历史已清除。")
                elif cmd == '/logs':
                    if last_logs:
                        print("\n📋 上次检索日志：")
                        for log in last_logs:
                            print(f"   {log}")
                    else:
                        print("暂无日志。")
                else:
                    print(f"❌ 未知命令：{cmd}，输入 /help 查看可用命令。")
                continue
            
            # 问答
            print("🤖 DocMind: ", end="", flush=True)
            
            # 流式输出
            full_answer = ""
            result = agent.chat(user_input)
            print(result["answer"])
            
            # 保存日志
            last_logs = result.get("retrieval_info", {}).get("logs", [])
            
            # 显示置信度
            confidence = result.get("confidence", "unknown")
            if confidence == "low":
                print("   ⚠️ [置信度较低，请谨慎参考]")
            elif confidence == "none":
                print("   ❌ [未找到相关信息]")
            
            # 显示来源
            sources = result.get("sources", [])
            if sources:
                print(f"   📚 参考来源：{len(sources)} 条")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误：{e}")


def main():
    parser = argparse.ArgumentParser(description="DocMind CLI")
    parser.add_argument('--add', type=str, help='添加文档到知识库')
    parser.add_argument('--eval', action='store_true', help='运行评估')
    parser.add_argument('--reset', action='store_true', help='重置知识库')
    parser.add_argument('--collection', type=str, default='docmind_knowledge_base', help='集合名称')
    
    args = parser.parse_args()
    
    # 初始化 Agent
    print("🔧 初始化 DocMind...")
    
    from docmind.agent import RAGAgent
    from docmind.retrieval import HybridRetriever, VectorStore, BM25Store
    from docmind.core.config import settings
    
    # 更新配置
    settings.retrieval.collection_name = args.collection
    
    agent = RAGAgent()
    
    # 处理命令行参数
    if args.add:
        result = agent.add_document(args.add)
        print(f"📄 {result['message']}")
        return
    
    if args.reset:
        confirm = input("⚠️ 确定要重置知识库吗？(y/n): ")
        if confirm.lower() == 'y':
            agent.reset_knowledge_base()
            print("✅ 知识库已重置。")
        return
    
    if args.eval:
        print("🧪 运行评估...")
        os.system(f"python -m evaluation.run_evaluation")
        return
    
    # 默认进入交互模式
    interactive_mode(agent)


if __name__ == "__main__":
    main()
