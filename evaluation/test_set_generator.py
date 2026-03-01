"""
测试集生成器

支持：
1. 手动标注测试集的加载
2. 基于 LLM 自动生成测试集
3. 混合模式
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class TestCase:
    """测试用例数据结构"""
    question: str
    ground_truth_answer: str
    ground_truth_contexts: List[str]
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "manual"  # manual, auto_generated
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TestSetGenerator:
    """测试集生成器"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM 客户端（用于自动生成）
        """
        self.llm = llm

    def load_from_json(self, file_path: str) -> List[Dict]:
        """从 JSON 文件加载测试集"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = data.get("test_cases", data)
        if isinstance(test_cases, dict):
            test_cases = [test_cases]

        print(f"✅ Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases

    def save_to_json(self, test_cases: List[Dict], file_path: str):
        """保存测试集到 JSON 文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        output = {
            "version": "1.0",
            "total_count": len(test_cases),
            "test_cases": test_cases
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(test_cases)} test cases to {file_path}")

    def auto_generate(
        self,
        chunks: List[Dict],
        num_samples: int = 20,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        基于知识库 chunks 自动生成测试集

        Args:
            chunks: 知识库切片列表，每个元素包含 content 和 metadata
            num_samples: 生成数量
            categories: 指定类别（如 ["命名规约", "并发处理"]）

        Returns:
            生成的测试用例列表
        """
        if self.llm is None:
            raise ValueError("LLM client is required for auto generation")

        test_cases = []

        # 随机采样 chunks
        sampled_chunks = random.sample(chunks, min(num_samples, len(chunks)))

        for i, chunk in enumerate(sampled_chunks):
            print(f"Generating test case {i+1}/{len(sampled_chunks)}...")

            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            try:
                qa_pair = self._generate_qa_from_chunk(content)
                if qa_pair:
                    test_case = TestCase(
                        question=qa_pair["question"],
                        ground_truth_answer=qa_pair["answer"],
                        ground_truth_contexts=[content],
                        category=metadata.get("h1", "general"),
                        difficulty=self._estimate_difficulty(content),
                        source="auto_generated",
                        metadata={"chunk_id": i}
                    )
                    test_cases.append(test_case.to_dict())
            except Exception as e:
                print(f"  ⚠️ Failed to generate for chunk {i}: {e}")
                continue

        print(f"✅ Auto-generated {len(test_cases)} test cases")
        return test_cases

    def _generate_qa_from_chunk(self, content: str) -> Optional[Dict]:
        """基于单个 chunk 生成问答对"""
        prompt = f"""请基于以下技术文档片段，生成1个真实用户可能会问的问题和标准答案。

文档内容：
{content[:800]}

要求：
1. 问题要自然、口语化（模拟真实用户提问，可以用"怎么"、"为什么"、"能不能"等）
2. 答案必须能从文档中直接找到依据
3. 问题不要太简单（如"什么是XXX"），要有一定思考深度

请严格按照以下 JSON 格式输出（不要有其他内容）：
{{
    "question": "用户问题",
    "answer": "基于文档的标准答案",
    "key_point": "答案的核心要点（一句话）"
}}"""

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.7,
                max_tokens=500
            )
            result_text = response.choices[0].message.content.strip()

            # 尝试解析 JSON
            # 处理可能的 markdown 代码块
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text)
        except Exception as e:
            print(f"  Parse error: {e}")
            return None

    def _estimate_difficulty(self, content: str) -> str:
        """估计问题难度"""
        # 简单启发式规则
        if "【强制】" in content:
            return "easy"
        elif "【推荐】" in content:
            return "medium"
        elif len(content) > 300 or "例如" in content or "比如" in content:
            return "hard"
        return "medium"

    def create_manual_template(self, output_path: str, num_templates: int = 10):
        """创建手动标注模板"""
        template = {
            "version": "1.0",
            "instructions": "请填写 question, ground_truth_answer, ground_truth_contexts 字段",
            "test_cases": []
        }

        for i in range(num_templates):
            template["test_cases"].append({
                "question": f"[请填写问题 {i+1}]",
                "ground_truth_answer": "[请填写标准答案]",
                "ground_truth_contexts": ["[请粘贴相关的文档原文]"],
                "category": "命名规约",  # 可选: 命名规约/常量定义/代码格式/并发处理/异常日志 等
                "difficulty": "medium",  # easy/medium/hard
                "source": "manual"
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

        print(f"✅ Created annotation template at: {output_path}")
        print(f"   Please fill in {num_templates} test cases manually.")


# ============ 预置的 Java 开发手册测试集 ============

JAVA_MANUAL_TEST_SET = [
    {
        "question": "POJO类中的布尔变量能用is开头吗？",
        "ground_truth_answer": "不能。POJO类中的布尔类型变量都不要加is前缀，否则部分框架解析会引起序列化错误。",
        "ground_truth_contexts": [
            "【强制】POJO 类中的布尔类型变量都不要加 is 前缀，否则部分框架解析会引起序列化错误"
        ],
        "category": "命名规约",
        "difficulty": "easy"
    },
    {
        "question": "为什么不推荐使用Executors创建线程池？",
        "ground_truth_answer": "线程池不允许使用Executors去创建，而是通过ThreadPoolExecutor的方式。因为Executors返回的线程池对象弊端：FixedThreadPool和SingleThreadPool允许的请求队列长度为Integer.MAX_VALUE，可能会堆积大量请求导致OOM；CachedThreadPool允许创建的线程数量为Integer.MAX_VALUE，可能会创建大量线程导致OOM。",
        "ground_truth_contexts": [
            "【强制】线程池不允许使用 Executors 去创建，而是通过 ThreadPoolExecutor 的方式"
        ],
        "category": "并发处理",
        "difficulty": "medium"
    },
    {
        "question": "SimpleDateFormat是线程安全的吗？怎么处理日期格式化？",
        "ground_truth_answer": "SimpleDateFormat是线程不安全的类，一般不要定义为static变量。如果定义为static，必须加锁，或者使用DateUtils工具类。JDK8及以上推荐使用Instant代替Date，使用DateTimeFormatter代替SimpleDateFormat。",
        "ground_truth_contexts": [
            "【强制】SimpleDateFormat 是线程不安全的类，一般不要定义为 static 变量"
        ],
        "category": "并发处理",
        "difficulty": "medium"
    },
    {
        "question": "包名怎么命名？",
        "ground_truth_answer": "包名统一使用小写，点分隔符之间有且仅有一个自然语义的英语单词。包名统一使用单数形式，但是类名如果有复数含义，类名可以使用复数形式。",
        "ground_truth_contexts": [
            "【强制】包名统一使用小写，点分隔符之间有且仅有一个自然语义的英语单词"
        ],
        "category": "命名规约",
        "difficulty": "easy"
    },
    {
        "question": "常量命名有什么规范？",
        "ground_truth_answer": "常量命名全部大写，单词间用下划线隔开，力求语义表达完整清楚，不要嫌名字长。",
        "ground_truth_contexts": [
            "【强制】常量命名全部大写，单词间用下划线隔开，力求语义表达完整清楚"
        ],
        "category": "命名规约",
        "difficulty": "easy"
    },
    {
        "question": "ArrayList的subList返回的结果能强转成ArrayList吗？",
        "ground_truth_answer": "不能。ArrayList的subList结果不可强转成ArrayList，否则会抛出ClassCastException异常。subList()返回的是ArrayList的内部类SubList，并不是ArrayList本身，而是ArrayList的一个视图。",
        "ground_truth_contexts": [
            "【强制】ArrayList 的 subList 结果不可强转成 ArrayList"
        ],
        "category": "集合处理",
        "difficulty": "medium"
    },
    {
        "question": "什么情况下不能在foreach里面进行元素的remove/add操作？",
        "ground_truth_answer": "不要在foreach循环里进行元素的remove/add操作，remove元素请使用Iterator方式。如果并发操作，需要对Iterator对象加锁。",
        "ground_truth_contexts": [
            "【强制】不要在 foreach 循环里进行元素的 remove / add 操作"
        ],
        "category": "集合处理",
        "difficulty": "medium"
    },
    {
        "question": "方法的参数个数有限制吗？",
        "ground_truth_answer": "方法的参数个数不宜过多，推荐不超过5个。如果超过5个，建议将多个参数封装成一个DTO对象传递。",
        "ground_truth_contexts": [
            "方法的参数不宜过多"
        ],
        "category": "代码格式",
        "difficulty": "easy"
    },
    {
        "question": "hashCode和equals方法需要同时重写吗？",
        "ground_truth_answer": "只要重写equals，必须重写hashCode。因为Set存储的是不重复的对象，依据hashCode和equals进行判断，所以Set存储的对象必须重写这两个方法。如果自定义对象作为Map的键，那么必须覆写hashCode和equals。",
        "ground_truth_contexts": [
            "只要重写 equals，必须重写 hashCode"
        ],
        "category": "OOP规约",
        "difficulty": "medium"
    },
    {
        "question": "异常捕获后可以什么都不做吗？",
        "ground_truth_answer": "不可以。捕获异常是为了处理它，不要捕获了却什么都不处理而抛弃之。如果不想处理它，请将该异常抛给它的调用者。最外层的业务使用者必须处理异常，将其转化为用户可以理解的内容。",
        "ground_truth_contexts": [
            "【强制】捕获异常是为了处理它，不要捕获了却什么都不处理而抛弃之"
        ],
        "category": "异常日志",
        "difficulty": "easy"
    },
    {
        "question": "事务场景中使用try-catch有什么注意事项？",
        "ground_truth_answer": "有try块放到了事务代码中，catch异常后，如果需要回滚事务，一定要注意手动回滚事务。",
        "ground_truth_contexts": [
            "有 try 块放到了事务代码中，catch 异常后，如果需要回滚事务，一定要注意手动回滚事务"
        ],
        "category": "异常日志",
        "difficulty": "hard"
    },
    {
        "question": "接口和实现类的命名规范是什么？",
        "ground_truth_answer": "对于Service和DAO类，基于SOA的理念，暴露出来的服务一定是接口，内部的实现类用Impl的后缀与接口区别。如果是形容能力的接口名称，取对应的形容词为接口名（通常是–able的形容词）。",
        "ground_truth_contexts": [
            "对于 Service 和 DAO 类，基于 SOA 的理念，暴露出来的服务一定是接口，内部的实现类用 Impl 的后缀与接口区别"
        ],
        "category": "命名规约",
        "difficulty": "easy"
    },
    {
        "question": "使用Map的keySet方法遍历时可以删除元素吗？",
        "ground_truth_answer": "使用entrySet遍历Map类集合KV，而不是keySet方式进行遍历。keySet其实是遍历了2次，一次是转为Iterator对象，另一次是从hashMap中取出key所对应的value。而entrySet只是遍历了一次就把key和value都放到了entry中，效率更高。",
        "ground_truth_contexts": [
            "【推荐】使用 entrySet 遍历 Map 类集合 KV，而不是 keySet 方式进行遍历"
        ],
        "category": "集合处理",
        "difficulty": "medium"
    },
    {
        "question": "代码中的魔法值是什么意思？要怎么处理？",
        "ground_truth_answer": "不允许任何魔法值（即未经预先定义的常量）直接出现在代码中。反例：String key = \"Id#taobao_\" + tradeId; 应该将\"Id#taobao_\"定义为常量。",
        "ground_truth_contexts": [
            "【强制】不允许任何魔法值（即未经预先定义的常量）直接出现在代码中"
        ],
        "category": "常量定义",
        "difficulty": "easy"
    },
    {
        "question": "Long类型赋值的时候有什么注意事项？",
        "ground_truth_answer": "在long或者Long赋值时，数值后使用大写字母L，不能是小写字母l，小写容易跟数字混淆，造成误解。",
        "ground_truth_contexts": [
            "【强制】在 long 或者 Long 赋值时，数值后使用大写字母 L"
        ],
        "category": "常量定义",
        "difficulty": "easy"
    },
]


def get_default_test_set() -> List[Dict]:
    """获取预置的测试集"""
    return JAVA_MANUAL_TEST_SET


def load_or_create_test_set(
    file_path: str,
    auto_generate: bool = False,
    llm=None,
    chunks=None
) -> List[Dict]:
    """
    加载或创建测试集

    Args:
        file_path: 测试集文件路径
        auto_generate: 如果文件不存在，是否自动生成
        llm: LLM 客户端
        chunks: 知识库切片

    Returns:
        测试用例列表
    """
    generator = TestSetGenerator(llm)

    if os.path.exists(file_path):
        return generator.load_from_json(file_path)

    # 文件不存在
    if auto_generate and llm and chunks:
        print(f"Test set not found at {file_path}, auto-generating...")
        test_cases = generator.auto_generate(chunks, num_samples=20)
        generator.save_to_json(test_cases, file_path)
        return test_cases
    else:
        # 使用预置测试集
        print(f"Test set not found at {file_path}, using default test set...")
        test_cases = get_default_test_set()
        generator.save_to_json(test_cases, file_path)
        return test_cases
