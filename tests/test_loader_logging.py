"""测试 DocumentLoader.load() 的时间日志功能"""
import os
import tempfile
import logging
from docmind.document.loader import DocumentLoader


def test_load_outputs_time_logs(caplog):
    """测试加载文档时会输出时间日志"""
    # 创建一个临时文本文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        loader = DocumentLoader()
        with caplog.at_level(logging.INFO):
            result = loader.load(temp_path)

        # 检查日志中是否包含开始和结束日志
        log_messages = [r.message for r in caplog.records]
        assert any("[DocumentLoader] Start loading" in msg for msg in log_messages)
        assert any("[DocumentLoader] Finished loading" in msg for msg in log_messages)
        assert result == "test content"
    finally:
        os.unlink(temp_path)