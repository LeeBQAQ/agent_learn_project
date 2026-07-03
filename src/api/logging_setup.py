import logging
import os

import structlog


def setup_logging():
    """统一配置项目所有日志输出"""

    # 1. 压制第三方库的噪音日志
    for lib in ["pymilvus", "milvus", "grpc", "httpx", "httpcore", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # 2. 压制 HuggingFace 模型加载报告
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

    # 3. 根日志级别
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 4. structlog 控制台输出（本地开发友好，非 JSON）
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)
