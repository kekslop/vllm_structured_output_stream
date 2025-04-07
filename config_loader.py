import argparse
import functools
import os
from envyaml import EnvYAML

@functools.cache
def get_config(argv=None):
    """
    Загружает конфигурацию из YAML файла с поддержкой переменных окружения.
    
    Args:
        argv: Аргументы командной строки (опционально)
        
    Returns:
        Объект с конфигурацией приложения
    """
    parser = argparse.ArgumentParser(description='RAG Stream Processor')
    parser.add_argument(
        "--app_config",
        dest="app_config",
        required=False,
        type=str,
        default=os.environ.get("APP_CONFIG", "config.yaml"),
    )
    parser.add_argument('--message-key', type=str, help='Key for the message to use')
    parser.add_argument('--stream', action='store_true', default=True, help='Enable streaming mode')

    args = parser.parse_args(argv)
    
    # Загружаем конфигурацию из YAML файла
    config_yaml = EnvYAML(args.app_config)
    
    # Добавляем конфигурацию к аргументам
    args.api_config = config_yaml.get('api', {})
    args.messages_config = config_yaml.get('messages', {})
    args.documents_config = config_yaml.get('documents', {})
    args.response_schema = config_yaml.get('response_schema', {})
    
    return args 