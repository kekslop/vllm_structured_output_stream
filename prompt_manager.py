import json

def get_system_message(config):
    """
    Возвращает форматированное системное сообщение с контекстом документов.
    
    Args:
        config: Объект конфигурации
        
    Returns:
        Отформатированное системное сообщение
    """
    # Загружаем данные документов из указанного файла
    with open(config.documents_config.get('data_file', 'rag_doc_data.json'), 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    
    # Форматируем данные документов как строку
    doc_context = json.dumps(doc_data, ensure_ascii=False)
    
    # Возвращаем системное сообщение с подставленным контекстом
    system_template = config.messages_config.get('system_template', '')
    return system_template.format(context=doc_context)

def get_user_message(config, message_key=None):
    """
    Возвращает сообщение пользователя по ключу или сообщение по умолчанию.
    
    Args:
        config: Объект конфигурации
        message_key: Опциональный ключ для выбора конкретного сообщения
        
    Returns:
        Выбранное сообщение или сообщение по умолчанию
    """
    # Всегда возвращаем сообщение по умолчанию из конфигурации
    return config.messages_config.get('default_user_message', '')

def prepare_messages(config, message_key=None):
    """
    Подготавливает сообщения для запроса к API.
    
    Args:
        config: Объект конфигурации
        message_key: Опциональный ключ для выбора конкретного сообщения
        
    Returns:
        Список сообщений в формате для запроса к API
    """
    # Получаем системное и пользовательское сообщения
    system_message = get_system_message(config)
    user_message = get_user_message(config, message_key)
    
    # Подготавливаем сообщения в формате для API
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ] 