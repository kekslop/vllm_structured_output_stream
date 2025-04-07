import json

def load_prompt_templates(file_path):
    """
    Загружает шаблоны промптов из JSON-файла.
    
    Args:
        file_path: Путь к файлу с шаблонами промптов
        
    Returns:
        Словарь с шаблонами промптов
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_system_message(config):
    """
    Возвращает форматированное системное сообщение с контекстом документов.
    
    Args:
        config: Объект конфигурации
        
    Returns:
        Отформатированное системное сообщение
    """
    # Загружаем шаблоны промптов
    templates_file = config.messages_config.get('templates_file', 'prompt_templates.json')
    templates = load_prompt_templates(templates_file)
    
    # Загружаем данные документов из указанного файла
    docs_file = config.documents_config.get('data_file', 'rag_doc_data.json')
    with open(docs_file, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    
    # Форматируем данные документов как строку
    doc_context = json.dumps(doc_data, ensure_ascii=False)
    
    # Возвращаем системное сообщение с подставленным контекстом
    system_template = templates.get('system_template', '')
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
    # Загружаем шаблоны промптов
    templates_file = config.messages_config.get('templates_file', 'prompt_templates.json')
    templates = load_prompt_templates(templates_file)
    
    # Проверяем, указан ли message_key в конфигурации, если не указан в аргументах
    if message_key is None:
        message_key = config.messages_config.get('message_key')
    
    # Если ключ указан и существует в списке сообщений, возвращаем соответствующее сообщение
    if message_key and 'user_messages' in templates and message_key in templates['user_messages']:
        return templates['user_messages'][message_key]
    
    # Иначе возвращаем сообщение по умолчанию
    return templates.get('default_user_message', '')

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