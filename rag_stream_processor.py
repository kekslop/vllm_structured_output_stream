import os
import json
import asyncio
import time
import codecs
from typing import Dict, List, Any
from openai import AsyncOpenAI
import tiktoken

# Импортируем наши модули
from config_loader import get_config
from prompt_manager import prepare_messages

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Оценивает количество токенов в тексте"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    return len(encoding.encode(text))

async def process_streaming_response(client, messages, model_name, completion_tokens, extra_body, message_key=None):
    """
    Обрабатывает потоковый ответ от API.
    
    Args:
        client: Клиент AsyncOpenAI
        messages: Список сообщений для запроса
        model_name: Название модели
        completion_tokens: Максимальное количество токенов для генерации
        extra_body: Дополнительные параметры запроса
        message_key: Опциональный ключ для выбора конкретного сообщения
    """
    # Обеспечиваем существование директории для ответов
    os.makedirs("responses", exist_ok=True)
    
    print("Streaming response:")
    full_response = ""

    # Переменные для измерения
    request_start_time = time.time()
    first_chunk_time = None
    chunk_timestamps = []
    chunk_sizes = []

    # Создаем потоковый запрос
    response_stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=completion_tokens,
        temperature=0.2,
        extra_body=extra_body,
        stream=True,
    )

    # Обрабатываем поток
    chunk_counter = 0
    debug_mode = False  # Можно включить для отладки структуры чанков

    # Создаем файл для сохранения сырых чанков
    raw_chunks_file = f"responses/raw_chunks_{message_key or 'default'}.jsonl"
    with open(raw_chunks_file, "w", encoding='utf-8') as raw_file:
        async for chunk in response_stream:
            current_time = time.time()
            chunk_counter += 1

            # Сохраняем полные данные чанка в файл
            raw_file.write(json.dumps({
                "timestamp": current_time,
                "chunk_num": chunk_counter,
                "data": chunk.model_dump()
            }, ensure_ascii=False) + "\n")
            raw_file.flush()  # Гарантируем запись даже при прерывании

            # Выводим полную структуру чанка при отладке
            if debug_mode:
                print(f"\n[CHUNK {chunk_counter}]")
                print(json.dumps(chunk.model_dump(), indent=2, ensure_ascii=False))

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                # Выводим все возможные поля дельты при отладке
                if debug_mode and hasattr(delta, 'model_dump'):
                    print(f"[DELTA {chunk_counter}]")
                    print(json.dumps(delta.model_dump(), indent=2, ensure_ascii=False))

                if delta.content is not None:
                    content = delta.content

                    # Если это первый чанк с контентом, запомним время
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        print(
                            f"\n[METRICS] Время до первого токена: {first_chunk_time - request_start_time:.3f} секунд\n")

                    # Записываем чанк и его размер
                    chunk_timestamps.append(current_time)
                    chunk_sizes.append(len(content))

                    full_response += content
                    print(content, end="", flush=True)

    # Подсчитываем метрики после получения всего ответа
    total_time = time.time() - request_start_time
    if first_chunk_time is None:
        first_chunk_time = time.time()  # На случай, если никаких чанков не получили
    time_to_first_chunk = first_chunk_time - request_start_time

    print("\n\n")
    print(f"[FULL RESPONSE]\n{full_response}\n")
    
    # Пытаемся распарсить JSON для более читаемого вывода
    try:
        json_result = json.loads(full_response)
        
        # Выводим запрос пользователя
        print("\n=== ЗАПРОС ===")
        print(json_result["reasoning"]["query_analysis"]["user_query"])
        
        # Выводим ответ
        print("\n=== ОТВЕТ ===")
        print(json_result["response"])
        
        # Выводим источники
        print("\n=== ИСТОЧНИКИ ===")
        for source in json_result["sources"]:
            print(f"Документ: {source['document_name']}, Страницы: {source['pages']}")
            if "citation" in source:
                print(f"Цитата: {source['citation']}")
            print("---")
    except json.JSONDecodeError:
        # Если не удалось распарсить JSON, выводим как есть
        print("\n=== ПОЛНЫЙ ОТВЕТ ===")
        print(full_response)

    # Технические детали запроса
    try:
        final_chunk_info = chunk
        print(f"\n[RAW FINAL CHUNK INFO]")
        print(json.dumps(final_chunk_info.model_dump(), indent=2, ensure_ascii=False))

        # Проверяем, есть ли информация о токенах в последнем чанке
        if hasattr(final_chunk_info, 'usage') and final_chunk_info.usage:
            print("\n[TOKEN USAGE FROM FINAL CHUNK]")
            print(json.dumps(final_chunk_info.usage.model_dump(), indent=2, ensure_ascii=False))
            prompt_tokens = final_chunk_info.usage.prompt_tokens
            completion_tokens = final_chunk_info.usage.completion_tokens
            total_tokens = final_chunk_info.usage.total_tokens

            token_count = completion_tokens
            has_token_info = True
        else:
            has_token_info = False
    except Exception as e:
        print(f"\n[ERROR] Could not dump final chunk info: {e}")
        has_token_info = False

    # Если нет информации о токенах, делаем запрос для получения метрик
    if not has_token_info:
        # Подсчёт токенов с помощью tiktoken
        token_count = count_tokens(full_response)

        # Запрос точного количества токенов через не-потоковый запрос
        print("\n[INFO] No token usage in final chunk. Fetching sample token usage with a test request...")
        try:
            # Используем короткое сообщение для получения метаданных
            token_check_response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Short response."},
                    {"role": "user", "content": "Say OK"}
                ],
                max_tokens=10,
                temperature=0.7,
                extra_body=extra_body,
                stream=False
            )

            print("\n[SAMPLE USAGE INFO]")
            print(json.dumps(token_check_response.usage.model_dump(), indent=2, ensure_ascii=False))

            # Делаем оценку для нашего реального запроса с аналогичной пропорцией
            # Формула: (prompt_tokens / completion_tokens в тестовом запросе) * текущие completion_tokens
            if hasattr(token_check_response, 'usage') and token_check_response.usage:
                test_ratio = token_check_response.usage.prompt_tokens / max(
                    token_check_response.usage.completion_tokens, 1)
                estimated_prompt_tokens = int(test_ratio * token_count)

                print(
                    f"\n[INFO] Based on test ratio, estimated prompt tokens for main request: ~{estimated_prompt_tokens}")
                print(f"[INFO] Estimated completion tokens: ~{token_count}")
                print(f"[INFO] Estimated total tokens: ~{estimated_prompt_tokens + token_count}")

        except Exception as e:
            print(f"\n[ERROR] Could not fetch sample token usage: {e}")

    # Подсчёт скорости
    tokens_per_second = token_count / (time.time() - first_chunk_time) if first_chunk_time != time.time() else 0

    print(f"\n[METRICS] Общее время ответа: {total_time:.3f} секунд")
    print(f"[METRICS] Время до первого токена: {time_to_first_chunk:.3f} секунд")
    print(f"[METRICS] Время генерации ответа: {total_time - time_to_first_chunk:.3f} секунд")
    print(f"[METRICS] Оценка количества токенов: {token_count}")
    print(f"[METRICS] Скорость генерации: {tokens_per_second:.2f} токенов/секунду")

    # Детальная информация о чанках (для анализа равномерности)
    if len(chunk_timestamps) > 1:
        chunk_intervals = [chunk_timestamps[i] - chunk_timestamps[i - 1] for i in
                          range(1, len(chunk_timestamps))]
        avg_chunk_interval = sum(chunk_intervals) / len(chunk_intervals)
        print(f"[METRICS] Average interval between chunks: {avg_chunk_interval:.3f} seconds")

    # Сохраняем ответ в файл
    response_file = f"responses/response_{message_key or 'default'}.txt"
    metrics_file = f"responses/metrics_{message_key or 'default'}.json"

    with open(response_file, "w", encoding='utf-8') as f:
        f.write(full_response)

    # Сохраняем метрики в отдельный файл
    metrics = {
        "total_time_seconds": total_time,
        "time_to_first_chunk_seconds": time_to_first_chunk,
        "token_count": token_count,
        "tokens_per_second": tokens_per_second,
        "total_chunks": len(chunk_timestamps),
        "total_raw_chunks": chunk_counter,
        "request_timestamp": request_start_time
    }

    print(f"[INFO] Total raw chunks received: {chunk_counter}")
    print(f"[INFO] Raw chunks saved to: {raw_chunks_file}")

    with open(metrics_file, "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Response saved to {response_file}")
    print(f"Metrics saved to {metrics_file}")
    
    return full_response

async def process_non_streaming_response(client, messages, model_name, completion_tokens, extra_body, message_key=None):
    """
    Обрабатывает не-потоковый ответ от API.
    
    Args:
        client: Клиент AsyncOpenAI
        messages: Список сообщений для запроса
        model_name: Название модели
        completion_tokens: Максимальное количество токенов для генерации
        extra_body: Дополнительные параметры запроса
        message_key: Опциональный ключ для выбора конкретного сообщения
    """
    # Обеспечиваем существование директории для ответов
    os.makedirs("responses", exist_ok=True)
    
    # Время начала запроса
    request_start_time = time.time()

    # Создаем запрос
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=completion_tokens,
        temperature=0.7,
        extra_body=extra_body,
        stream=False
    )

    # Получено время ответа
    response_time = time.time() - request_start_time

    # Получаем содержимое ответа
    result = response.choices[0].message.content

    # Считаем токены
    token_count = count_tokens(result)
    tokens_per_second = token_count / response_time

    # Выводим метрики
    print(f"[METRICS] Общее время ответа: {response_time:.3f} секунд")
    print(f"[METRICS] Оценка количества токенов: {token_count}")
    print(f"[METRICS] Скорость генерации: {tokens_per_second:.2f} токенов/секунду")

    # Парсим как JSON для красивого вывода
    try:
        json_result = json.loads(result)
        
        # Выводим запрос пользователя
        print("\n=== ЗАПРОС ===")
        print(json_result["reasoning"]["query_analysis"]["user_query"])
        
        # Выводим ответ
        print("\n=== ОТВЕТ ===")
        print(json_result["response"])
        
        # Выводим источники
        print("\n=== ИСТОЧНИКИ ===")
        for source in json_result["sources"]:
            print(f"Документ: {source['document_name']}, Страницы: {source['pages']}")
            if "citation" in source:
                print(f"Цитата: {source['citation']}")
            print("---")
        
        # Сохраняем полный JSON для отладки
        formatted_result = json.dumps(json_result, indent=2, ensure_ascii=False)
        print("\n=== ПОЛНЫЙ JSON (для отладки) ===")
        print(formatted_result)
    except json.JSONDecodeError:
        # Если не удалось распарсить JSON, выводим как есть
        print(result)

    # Сохраняем ответ в файл
    response_file = f"responses/response_{message_key or 'default'}.json"
    metrics_file = f"responses/metrics_{message_key or 'default'}.json"

    with open(response_file, "w", encoding='utf-8') as f:
        f.write(result)

    # Сохраняем метрики в отдельный файл
    metrics = {
        "total_time_seconds": response_time,
        "time_to_first_chunk_seconds": response_time,  # У нас нет чанков в режиме без стриминга, значение формальное
        "token_count": token_count,
        "tokens_per_second": tokens_per_second,
        "request_timestamp": request_start_time
    }

    with open(metrics_file, "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Response saved to {response_file}")
    print(f"Metrics saved to {metrics_file}")
    
    return result

async def send_llm_request(config, message_key=None, stream=True):
    """
    Отправляет запрос к API LLM и обрабатывает ответ.
    
    Args:
        config: Объект конфигурации
        message_key: Опциональный ключ для выбора сообщения
        stream: Использовать ли потоковую передачу (по умолчанию True)
    """
    # Получаем сообщения из менеджера промптов
    messages = prepare_messages(config, message_key)
    
    # Получаем настройки API из конфигурации
    api_config = config.api_config
    base_url = api_config.get('base_url', '')
    api_key = api_config.get('token', '')
    model_name = api_config.get('model_name', '')
    completion_tokens = api_config.get('completion_tokens', 2000)
    
    # Получаем настройки ответа из конфигурации
    response_config = config.response_config if hasattr(config, 'response_config') else {}
    
    # Загружаем схему ответа из файла
    schema_file = response_config.get('schema_file', 'response_schema.json')
    with open(schema_file, 'r', encoding='utf-8') as f:
        response_schema = json.load(f)
    
    # Получаем дополнительные параметры декодирования
    guided_decoding_backend = response_config.get('guided_decoding_backend', 'xgrammar')
    repetition_penalty = response_config.get('repetition_penalty', 1.0)
    
    # Подготавливаем клиент
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Экстра-параметры с схемой
    extra_body = {
        "repetition_penalty": repetition_penalty,
        "guided_json": json.dumps(response_schema),
        "guided_decoding_backend": guided_decoding_backend
    }
    
    # Выводим информацию о запросе
    user_message = messages[1]['content']
    print(f"Sending request for: {user_message}")
    
    try:
        # Обрабатываем запрос в зависимости от режима
        if stream:
            return await process_streaming_response(
                client, messages, model_name, completion_tokens, extra_body, message_key
            )
        else:
            return await process_non_streaming_response(
                client, messages, model_name, completion_tokens, extra_body, message_key
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def main_async():
    """Асинхронная точка входа"""
    # Загружаем конфигурацию
    config = get_config()
    
    # Отправляем запрос с параметрами из конфигурации
    await send_llm_request(config, config.message_key, config.stream)

def main():
    """Точка входа для выполнения скрипта"""
    # Устанавливаем кодировку для stdout
    import sys
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    
    # Запускаем асинхронную функцию
    asyncio.run(main_async())

if __name__ == "__main__":
    main()