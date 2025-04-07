import os
import json
import asyncio
import time
from typing import Dict, List, Any
from openai import AsyncOpenAI
import tiktoken
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# API configuration
BASE_URL = config['api']['base_url']
API_KEY = config['api']['token']

# Model settings
MODEL_NAME = config['api']['model_name']
COMPLETION_TOKENS = config['api']['completion_tokens']

# Document schema for response validation
DOCUMENT_SCHEMA = config['response_schema']

def get_system_message():
    """Returns the formatted system message with the document context"""
    # Load document data from the specified file
    with open(config['documents']['data_file'], 'r') as f:
        doc_data = json.load(f)
    
    # Format the document data as a string
    doc_context = json.dumps(doc_data)
    
    return config['messages']['system_template'].format(context=doc_context)

def get_user_message(message_key=None):
    """
    Returns a user message by key or the default message if no key is provided
    
    Args:
        message_key: Optional key to select a specific message from USER_MESSAGES
        
    Returns:
        The selected message or the default message
    """
    return config['messages']['default_user_message']

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Оценивает количество токенов в тексте"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    return len(encoding.encode(text))


async def send_llm_request(message_key=None, stream=True):
    """
    Send a request to the LLM API endpoint using AsyncOpenAI client

    Args:
        message_key: Optional key to select a specific message from USER_MESSAGES
        stream: Whether to stream the response (default True)
    """
    # Ensure responses directory exists
    os.makedirs("responses", exist_ok=True)

    # Get user message
    user_message = get_user_message(message_key)

    # Get system message from document_context.py
    system_message = get_system_message()

    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Create the client
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Extra body parameters with schema
    extra_body = {
        "repetition_penalty": 1,
        "guided_json": json.dumps(DOCUMENT_SCHEMA),
        "guided_decoding_backend": "xgrammar"
    }

    try:
        print(f"Sending request for: {user_message}")

        # Send request
        if stream:
            # Handle streaming response
            print("Streaming response:")
            full_response = ""

            # Переменные для измерения
            request_start_time = time.time()
            first_chunk_time = None
            chunk_timestamps = []
            chunk_sizes = []

            # Create the streaming request
            response_stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=COMPLETION_TOKENS,
                temperature=0.2,
                extra_body=extra_body,
                stream=True,
            )

            # Process the stream
            chunk_counter = 0
            debug_mode = False  # Можно включить для отладки структуры чанков

            # Создадим файл для сохранения сырых чанков
            raw_chunks_file = f"responses/raw_chunks_{message_key or 'default'}.jsonl"
            with open(raw_chunks_file, "w") as raw_file:
                async for chunk in response_stream:
                    current_time = time.time()
                    chunk_counter += 1

                    # Сохраняем полные данные чанка в файл
                    raw_file.write(json.dumps({
                        "timestamp": current_time,
                        "chunk_num": chunk_counter,
                        "data": chunk.model_dump()
                    }) + "\n")
                    raw_file.flush()  # Гарантируем запись даже при прерывании

                    # Выводим полную структуру чанка при отладке
                    if debug_mode:
                        print(f"\n[CHUNK {chunk_counter}]")
                        print(json.dumps(chunk.model_dump(), indent=2))

                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta

                        # Выводим все возможные поля дельты при отладке
                        if debug_mode and hasattr(delta, 'model_dump'):
                            print(f"[DELTA {chunk_counter}]")
                            print(json.dumps(delta.model_dump(), indent=2))

                        if delta.content is not None:
                            content = delta.content

                            # Если это первый чанк с контентом, запомним время
                            if first_chunk_time is None:
                                first_chunk_time = current_time
                                print(
                                    f"\n[METRICS] Time to first chunk: {first_chunk_time - request_start_time:.3f} seconds\n")

                            # Записываем чанк и его размер
                            chunk_timestamps.append(current_time)
                            chunk_sizes.append(len(content))

                            full_response += content
                            print(content, end="")

            # Посчитаем метрики после получения всего ответа
            total_time = time.time() - request_start_time
            if first_chunk_time is None:
                first_chunk_time = time.time()  # На случай, если никаких чанков не получили
            time_to_first_chunk = first_chunk_time - request_start_time

            print("\n\n")
            print(f"[FULL RESPONSE]\n{full_response}\n")

            # Технические детали запроса
            try:
                final_chunk_info = chunk
                print(f"\n[RAW FINAL CHUNK INFO]")
                print(json.dumps(final_chunk_info.model_dump(), indent=2))

                # Проверяем, есть ли информация о токенах в последнем чанке
                if hasattr(final_chunk_info, 'usage') and final_chunk_info.usage:
                    print("\n[TOKEN USAGE FROM FINAL CHUNK]")
                    print(json.dumps(final_chunk_info.usage.model_dump(), indent=2))
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

                # Запрос точного количества токенов через неpотоковый запрос
                print("\n[INFO] No token usage in final chunk. Fetching sample token usage with a test request...")
                try:
                    # Используем короткое сообщение для получения метаданных
                    token_check_response = await client.chat.completions.create(
                        model=MODEL_NAME,
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
                    print(json.dumps(token_check_response.usage.model_dump(), indent=2))

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

            print(f"\n[METRICS] Total response time: {total_time:.3f} seconds")
            print(f"[METRICS] Time to first chunk: {time_to_first_chunk:.3f} seconds")
            print(f"[METRICS] Estimated completion tokens: {token_count}")
            print(f"[METRICS] Generation speed: {tokens_per_second:.2f} tokens/second")

            # Детальная информация о чанках (для анализа равномерности)
            if len(chunk_timestamps) > 1:
                chunk_intervals = [chunk_timestamps[i] - chunk_timestamps[i - 1] for i in
                                   range(1, len(chunk_timestamps))]
                avg_chunk_interval = sum(chunk_intervals) / len(chunk_intervals)
                print(f"[METRICS] Average interval between chunks: {avg_chunk_interval:.3f} seconds")

            # Save response to file
            response_file = f"responses/response_{message_key or 'default'}.txt"
            metrics_file = f"responses/metrics_{message_key or 'default'}.json"

            with open(response_file, "w") as f:
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

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Response saved to {response_file}")
            print(f"Metrics saved to {metrics_file}")
        else:
            # Handle non-streaming response with guided json format
            request_start_time = time.time()

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=COMPLETION_TOKENS,
                temperature=0.7,
                extra_body=extra_body,
                stream=False
            )

            # Получено время ответа
            response_time = time.time() - request_start_time

            # Get response content
            result = response.choices[0].message.content

            # Считаем токены
            token_count = count_tokens(result)
            tokens_per_second = token_count / response_time

            # Выводим метрики
            print(f"[METRICS] Total response time: {response_time:.3f} seconds")
            print(f"[METRICS] Estimated token count: {token_count}")
            print(f"[METRICS] Generation speed: {tokens_per_second:.2f} tokens/second")

            # Parse as JSON for pretty printing
            try:
                json_result = json.loads(result)
                formatted_result = json.dumps(json_result, indent=2)
                print(formatted_result)
            except json.JSONDecodeError:
                # Fall back to plain text if not valid JSON
                print(result)

            # Save response to file
            response_file = f"responses/response_{message_key or 'default'}.json"
            metrics_file = f"responses/metrics_{message_key or 'default'}.json"

            with open(response_file, "w") as f:
                f.write(result)

            # Сохраняем метрики в отдельный файл
            metrics = {
                "total_time_seconds": response_time,
                "time_to_first_chunk_seconds": response_time,  # У нас нет чанков в режиме без стриминга
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "request_timestamp": request_start_time
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Response saved to {response_file}")
            print(f"Metrics saved to {metrics_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


async def main_async():
    # Parse command line arguments
    import sys

    # Default to chairman query with streaming
    message_key = "chairman"
    stream = True

    # Check for command line arguments
    if len(sys.argv) > 1:
        message_key = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2].lower() == "nostream":
        stream = False

    # Run the query
    await send_llm_request(message_key, stream)


def main():
    """Entry point for script execution"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()