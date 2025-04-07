# RAG Stream Processor

Проект для обработки запросов с использованием RAG (Retrieval-Augmented Generation) и потоковой передачи ответов от LLM API.

## Описание

Этот проект позволяет отправлять запросы к LLM API с использованием контекста из документов (RAG) и получать ответы в потоковом режиме. Система анализирует документы, находит релевантную информацию и генерирует ответы на основе найденных данных.

## Структура проекта

- `rag_stream_processor.py` - основной файл для обработки запросов с использованием RAG и потоковой передачи
- `config_loader.py` - модуль для загрузки конфигурации из YAML файла
- `prompt_manager.py` - модуль для работы с системными и пользовательскими сообщениями
- `config.yaml` - файл конфигурации с настройками API
- `config.yaml.example` - пример файла конфигурации
- `prompt_templates.json` - шаблоны системных и пользовательских сообщений
- `response_schema.json` - схема ответа для структурированного вывода
- `rag_doc_data.json` - файл с данными документов в формате JSON
- `requirements.txt` - список зависимостей проекта

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/kekslop/vllm_structured_output_stream.git
cd vllm_structured_output_stream
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Настройка

1. Скопируйте файл `config.yaml.example` в `config.yaml`:
```bash
cp config.yaml.example config.yaml
```

2. Отредактируйте `config.yaml`, заполнив следующие параметры:
   - `api.base_url` - URL вашего API vLLM
   - `api.token` - ваш токен API
   - `api.model_name` - название модели (рекомендуется не менее 7B)
   - `api.completion_tokens` - максимальное количество токенов для генерации
   - `messages.templates_file` - путь к файлу с шаблонами промптов
   - `messages.message_key` - ключ сообщения (опционально)
   - `documents.data_file` - путь к файлу с данными документов
   - `response.schema_file` - путь к файлу со схемой ответа

3. При необходимости отредактируйте:
   - `prompt_templates.json` - для изменения шаблонов сообщений
   - `response_schema.json` - для изменения структуры ответа

## Использование

Запустите основной скрипт с параметрами:

```bash
python rag_stream_processor.py [--message-key KEY] [--stream] [--app_config CONFIG_PATH]
```

Параметры:
- `--message-key` - ключ сообщения из `prompt_templates.json` (опционально)
- `--stream` - включить потоковую передачу (по умолчанию включена)
- `--app_config` - путь к файлу конфигурации (по умолчанию `config.yaml`)

## Преимущества структуры проекта

- Разделение кода на модули по функциональности
- Использование `envyaml` для загрузки конфигурации с поддержкой переменных окружения
- Отдельные файлы для промптов, схемы ответа и конфигурации
- Отдельные функции для потокового и не-потокового режимов
- Поддержка множества готовых запросов через `message_key`
- Централизованная обработка конфигурации

## Лицензия

Этот проект распространяется под лицензией MIT. Вы можете свободно использовать, модифицировать и распространять этот код, при условии сохранения текста лицензии и указания авторов.

MIT License

Copyright (c) 2024 Neural Deep Tech

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 