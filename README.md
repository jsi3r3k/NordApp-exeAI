# NordApp RAG API

To jest moje rozwiązanie zadania praktycznego exeAI. Aplikacja udostępnia małe REST API, które odpowiada na pytania dotyczące NordApp wyłącznie na podstawie dostarczonej bazy wiedzy.

Projekt wykorzystuje podejście RAG:

1. dokument jest dzielony na logiczne fragmenty,
2. aplikacja wyszukuje fragmenty najlepiej pasujące do pytania,
3. Gemini generuje odpowiedź na podstawie znalezionego kontekstu,
4. API zwraca odpowiedź razem z wykorzystanymi źródłami.

Jeżeli dokument nie zawiera potrzebnej informacji, aplikacja nie próbuje zgadywać i zwraca kontrolowaną odmowę.

## Uruchomienie

Do uruchomienia potrzebne są:

- Docker z obsługą Docker Compose,
- klucz do Gemini API.

Najpierw należy utworzyć lokalny plik konfiguracyjny:

```bash
cp .env.example .env
```

Następnie uzupełnić klucz w `.env`:

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-3.6-flash
```

Aplikację można uruchomić jedną komendą:

```bash
docker compose up --build
```

Przy pierwszym uruchomieniu pobierany jest lokalny model embeddingowy, dlatego start może potrwać trochę dłużej. Model jest zapisywany w wolumenie Dockera i nie musi być pobierany ponownie przy każdym uruchomieniu.

Po uruchomieniu dostępne są:

- Swagger UI: http://localhost:8000/docs
- healthcheck: http://localhost:8000/health
- endpoint pytań: `POST http://localhost:8000/ask`

Zatrzymanie aplikacji:

```bash
docker compose down
```

## Przykładowe użycie

Zapytanie:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Ile kosztuje plan Team?"}'
```

Odpowiedź:

```json
{
  "answer": "Plan Team kosztuje 149 zł netto miesięcznie. Przy rozliczeniu rocznym obowiązuje 20% rabatu.",
  "grounded": true,
  "sources": [
    {
      "chunk_id": "plany-i-ceny",
      "section": "Plany i ceny",
      "content": "NordApp jest dostępny w trzech planach...",
      "score": 0.8675
    }
  ]
}
```

Przykład pytania, na które dokument nie odpowiada:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Czy NordApp integruje się z Jira?"}'
```

Odpowiedź:

```json
{
  "answer": "Nie znalazłem tej informacji w bazie wiedzy NordApp.",
  "grounded": false,
  "sources": []
}
```

## Jak działa aplikacja

```text
Pytanie użytkownika
        ↓
Walidacja Pydantic
        ↓
Embedding pytania
        ↓
Wyszukanie dwóch najlepiej dopasowanych fragmentów
        ↓
Gemini z ograniczonym kontekstem
        ↓
Walidacja źródeł po stronie backendu
        ↓
Odpowiedź albo kontrolowana odmowa
```

Dokument jest przetwarzany podczas uruchamiania aplikacji. Wtedy powstają jego fragmenty i embeddingi. Dzięki temu nie są one obliczane ponownie przy każdym zapytaniu.

## Podjęte decyzje

### Podział dokumentu

Zdecydowałam się dzielić dokument według nagłówków drugiego poziomu Markdown. Dostarczona baza wiedzy ma osiem wyraźnych sekcji, takich jak ceny, limity, integracje czy bezpieczeństwo.

W tym przypadku podział według nagłówków zachowuje znaczenie całych sekcji i jest czytelniejszy niż cięcie tekstu co określoną liczbę znaków.

Dla większych lub mniej uporządkowanych dokumentów zastosowałabym podział według liczby tokenów, prawdopodobnie z częściowym nakładaniem się fragmentów.

### Wyszukiwanie in-memory

Po podziale dokument zawiera tylko osiem fragmentów. Z tego powodu nie użyłam osobnej bazy wektorowej. Embeddingi są przechowywane w pamięci, a podobieństwo jest liczone bezpośrednio za pomocą NumPy.

Dodanie FAISS, Chromy albo osobnej usługi bazodanowej zwiększyłoby liczbę zależności, ale nie przyniosłoby tutaj realnej korzyści. Przy większej bazie wiedzy warstwę in-memory można zastąpić np. pgvector lub Qdrantem bez zmiany publicznego API.

### Embeddingi

Do wyszukiwania wykorzystuję lokalny model:

```text
intfloat/multilingual-e5-small
```

Wybrałam go ze względu na obsługę języka polskiego i stosunkowo niewielki rozmiar. Pytania otrzymują prefiks `query:`, natomiast fragmenty dokumentu `passage:`, zgodnie ze sposobem działania modelu E5.

### Liczba pobieranych fragmentów

Do Gemini przekazuję dwa najlepiej dopasowane fragmenty.

Podczas ręcznego sprawdzania retrievalu pytanie o liczbę użytkowników planu Starter umieściło właściwą sekcję na drugim miejscu. Samo `top_k=1` mogłoby więc odrzucić potrzebny kontekst.

## Zapobieganie halucynacjom

Początkowo rozważałam wykorzystanie minimalnego wyniku podobieństwa jako głównego warunku odmowy. Testy pokazały jednak, że wyniki modelu embeddingowego nie pozwalają łatwo oddzielić pytań obsługiwanych od nieobsługiwanych.

Przykładowo pytanie o nieistniejącą integrację z Jira uzyskało wynik `0.8752`, podczas gdy poprawne pytanie o cenę planu Team uzyskało `0.8675`. Ustawienie progu pomiędzy tymi wartościami odrzucałoby również poprawne pytania.

Zamiast arbitralnego progu zastosowałam kilka innych zabezpieczeń:

1. Gemini otrzymuje wyłącznie dwa fragmenty znalezione przez retrieval.
2. Instrukcja systemowa zabrania korzystania z wiedzy zewnętrznej.
3. Brak informacji nie może zostać zinterpretowany jako odpowiedź przecząca.
4. Gemini zwraca ustrukturyzowany wynik zawierający:
   - `answerable`,
   - `answer`,
   - `source_ids`.
5. Backend sprawdza, czy wskazane źródła rzeczywiście należą do fragmentów zwróconych przez retrieval.
6. Odpowiedź bez prawidłowego źródła jest odrzucana.
7. Przy braku danych API zwraca stały komunikat odmowy.

Dzięki temu ostateczna decyzja nie jest oparta wyłącznie na deklaracji modelu.

## Testy

Testy jednostkowe nie pobierają modelu embeddingowego i nie wywołują Gemini API. Retriever i generator odpowiedzi są zastępowane mockami, dzięki czemu testy są szybkie i deterministyczne.

Do uruchomienia testów potrzebny jest Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

Obecne testy sprawdzają:

- poprawną odpowiedź ze źródłem,
- odmowę, gdy kontekst nie zawiera odpowiedzi,
- odrzucenie odpowiedzi wskazującej nieistniejące źródło.

## Ograniczenia

Ze względu na zakres i limit czasowy zadania przyjęłam kilka uproszczeń:

- aplikacja pracuje na jednym statycznym dokumencie,
- embeddingi istnieją tylko w pamięci procesu,
- generowanie odpowiedzi wymaga dostępu do Gemini API,
- pierwsze uruchomienie wymaga pobrania lokalnego modelu,
- API nie ma uwierzytelniania ani rate limitingu,
- decyzja modelu językowego nadal ma charakter probabilistyczny.

Nie dodawałam opcjonalnego tool callingu. Priorytetem było dla mnie ukończenie i przetestowanie obowiązkowego przepływu RAG oraz zabezpieczenie odpowiedzi przed halucynacjami.

## Jak mierzyłabym jakość

W kolejnym kroku przygotowałabym mały, wersjonowany zbiór pytań testowych zawierający:

- pytania z bezpośrednią odpowiedzią,
- parafrazy tych samych pytań,
- pytania wymagające pobrania dwóch sekcji,
- pytania podobne tematycznie, ale bez odpowiedzi w dokumencie,
- pytania całkowicie niezwiązane z NordApp.

Dla retrievalu mierzyłabym `Recall@2`, czyli jak często właściwy fragment znajduje się w dwóch najlepszych wynikach.

Dla całej aplikacji mierzyłabym:

- poprawność odpowiedzi,
- zgodność odpowiedzi ze źródłami,
- skuteczność rozpoznawania pytań bez odpowiedzi,
- liczbę nieuzasadnionych odmów,
- czas odpowiedzi.

## Struktura projektu

```text
.
├── app
│   ├── chunking.py       # podział dokumentu
│   ├── llm.py            # komunikacja z Gemini
│   ├── main.py           # aplikacja FastAPI
│   ├── retrieval.py      # embeddingi i wyszukiwanie
│   ├── schemas.py        # modele Pydantic
│   └── service.py        # połączenie retrievalu i LLM
├── data
│   └── nordapp_baza_wiedzy.md
├── tests
│   └── test_service.py
├── .dockerignore
├── .env.example
├── .gitignore
├── compose.yaml
├── Dockerfile
├── requirements-dev.txt
└── requirements.txt
```