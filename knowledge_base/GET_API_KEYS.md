# Источники данных для FDV Predictor

## DefiLlama API ✅ РАБОТАЕТ

**Лучший бесплатный источник!** Без ключа, без лимитов.

### Что дает:
- 6763+ раундов инвестиций
- 1338 недавних раундов (18 месяцев)
- 67 категорий проектов
- 200+ протоколов с TVL
- Lead investors по каждому раунду

### Как использовать:
```bash
python defillama_collector.py
```

### Собранные данные:
- `defillama_raises.json` - все раунды инвестиций
- `defillama_recent_raises.json` - недавние раунды
- `defillama_category_stats.json` - статистика по категориям
- `defillama_protocols.json` - протоколы с TVL

---

## CoinGecko API ✅ РАБОТАЕТ

Бесплатный без ключа, но с лимитами.

### Лимиты:
- 10-30 запросов в минуту
- Нет funding rounds
- Нет unlock schedules

### Что дает:
- 500+ топ монет с FDV
- Market cap, price, supply
- ATH данные
- Social metrics (Twitter, Telegram)

### Как использовать:
```bash
python coingecko_collector.py
```

### Собранные данные:
- `coingecko_top_coins.json` - топ 500 монет
- `coingecko_fdv_tokens.json` - детальные данные по токенам

---

## DropsTab API (платный)

Требует API ключ. Планы от $19/месяц.

### Что дает:
- Token unlocks schedules
- Детальные funding rounds
- Investors/VCs data

### Как получить:
1. Зайти на https://dropstab.com/products/commercial-api
2. Выбрать план (Starter $19/mo или выше)
3. Или написать на hello@dropstab.com о Builders Program

### Как использовать:
```python
# В dropstab_collector.py:
HEADERS = {
    "accept": "*/*",
    "x-dropstab-api-key": "ТВОЙ_КЛЮЧ"
}
```

---

## Другие источники

| Источник | Статус | Что дает |
|----------|--------|----------|
| Messari | Платный | Хорошие данные, дорого |
| CryptoRank | Бесплатный tier | Рейтинги, fundraising |
| Token Unlocks | Нет API | Unlock schedules (парсинг) |

---

## Итого собрано данных

После запуска всех коллекторов:

```
DefiLlama:
  - 6763 fundraising rounds
  - 1338 recent (18 mo)
  - 67 categories

CoinGecko:
  - 500 top coins
  - 8 detailed tokens (rate limited)
```

**Этого достаточно для начала!**
