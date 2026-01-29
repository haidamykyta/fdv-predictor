# FDV Knowledge Base

База знаний для прогнозирования FDV токенов при запуске.

## Структура

```
knowledge_base/
├── data/
│   ├── token_launches.json      # Исторические запуски токенов
│   ├── fundraising_rounds.json  # Данные о раундах финансирования
│   ├── sector_benchmarks.json   # Бенчмарки по секторам
│   └── vc_tiers.json            # Рейтинг VC инвесторов
├── formulas/
│   └── fdv_calculation.md       # Формулы расчета FDV
├── articles/
│   └── sources.md               # Ссылки на статьи и исследования
└── scripts/
    └── collect_data.py          # Скрипт сбора данных
```

## Ключевые факторы FDV

1. **Fundraising History** - Сумма привлеченных средств и оценка
2. **Tokenomics** - Начальная циркуляция, unlock schedule
3. **Sector** - L1/L2/DeFi/Gaming и средние FDV по сектору
4. **Investor Tier** - Качество инвесторов (a16z, Paradigm = tier 1)
5. **Market Conditions** - BTC price, общий sentiment
6. **Twitter/Social** - Размер комьюнити

## Формула прогноза

```
Predicted_FDV = Base_Sector_FDV
                × Fundraising_Multiplier
                × Investor_Tier_Factor
                × Market_Condition_Factor
                × Hype_Factor
```
