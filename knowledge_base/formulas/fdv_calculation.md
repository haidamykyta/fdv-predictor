# Формулы расчета FDV

## Базовая формула

```
FDV = Token_Price × Total_Supply
```

Где:
- **Token_Price** = цена токена в момент TGE
- **Total_Supply** = максимальное количество токенов

## Альтернативный расчет через Market Cap

```
FDV = Market_Cap / Circulating_Pct

Где:
Market_Cap = Token_Price × Circulating_Supply
Circulating_Pct = Circulating_Supply / Total_Supply
```

---

## Прогнозирование FDV

### Метод 1: Fundraising Multiple

```
Predicted_FDV = Last_Valuation × Multiple

Типичные Multiple по раунду:
- Seed → TGE: 10-50x
- Series A → TGE: 5-20x
- Series B → TGE: 2-10x
```

**Пример:**
- Проект поднял $50M при оценке $500M (Series A)
- Типичный multiple для Series A: 8x
- Predicted FDV = $500M × 8 = $4B

### Метод 2: Comparable Analysis

```
Predicted_FDV = Sector_Median_FDV × Adjustment_Factor

Adjustment_Factor =
    Investor_Tier_Mult
    × Market_Condition_Mult
    × Hype_Mult
```

**Пример:**
- Сектор: L2 ZK Rollup (median FDV = $10B)
- Tier 1 investors: ×1.5
- Bull market: ×1.3
- High hype: ×1.2
- Predicted FDV = $10B × 1.5 × 1.3 × 1.2 = $23.4B

### Метод 3: Revenue/TVL Multiple

```
Для DeFi протоколов:
FDV = TVL × TVL_Multiple
FDV = Annual_Revenue × Revenue_Multiple

Типичные TVL Multiples:
- DEX: 0.3-1x TVL
- Lending: 0.2-0.5x TVL
- Staking/Restaking: 0.1-0.3x TVL

Типичные Revenue Multiples:
- DEX: 50-200x revenue
- Lending: 30-100x revenue
```

---

## Факторы корректировки

### 1. Investor Tier Factor

| Tier | Multiplier | Примеры |
|------|------------|---------|
| Tier 1 | 1.5x | a16z, Paradigm, Sequoia |
| Tier 2 | 1.2x | Dragonfly, Multicoin |
| Tier 3 | 1.0x | Small VCs |
| No VC | 0.7x | Community/fair launch |

### 2. Market Condition Factor

| Условие | Multiplier |
|---------|------------|
| Bull market (BTC > prev ATH) | 1.5x |
| Neutral | 1.0x |
| Bear market (BTC < 50% ATH) | 0.5x |

### 3. Initial Circulation Factor

```
Low_Circ_Premium = 1 / sqrt(Initial_Circ_Pct)

Примеры:
- 5% unlock: premium = 1/√0.05 = 4.5x
- 15% unlock: premium = 1/√0.15 = 2.6x
- 30% unlock: premium = 1/√0.30 = 1.8x
```

### 4. Hype/Social Factor

```
Hype_Factor = log10(Twitter_Followers) / 5

Примеры:
- 10K followers: factor = 0.8
- 100K followers: factor = 1.0
- 1M followers: factor = 1.2
```

---

## Итоговая формула прогноза

```python
def predict_fdv(
    sector_median_fdv,
    last_valuation=None,
    fundraising_round="series_a",
    investor_tier=2,
    market_condition="neutral",
    initial_circ_pct=0.15,
    twitter_followers=100000
):
    # Базовый FDV
    if last_valuation:
        multiples = {"seed": 30, "series_a": 8, "series_b": 4}
        base_fdv = last_valuation * multiples.get(fundraising_round, 5)
    else:
        base_fdv = sector_median_fdv

    # Факторы
    tier_mult = {1: 1.5, 2: 1.2, 3: 1.0, 0: 0.7}[investor_tier]
    market_mult = {"bull": 1.5, "neutral": 1.0, "bear": 0.5}[market_condition]
    circ_premium = 1 / (initial_circ_pct ** 0.5)
    hype_factor = min(max(log10(twitter_followers) / 5, 0.5), 1.5)

    # Итог
    predicted_fdv = base_fdv * tier_mult * market_mult * hype_factor

    # Корректировка на циркуляцию (не прямой множитель)
    predicted_fdv = predicted_fdv * (1 + (circ_premium - 1) * 0.3)

    return predicted_fdv
```

---

## Примеры расчетов

### Eigenlayer
- Sector: Restaking (median $4B)
- Last valuation: $500M (Series B)
- Investors: Tier 1 (a16z, Polychain)
- Initial circ: 5.5%
- Market: Bull

```
Base = $500M × 4 = $2B (Series B multiple)
Tier factor = 1.5
Market factor = 1.5
Hype factor = 1.2

Predicted = $2B × 1.5 × 1.5 × 1.2 = $5.4B
Actual = $1.29B (рынок перекупил, коррекция)
```

### friend.tech
- Sector: SocialFi (median $200M)
- Last valuation: ~$50M (implied)
- Investors: Tier 1 (Paradigm)
- Initial circ: 100% (fair launch)
- Market: Bull but cooling

```
Base = $200M (sector median, no fundraising)
Tier factor = 1.5
Market factor = 1.0
Circ factor = 1.0 (full unlock = no premium)

Predicted = $200M × 1.5 × 1.0 = $300M
Actual = $10M (massive dump, dead project)
```

---

## Ключевые инсайты

1. **VCs matter** - Tier 1 investors = +50% FDV premium
2. **Market timing** - Bull market = 1.5x, Bear = 0.5x
3. **Low float = premium** - 5% unlock = higher initial price
4. **Sector matters** - L1/L2 >> SocialFi/GameFi
5. **Revenue/TVL** - Для DeFi лучше использовать метрики, не hype
