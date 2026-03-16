# Data Cleaning Report — posting_2026

_Generated: 2026-03-13 13:49:41_

Parsed numeric/typed columns created as DuckDB views (`<table>_clean`).


---
## Validation Results


### 招聘_clean — salary_type distribution (sample 10M)
```
ERROR: Parser Error: syntax error at or near "GROUP"

LINE 5:         GROUP BY 1 ORDER BY 2 DESC
            ^
```

### 招聘_clean — salary_lo sanity (range rows: lo < hi)
```
  4292685  4276003  16682
```
_(456.4s)_

### 工商信息_clean — reg_capital_wan null rate
```
  5000000  2142790  42.86
```
_(187.1s)_

### 工商信息_clean — currency distribution
```
ERROR: Parser Error: syntax error at or near "GROUP"

LINE 6:         GROUP BY 1 ORDER BY 2 DESC
        ^
```

### 年报_clean — insured_n: parsed vs null (sample)
```
  5000000  2335484  46.71  0  2000000
```
_(190.1s)_

### 年报_clean — total_assets_wan: parsed vs null (sample)
```
  5000000  719309  14.39
```
_(212.3s)_

### 年报_clean — net_profit_wan: negative check
```
  186198  599356  -1466860000.0  4175275386.86
```
_(196.7s)_

### 股东信息_clean — share_pct range check
```
  5000000  5000000  0
```
_(117.3s)_

### 股东信息_clean — subscribed_currency distribution
```
ERROR: Parser Error: syntax error at or near "GROUP"

LINE 6:         GROUP BY 1 ORDER BY 2 DESC
        ^
```

### 股权出质_clean — pledge_unit distribution
```
  万元  823490
  万股  223221
  元  165007
  None  55365
```
_(0.1s)_

---
## Clean Views Created

- `招聘_clean`
- `工商信息_clean`
- `年报社保财报信息_clean`
- `股东信息_clean`
- `股权出质_clean`


## New Columns Summary

| View | New Columns |
|------|-------------|
| 招聘_clean | salary_lo (元/月), salary_hi (元/月), salary_type |
| 工商信息_clean | reg_capital_wan (万元), reg_capital_currency |
| 年报社保财报信息_clean | insured_n (人), total_assets_wan (万元), net_profit_wan (万元), total_tax_wan (万元) |
| 股东信息_clean | share_pct (%), subscribed_wan (万元), subscribed_currency, paid_wan (万元), paid_currency |
| 股权出质_clean | pledge_amount, pledge_unit (元/万元/万股) |