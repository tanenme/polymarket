<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# berikan library untuk menemukan market btc up / down 15 menit di polymarket

Untuk menemukan market **“BTC Up / Down 15 menit”** lewat Python, yang Anda butuhkan bukan library khusus yang sangat sempit, tapi kombinasi:

1. **Polymarket HTTP API (gamma-api)** untuk fetch market by slug / filter, dan
2. Library/helper Python yang membungkus pola slug deterministik untuk BTC 5m/15m.

Ada beberapa opsi praktis yang bisa Anda pakai.

***

## 1. Polymarket API resmi (gamma-api)

Dokumentasi Polymarket menjelaskan cara **fetch markets by slug** via `gamma-api.polymarket.com`.[^1]

- Endpoint umum:
    - `GET https://gamma-api.polymarket.com/markets?slug=...`
    - atau `GET https://gamma-api.polymarket.com/markets/slug/...`[^1]
- Dengan pola slug deterministik BTC up/down (5m/15m), Anda bisa langsung memanggil event yang tepat.

Untuk BTC up/down, komunitas menemukan pola slug yang sifatnya **deterministik dengan timestamp**. Contoh untuk 5 menit:[^2]

```python
window_ts = now - (now % 300)  # start window 5 menit
slug = f"btc-updown-5m-{window_ts}"
```

Lalu panggil:

```python
import requests

resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"slug": slug},
)
m = resp.json()
```

Untuk **15 menit**, polanya mirip tapi dengan interval 900 detik:

```python
import time
import requests

now = int(time.time())
window_ts = now - (now % 900)  # awal window 15 menit
slug = f"btc-updown-15m-{window_ts}"  # penting: '15m'

resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"slug": slug},
)
data = resp.json()
print(data)
```

Slug `btc-updown-15m-...` sesuai contoh URL market resmi Polymarket seperti:[^3]

- `https://polymarket.com/id/event/btc-updown-15m-1769259600`

Jadi, “library” dasarnya di sini adalah **requests** + pola slug + endpoint `markets?slug=...`.[^1]

***

## 2. Utility Python open-source: *Polymarket-Market-Finder*

Ada repo GitHub yang benar-benar dibuat untuk masalah Anda:

- `handiko/Polymarket-Market-Finder` → “A Python utility to deterministically discover active Polymarket Bitcoin (BTC) 5-minute and 15-minute 'Up/Down' prediction markets in real-time.”[^4]

Fitur inti yang biasanya ada:

- Fungsi untuk **menghitung slug aktif** BTC up/down 5m dan 15m berdasar waktu sekarang (atau window tertentu).[^4]
- Fungsi untuk memanggil **gamma-api** dan mengembalikan market ID, token YES/NO, dan harga.[^4]

Ini contoh gaya pakaiannya (sketsa, menyesuaikan repo aslinya):

```python
from polymarket_market_finder import get_btc_updown_15m_market

market = get_btc_updown_15m_market()

print(market["slug"])
print(market["id"])
print(market["yes_token_id"], market["no_token_id"])
print(market["outcomePrices"])
```

Kelebihan pendekatan ini:

- Anda tidak perlu memikirkan slug dan window sendiri.
- Library sudah fokus pada BTC 5m/15m Up/Down markets, sesuai kebutuhan Anda.[^4]

***

## 3. Alternatif: filter lewat tag “crypto” + judul

Kalau Anda tidak mau mengandalkan pola slug, bisa juga pakai strategi **fetching markets** dari dokumentasi:[^1]

- `GET https://gamma-api.polymarket.com/markets?limit=...&active=true&tag_id=...`
- Lalu filter di Python:
    - `question` atau `title` berisi `"Bitcoin Up or Down"` dan `"15m"`
    - `active == True`
    - `closed == False`

Contoh pseudo-code:

```python
import requests

def find_btc_updown_15m_markets():
    resp = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "limit": 100, "order": "start_date"},
    )
    markets = resp.json()
    results = []
    for m in markets:
        title = (m.get("question") or m.get("title") or "").lower()
        if "bitcoin up or down" in title and "15m" in title:
            results.append(m)
    return results
```

Ini lebih “brute force”, tapi tetap memakai API resmi dengan pattern “fetching markets” yang didokumentasikan.[^1]

***

## 4. Penyedia pihak ketiga

Jika fokus Anda hanya pada **data BTC up/down 15m** (bukan mapping langsung ke Polymarket API), ada API pihak ketiga seperti:

- SynthData: endpoint `GET /insights/polymarket/up-down/15min?asset=BTC`.[^5]
- PredictAPI: endpoint agregat untuk market Polymarket, termasuk BTC up/down horizon tertentu.[^6]

Ini berguna jika Anda ingin **fitur data** untuk ML tanpa harus mengurusi slug, token ID, dll, tapi untuk **eksekusi real trade** tetap sebaiknya lewat API resmi Polymarket.[^5][^6]

***

## Rekomendasi praktis

Untuk kebutuhan Anda (“menemukan market BTC up/down 15 menit di Polymarket” dalam script Python):

- Jika mau **langsung siap pakai** dan khusus BTC up/down:
    - Gunakan repo `Polymarket-Market-Finder` dari GitHub.[^4]
- Jika mau **kendali penuh** dan mudah di-porting:
    - Implementasikan sendiri pola slug `btc-updown-15m-{window_ts}`
    - Panggil `https://gamma-api.polymarket.com/markets?slug=...` dengan `requests`.[^2][^3][^1]

Kalau Anda mau, saya bisa tuliskan **fungsi Python lengkap** `get_current_btc_updown_15m()` yang:

1. Hitung window 15 menit aktif sekarang,
2. Build slug,
3. Hit API Polymarket,
4. Return market ID + token YES/NO + harga, siap dipakai di bot trading.
<span style="display:none">[^10][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://docs.polymarket.com/market-data/fetching-markets

[^2]: https://gist.github.com/Archetapp/7680adabc48f812a561ca79d73cbac69

[^3]: https://polymarket.com/id/event/btc-updown-15m-1769259600

[^4]: https://github.com/handiko/Polymarket-Market-Finder

[^5]: https://docs.synthdata.co/insights/polymarket

[^6]: https://www.predictapi.dev

[^7]: https://www.reddit.com/r/PredictionsMarkets/comments/1q9uwxr/how_to_fetch_15minute_crypto_markets_via/nyyi4n8/

[^8]: https://www.ainvest.com/news/polymarket-launches-5-minute-btc-prediction-event-bearish-sentiment-grows-2602/

[^9]: https://robottraders.io/blog/polymarket-trading-bot-python

[^10]: https://www.cointech2u.com/polymarket-has-launched-the-5-minute-btc-up-down-prediction-event/

