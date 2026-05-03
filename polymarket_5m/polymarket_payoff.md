<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# jika saya sedang membuat model machine learning untuk prediction di polymarket , bagaimana menentukan payoff saat saya membuat script live trade (dana percobaan / dummy) namun saya ingin membuat payoff real seperti eksekusi bet di pasar polymarket

Ya — untuk model machine learning + **paper trading/dummy live trade** di Polymarket, payoff yang “realistis” harus dimodelkan sebagai **mark-to-market dari order book**, lalu saat resolusi baru disettle menjadi **1 atau 0 per share**. Harga Anda bukan harga tampilan di UI, tetapi **harga eksekusi aktual** dari bid/ask, dan fee juga harus masuk ke simulasi.[^1][^2]

## Inti model

Polymarket memakai **Central Limit Order Book (CLOB)**, jadi harga terbentuk dari supply-demand, bukan ditentukan platform. Harga yang tampil di UI biasanya **midpoint bid-ask**; tetapi saat beli Anda membayar **ask**, dan saat jual Anda menerima **bid**.[^1]

Untuk script dummy yang meniru eksekusi nyata, payoff/live PnL sebaiknya dipisah jadi dua layer:

- **Unrealized PnL** saat market masih hidup: nilai posisi berdasar harga pasar sekarang, biasanya pakai bid untuk posisi long yang mau ditutup sekarang.[^1]
- **Realized/final payoff** saat exit atau resolusi: jika hold sampai resolve, share outcome benar ditebus 1, outcome salah jadi 0.[^1]


## Rumus payoff

Untuk satu posisi **BUY YES** sebanyak $q$ share di harga eksekusi $p_{in}$:

- **Biaya masuk** = $q \times p_{in} + fee_{in}$.[^3][^1]
- Jika Anda tutup sebelum resolve pada harga jual $p_{out}$, hasil keluar = $q \times p_{out} - fee_{out}$.[^4][^1]
- **Realized PnL** = hasil keluar − biaya masuk.[^1]

Jika Anda tahan sampai resolve:

- Jika outcome YES benar, **gross payoff** = $q \times 1$.[^1]
- Jika outcome YES salah, **gross payoff** = 0.[^1]
- **Final PnL** = gross payoff − biaya masuk.[^1]

Maka payoff per share untuk BUY YES adalah:

- Menang saat resolve: $1 - p_{in}$ sebelum fee.[^1]
- Kalah saat resolve: $-p_{in}$ sebelum fee.[^1]

Untuk **BUY NO**, logikanya simetris; yang penting outcome benar tetap settle ke 1 per share untuk token yang menang.[^1]

## Harga yang harus dipakai

Agar simulasi dummy terasa seperti Polymarket sungguhan, gunakan aturan ini:

- **Jangan entry di displayed price/midpoint**, karena itu hanya angka tampilan. Saat BUY, gunakan **best ask**; saat SELL, gunakan **best bid**.[^1]
- Jika size order Anda lebih besar dari kuantitas di level teratas, hitung **slippage** dengan menyapu beberapa level order book sampai size terpenuhi. Dokumentasi menekankan order besar bisa menggerakkan harga dan perlu cek depth.[^1]
- Kalau Anda membuat “market order” di simulator, implementasinya tetap sebagai limit agresif yang langsung makan liquidity di book, karena di Polymarket market order pada dasarnya adalah limit order yang dipasang cukup agresif untuk langsung terisi.[^1]

Praktiknya, untuk paper trading:

1. Ambil snapshot order book live.[^2][^1]
2. Hitung **VWAP eksekusi** untuk jumlah share yang ingin dibeli/jual.
3. Catat fee, timestamp, dan side.
4. Revalue posisi tiap tick memakai **harga likuidasi realistis**:
    - posisi YES long → mark ke **best bid YES** kalau ingin simulasi “kalau saya keluar sekarang”.[^1]
    - alternatif konservatif: mark ke bid/ask sesuai sisi posisi, bukan midpoint.[^1]

## Fee dalam simulator

Dokumentasi CLOB menyebut ada bagian fee, dan sumber-sumber fee 2026 menjelaskan bahwa:

- **Maker orders** umumnya fee 0 di platform global.[^3][^4]
- **Taker orders** kena fee dinamis menurut kategori market dan harga share saat eksekusi.[^5][^3]
- Ada juga kemungkinan **builder fee** tambahan bila order dirutekan lewat aplikasi dengan builder code; fee ini menumpuk di atas platform fee.[^6]

Untuk dummy engine, minimal modelkan:

- `fee_in = f(category, side, p_in, q, maker_or_taker)`
- `fee_out = f(category, side, p_out, q, maker_or_taker)`

Kalau Anda belum punya formula resmi lengkap per market dari endpoint saat itu, gunakan asumsi konservatif:

- maker = 0
- taker = persen kategori × notional efektif, atau pakai tabel kategori yang sedang berlaku.[^4][^3]

Ini akan jauh lebih realistis daripada mengabaikan fee, terutama untuk crypto short-duration markets yang fee takernya cenderung lebih tinggi.[^5][^3]

## Arsitektur simulasi

Struktur paper-trading engine yang bagus untuk ML:

- **Signal layer**: model mengeluarkan probability internal, misalnya `p_model = 0.63`.
- **Execution layer**: bandingkan `p_model` dengan harga pasar dan spread, lalu putuskan entry hanya jika ada edge setelah fee. Harga pasar Polymarket merepresentasikan probabilitas implisit.[^1]
- **Position layer**: simpan `side`, `qty`, `entry_vwap`, `entry_fee`, `market_id`, `token_id`, `opened_at`.
- **Marking layer**: update unrealized PnL dari order book live.
- **Settlement layer**: saat resolve, ganti mark value menjadi 1 atau 0 per share sesuai hasil market.[^1]

Aturan keputusan yang umum:

- BUY YES jika `p_model > ask_yes + total_cost_buffer`
- BUY NO jika `1 - p_model > ask_no + total_cost_buffer`

Buffer penting karena displayed probability bukan harga eksekusi aktual dan ada spread + fee.[^3][^1]

## Contoh logika Python

Berikut contoh sederhana untuk dummy payoff yang mendekati eksekusi nyata:

```python
from dataclasses import dataclass

@dataclass
class Position:
    side: str          # "YES" or "NO"
    qty: float
    entry_price: float
    entry_fee: float
    market_id: str

def realized_pnl_close(pos, exit_price, exit_fee):
    gross_in = pos.qty * pos.entry_price + pos.entry_fee
    gross_out = pos.qty * exit_price - exit_fee
    return gross_out - gross_in

def final_pnl_resolve(pos, outcome_yes: bool):
    won = (pos.side == "YES" and outcome_yes) or (pos.side == "NO" and not outcome_yes)
    gross_payoff = pos.qty * (1.0 if won else 0.0)
    gross_in = pos.qty * pos.entry_price + pos.entry_fee
    return gross_payoff - gross_in

def unrealized_pnl_mark_to_bid(pos, best_bid_same_outcome):
    mark_value = pos.qty * best_bid_same_outcome
    cost = pos.qty * pos.entry_price + pos.entry_fee
    return mark_value - cost
```

Interpretasinya:

- `entry_price` harus berasal dari **ask/VWAP ask** saat buy, bukan midpoint.[^1]
- `best_bid_same_outcome` dipakai untuk menilai “kalau keluar sekarang, saya benar-benar dapat berapa”.[^1]
- Jika model Anda ingin simulasi close agresif, tambahkan `exit_fee` dan slippage saat jual.[^3][^1]


## Hal yang sering salah

- Memakai **midpoint** sebagai harga fill masuk/keluar; ini membuat backtest terlalu optimistis.[^1]
- Mengabaikan **partial fills**; di CLOB, order bisa terisi sebagian di beberapa level.[^1]
- Menghitung payoff live sebagai $1 - entry$ sebelum resolve; itu salah, karena sebelum resolve nilai posisi Anda adalah **harga pasar sekarang**, bukan payout final.[^1]
- Tidak membedakan **mark-to-market** dan **settlement payoff**; untuk live trading ML, keduanya harus dipisah.[^1]


## Rekomendasi praktis

Untuk script dummy yang paling mendekati Polymarket nyata, gunakan definisi payoff berikut:

- **Live value posisi** = `qty × executable_exit_price`[^1]
- **Live PnL** = `live value − entry cost − estimated exit fee`[^3][^1]
- **Final payoff saat resolve** = `qty × {1 atau 0} − entry cost`[^1]

Jika Anda ingin, saya bisa lanjutkan dengan **template Python lengkap** untuk:

- ingest order book Polymarket,
- simulasi market/limit fill,
- hitung fee,
- update live PnL,
- dan settle final payoff seperti market asli.
<span style="display:none">[^10][^11][^12][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/Polymarket/polymarket-us-python

[^2]: https://oddspapi.io/blog/polymarket-api-kalshi-api-vs-sportsbooks-the-developers-guide/

[^3]: https://www.predictionhunt.com/blog/polymarket-fees-complete-guide

[^4]: https://www.kucoin.com/blog/in-polymarket-fees-trading-guide-2026

[^5]: https://www.kucoin.com/blog/polymarket-fees-trading-guide-2026

[^6]: https://docs.polymarket.com/builders/fees

[^7]: https://www.polymarketarbitragebot.net/fil/guides/polymarket-fees-explained

[^8]: https://deadspin.com/prediction-markets/polymarket/fees/

[^9]: https://docs.polymarket.com/concepts/prices-orderbook

[^10]: https://docs.polymarket.com/developers/CLOB/introduction

[^11]: https://www.pokernews.com/prediction-markets/news/2026/04/polymarket-blunder-prompts-quick-u-turn-new-polymarket-fees-50947.htm

[^12]: https://www.polymarketexchange.com/fees-hours.html

