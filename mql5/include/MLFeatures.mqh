//+------------------------------------------------------------------+
//| MLFeatures.mqh — CatBoost EA用 9特徴量計算ライブラリ              |
//|                                                                  |
//| Python側パイプライン(conditional_direction_diagnostic.py)と      |
//| 厳密一致することをサンドボックスの鏡実装で検証済み(2026-08-08)。  |
//| 変更する場合は必ず verify_mql5_features.py で再検証すること。     |
//|                                                                  |
//| 特徴量の順序(モデル入力順・変更禁止):                             |
//|   [0] ret_1  [1] ret_3  [2] ret_6  [3] dist_ma50  [4] dist_ma200 |
//|   [5] rsi    [6] atr_ratio        [7] hour        [8] dayofweek  |
//|                                                                  |
//| 重要な規約:                                                       |
//|  - 配列は昇順(index 0 = 最古)。CopyRates後に                     |
//|    ArraySetAsSeries(rates, false) を必ず呼ぶこと。               |
//|  - dayofweek は Python規約(月曜=0)。MQL5のday_of_week(日曜=0)   |
//|    から (dow+6)%7 で変換している。ここを触ると全予測が壊れる。    |
//|  - 未定義値は MLF_NAN。ウォームアップに最低320本、               |
//|    ATR/RSIの種子減衰を考えると1000本以上の履歴で使うこと。        |
//+------------------------------------------------------------------+
#ifndef __MLFEATURES_MQH__
#define __MLFEATURES_MQH__

#define MLF_NAN (EMPTY_VALUE)
#define MLF_NFEAT 9

bool MLF_IsValid(const double v) { return (v != MLF_NAN); }

//+------------------------------------------------------------------+
//| 全バーの特徴量を昇順配列で計算する                                 |
//| rates: 昇順のMqlRates配列 / n: 本数                               |
//| feat[n][MLF_NFEAT] に出力(未定義は MLF_NAN)                      |
//+------------------------------------------------------------------+
bool MLF_ComputeAll(const MqlRates &rates[], const int n,
                    double &feat[][MLF_NFEAT])
{
   const int P = 14;          // ATR/RSI期間
   const int W50 = 50, W200 = 200, W100 = 100;
   if(n < W200 + 20)
      return false;
   if(ArrayRange(feat, 0) != n)
      ArrayResize(feat, n);

   //--- True Range と Wilder ATR(Python側と同一の種付け:
   //    atr[13] = mean(tr[0..13]), 以降 (prev*13 + tr)/14)
   double tr[];  ArrayResize(tr, n);
   tr[0] = rates[0].high - rates[0].low;
   for(int i = 1; i < n; i++)
   {
      double pc = rates[i-1].close;
      double a = rates[i].high - rates[i].low;
      double b = MathAbs(rates[i].high - pc);
      double c = MathAbs(rates[i].low  - pc);
      tr[i] = MathMax(a, MathMax(b, c));
   }
   double atr[]; ArrayResize(atr, n); ArrayInitialize(atr, MLF_NAN);
   {
      double s = 0.0;
      for(int i = 0; i < P; i++) s += tr[i];
      atr[P-1] = s / P;
      for(int i = P; i < n; i++)
         atr[i] = (atr[i-1] * (P-1) + tr[i]) / P;
   }

   //--- Wilder RSI(Python側と同一: ag[14]=mean(gain[1..14]), 以降再帰)
   double rsi[]; ArrayResize(rsi, n); ArrayInitialize(rsi, MLF_NAN);
   {
      double sg = 0.0, sl = 0.0;
      for(int i = 1; i <= P; i++)
      {
         double d = rates[i].close - rates[i-1].close;
         sg += MathMax(d, 0.0);
         sl += MathMax(-d, 0.0);
      }
      double ag = sg / P, al = sl / P;
      rsi[P] = (al == 0.0) ? ((ag > 0.0) ? 100.0 : 50.0)
                           : 100.0 - 100.0 / (1.0 + ag / al);
      for(int i = P + 1; i < n; i++)
      {
         double d = rates[i].close - rates[i-1].close;
         ag = (ag * (P-1) + MathMax(d, 0.0)) / P;
         al = (al * (P-1) + MathMax(-d, 0.0)) / P;
         rsi[i] = (al == 0.0) ? ((ag > 0.0) ? 100.0 : 50.0)
                              : 100.0 - 100.0 / (1.0 + ag / al);
      }
   }

   //--- 移動平均は走査和で O(n)
   double sum50 = 0.0, sum200 = 0.0, sumatr = 0.0;
   int natr = 0;               // 走査和に入っているATRの本数
   for(int i = 0; i < n; i++)
   {
      sum50  += rates[i].close;
      sum200 += rates[i].close;
      if(i >= W50)  sum50  -= rates[i-W50].close;
      if(i >= W200) sum200 -= rates[i-W200].close;
      if(MLF_IsValid(atr[i])) { sumatr += atr[i]; natr++; }
      if(i >= W100 && MLF_IsValid(atr[i-W100])) { sumatr -= atr[i-W100]; natr--; }

      MqlDateTime st;
      TimeToStruct(rates[i].time, st);

      double a = atr[i];
      bool aok = MLF_IsValid(a) && a > 0.0;
      double c0 = rates[i].close;

      feat[i][0] = (aok && i >= 1) ? (c0 - rates[i-1].close) / a : MLF_NAN;
      feat[i][1] = (aok && i >= 3) ? (c0 - rates[i-3].close) / a : MLF_NAN;
      feat[i][2] = (aok && i >= 6) ? (c0 - rates[i-6].close) / a : MLF_NAN;
      feat[i][3] = (aok && i >= W50-1)  ? (c0 - sum50  / W50)  / a : MLF_NAN;
      feat[i][4] = (aok && i >= W200-1) ? (c0 - sum200 / W200) / a : MLF_NAN;
      feat[i][5] = rsi[i];
      feat[i][6] = (aok && natr == W100) ? a / (sumatr / W100) : MLF_NAN;
      feat[i][7] = (double)st.hour;                      // サーバー時刻
      feat[i][8] = (double)((st.day_of_week + 6) % 7);   // 月曜=0 へ変換
   }
   return true;
}

//+------------------------------------------------------------------+
//| 直近の確定バーの特徴量ベクトルを取得する(EA用ヘルパ)              |
//| 戻り値: 全要素が定義済みなら true                                 |
//+------------------------------------------------------------------+
bool MLF_LastClosedVector(const string symbol, const ENUM_TIMEFRAMES tf,
                          const int history, double &vec[])
{
   MqlRates rates[];
   int got = CopyRates(symbol, tf, 1, history, rates);   // shift1 = 確定バー
   if(got < 320) return false;
   ArraySetAsSeries(rates, false);                       // 昇順を保証
   static double feat[][MLF_NFEAT];
   if(!MLF_ComputeAll(rates, got, feat)) return false;
   ArrayResize(vec, MLF_NFEAT);
   for(int k = 0; k < MLF_NFEAT; k++)
   {
      vec[k] = feat[got-1][k];
      if(!MLF_IsValid(vec[k])) return false;
   }
   return true;
}

#endif // __MLFEATURES_MQH__
