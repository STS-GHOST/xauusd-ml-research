//+------------------------------------------------------------------+
//| XAU_CatBoost_EA.mq5 — CatBoost direction model for XAUUSD H1      |
//|                                                                  |
//| ████ RESEARCH ARTIFACT — NO EDGE WAS FOUND. DO NOT TRADE. ████    |
//|                                                                  |
//| This EA is published as the executable half of a negative        |
//| research result. Walk-forward out-of-sample evaluation found no  |
//| exploitable edge. It is kept here because the *process* — and    |
//| the errors corrected along the way — is the point, not the       |
//| strategy. See README.md and RESEARCH_LOG.md.                     |
//|                                                                  |
//| Headline (2026-08-21, purge/embargo walk-forward, 2021-01〜2026-07):|
//|   partial IC = -0.0002,  95% CI [-0.0195, +0.0193],  n = 32,947  |
//|   Required for viability: IC >= 0.029 — outside the CI.          |
//|   Signal upper bound  +0.059 ATR/trade                           |
//|   Friction floor      +0.053 ATR/trade (spread + stop-loss)      |
//|   => break-even at best. Not tradeable.                          |
//|                                                                  |
//| ████ 検証の結果エッジは確認できませんでした。実運用しないでください ████ |
//|                                                                  |
//| 本EAは「否定的な研究結果」の実装側として公開しています。            |
//| ウォークフォワードOOSで運用に足るエッジは確認できませんでした。      |
//| 残しているのは戦略ではなく、途中で見つけて直した誤りを含む「過程」    |
//| のためです。詳細は README.ja.md / RESEARCH_LOG.ja.md を参照。      |
//|                                                                  |
//| NOT INVESTMENT ADVICE. Trading leveraged FX/CFD can lose more    |
//| than the deposited amount. Use at your own risk.                 |
//| 投資助言ではありません。レバレッジ取引は預託額を超える損失の        |
//| 可能性があります。利用は自己責任で。                               |
//+------------------------------------------------------------------+
//| 仕様(2026-08-18 のベースライン比較で改訂):                        |
//|  シグナル : MLModel.mqh(埋め込みCatBoost、Python側と一致検証済み) |
//|  ゲート   : conf >= 0.60 かつ atr_ratio >= 1.10                  |
//|  方向     : BUY のみ。SELL判定は発注せず見送る                    |
//|  エントリー: バー確定直後に成行(検証の「次バー始値」に対応)        |
//|  決済     : 6時間タイムアウト + 3.0 ATR 安全SL                    |
//|  ロット   : 固定・入力パラメータ                                  |
//|  拒否権   : スプレッドが上限超なら見送り(既定200pt=2pips)        |
//|                                                                  |
//| v1.10 での変更点と根拠(model_vs_baseline_test.py):                |
//|  同一バー集合での分解により、方向判定は有害と判明した。            |
//|    M(選択+方向) EV=+0.1043 / B(同一バー・常にBUY) EV=+0.1583      |
//|    方向判定の寄与 M-B = -0.0540 ATR (t=-1.84, SELL 48/881件)      |
//|    → SELLを廃止するとEVが52%改善する。O2(半減)では不十分。        |
//|  一方、場面選択には価値がある。                                    |
//|    B - N(ゲート全バーでBUY) = +0.0999 ATR, 経験的p=0.0150         |
//|    → モデルは「今買うのが良い場面か」を採点する関数であり、        |
//|       低スコアは売り時を意味しない。                              |
//|  SELLを出さないためO1(hour=1のSELL見送り)は不要になり削除した。   |
//|                                                                  |
//| 期待値(全バー評価・先読みなし): +0.1583 ATR/トレード              |
//|  ※引継書の +0.2933 は中立帯を評価から除いた先読み込みの値だった。  |
//+------------------------------------------------------------------+
//| ===== v1.11 (2026-08-21) 上記 v1.10 の根拠はすべて撤回された =====  |
//|                                                                  |
//| v1.10 の数値を出したモデルは 2023-01-03 以降のデータで学習され、   |
//| ホールドアウトを取っていなかった。評価区間 2023-26 は全て          |
//| in-sample であり、+0.1583 ATR も +0.0999 ATR も OOS の値ではない。 |
//|                                                                  |
//| purge/embargo 付きウォークフォワードで全期間を OOS 化して再評価:   |
//|   ドリフト条件付き偏IC  全期間       -0.0002 [-0.0195, +0.0193]   |
//|                         2021-22横ばい -0.0147 [-0.0516, +0.0138]  |
//|                         2023-26上昇   +0.0099 [-0.0139, +0.0344]  |
//|   → 必要水準 IC 0.029 は CI の外。方向性エッジは検出されない。     |
//|                                                                  |
//| v1.10 の個別の結論についても:                                     |
//|  ・「場面選択に価値がある(+0.0999)」→ in-sample。OOS では消えた。  |
//|  ・「SELL 廃止で EV +52%」→ 同じ評価系の上の比較なので再検証要。   |
//|    ただし BUY 専用という設計自体は現状維持(SELL を足す根拠もない)。|
//|  ・エントリー規約(バー t 確定 → 次バー始値)のみ検証で追認された。  |
//|                                                                  |
//| InpAtrRatioMin の既定値を 1.10 → 0.0(無効)に変更:                |
//|   3水準 × 2レジームの段階分解で、絞るほど最終 EV が悪化し、        |
//|   向きが両レジームで一致した。支持する証拠がなく標本を 54% 捨てる。|
//|     フィルタなし  2020 -0.026 / 2021-22 -0.001                    |
//|     >= 0.90       2020 -0.019 / 2021-22 -0.003                    |
//|     >= 1.10       2020 -0.094 / 2021-22 -0.038                    |
//|   統計的勝利ではなく設計判断。個別セルはいずれも判定不能。         |
//|                                                                  |
//| 実測コスト: 中央値 0.0156〜0.0245 ATR/トレード。                   |
//|   研究コードが想定していた COST_ATR=0.0287 は約 1.3 倍過大だった。 |
//+------------------------------------------------------------------+
//| 全決定を MQL5/Files/xau_ml_decisions.csv に記録する。             |
//+------------------------------------------------------------------+
#property strict
#property version   "1.11"

#include <Trade/Trade.mqh>
#include <MLFeatures.mqh>
#include <MLModel.mqh>

//--- 入力パラメータ
input double InpLot             = 0.01;      // ロット(BUY専用)
input double InpConfMin         = 0.60;      // 信頼度ゲート
input double InpAtrRatioMin     = 0.00;      // ボラレジームゲート(v1.11で無効化。旧既定1.10)
input int    InpMaxSpreadPoints = 200;       // スプレッド上限(pt) O3
input int    InpHoldHours       = 6;         // タイムアウト(時間)
input double InpSL_ATR          = 3.0;       // 安全SL(ATR倍)
input int    InpHistoryBars     = 1500;      // 特徴量計算に使う履歴本数
input long   InpMagic           = 20260811;  // マジックナンバー
input string InpLogFile         = "xau_ml_decisions.csv";
input string InpSymbolMustContain = "XAUUSD";  // 銘柄の安全確認(空で無効)
input bool   InpShowPanel       = true;      // チャートに状態パネルを表示
input int    InpPanelX          = 12;        // パネル位置X
input int    InpPanelY          = 24;        // パネル位置Y
input int    InpPanelFontSize   = 9;         // フォントサイズ

CTrade  trade;
int     atr_handle = INVALID_HANDLE;
datetime last_bar  = 0;

//--- パネル表示用の最新判定値(毎時更新)
double   g_p1 = 0, g_conf = 0, g_atr_ratio = 0;
int      g_dir = 0;
datetime g_sig_bar = 0;
string   g_last_action = "(起動直後)";
int      g_n_decisions = 0, g_n_trades = 0, g_n_skip_sell = 0, g_n_skip_spr = 0;
long     g_spr_max = 0;

#define PANEL_PREFIX "MLPANEL_"

//+------------------------------------------------------------------+
int OnInit()
{
   if(StringLen(InpSymbolMustContain) > 0
      && StringFind(_Symbol, InpSymbolMustContain) < 0)
   {
      PrintFormat("★中止: このEAは %s 専用です(現在のチャート: %s)。"
                  " 別銘柄で使うと学習していない分布に適用され危険です。",
                  InpSymbolMustContain, _Symbol);
      return INIT_FAILED;
   }
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);
   atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(atr_handle == INVALID_HANDLE)
   {
      Print("iATR作成失敗");
      return INIT_FAILED;
   }
   Print("=========================================================");
   Print("RESEARCH ARTIFACT — no out-of-sample edge was found.");
   Print("  walk-forward partial IC = -0.0002, 95% CI [-0.0195,+0.0193]");
   Print("  Not intended for live trading. See README.md.");
   Print("検証の結果エッジは確認できていません。実運用は想定していません。");
   Print("=========================================================");
   PrintFormat("起動 v1.11 BUY専用: %s model=%d本 lot=%.2f conf>=%.2f ar>=%.2f "
               "spread<=%dpt hold=%dh sl=%.1fATR",
               _Symbol, MLM_NTREES, InpLot, InpConfMin, InpAtrRatioMin,
               InpMaxSpreadPoints, InpHoldHours, InpSL_ATR);
   PanelUpdate();
   EventSetTimer(2);          // ポジションなし時もスプレッド表示を更新
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   PanelClear();
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
}

//+------------------------------------------------------------------+
//| ティックが来ない時間帯でもパネルを更新する                          |
//+------------------------------------------------------------------+
void OnTimer()
{
   PanelUpdate();
}

//+------------------------------------------------------------------+
//| 自EAのポジションを探す(なければ-1)                                 |
//+------------------------------------------------------------------+
int FindMyPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol
         && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return (int)ticket;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| パネル: 1行を描画(なければ作成)                                    |
//+------------------------------------------------------------------+
void PanelLine(const int idx, const string text, const color clr)
{
   string name = PANEL_PREFIX + IntegerToString(idx);
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, InpPanelX);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE,
                       InpPanelY + idx * (InpPanelFontSize + 7));
      ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, InpPanelFontSize);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
   }
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
}

//+------------------------------------------------------------------+
//| パネル: 全消去                                                     |
//+------------------------------------------------------------------+
void PanelClear()
{
   ObjectsDeleteAll(0, PANEL_PREFIX);
}

//+------------------------------------------------------------------+
//| パネル: 更新                                                       |
//|  判定値(conf/atr_ratio)は毎時更新、スプレッドと経過時間はティック  |
//+------------------------------------------------------------------+
void PanelUpdate()
{
   if(!InpShowPanel) return;

   const color C_OK   = clrLimeGreen;
   const color C_NG   = clrTomato;
   const color C_INFO = clrSilver;
   const color C_HEAD = clrGold;

   long spr = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spr > g_spr_max) g_spr_max = spr;

   bool ok_conf = (g_conf >= InpConfMin);
   bool ok_ar   = (g_atr_ratio >= InpAtrRatioMin);
   bool ok_spr  = (spr <= InpMaxSpreadPoints);

   int i = 0;
   PanelLine(i++, "XAU CatBoost EA v1.11 BUY専用  [model " +
                  IntegerToString(MLM_NTREES) + "本]", C_HEAD);
   PanelLine(i++, " ※研究用 / OOSでエッジ未確認 (IC -0.0002)", C_NG);

   if(g_sig_bar == 0)
      PanelLine(i++, " 判定待ち(次のH1バー確定時に更新)", C_INFO);
   else
      PanelLine(i++, " 直近判定: " +
                TimeToString(g_sig_bar, TIME_DATE | TIME_MINUTES) + "  p1=" +
                DoubleToString(g_p1, 4) + "  " +
                (g_dir > 0 ? "BUY" : "SELL(見送り)"), C_INFO);

   PanelLine(i++, StringFormat(" conf      %.4f / %.2f   %s",
                  g_conf, InpConfMin, ok_conf ? "OK" : "-- 不足"),
             ok_conf ? C_OK : C_NG);
   PanelLine(i++, StringFormat(" atr_ratio %.4f / %.2f   %s",
                  g_atr_ratio, InpAtrRatioMin, ok_ar ? "OK" : "-- 不足"),
             ok_ar ? C_OK : C_NG);
   PanelLine(i++, StringFormat(" spread    %d pt / %d      %s (最大 %d)",
                  (int)spr, InpMaxSpreadPoints, ok_spr ? "OK" : "-- 超過",
                  (int)g_spr_max),
             ok_spr ? C_OK : C_NG);

   //--- ポジション状態(ティックごとに経過時間を更新)
   int ticket = FindMyPosition();
   if(ticket >= 0 && PositionSelectByTicket((ulong)ticket))
   {
      long   ptype = PositionGetInteger(POSITION_TYPE);
      double vol   = PositionGetDouble(POSITION_VOLUME);
      double prof  = PositionGetDouble(POSITION_PROFIT);
      datetime pot = (datetime)PositionGetInteger(POSITION_TIME);
      datetime pot_bar = pot - (pot % 3600);
      double left = (double)(pot_bar + InpHoldHours * 3600 - TimeCurrent()) / 3600.0;
      if(left < 0) left = 0;
      PanelLine(i++, StringFormat(" 保有 %s %.2f lot  損益 %.2f  残り %.1fh",
                     ptype == POSITION_TYPE_BUY ? "BUY " : "SELL", vol, prof, left),
                prof >= 0 ? C_OK : C_NG);
   }
   else
   {
      PanelLine(i++, " ポジションなし", C_INFO);
   }

   PanelLine(i++, StringFormat(" 判定%d回  発注%d  SELL見送り%d  スプ見送り%d",
                  g_n_decisions, g_n_trades, g_n_skip_sell, g_n_skip_spr), C_INFO);
   PanelLine(i++, " 直近: " + g_last_action, C_INFO);
   ChartRedraw(0);
}

//+------------------------------------------------------------------+
//| 決定ログをCSVへ追記                                                |
//+------------------------------------------------------------------+
void LogDecision(const datetime bar_time, const double p1, const double conf,
                 const int dir, const double atr_ratio, const long spread,
                 const string action, const double lot)
{
   int h = FileOpen(InpLogFile, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   if(FileSize(h) == 0)
      FileWriteString(h, "decision_time,signal_bar,p1,conf,dir,atr_ratio,"
                         "spread_pt,action,lot\n");
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + ","
                   + TimeToString(bar_time, TIME_DATE | TIME_MINUTES) + ","
                   + DoubleToString(p1, 6) + "," + DoubleToString(conf, 6) + ","
                   + IntegerToString(dir) + "," + DoubleToString(atr_ratio, 4) + ","
                   + IntegerToString(spread) + "," + action + ","
                   + DoubleToString(lot, 2) + "\n");
   FileClose(h);
}

//+------------------------------------------------------------------+
//| タイムアウト決済: 建玉バーの開始から InpHoldHours 経過で成行決済    |
//+------------------------------------------------------------------+
void ManageTimeout(const datetime bar_time)
{
   int ticket = FindMyPosition();
   if(ticket < 0) return;
   if(!PositionSelectByTicket((ulong)ticket)) return;
   datetime pot = (datetime)PositionGetInteger(POSITION_TIME);
   datetime pot_bar = pot - (pot % 3600);            // 建玉の属するH1バー開始
   if(bar_time >= pot_bar + InpHoldHours * 3600)
   {
      if(trade.PositionClose((ulong)ticket))
         PrintFormat("タイムアウト決済 ticket=%d (%d時間経過)", ticket, InpHoldHours);
      else
         PrintFormat("決済失敗 ticket=%d err=%d", ticket, GetLastError());
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   //--- ティックごと: スプレッド・ポジション経過時間の表示を更新
   PanelUpdate();

   //--- 新しいH1バーの検出
   datetime bar_time = iTime(_Symbol, PERIOD_H1, 0);
   if(bar_time == last_bar) return;
   last_bar = bar_time;

   //--- 先にタイムアウト管理(同一バーでの決済→新規は許可)
   ManageTimeout(bar_time);

   //--- 特徴量(直近確定バー)
   double vec[];
   if(!MLF_LastClosedVector(_Symbol, PERIOD_H1, InpHistoryBars, vec))
   {
      Print("特徴量計算失敗(履歴不足の可能性)");
      return;
   }
   int dir = 0;
   double p1 = MLM_PredictP1(vec);
   double conf = MLM_Confidence(vec, dir);
   double atr_ratio = vec[6];
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   datetime sig_bar = iTime(_Symbol, PERIOD_H1, 1);

   //--- パネル用に最新判定を保存
   g_p1 = p1; g_conf = conf; g_dir = dir;
   g_atr_ratio = atr_ratio; g_sig_bar = sig_bar;
   g_n_decisions++;

   //--- ゲート
   if(conf < InpConfMin || atr_ratio < InpAtrRatioMin)
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "SKIP_GATE", 0);
      g_last_action = (conf < InpConfMin) ? "ゲート落ち(conf不足)"
                                          : "ゲート落ち(atr_ratio不足)";
      PanelUpdate();
      return;
   }
   //--- 方向: SELLは発注しない(2026-08-18 の分解で方向判定は有害と判明)
   //    モデルは「今買うのが良い場面か」の採点器であり、
   //    低スコアは売り時を意味しない。SELL判定は単に見送る。
   if(dir < 0)
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "SKIP_SELL", 0);
      g_n_skip_sell++;
      g_last_action = StringFormat("SELL判定のため見送り (p1=%.3f)", p1);
      PanelUpdate();
      return;
   }
   //--- O3: スプレッドガード
   if(spread > InpMaxSpreadPoints)
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "SKIP_SPREAD", 0);
      PrintFormat("O3発動: spread=%dpt > %dpt で見送り", spread, InpMaxSpreadPoints);
      g_n_skip_spr++;
      g_last_action = "O3発動: スプレッド" + IntegerToString((int)spread) + "pt で見送り";
      PanelUpdate();
      return;
   }
   //--- 既存ポジションがあれば新規は見送り(重複建玉なし)
   if(FindMyPosition() >= 0)
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "SKIP_POSITION", 0);
      g_last_action = "既存ポジションあり(新規見送り)";
      PanelUpdate();
      return;
   }

   //--- SL距離: Wilder ATR(iATR、種子差はSL用途では無視できる)
   double atr_buf[1];
   if(CopyBuffer(atr_handle, 0, 1, 1, atr_buf) != 1)
   {
      Print("ATR取得失敗");
      return;
   }
   double atr = atr_buf[0];
   double sl_dist = InpSL_ATR * atr;

   //--- 発注(BUYのみ)
   double lot = InpLot;
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool ok = trade.Buy(lot, _Symbol, 0.0,
                       NormalizeDouble(ask - sl_dist, _Digits), 0.0,
                       "ML v1.11 BUY");
   if(ok)
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "TRADE", lot);
      PrintFormat("発注: BUY lot=%.2f conf=%.3f ar=%.2f spread=%dpt sl=%.2f",
                  lot, conf, atr_ratio, spread, sl_dist);
      g_n_trades++;
      g_last_action = StringFormat("発注 BUY %.2f lot (conf=%.3f)", lot, conf);
   }
   else
   {
      LogDecision(sig_bar, p1, conf, dir, atr_ratio, spread, "ORDER_FAIL", lot);
      PrintFormat("発注失敗: err=%d retcode=%d", GetLastError(),
                  trade.ResultRetcode());
      g_last_action = "発注失敗 retcode=" + IntegerToString(trade.ResultRetcode());
   }
   PanelUpdate();
}