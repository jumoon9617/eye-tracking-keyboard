# アイトラッキングでタイピング効果測定

## 概要
タイピング中の視線計測を行い、タッチタイピングの習熟度などを測る

## 目標
- [x] 環境構築
- [x] サンプルコードを実行してみる
- [x] 改造して視線を数値かしてを入手できるようにする

## 参考URL
* [DepthaiV3のXLinkOut対応] (https://discuss.luxonis.com/d/6204-depthai-v3-new-practice-for-xlinkout)
* [oak-dサンプル] (https://github.com/luxonis/oak-examples/tree/main/neural-networks)

## 動作方法
* python(3.10のもの)を構築
* requirements.txtでパッケージインストール
* python3 main.pyで実行
* [localhost:8082] (http://localhost:8082/)　にアクセスすると視線を可視化した状態で画面に写せる
* 必要に応じてコンソールの方で出力されている数値を確認(数値は縦横0~255までの範囲でどこを見ているか測定)

## 更新内容
### 2025/7/25
* プロジェクト開始
* 環境構築

### 2025/7/29
* サンプルコードを少し改変して動かしてみた

### 2025/8/3
* カメラから視線のデータ取得をコンソール出力するようにした