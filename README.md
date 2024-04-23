# 22.B.Udou
B22 有働のSAGTCNNのリポジトリ

## 期間, タイトル, 担当者
2022年4月-2023年2月, 自己注意によるコンテクスト抽出を用いた画像ノイズ除去手法, 有働和矢

## フォルダの説明
root/

├ 21.M.Imai-main/　　　　　　　　　　　　　:今井先輩のリポジトリをコピー

│　　　　　└　DN/　　　　　　　　　　　　:SAGTCNNでの生成ノイズ除去

│　　　　　└　Pub/修論　　　　　　　　　　:今井先輩の修論Tex

│　　　　　└  RN/　　　　　　　　　　　　　:GTCNNでのリアルノイズ除去（手を付けてない）

│　　　　　└　etc...　　　　　　　　　　　　:その他ファイル

├ Publication/　　　　　　　　　　　　　　　:卒論pdfとTex, および発表資料

## ソースコードの動かし方

雑な解説

1. このリポジトリをクローンする。
2. DN/がカレントディレクトリになるように移動。
3. NASの今井先輩のフォルダからDetasetsというフォルダをダウンロード(60GB)。
4. main.py　を動かす。その際、には以下の呪文を唱える。

・trainの場合

python main.py confs/GTCNN experiment.color=1 experiment.random_corp=True experiment.large_size=512 experiment.stride=512 dataset.test_set=[Set12,BSD68,Urban100] dataset.train_set=[DIV2K] dataset.val_set=[BSD400] dataset.test_root={testのパス} dataset.train_root={trainのパス} dataset.val_root={valのパス} experiment.epoch_num=600 experiment.sigma=50

・testの場合

python main.py confs/GTCNN experiment.color=1 experiment.test_only=True dataset.test_set=[Set12,BSD68,Urban100] dataset.test_root={testのパス} experiment.best_model={動かしたいモデルのパス}
