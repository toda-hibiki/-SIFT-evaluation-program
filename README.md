# SIFT-evaluation-program

Overview
-

SIFTによる画像感のマッチングの精度評価用プログラムです。

SIFTによるマッチングの手順は以下のようになっています。  
*RANSAC適用、入力画像二枚の場合

1. 入力画像のインプット
1. それぞれの画像に対して、特徴点を取得する
1. 総当りマッチングで特徴点の対応付けを行う
1. 点同士の距離を閾値とし、信頼度の高い特徴点を選択
1. RANSACによる対応付けの交差除去
1. 結果画像の出力

以上の処理では、二度の特徴点選択を行っており、最初に取得した点がどれだけ残っているかが精度評価において重要です。  
*入力画像の二枚が同じ対象を撮影したものとする、回転や角度の違いは大丈夫です

そのため、今回は、3.でマッチングした点と4.及び5.で残った点の割合を比較することで精度評価とします。
また、各工程における、処理時間も取得します。　　
これにより、距離による閾値処理及びRANSACの有効性についての確認も期待します。
プログラムの説明はソースコードにコメントアウトで記述しています。

評価式:
ratio1 = (閾値処理により残った点数) / (最初にマッチングした点数)
ratio2 = (RANSACにより残った点数) / (最初にマッチングした点数)

実験条件
-

お好みで変更してください、デフォルトでは、 

入力画像:60枚(30ペア)  
*1から30までの名前のフォルダの中に"1.jpg"と".2.jpg"という名前で保存  
距離の閾値:0.5から0.9の間で0.1刻みで変動  
画像サイズ:お好み

Environment
-

- Python:3.6  
- OpenCV-python:3.4.2.17  
- OpenCV-contrib-python:3.4.2.17  
*Opencv-pythonとOpenCV-contribのバージョンは合わせる必要がある  
- numpy:最新  
- matplotlib:最新  

