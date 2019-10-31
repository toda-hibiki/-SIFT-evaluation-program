import numpy as np
import cv2
from matplotlib import pyplot as plt
import time #時間取得用
import csv #精度出力用

size = 0.25 # サイズを選択(リサイズ用) 
MIN_MATCH_COUNT = 10 #最低限必要な点の数

with open('data'+str(size)+'.csv','w') as f: #CSVファイル(なければ作成)-インデント内で有効
    writer = csv.writer(f)
    #グラフ横軸用意
    writer.writerow(['画像番号','しきい値','kp1','kp2','matches','matches time','good','good time','RANSAC','RANSAC time','total time','RANSAC/good','RANSAC/matches'])

    for x in range(1, 31):#フォルダ数に応じて変動
    #画像の読み込み
      img1 = cv2.imread(f'./data/{x}/1.jpg',0) # queryImage
      img2 = cv2.imread(f'./data/{x}/2.jpg',0) # trainImage
    #画像のresize
      img1 = cv2.resize(img1, dsize=None, fx=size, fy=size)
      img2 = cv2.resize(img2, dsize=None, fx=size, fy=size)
      print(x)
      t1 = time.time()
    # Initiate SIFT detector
      sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
      kp1, des1 = sift.detectAndCompute(img1,None)
      kp2, des2 = sift.detectAndCompute(img2,None)
    #キーポイントの数出力
      print('kp1:',len(kp1))
      print('kp2:',len(kp2))

      t11 = time.time()
    
      bf = cv2.BFMatcher()
      matches = bf.knnMatch(des1,des2, k=2)
    
    #マッチング数
      print('matching:',len(matches))
      t2 = time.time()
    #マッチングにかかった時間(総当り)
      t = t2 - t11
      print(f"経過時間:{t}")

    # store all the good matches as per Lowe's ratio test.
      good = []
    #しきい値(0.5-0.9)
      for th in range(50,100,10):
        th = th * 0.01
        th = round(th,1)#四捨五入
        for m,n in matches:
          if m.distance < th*n.distance:
              good.append(m)
        print(th)#しきい値表示
        t3 = time.time()
        #しきい値処理の時間
        tt = t3 - t2
        print('good:',len(good))
        print(f"経過時間:{tt}")

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #十分な特徴点が得られない場合、配列内を0の配列にする(デフォルトはFalseが入り後でエラーになるため)
        else:
            #print ("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
            matchesMask = [0]*len(good)

        t4 = time.time()

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                         singlePointColor = None,
                         matchesMask = matchesMask, # draw only inliers
                         flags = 2)
        #RANSACでマッチングした数
        print('RANSAC:',matchesMask.count(1))
        #RANSACの処理時間
        ttt = t4 - t3
        print(f"経過時間:{ttt}")

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        t5 = time.time()
        tttt = t5 - t1
        print(f"終了までの経過時間:{tttt}")

        if len(good)>0:
          ratio1 = matchesMask.count(1) / len(good)
          ratio2 = matchesMask.count(1) / len(matches)
        #RANSACによる結果が0の場合精度を0とする
        else:
          ratio1 = 0
          ratio2 = 0
        #ファイル出力
        writer.writerow([x,th,len(kp1),len(kp2),len(matches),(t),len(good),(tt),matchesMask.count(1),(ttt),(tttt),(ratio1),(ratio2)])

        #plt.imshow(img3, 'gray'),plt.show()
        cv2.imwrite(f'result/{x}/{size}/{size}_result_{th}.png',img3)
