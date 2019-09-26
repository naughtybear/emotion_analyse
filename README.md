# 智慧敘事情緒分析
>這個專案會根據你說出的句子判斷是否有心裡疾病或情緒問題，主要使用的架構是LSTM，並且以自為單位做word embedding

## 開發環境
* OS: Ubuntu 18.04 LTS
* GPU: GTX 1060 6G
* CPU: i5-6700
* RAM: 16G

## 主要使用套件
* keras
* numpy
* pickle

## 執行方法
### 第一步
執行pretreat.py檔案將資料進行預處理
### 第二步
執行keras_lsth.py檔案進行訓練
### 第三步
執行test_model.py檔案查看結果
