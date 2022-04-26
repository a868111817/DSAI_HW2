# DSAI_HW2

NCKU DSAI course homework 2 - Auto Trading

## Requirements
Using pipenv
```
pipenv install
```

```
pipenv shell
```

## Execution

```
python main.py --training training.csv --testing testing.csv --output output.csv --model rnn
```
## Dataset

* 下圖為dataset的close price走勢。
* 此次作業將dataset切分成training以及testing，各為8:2，且feature抓取 close price當作訓練特徵。

![](https://i.imgur.com/kM0Bfxs.png)

## Model

### RNN

* 在測試階段時，RNN model與實際close price之折線圖。
![](https://i.imgur.com/7rTwM5W.png)

### LSTM

* 在測試階段時，LSTM model與實際close price之折線圖。
![](https://i.imgur.com/1lw1KiF.png)

最後選用RNN當作最後輸出模型，為many-to-one的RNN模型，用前五天的資料去預測明天的股價。

![](https://i.imgur.com/kpcRKKV.png)

圖片來源: https://medium.com/swlh/a-technical-guide-on-rnn-lstm-gru-for-stock-price-prediction-bce2f7f30346

*  模型結構

![](https://i.imgur.com/BXeza8I.png)


## Trading Strategy

```
        # sell
        if (today > tomorrow ):
            if stock == 0:
                action.append(-1)
                stock -= 1
            elif stock == 1:
                action.append(-1)
                stock -= 1
            elif stock == -1:
                action.append(0)
        # hold
        elif (today == tomorrow ):
            action.append(0)
        # buy
        else:
            if stock == 0:
                action.append(1)
                stock += 1
            elif stock == 1:
                action.append(0)
            elif stock == -1:
                action.append(1)
                stock += 1
```

## Final testing

* 最終測試階段的購買策略
    1.前5天: 抓取前N天的平均當作明天的股價，再依照上述購買策略去決定是否購買/放空/不動作。
    2.剩餘天數: 依照當天之前五天的股價，用模型預測明天股價，再依照上述購買策略去決定是否購買/放空/不動作。

* 最終結果輸出成 **output.csv**

## Reference:

https://medium.com/swlh/a-technical-guide-on-rnn-lstm-gru-for-stock-price-prediction-bce2f7f30346