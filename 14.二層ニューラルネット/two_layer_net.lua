--Copyright (C) 2017  Kazuki Nakamae
--Released under MIT License
--license available in LICENSE file

common = require './common'

--[[***クラスの定義*******************************************************************]]

--- simpleNetクラス（オブジェクト）
-- 単純なニューラルネットワークを生成する
-- @param input_size 入力層のニューロン数{Type:Tensor}
-- @param hidden_size 隠れ層内のニューロン数{Type:Tensor}
-- @param output_size 出力層のニューロン数{Type:Tensor}
TwoLayerNet={}
TwoLayerNet.new = function(input_size, hidden_size, output_size, weight_init_std)

        --デフォルト引数
        if not weight_init_std then
            weight_init_std = 0.01
        end

        local obj = {}

        --メンバ変数
        --重みとバイアスを設定
        obj.param = {}
        obj.param['W1'] = weight_init_std * torch.randn(input_size, hidden_size)
        obj.param['b1'] = torch.Tensor(hidden_size):zero()
        obj.param['W2'] = weight_init_std * torch.randn(hidden_size, output_size)
        obj.param['b2'] = torch.Tensor(output_size):zero()

        -- @function self.predict()
        -- 推論処理を行う。
        -- @param x 入力データ{Type:Tensor}
        -- @return 出力：推定される確率{Type:Tensor}
        obj.predict = function(self, x)
            local W1, W2 = self.param['W1']:clone(), self.param['W2']:clone()
            local b1, b2 = self.param['b1']:clone(), self.param['b2']:clone()

            local a1 = common.mulTensor(x, W1) + common.makeIterTensor(b1,common.mulTensor(x, W1):size()[1])
            local z1 = common.sigmoid(a1)
            local a2 = common.mulTensor(z1,W2) + common.makeIterTensor(b2,common.mulTensor(z1,W2):size()[1])
            local y = common.softmax(a2)

            return y
        end

        -- @function self.loss()
        -- 入力データと教師データとの誤差を算出する
        -- @param x 入力データ{Type:Tensor}
        -- @param t 教師データ{Type:Tensor}
        -- @return 損失関数の出力値{Type:Tensor}
        obj.loss = function(self, x, t)
            local y = self:predict(x)
            local loss = common.cross_entropy_error(y, t)

            return loss
        end

        -- @function self.accuracy()
        -- 入力データと教師データとの間の精度を算出する
        -- @param x 入力データ{Type:Tensor}
        -- @param t 教師データ{Type:Tensor}
        -- @return 入力データと教師データとの間の精度{Type:number}
        obj.accuracy = function(self, x, t)
            local y = self:predict(x)
            local y_value, y_indices = torch.max(y, 2)
            local t_value, t_indices = torch.max(t, 2)
            local accuracy_cnt = 0
            --要素それぞれを比較して合計を計算
            accuracy_cnt = accuracy_cnt + torch.sum(torch.eq(y_indices:byte(), t_indices:byte()))
            return accuracy_cnt/x:size()[1]
        end

        -- @function self.numerical_gradient()
        -- 入力データ・教師データでの損失関数でのパラメータに対する勾配を計算する。
        -- @param x 入力データ{Type:Tensor}
        -- @param t 教師データ{Type:Tensor}
        -- @return 各パラメータの勾配{Type:table}
        obj.numerical_gradient = function(self, x, t)
            local loss_W = (function(w) return self:loss(x, t) end)

            local grads = {}
            print("W1の勾配計算...")
            grads['W1'] = common.numerical_gradient(loss_W, self.param['W1'])
            print("b1の勾配計算...")
            grads['b1'] = common.numerical_gradient(loss_W, self.param['b1'])
            print("W2の勾配計算...")
            grads['W2'] = common.numerical_gradient(loss_W, self.param['W2'])
            print("b2の勾配計算...")
            grads['b2'] = common.numerical_gradient(loss_W, self.param['b2'])
            print("計算完了")

            return grads
        end

        return obj
end



--[[***処理部*******************************************************************]]

torch.manualSeed(2017) --シード

--二層NNを作成
local net = TwoLayerNet.new(10, 100, 10)

--パラメータのサイズの確認
print(net.param['W1']:size())
print(net.param['b1']:size())
print(net.param['W2']:size())
print(net.param['b2']:size())

--テスト用のダミーデータ
local t = thresMaxTensor(torch.rand(100, 10)) --ダミーの正解ラベルデータ(100枚分)
local x = torch.rand(100, 10) --ダミーの入力データ(100枚分)

--推論処理の実行
print("predict")
print(net:predict(x))

--精度計算の実行
print("accuracy")
print(net:accuracy(x, t))

--勾配計算の実行
print("gradient")
local grads = net:numerical_gradient(x, t)
print("W1の勾配")
print(grads['W1'])
print("b1の勾配")
print(grads['b1'])
print("W2の勾配")
print(grads['W2'])
print("b2の勾配")
print(grads['b2'])