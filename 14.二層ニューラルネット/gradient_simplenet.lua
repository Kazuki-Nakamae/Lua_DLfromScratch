--Copyright (C) 2017  Kazuki Nakamae
--Released under MIT License
--license available in LICENSE file

common = require './common'


--- simpleNetクラス（オブジェクト）
-- 単純なニューラルネットワークを生成する
-- @param isConstW 重みを固定 {Type:Bool}
simpleNet={}
simpleNet.new = function(isConstW)
        local obj={}

        --メンバ変数
        if isConstW then
            --原書で紹介されている重みを使用
            obj.W = torch.Tensor({{0.47355232, 0.9977393, 0.84668094}, {0.85557411, 0.03563661, 0.69422093}})
        else
            obj.W = torch.randn(2,3)
        end
        print("重みパラメータ : ")
        print(obj.W:double())

        --メソッド
        obj.predict = function(self, x)
            return common.mulTensor(x, self.W)
        end
        obj.loss = function(self, x, t)
            local z = self:predict(x)
            local y = common.softmax(z)
            local loss = common.cross_entropy_error(y, t)

            return loss
        end

        return obj
	  end

--入力
local x = torch.Tensor({0.6, 0.9})
--正解ラベル
local t = torch.Tensor({0, 0, 1})

--NNを作成（原書の重みを使用）
local net = simpleNet.new(true)

--推定と損失関数計算の確認
local p = net:predict(x)
print("推定値 : ")
print(p)
print("推定値の最大値とそのインデックス : ")
print(torch.max(p, 1))
print("損失関数の値 : "..net:loss(x, t).."\n")

--損失関数の勾配を計算
local f = (function(w) return net:loss(x, t) end)
local dW = common.numerical_gradient(f, net.W)

print("入力値での損失関数に対する重みの勾配")
print(dW)