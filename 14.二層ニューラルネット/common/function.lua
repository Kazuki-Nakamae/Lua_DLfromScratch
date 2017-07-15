---ソフトマックス関数.
-- 入力値を確率に変換する
-- @param x 入力 (Type：torch.DoubleTensor)
-- @return 0-1 (Type：number)
function softmax(x)
    local c = torch.max(x)
    local exp_x = torch.exp(x - c)
    local sum_exp_x = torch.sum(exp_x)
    local y = exp_x / sum_exp_x

    return y
end

---シグモイド関数.
-- 入力の各成分に対するゲイン1の標準シグモイド関数（1/(1+exp(x))）の値を返す
-- @param x 入力 (Type：torch.DoubleTensor)
-- @return torch.DoubleTensor
function sigmoid(x)
    y = x:clone():fill(1)
    return torch.cdiv(y , (1 + torch.exp(-x)))
end

---ReLU(Rectified Linear Unit)関数.
-- 入力の各成分が0以下ならば0、0よりも高いならばそのままの値を返す
-- @param x 入力 (Type：torch.DoubleTensor)
-- @return torch.DoubleTensor
function relu(x)
    y = x:clone():fill(0)
    return torch.cmax(y, x)
end

---バッチ対応版交差エントロピー誤差算出関数
-- テンソル同士の交差エントロピー誤差(-∑tilogyi)を求める
-- @param y 入力１、今回はNNが出力する確率リスト {Type:Tensor}
-- @param t 入力２、今回は正解ラベルリスト {Type:ByteTensor}
-- @return 交差エントロピー誤差 {Type:number}
function cross_entropy_error(y, t)
    if y:dim() == 1 then
        y = y:resize(1,y:nElement())
    end
    local batch_size = y:size()[1]
    return -( y:log():cmul(t) ):sum() / batch_size
end