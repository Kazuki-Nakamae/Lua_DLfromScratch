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

---交差エントロピー誤差算出関数
-- テンソル同士の交差エントロピー誤差(-∑tilogyi)を求める
-- @param y 入力１、今回はNNが出力する確率リスト {Type:Tensor}
-- @param t 入力２、今回は正解ラベルリスト {Type:ByteTensor}
-- @return 交差エントロピー誤差 {Type:number}
function cross_entropy_error(y, t)
    local delta = 1e-7
    return -torch.cmul(t:double(), ( y:double() + delta ):log() ):sum()
end