---勾配算出関数.
-- 入力値に対する多変数関数の勾配を求める
-- @param f 多変数関数 (Type：function)
-- @param x 入力値 (Type：Tensor ※1D Tensor)
-- @return 入力値に対する勾配の値 (Type：Tensor)
function _numerical_gradient_no_batch(f, x)
    local h = 1e-4 -- 0.0001
    local grad = x:clone():zero()

    for idx = 1, x:size()[1] do
        local tmp_val = x:float()[idx]
        x[idx] = tmp_val + h --一つの要素だけ動かす
        local fxh1 = f(x) -- f(x+h)

        x[idx] = tmp_val - h 
        local fxh2 = f(x) -- f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val -- 値を元に戻す
    end
    
    return grad
end

---勾配算出関数.
-- 入力値（複数）に対する多変数関数の勾配を求める
-- @param f 多変数関数 (Type：function)
-- @param x 入力値 (Type：Tensor)
-- @return 入力値に対する勾配の値 (Type：Tensor)
function numerical_gradient(f, X)
    if X:dim() == 1 then
        return _numerical_gradient_no_batch(f, X)
    else
        local grad = X:clone():zero()

        for idx = 1, X:size()[1] do
            grad[idx] = _numerical_gradient_no_batch(f, X[idx]) --1Dずつ勾配を計算
        end

        return grad
    end
end