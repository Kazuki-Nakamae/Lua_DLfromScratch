---テンソル間の積の計算関数
-- 各次元に対応したテンソルの積ABを行う。
-- @param A A (Type：Tensor)
-- @param B B (Type：Tensor)
-- @return AB (Type：torch.DoubleTensor)
function mulTensor(A, B)
    A = A:double()
    B = B:double()
    local AB = nil;
    if (A:dim() == 1 and B:dim() ~= 1) then
        --1Dvector・matrix
        AB = torch.mv(B:t(), A)
    else
        --others
        AB = A*B
    end
    return AB
end

---テンソルをスカラー変換関数
-- 1x1テンソルをスカラーへ変換する。
-- @param tensor 1x1テンソル (Type：Tensor)
-- @return スカラー (Type：number)
function tensor2scalar(tensor)
    return tensor[1]
end

---行反復行列生成関数
-- 入力されたベクトルをN行反復する行列を生成する
-- @param vector 一次元ベクトル (Type：Tensor)
-- @param iter 反復数 (Type：byte)
-- @return スカラー (Type：number)
function makeIterTensor(vector,iter)
    local iterTensor = torch.DoubleTensor(vector:size()[1],iter)
    local i = 0
    iterTensor:apply(function() --applyで各要素へ値を代入
        i = i + 1
        if vector:size()[1]<i then --ベクトルのインデックスを超えたら初期化
            i = 1
        end
        return vector[i]
    end)

    return iterTensor
end

---ランダムインデックス取得関数
-- 入力されたサイズ内で指定した数の整数を取得する
-- @param datasize データサイズ (Type：number)
-- @param getsize 取得サイズ (Type：number)
-- @param seed 乱数のシード (Type：number)
-- @return インデックスリスト (Type：long Tensor)
function getRandIndex(datasize,getsize,seed)
    torch.manualSeed(seed)
    --データサイズ分の整数のランダムな順列からgetsize分切り取る
    return torch.randperm(datasize):sub(1,getsize):long()
end

---任意要素アクセス関数
-- 入力されたテンソルに対して、次元分の指定テンソルの順序でアクセスした要素のリストを返す。
-- @param readTensor データサイズ (Type：tensor)
-- @param ... 指定テンソルを内包するテーブル (Type：table)
-- @return 取得した要素のリスト (Type：Tensor)
function getElement(readTensor,...)
    local args = {...}
    local elelist = {}
    for order = 1, args[1]:size()[1] do
        local indexlist = {}
        for dim = 1, readTensor:dim() do
            table.insert(indexlist,(args[dim])[order])
        end
        table.insert(elelist,readTensor[indexlist])
    end
    return torch.Tensor(elelist)
end

---最大値二値化関数
-- 各行に対して最大値の要素を1,それ以外の要素を0とする
-- @param t 二値化するテンソル (Type：tensor)
-- @return 二値化されたテンソル (Type：Tensor)
function thresMaxTensor(t)
    local t_value, t_indices = torch.max(t, 2)
    for row = 1, t:size()[1] do
        for col = 1, t:size()[2] do
            if col == t_indices[row][1] then
                t[row][col] = 1
            else
                t[row][col] = 0
            end
        end
    end
    return t
end