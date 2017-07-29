--Copyright (C) 2017  Kazuki Nakamae
--Released under MIT License
--license available in LICENSE file

--- MulLayerクラス（オブジェクト）
-- 乗算レイヤを実装
MulLayer={}
MulLayer.new = function()
    local obj = {}

    --メンバ変数
    obj.x = nil
    obj.y = nil

    -- @function self.forward()
    -- 順伝搬
    -- @param x 入力１{Type:number}
    -- @param x 入力２{Type:number}
    -- @return 乗算結果{Type:number}
    obj.forward = function(self, x, y)
        self.x = x
        self.y = y
        local out = x * y

        return out
    end

    -- @function self.backward()
    -- 逆伝搬
    -- @param dout 微分{Type:number}
    -- @return 各要素からの増分{Type:number}
    obj.backward = function(self, dout)
        local dx = dout * self.y
        local dy = dout * self.x

        return dx, dy
    end

    return obj
end

--- AddLayerクラス（オブジェクト）
-- 加算レイヤを実装
AddLayer={}
AddLayer.new = function()
    local obj = {}

    -- @function self.forward()
    -- 順伝搬
    -- @param x 入力１{Type:number}
    -- @param x 入力２{Type:number}
    -- @return 加算結果{Type:number}
    obj.forward = function(self, x, y)
        local out = x + y

        return out
    end

    -- @function self.backward()
    -- 逆伝搬
    -- @param dout 微分{Type:number}
    -- @return 各要素からの増分{Type:number}
    obj.backward = function(self, dout)
        local dx = dout * 1
        local dy = dout * 1

        return dx, dy
    end

    return obj
end