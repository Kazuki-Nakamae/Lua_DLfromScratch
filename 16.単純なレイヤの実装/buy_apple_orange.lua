--Copyright (C) 2017  Kazuki Nakamae
--Released under MIT License
--license available in LICENSE file

require './layer_naive'

local apple = 100
local apple_num = 2
local orange = 150
local orange_num = 3
local tax = 1.1

-- layerの宣言
local mul_apple_layer = MulLayer.new()
local mul_orange_layer = MulLayer.new()
local add_apple_orange_layer = AddLayer.new()
local mul_tax_layer = MulLayer.new()

-- forward
local apple_price = mul_apple_layer:forward(apple, apple_num)  -- (1層)
local orange_price = mul_orange_layer:forward(orange, orange_num)  -- (2層)
local all_price = add_apple_orange_layer:forward(apple_price, orange_price)  -- (3層)
local price = mul_tax_layer:forward(all_price, tax)  -- (4層)

-- backward
local dprice = 1
local dall_price, dtax = mul_tax_layer:backward(dprice)  -- (4層)
local dapple_price, dorange_price = add_apple_orange_layer:backward(dall_price)  -- (3層)
local dorange, dorange_num = mul_orange_layer:backward(dorange_price)  -- (2層)
local dapple, dapple_num = mul_apple_layer:backward(dapple_price)  -- (1層)

-- 結果
print("price:"..price)
print("dApple:"..dapple)
print("dApple_num:"..dapple_num)
print("dOrange:"..dorange)
print("dOrange_num:"..dorange_num)
print("dTax:"..dtax)
