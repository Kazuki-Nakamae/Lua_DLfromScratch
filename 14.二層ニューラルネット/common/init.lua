--Copyright (C) 2017  Kazuki Nakamae
--Released under MIT License
--license available in LICENSE file

require './function.lua'
require './gradient.lua'
require './exTorch.lua'

local common = {}

local help = {
softmax = [[softmax(x) -- Normalize input Tencor]],
cross_entropy_error = [[cross_entropy_error(y, t) -- Calculate the cross entropy between y and t]],
numerical_gradient = [[numerical_gradient(f, X) -- Calculate gradient of a given function f(X)]],
mulTensor = [[mulTensor(A, B) -- Calculate multiple of tensor A and tensor B]],
tensor2scalar = [[tensor2scalar(tensor) -- Convert tensor to scalar]],
makeIterTensor = [[makeIterTensor(vector,iter) -- Generate tensor whose rows are repeated]],
getRandIndex = [[getRandIndex(datasize,getsize,seed) -- Get random index of tensor which have elements of datasize]],
getElement = [[getElement(readTensor,...) -- Get value of readTensor[...]. When #{...}>=2, Access value of readTensor according to each value of elements in {...}]]
}

common.softmax = function(x)
  if not x then
    xlua.error('x must be supplied',
                'common.softmax', 
                help.softmax)
  end
  return softmax(x)
end

common.cross_entropy_error = function(y, t)
    if not y then
      xlua.error('y must be supplied',
                  'common.cross_entropy_error', 
                  help.cross_entropy_error)
    elseif not t then
      xlua.error('t must be supplied',
                  'common.cross_entropy_error', 
                  help.cross_entropy_error)
    end
    return cross_entropy_error(y, t)
end

common.numerical_gradient = function(f, X)
  if not f then
    xlua.error('f must be supplied', 
        'common.numerical_gradient', 
        help.numerical_gradient)
  elseif not X then
    xlua.error('X must be supplied', 
        'common.numerical_gradient', 
        help.numerical_gradient)
  end
  return numerical_gradient(f, X)
end

common.mulTensor = function(A, B)
  if not A then
    xlua.error('A must be supplied', 
        'common.mulTensor', 
        help.mulTensor)
  elseif not B then
    xlua.error('B must be supplied', 
        'common.mulTensor', 
        help.mulTensor)
  end
  return mulTensor(A, B)
end

common.tensor2scalar = function(tensor)
  if not A then
    xlua.error('tensor must be supplied', 
        'common.tensor2scalar', 
        help.tensor2scalar)
  end
  return tensor2scalar(tensor)
end

common.makeIterTensor = function(vector,iter)
  if not vector then
    xlua.error('vector must be supplied', 
        'common.makeIterTensor', 
        help.makeIterTensor)
  elseif not iter then
    xlua.error('iter must be supplied', 
        'common.makeIterTensor', 
        help.makeIterTensor)
  end
  return makeIterTensor(vector,iter)
end

common.getRandIndex = function(datasize,getsize,seed)
  if not datasize then
    xlua.error('datasize must be supplied', 
        'common.getRandIndex', 
        help.getRandIndex)
  elseif not getsize then
    xlua.error('getsize must be supplied', 
        'common.getRandIndex', 
        help.getRandIndex)
  elseif not seed then
    xlua.error('seed must be supplied', 
        'common.getRandIndex', 
        help.getRandIndex)
  end
  return getRandIndex(datasize,getsize,seed)
end

common.getElement = function(readTensor,...)
  if not readTensor then
    xlua.error('readTensor must be supplied', 
        'common.getElement', 
        help.getElement)
  end
  return getElement(readTensor,...)
end

return common
