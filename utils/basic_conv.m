
function tmp_layers = basic_conv(filterSize , numFilters ,   tmp_act , stride)
    tmp_layers = [
    convolution2dLayer(filterSize,numFilters , 'Stride' , stride , 'Padding' , 'same')
    batchNormalizationLayer
    tmp_act];
end