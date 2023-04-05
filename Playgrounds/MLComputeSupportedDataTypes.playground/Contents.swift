import MLCompute

/*:
# Supported Data Types Playground
 
 This playground is a reference for the data types that MLCLayers support.
 Basically only _float32_ types are supported, so MimicTense should not allow
 anything, but _Float_ types in a neural network model (or else the backend compute engine needs
 to convert other data types to _float32_ values). The exception is that
 Apple's neural engine (ane) will support _float16_ data types, and the main reason
 for using MLCompute is for access to this device.
*/

/*:
 #### GPU supported data types
 */
if let gpu = MLCDevice.gpu() {
    MLCArithmeticLayer.supportsDataType(.boolean, on: gpu)
    MLCArithmeticLayer.supportsDataType(.float16, on: gpu)
    MLCArithmeticLayer.supportsDataType(.float32, on: gpu)
    MLCArithmeticLayer.supportsDataType(.int8, on: gpu)
    MLCArithmeticLayer.supportsDataType(.int32, on: gpu)
    MLCArithmeticLayer.supportsDataType(.int64, on: gpu)
    MLCArithmeticLayer.supportsDataType(.uint8, on: gpu)
    
    MLCFullyConnectedLayer.supportsDataType(.boolean, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.float16, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.float32, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.int8, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.int32, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.int64, on: gpu)
    MLCFullyConnectedLayer.supportsDataType(.uint8, on: gpu)
    
    MLCConvolutionLayer.supportsDataType(.boolean, on: gpu)
    MLCConvolutionLayer.supportsDataType(.float16, on: gpu)
    MLCConvolutionLayer.supportsDataType(.float32, on: gpu)
    MLCConvolutionLayer.supportsDataType(.int8, on: gpu)
    MLCConvolutionLayer.supportsDataType(.int32, on: gpu)
    MLCConvolutionLayer.supportsDataType(.int64, on: gpu)
    MLCConvolutionLayer.supportsDataType(.uint8, on: gpu)
}

/*:
 #### CPU supported data types
 */
let cpu = MLCDevice.cpu()
MLCArithmeticLayer.supportsDataType(.boolean, on: cpu)
MLCArithmeticLayer.supportsDataType(.float16, on: cpu)
MLCArithmeticLayer.supportsDataType(.float32, on: cpu)
MLCArithmeticLayer.supportsDataType(.int8, on: cpu)
MLCArithmeticLayer.supportsDataType(.int32, on: cpu)
MLCArithmeticLayer.supportsDataType(.int64, on: cpu)
MLCArithmeticLayer.supportsDataType(.uint8, on: cpu)

MLCFullyConnectedLayer.supportsDataType(.boolean, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.float16, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.float32, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.int8, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.int32, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.int64, on: cpu)
MLCFullyConnectedLayer.supportsDataType(.uint8, on: cpu)

MLCConvolutionLayer.supportsDataType(.boolean, on: cpu)
MLCConvolutionLayer.supportsDataType(.float16, on: cpu)
MLCConvolutionLayer.supportsDataType(.float32, on: cpu)
MLCConvolutionLayer.supportsDataType(.int8, on: cpu)
MLCConvolutionLayer.supportsDataType(.int32, on: cpu)
MLCConvolutionLayer.supportsDataType(.int64, on: cpu)
MLCConvolutionLayer.supportsDataType(.uint8, on: cpu)

/*:
 #### Apple's neural engine (ane) data types
 */
if let ane = MLCDevice.ane() {
    MLCArithmeticLayer.supportsDataType(.boolean, on: ane)
    MLCArithmeticLayer.supportsDataType(.float16, on: ane)
    MLCArithmeticLayer.supportsDataType(.float32, on: ane)
    MLCArithmeticLayer.supportsDataType(.int8, on: ane)
    MLCArithmeticLayer.supportsDataType(.int32, on: ane)
    MLCArithmeticLayer.supportsDataType(.int64, on: ane)
    MLCArithmeticLayer.supportsDataType(.uint8, on: ane)
    
    MLCFullyConnectedLayer.supportsDataType(.boolean, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.float16, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.float32, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.int8, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.int32, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.int64, on: ane)
    MLCFullyConnectedLayer.supportsDataType(.uint8, on: ane)

    MLCConvolutionLayer.supportsDataType(.boolean, on: ane)
    MLCConvolutionLayer.supportsDataType(.float16, on: ane)
    MLCConvolutionLayer.supportsDataType(.float32, on: ane)
    MLCConvolutionLayer.supportsDataType(.int8, on: ane)
    MLCConvolutionLayer.supportsDataType(.int32, on: ane)
    MLCConvolutionLayer.supportsDataType(.int64, on: ane)
    MLCConvolutionLayer.supportsDataType(.uint8, on: ane)
}


