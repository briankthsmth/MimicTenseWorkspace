import Foundation
import MimicTense

NSSetUncaughtExceptionHandler { exception in
    print("Exception thrown: \(exception)")
}

let inference = try await Inference<Float> {
    InferenceDataSet(batchSize: 1) {
        InputData { Tensor<Float>([1, 1, 1]) }
        InputData { Tensor<Float>([1, 1, 1]) }
        InputData { Tensor<Float>([1, 1, 1]) }
    }
    Sequential<Float> {
        Arithmetic<Float>(.add) {
            Inputs {
                Tensor<Float>(shape: [1,3])
                Tensor<Float>(shape: [1,3])
            }
        }
        Arithmetic<Float>(.add) {
            Inputs {
                Tensor<Float>(shape: [1, 3])
            }
        }
    }
}
    .compile(device: .gpu)

for try await outputTensors in inference.outputStream {
    let outputTensor = outputTensors[0]
    print(outputTensor.shape)
    let outputVector = outputTensor.rank2Data?.joined().map { $0 } ?? []
    print(outputVector)
}

print("The end")
