import * as tf from '@tensorflow/tfjs'
import {
  decodeYoloPose,
  DecodeYoloPoseArgs,
  getModelInputShape,
  preprocessInput,
} from './common'
export * from './common'

export type PredictYoloPoseArgs = {
  model: tf.InferenceModel
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeYoloPoseArgs, 'output'> &
  ImageInput

export type ImageInput =
  | { pixels: Parameters<typeof tf.browser.fromPixels>[0] }
  | {
      /**
       * input shape: [height, width, channels] or [batch, height, width, channels]
       *
       * the pixel values should be in the range of [0, 255]
       */
      tensor: tf.Tensor
    }

/**
 * @description
 * output shape: [batch, features, boxes]
 *
 * features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 * - num_keypoints * 3: keypoint x, y, visibility
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 */
export async function predictYoloPose(args: PredictYoloPoseArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let result = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor
  })

  let output = (await result.array()) as number[][][]
  result.dispose()

  let predictions = await decodeYoloPose({
    ...args,
    output,
  })
  return predictions
}
