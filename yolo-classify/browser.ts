import * as tf from '@tensorflow/tfjs'
import { ClassifyResult, decodeClassify, DecodeClassifyArgs } from './common'
import { ImageInput } from '../tensorflow/browser'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type ClassifyArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeClassifyArgs, 'output'> &
  ImageInput

/**
 * @description
 * output shape: [batch, features, boxes]
 *
 * features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 */
export async function classifyImage(
  args: ClassifyArgs,
): Promise<ClassifyResult> {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let result = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor
  })

  let output = (await result.array()) as number[][]
  result.dispose()

  return decodeClassify({
    ...args,
    output,
  })
}

/**
 * Sync version of `detectClassify`.
 */
export function classifyImageSync(args: ClassifyArgs): ClassifyResult {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let output = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor
    return result.arraySync() as number[][]
  })

  return decodeClassify({
    ...args,
    output,
  })
}
