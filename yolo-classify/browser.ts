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
 * image features:
 *   - confidence of all classes
 *   - highest confidence, class_index
 *
 * The confidence are already normalized between 0 to 1, and sum up to 1.
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
