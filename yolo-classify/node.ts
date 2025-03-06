import * as tf from '@tensorflow/tfjs-node'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
import { ImageInput } from '../tensorflow/node'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
import { decodeClassify, DecodeClassifyArgs } from './common'
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
export async function classifyImage(args: ClassifyArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? await readFile(args.file) : null

  let result = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
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
 * Sync version of `detectBox`.
 */
export function classifyImageSync(args: ClassifyArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor
    return result.arraySync() as number[][]
  })

  return decodeClassify({
    ...args,
    output,
  })
}
