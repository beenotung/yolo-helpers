import * as tf from '@tensorflow/tfjs-node'
import { decodeBox, DecodeBoxArgs, decodeBoxSync } from './common'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
import { ImageInput } from '../tensorflow/node'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type DetectBoxArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeBoxArgs, 'output'> &
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
export async function detectBox(args: DetectBoxArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? await readFile(args.file) : null

  let result = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor
  })

  let output = (await result.array()) as number[][][]
  result.dispose()

  return await decodeBox({
    ...args,
    output,
  })
}

/**
 * Sync version of `detectBox`.
 */
export function detectBoxSync(args: DetectBoxArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor
    return result.arraySync() as number[][][]
  })

  return decodeBoxSync({
    ...args,
    output,
  })
}
