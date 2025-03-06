import * as tf from '@tensorflow/tfjs-node'
import { decodeSegment, DecodeSegmentArgs, decodeSegmentSync } from './common'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
import { ImageInput } from '../tensorflow/node'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type DetectSegmentArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeSegmentArgs, 'output'> &
  ImageInput

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
export async function detectSegment(args: DetectSegmentArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? await readFile(args.file) : null

  let result = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor[]
  })

  let output_boxes = result[0].array().then(data => {
    result[0].dispose()
    return data as number[][][]
  })

  let output_masks = result[1].array().then(data => {
    result[1].dispose()
    return data as number[][][][]
  })

  return await decodeSegment({
    ...args,
    input_shape,
    output_boxes: await output_boxes,
    output_masks: await output_masks,
  })
}

/**
 * Sync version of `detectSegment`.
 */
export function detectSegmentSync(args: DetectSegmentArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor[]
    let output_boxes = result[0].arraySync() as number[][][]
    let output_masks = result[1].arraySync() as number[][][][]
    return {
      output_boxes,
      output_masks,
    }
  })

  return decodeSegmentSync({
    ...args,
    input_shape,
    ...output,
  })
}
