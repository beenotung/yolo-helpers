import * as tf from '@tensorflow/tfjs'
import {
  decodeSegment,
  DecodeSegmentArgs,
  decodeSegmentSync,
  SegmentResult,
} from './common'
import { ImageInput } from '../tensorflow/browser'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type DetectSegmentArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeSegmentArgs, 'output_boxes' | 'output_masks'> &
  ImageInput

/**
 * @description
 * output shape: [batch, features, boxes] and [batch, mask_height, mask_width, 32]
 *
 * first output features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 * - 32: instance confidence
 *
 * second output binary mask
 * - mask_height x mask_width: 0 for background, 1 for object
 * - 32: instance index
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 */
export async function detectSegment(
  args: DetectSegmentArgs,
): Promise<SegmentResult> {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let result = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
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
export function detectSegmentSync(args: DetectSegmentArgs): SegmentResult {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let output = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
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
