import * as tf from '@tensorflow/tfjs-node'
import {
  decodeSegment,
  DecodeSegmentArgs,
  decodeSegmentSync,
  getSegmentOutputTensors,
} from './common'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
import { ImageInput } from '../tensorflow/node'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type DetectSegmentModel = tf.InferenceModel & {
  segment_output_format?: DecodeSegmentArgs['output_format']
}

export type DetectSegmentArgs = {
  model: DetectSegmentModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodeSegmentArgs, 'output_boxes' | 'output_masks'> &
  ImageInput

/**
 * boxes features:
 *   - x, y, width, height
 *   - highest confidence, class_index
 *   - mask coefficients for each channel
 *
 * mask features:
 * - [height, width, channel]: 0 for background, 1 for object
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
export async function detectSegment(args: DetectSegmentArgs) {
  let { model } = args
  let output_format = args.output_format ?? model.segment_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? await readFile(args.file) : null

  let result = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor[]
  })

  let output = getSegmentOutputTensors(result)
  let output_boxes = output.boxes.array().then(data => {
    output.boxes.dispose()
    return data as number[][][]
  })

  let output_masks = output.masks.array().then(data => {
    output.masks.dispose()
    return data as number[][][][]
  })

  result
    .filter(tensor => tensor !== output.boxes && tensor !== output.masks)
    .forEach(tensor => tensor.dispose())

  return await decodeSegment({
    ...args,
    output_format,
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
  let output_format = args.output_format ?? model.segment_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor[]
    let output = getSegmentOutputTensors(result)
    let output_boxes = output.boxes.arraySync() as number[][][]
    let output_masks = output.masks.arraySync() as number[][][][]
    result.forEach(tensor => tensor.dispose())
    return {
      output_boxes,
      output_masks,
    }
  })

  return decodeSegmentSync({
    ...args,
    output_format,
    input_shape,
    ...output,
  })
}
