import * as tf from '@tensorflow/tfjs-node'
import {
  decodePose,
  DecodePoseArgs,
  decodePoseSync,
  getPoseOutputTensor,
} from './common'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
import { ImageInput } from '../tensorflow/node'
import { getModelInputShape, preprocessInput } from '../tensorflow/common'
export * from './common'

export type DetectPoseModel = tf.InferenceModel & {
  pose_output_format?: DecodePoseArgs['output_format']
}

export type DetectPoseArgs = {
  model: DetectPoseModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodePoseArgs, 'output'> &
  ImageInput

/**
 * box features:
 *   - x, y, width, height
 *   - highest confidence, class_index
 *   - keypoints
 *
 * keypoint features:
 *   - x, y, visibility
 *
 * The x, y, width, height are in pixel unit, NOT normalized in the range of [0, 1].
 * The the pixel units are scaled to the input_shape.
 *
 * The confidence are already normalized between 0 to 1.
 */
export async function detectPose(args: DetectPoseArgs) {
  let { model } = args
  let output_format = args.output_format ?? model.pose_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? await readFile(args.file) : null

  let result = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    return model.predict(input, {}) as tf.Tensor | tf.Tensor[]
  })

  let output_tensor = getPoseOutputTensor(result)
  let output = (await output_tensor.array()) as number[][][]
  if (Array.isArray(result)) {
    result.forEach(tensor => tensor.dispose())
  } else {
    result.dispose()
  }

  return await decodePose({
    ...args,
    output_format,
    output,
  })
}

/**
 * Sync version of `detectPose`.
 */
export function detectPoseSync(args: DetectPoseArgs) {
  let { model } = args
  let output_format = args.output_format ?? model.pose_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor | tf.Tensor[]
    let output_tensor = getPoseOutputTensor(result)
    let output = output_tensor.arraySync() as number[][][]
    if (Array.isArray(result)) {
      result.forEach(tensor => tensor.dispose())
    } else {
      result.dispose()
    }
    return output
  })

  return decodePoseSync({
    ...args,
    output_format,
    output,
  })
}
