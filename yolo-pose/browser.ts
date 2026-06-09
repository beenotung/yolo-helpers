import * as tf from '@tensorflow/tfjs'
import {
  decodePose,
  DecodePoseArgs,
  decodePoseSync,
  getPoseOutputTensor,
  PoseResult,
} from './common'
import { ImageInput } from '../tensorflow/browser'
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
export async function detectPose(args: DetectPoseArgs): Promise<PoseResult> {
  let { model } = args
  let output_format = args.output_format ?? model.pose_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let result = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
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
export function detectPoseSync(args: DetectPoseArgs): PoseResult {
  let { model } = args
  let output_format = args.output_format ?? model.pose_output_format

  let input_shape = args.input_shape || getModelInputShape(model)

  let output = tf.tidy(() => {
    let input =
      'tensor' in args ? args.tensor : tf.browser.fromPixels(args.pixels)
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
