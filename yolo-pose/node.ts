import * as tf from '@tensorflow/tfjs-node'
import {
  decodePose,
  DecodePoseArgs,
  decodePoseSync,
  getModelInputShape,
  preprocessInput,
} from './common'
import { readFile } from 'fs/promises'
import { readFileSync } from 'fs'
export * from './common'

export async function loadYoloModel(
  /**
   * Can be with or without `/model.json`.
   *
   * Examples:
   * - "./saved_model/yolo11n-pose_web_model/model.json"
   * - "./saved_model/yolo11n-pose_web_model"
   * - "file://path/to/model.json"
   * - "http://localhost:8100/saved_models/yolo11n-pose_web_model"
   * - "https://domain.net/saved_models/yolo11n-pose_web_model/model.json"
   * */
  modelPath: string,
) {
  if (!modelPath.endsWith('/model.json')) {
    modelPath = modelPath + '/model.json'
  }
  if (!modelPath.includes('://')) {
    modelPath = 'file://' + modelPath
  }
  let model = await tf.loadGraphModel(modelPath)
  return model
}

export type DetectPoseArgs = {
  model: tf.InferenceModel
  /** used for image resize when necessary, auto inferred from model shape */
  input_shape?: {
    width: number
    height: number
  }
} & Omit<DecodePoseArgs, 'output'> &
  ImageInput

export type ImageInput =
  | {
      /** path to image file */
      file: string
    }
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
export async function detectPose(args: DetectPoseArgs) {
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

  return await decodePose({
    ...args,
    output,
  })
}

/**
 * Sync version of `detectPose`.
 */
export function detectPoseSync(args: DetectPoseArgs) {
  let { model } = args

  let input_shape = args.input_shape || getModelInputShape(model)

  let buffer = 'file' in args ? readFileSync(args.file) : null

  let output = tf.tidy(() => {
    let input = 'tensor' in args ? args.tensor : tf.node.decodeImage(buffer!)
    input = preprocessInput(input, input_shape)
    let result = model.predict(input, {}) as tf.Tensor
    return result.arraySync() as number[][][]
  })

  return decodePoseSync({
    ...args,
    output,
  })
}
