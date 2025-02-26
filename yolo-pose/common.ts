import type { InferenceModel, Tensor } from '@tensorflow/tfjs'
import type * as tf_type from '@tensorflow/tfjs'

export function checkPoseOutput(args: {
  input_shape: { width: number; height: number }
  /** e.g. `1` for single class */
  num_classes: number
  /** e.g. `17` for 17 keypoints */
  num_keypoints: number
  /** [batch, features, boxes] e.g. 1x17x8400 */
  output: number[][][]
}) {
  let {
    input_shape: { width, height },
    num_classes,
    num_keypoints,
  } = args
  let length = 4 + num_classes + num_keypoints * 3
  let num_boxes =
    (width / 8) * (height / 8) +
    (width / 16) * (height / 16) +
    (width / 32) * (height / 32)

  // e.g. 1x17x8400
  let batches = args.output
  if (!Array.isArray(batches)) {
    throw new Error('data must be 3D array')
  }
  if (batches[0].length === 0) {
    throw new Error('no a single batch')
  }
  if (!Array.isArray(batches[0])) {
    throw new Error('data must be 3D array')
  }
  if (batches[0].length !== length) {
    throw new Error('data[batch].length must be ' + length)
  }
  if (!Array.isArray(batches[0][0]) || typeof batches[0][0][0] !== 'number') {
    throw new Error('data must be 3D array')
  }
  for (let batch of batches) {
    for (let i = 0; i < length; i++) {
      if (batch[i].length !== num_boxes) {
        throw new Error('data[batch][${i}].length must be ' + num_boxes)
      }
    }
  }
}

export type BoundingBox = {
  /** center x of bounding box in px */
  x: number
  /** center y of bounding box in px */
  y: number
  /** width of bounding box in px */
  width: number
  /** height of bounding box in px */
  height: number
  /** class index with highest confidence */
  class_index: number
  /** confidence of the class with highest confidence */
  confidence: number
  /** confidence of all classes */
  all_confidences: number[]
}
export type Keypoint = {
  /** x of keypoint in px */
  x: number
  /** y of keypoint in px */
  y: number
  /** confidence of keypoint */
  visibility: number
}
export type BoundingBoxWithKeypoints = BoundingBox & {
  keypoints: Keypoint[]
}

/**
 * output shape: [batch, box]
 *
 * Array of batches, each containing array of detected bounding boxes
 * */
export type PoseResult = BoundingBoxWithKeypoints[][]

export type DecodePoseArgs = {
  /**
   * tensorflow runtime:
   * - browser: `import * as tf from '@tensorflow/tfjs'`
   * - nodejs: `import * as tf from '@tensorflow/tfjs-node'`
   */
  tf: typeof tf_type
  /** e.g. `1` for single class */
  num_classes: number
  /** e.g. `17` for 17 keypoints */
  num_keypoints: number
  /** batched predict result, e.g. 1x17x8400 */
  output: number[][][]
  /**
   * Number of boxes to return using non-max suppression.
   * If not provided, all boxes will be returned
   *
   * e.g. `1` for only selecting the bounding box with highest confidence.
   */
  maxOutputSize?: number
  /**
   * the threshold for deciding whether boxes overlap too much with respect to IOU.
   *
   * default: `0.5`
   */
  iouThreshold?: number
  /**
   * the threshold for deciding whether a box is a valid detection.
   *
   * default: `-Infinity`
   */
  scoreThreshold?: number
}

export async function decodePose(args: DecodePoseArgs): Promise<PoseResult> {
  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * 3

  // e.g. 1x17x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error('data[0].length must be ' + length)
  }

  let num_boxes = batches[0][0].length

  let result: PoseResult = []
  for (let batch of batches) {
    // e.g. 17x8400

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []
    let cls_indices: number[] = []
    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let box_score = batch[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let cls_score = batch[4 + i][i]
        if (cls_score > box_score) {
          box_score = cls_score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])
      scores.push(box_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      let box_indices_tensor = await tf.image.nonMaxSuppressionAsync(
        boxes,
        scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
      )
      box_indices = await box_indices_tensor.array()
      box_indices_tensor.dispose()
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences = []
      for (let i = 0; i < num_classes; i++) {
        all_confidences.push(batch[4 + i][box_index])
      }
      let keypoints = []
      for (let offset = 4 + num_classes; offset + 2 < length; offset += 3) {
        let x = batch[offset + 0][box_index]
        let y = batch[offset + 1][box_index]
        let visibility = batch[offset + 2][box_index]
        keypoints.push({ x, y, visibility })
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
        keypoints,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}

/**
 * Sync version of `decodePose`.
 */
export function decodePoseSync(args: DecodePoseArgs): PoseResult {
  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * 3

  // e.g. 1x17x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error('data[0].length must be ' + length)
  }

  let num_boxes = batches[0][0].length

  let result: PoseResult = []
  for (let batch of batches) {
    // e.g. 17x8400

    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []
    let cls_indices: number[] = []
    for (let box_index = 0; box_index < num_boxes; box_index++) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]

      let x1 = x - width / 2
      let y1 = y - height / 2
      let x2 = x + width / 2
      let y2 = y + height / 2

      let box_score = batch[4][box_index]
      let cls_index = 0
      for (let i = 1; i < num_classes; i++) {
        let cls_score = batch[4 + i][i]
        if (cls_score > box_score) {
          box_score = cls_score
          cls_index = i
        }
      }

      boxes.push([x1, y1, x2, y2])
      scores.push(box_score)
      cls_indices.push(cls_index)
    }

    let box_indices: number[]
    if (maxOutputSize) {
      box_indices = tf.tidy(() =>
        tf.image
          .nonMaxSuppression(
            boxes,
            scores,
            maxOutputSize,
            iouThreshold,
            scoreThreshold,
          )
          .arraySync(),
      )
    } else {
      box_indices = Array.from({ length: num_boxes }, (_, i) => i)
    }

    let bounding_boxes = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences = []
      for (let i = 0; i < num_classes; i++) {
        all_confidences.push(batch[4 + i][box_index])
      }
      let keypoints = []
      for (let offset = 4 + num_classes; offset + 2 < length; offset += 3) {
        let x = batch[offset + 0][box_index]
        let y = batch[offset + 1][box_index]
        let visibility = batch[offset + 2][box_index]
        keypoints.push({ x, y, visibility })
      }
      bounding_boxes.push({
        x,
        y,
        width,
        height,
        class_index,
        confidence,
        all_confidences,
        keypoints,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}

export function getModelInputShape(model: InferenceModel) {
  if (model.inputs.length !== 1) {
    throw new Error(
      `model should have 1 input, but having ${model.inputs.length} inputs`,
    )
  }
  let shape = model.inputs[0].shape
  if (!shape) {
    throw new Error(`model input shape is not defined`)
  }
  let height = shape[1]
  let width = shape[2]
  return { height, width }
}

/** normalize color and expand into shape: [batch, height, width, channels] */
export function preprocessInput(
  /**
   * input shape: [height, width, channels] or [batch, height, width, channels]
   *
   * the pixel values should be in the range of [0, 255]
   */
  input: Tensor,
  input_shape: { height: number; width: number },
) {
  // resize input to input_shape if necessary
  let input_height = input.shape[1]
  let input_width = input.shape[2]
  if (
    input_width !== input_shape.width ||
    input_height !== input_shape.height
  ) {
    input = input.resizeBilinear([input_shape.width, input_shape.height])
  }

  // expand batch dimension if input is 2D
  if (input.rank === 3) {
    input = input.expandDims()
  }

  // normalize to 0..1
  input = input.div(255.0)

  return input
}
