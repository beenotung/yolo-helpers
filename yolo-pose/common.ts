import type * as tf_type from '@tensorflow/tfjs'
import { BoundingBox } from '../yolo-box/common'

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
 * Array of batches, each containing array of detected bounding boxes with keypoints
 * */
export type PoseResult = BoundingBoxWithKeypoints[][]

export type PoseOutputFormat = 'auto' | 'yolo' | 'yolo26' | 'end2end'

export type PoseOutputTensor = tf_type.Tensor | tf_type.Tensor[]

export function getPoseOutputTensor(output: PoseOutputTensor): tf_type.Tensor {
  if (!Array.isArray(output)) return output
  let firstRank3Tensor = output.find(tensor => tensor.shape.length === 3)
  if (firstRank3Tensor) return firstRank3Tensor
  throw new Error('pose output tensor must be rank 3')
}

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
  /** for each keypoints, are them {x,y} or {x,y,visibility} */
  visibility: boolean
  /** batched predict result, e.g. 1x17x8400 */
  output: number[][][]
  /**
   * Format of the model output.
   *
   * - `yolo`: [batch, features, boxes]
   * - `end2end`/`yolo26`: reserved for end-to-end pose exports
   * - `auto`: infer from output shape
   *
   * default: `auto`
   */
  output_format?: PoseOutputFormat
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

function getPoseOutputShape(output: number[][][]) {
  return `[${output.length},${output[0]?.length ?? 0},${output[0]?.[0]?.length ?? 0}]`
}

function createAllConfidences(
  num_classes: number,
  class_index: number,
  confidence: number,
): number[] {
  let all_confidences = new Array(num_classes).fill(0)
  if (class_index >= 0 && class_index < num_classes) {
    all_confidences[class_index] = confidence
  }
  return all_confidences
}

function getEnd2EndPoseLength(args: DecodePoseArgs) {
  return 6 + args.num_keypoints * (args.visibility ? 3 : 2)
}

function shouldDecodeEnd2EndPose(args: DecodePoseArgs): boolean {
  let format = args.output_format ?? 'auto'
  if (format === 'yolo26' || format === 'end2end') return true
  if (format === 'yolo') return false

  let firstBatch = args.output[0]
  if (!firstBatch || firstBatch.length === 0) return false
  return firstBatch[0]?.length === getEnd2EndPoseLength(args)
}

function assertSupportedPoseOutput(args: DecodePoseArgs, length: number) {
  let format = args.output_format ?? 'auto'
  if (format === 'yolo') return
  if (format === 'yolo26' || format === 'end2end') {
    throw new Error(
      `end-to-end pose output is not supported yet; got output shape ${getPoseOutputShape(args.output)}. Run npm run inspect:yolo26 -- path/to/model to inspect the real output shape.`,
    )
  }
  if (args.output[0]?.length === length) return
  throw new Error(
    `unsupported pose output shape ${getPoseOutputShape(args.output)}; expected [batch,${length},boxes] for yolo output. Run npm run inspect:yolo26 -- path/to/model if this is an end-to-end export.`,
  )
}

/**
 * tensorflow output: [batch, boxes, 6 + num_keypoints * 2/3]
 * box features:
 * - x1, y1, x2, y2, confidence, class_index
 * - keypoint x, y, visibility/confidence
 *
 * This is the end-to-end pose output used by YOLO26 pose exports.
 */
export async function decodeEnd2EndPose(
  args: DecodePoseArgs,
): Promise<PoseResult> {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let length = getEnd2EndPoseLength(args)

  let batches = args.output
  if (batches[0].length === 0) {
    return []
  }
  if (batches[0][0].length !== length) {
    throw new Error(`data[batch][box].length must be ${length}`)
  }

  let result: PoseResult = []
  for (let batch of batches) {
    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []

    for (let box_index = 0; box_index < batch.length; box_index++) {
      let box = batch[box_index]
      boxes.push([box[0], box[1], box[2], box[3]])
      scores.push(box[4])
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
      box_indices = Array.from({ length: batch.length }, (_, i) => i)
    }

    let bounding_boxes: BoundingBoxWithKeypoints[] = []
    for (let box_index of box_indices) {
      let box = batch[box_index]
      let [x1, y1, x2, y2, confidence, raw_class_index] = box
      let class_index = Math.trunc(raw_class_index)
      let width = x2 - x1
      let height = y2 - y1
      let keypoints: Keypoint[] = []
      for (
        let offset = 6;
        offset < length;
        offset += args.visibility ? 3 : 2
      ) {
        keypoints.push({
          x: box[offset],
          y: box[offset + 1],
          visibility: args.visibility ? box[offset + 2] : 1,
        })
      }
      bounding_boxes.push({
        x: x1 + width / 2,
        y: y1 + height / 2,
        width,
        height,
        class_index,
        confidence,
        all_confidences: createAllConfidences(
          num_classes,
          class_index,
          confidence,
        ),
        keypoints,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}

/**
 * Sync version of `decodeEnd2EndPose`.
 */
export function decodeEnd2EndPoseSync(args: DecodePoseArgs): PoseResult {
  let { tf, num_classes, maxOutputSize, iouThreshold, scoreThreshold } = args
  let length = getEnd2EndPoseLength(args)

  let batches = args.output
  if (batches[0].length === 0) {
    return []
  }
  if (batches[0][0].length !== length) {
    throw new Error(`data[batch][box].length must be ${length}`)
  }

  let result: PoseResult = []
  for (let batch of batches) {
    let boxes: [x1: number, y1: number, x2: number, y2: number][] = []
    let scores: number[] = []

    for (let box_index = 0; box_index < batch.length; box_index++) {
      let box = batch[box_index]
      boxes.push([box[0], box[1], box[2], box[3]])
      scores.push(box[4])
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
      box_indices = Array.from({ length: batch.length }, (_, i) => i)
    }

    let bounding_boxes: BoundingBoxWithKeypoints[] = []
    for (let box_index of box_indices) {
      let box = batch[box_index]
      let [x1, y1, x2, y2, confidence, raw_class_index] = box
      let class_index = Math.trunc(raw_class_index)
      let width = x2 - x1
      let height = y2 - y1
      let keypoints: Keypoint[] = []
      for (
        let offset = 6;
        offset < length;
        offset += args.visibility ? 3 : 2
      ) {
        keypoints.push({
          x: box[offset],
          y: box[offset + 1],
          visibility: args.visibility ? box[offset + 2] : 1,
        })
      }
      bounding_boxes.push({
        x: x1 + width / 2,
        y: y1 + height / 2,
        width,
        height,
        class_index,
        confidence,
        all_confidences: createAllConfidences(
          num_classes,
          class_index,
          confidence,
        ),
        keypoints,
      })
    }
    result.push(bounding_boxes)
  }
  return result
}

/**
 * tensorflow output: [batch, features, instances]
 * features:
 * - 4: x, y, width, height
 * - num_classes: class confidence
 * - num_keypoints * 3: keypoint x, y, visibility
 *
 * e.g. 1x17x8400 for 1 batch of 8400 instances with 4 keypoints and 1 class
 * (17 = 4 + 1 + 4 * 3)
 *
 * The confidence are already normalized between 0 to 1.
 */
export async function decodePose(args: DecodePoseArgs): Promise<PoseResult> {
  if (shouldDecodeEnd2EndPose(args)) {
    return decodeEnd2EndPose(args)
  }

  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * (args.visibility ? 3 : 2)
  assertSupportedPoseOutput(args, length)

  // e.g. 1x17x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error(`data[batch].length must be ${length}`)
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
        let cls_score = batch[4 + i][box_index]
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

    let bounding_boxes: BoundingBoxWithKeypoints[] = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch[4 + i][box_index]
      }
      let keypoints: Keypoint[] = []
      for (
        let offset = 4 + num_classes;
        offset < length;
        offset += args.visibility ? 3 : 2
      ) {
        let x = batch[offset + 0][box_index]
        let y = batch[offset + 1][box_index]
        let visibility = args.visibility ? batch[offset + 2][box_index] : 1
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
  if (shouldDecodeEnd2EndPose(args)) {
    return decodeEnd2EndPoseSync(args)
  }

  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * (args.visibility ? 3 : 2)
  assertSupportedPoseOutput(args, length)

  // e.g. 1x17x8400
  let batches = args.output

  if (batches[0].length === 0) {
    // no a single batch
    return []
  }
  if (batches[0].length !== length) {
    throw new Error(`data[batch].length must be ${length}`)
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
        let cls_score = batch[4 + i][box_index]
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

    let bounding_boxes: BoundingBoxWithKeypoints[] = []
    for (let box_index of box_indices) {
      let x = batch[0][box_index]
      let y = batch[1][box_index]
      let width = batch[2][box_index]
      let height = batch[3][box_index]
      let class_index = cls_indices[box_index]
      let confidence = batch[4 + class_index][box_index]
      let all_confidences: number[] = new Array(num_classes)
      for (let i = 0; i < num_classes; i++) {
        all_confidences[i] = batch[4 + i][box_index]
      }
      let keypoints: Keypoint[] = []
      for (
        let offset = 4 + num_classes;
        offset < length;
        offset += args.visibility ? 3 : 2
      ) {
        let x = batch[offset + 0][box_index]
        let y = batch[offset + 1][box_index]
        let visibility = args.visibility ? batch[offset + 2][box_index] : 1
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
