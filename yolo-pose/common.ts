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
  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * (args.visibility ? 3 : 2)

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
        offset + 2 < length;
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
  let {
    tf,
    num_classes,
    num_keypoints,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  } = args
  let length = 4 + num_classes + num_keypoints * (args.visibility ? 3 : 2)

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
        offset + 2 < length;
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
